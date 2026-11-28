# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monitoring tools for Kubeflow Pipelines runs."""

import logging
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api, get_pipelines_client
from kubeflow_mcp.core.security import truncate_log_output, validate_k8s_name
from kubeflow_mcp.pipelines.api._common import (
    ARGO_NODE_ID_ANNOTATION,
    RUN_ID_LABEL,
    RUN_STATES,
    check_run_namespace,
    page_fields,
    pipelines_error,
    resolve_namespace,
    validate_display_name,
    validate_limit,
    validate_run_id,
)
from kubeflow_mcp.pipelines.types import run_node_ids, run_to_dict

logger = logging.getLogger(__name__)

MAX_TAIL_LINES = 1000
MAX_LOG_PODS = 10
# Total characters of log text across all pods in one response.
MAX_LOG_CHARS = 10_000
# Server-side byte cap per pod log read so a single long line cannot load an
# unbounded payload into memory before truncate_log_output() runs.
MAX_LOG_BYTES = 1024 * 1024
# Argo, which runs KFP tasks, puts the user's step in the "main" container.
_STEP_CONTAINER = "main"
# KFP annotates launcher (user step executor) pods with this runtime role.
_LAUNCHER_ROLE = "launcher"
_ROLE_ANNOTATION = "pipelines.kubeflow.org/runtime-role"
# Argo records the node's full path, e.g. "wf(0).root(0).train-model(0).executor(0)".
_NODE_NAME_ANNOTATION = "workflows.argoproj.io/node-name"
_NODE_INDEX_RE = re.compile(r"\(\d+\)$")


def _strip_index(name: str) -> str:
    """Drop Argo's iteration suffix: ``train-model(0)`` -> ``train-model``."""
    return _NODE_INDEX_RE.sub("", name)


def _task_from_node_name(node_name: str | None) -> str | None:
    """Return the user task a pod runs, from its Argo node path.

    KFP records launcher pods under a task called ``executor``; the node path names
    the task that owns it (``...train-model(0).executor(0)`` -> ``train-model``).
    """
    parts = [_strip_index(part) for part in (node_name or "").split(".") if part]
    if not parts:
        return None
    if parts[-1] == "executor" and len(parts) > 1:
        return parts[-2]
    return parts[-1]


def _resolve_run_pods(run_id: str, run: Any, namespace: str) -> list[tuple[str | None, str, bool]]:
    """Map a run's Argo node IDs to real K8s pod names.

    KFP v2 / Argo 3.4+ uses ``POD_NAMES=v2``, where the pod names include the
    template name and differ from the Argo node IDs stored in the run's task details.
    This function lists live pods by the ``pipeline/runid`` label and maps them back
    to the run's recorded node IDs, naming each live pod after the task its Argo node
    path names (launcher pods are recorded under a task called ``executor``).

    Returns ``(task_display_name, pod, missing)`` triples. Workload (launcher) pods
    come before internal pods (drivers, cleaned-up IDs), so a DAG's internal entries
    never exhaust the read budget before the user's steps get read. ``missing`` is
    True for a recorded node no live pod of the run carries: once the run's pods
    were listed, such a node is a cleaned-up pod or a DAG node with no pod at all.
    """
    node_entries = run_node_ids(run)
    if not node_entries:
        return []

    try:
        api = get_core_v1_api()
        pod_list = api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"{RUN_ID_LABEL}={run_id}",
            _request_timeout=K8S_TIMEOUT,
        )
    except Exception as e:
        logger.warning("Could not list pods for run %s: %s", run_id, e)
        raise

    # Index live pods by their Argo node ID annotation. A single node ID can map to
    # multiple pods (driver + launcher); keep the launcher apart so the user's step
    # executor wins over the internal driver when a node ID maps to both.
    launcher_pods: dict[str, tuple[str, str | None]] = {}
    any_pods: dict[str, tuple[str, str | None]] = {}
    for pod in getattr(pod_list, "items", []) or []:
        meta = getattr(pod, "metadata", None)
        if meta is None:
            continue
        pod_name = getattr(meta, "name", None)
        if not pod_name:
            continue
        annotations = getattr(meta, "annotations", None) or {}
        node_id = annotations.get(ARGO_NODE_ID_ANNOTATION) or pod_name
        live = (pod_name, _task_from_node_name(annotations.get(_NODE_NAME_ANNOTATION)))
        if annotations.get(_ROLE_ANNOTATION, "") == _LAUNCHER_ROLE:
            launcher_pods.setdefault(node_id, live)
        any_pods.setdefault(node_id, live)
        any_pods.setdefault(pod_name, live)

    workload: list[tuple[str | None, str, bool]] = []
    unresolved: list[tuple[str | None, str, bool]] = []
    internal: list[tuple[str | None, str, bool]] = []
    for task_name, node_id in node_entries:
        live = launcher_pods.get(node_id)
        if live is not None:
            workload.append((live[1] or task_name, live[0], False))
            continue
        live = any_pods.get(node_id)
        if live is not None:
            internal.append((live[1] or task_name, live[0], False))
        else:
            # No live pod carries this node ID. Keep the recorded ID so it is reported
            # instead of silently skipped. When the run's pods could be listed it is
            # known to be gone; otherwise (older KFP, where recorded IDs are pod names)
            # it is still read.
            unresolved.append((task_name, node_id, bool(any_pods)))
    return [*workload, *unresolved, *internal]


def list_pipeline_runs(
    pipeline: str | None = None,
    experiment: str | None = None,
    status: str | None = None,
    limit: int = 50,
    page_token: str = "",
    namespace: str | None = None,
) -> dict[str, Any]:
    """List pipeline runs, newest pages first, optionally filtered.

    Args:
        pipeline: Only runs of this pipeline (display name).
        experiment: Only runs in this experiment (display name).
        status: Only runs in this state: ``pending``, ``running``, ``succeeded``,
            ``failed``, ``skipped``, ``canceling``, ``canceled`` or ``paused``.
        limit: Runs per page, 1 to 200. Defaults to 50.
        page_token: ``next_page_token`` from a previous call, to read the next page.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing ``runs`` (``run_id``, ``name``, ``state``,
        ``created_at``, ``finished_at``, ``error``, ...), ``next_page_token``,
        ``has_more`` and ``total``. The pipeline filter is applied after the
        server returns a page, so a page can hold fewer runs than ``limit``
        while ``has_more`` is still true. Runs whose namespace cannot be
        verified against the requested one are dropped (fail-closed) and the
        count is reported in ``runs_filtered``.
    """
    for value, field in ((pipeline, "pipeline"), (experiment, "experiment")):
        if value is not None:
            err = validate_display_name(value, field)
            if err is not None:
                return err.model_dump()

    normalized_status = status.strip().lower() if isinstance(status, str) else status
    if normalized_status is not None and normalized_status not in RUN_STATES:
        return ToolError(
            error=f"Unsupported status '{status}'",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"supported_statuses": sorted(RUN_STATES)},
        ).model_dump()

    limit_err = validate_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        response = client.list_runs(
            pipeline=pipeline,
            experiment=experiment,
            status=normalized_status,
            page_token=page_token or "",
            page_size=limit,
        )
        # The same fail-closed namespace verification get_pipeline_run applies: the
        # SDK can return runs whose experiment belongs to another namespace (or carries
        # none at all on standalone KFP), and listing must not expose what the direct
        # read would deny. Unverifiable runs are dropped, never reported.
        runs: list[dict[str, Any]] = []
        filtered = 0
        for raw_run in getattr(response, "runs", None) or []:
            if check_run_namespace(client, raw_run, ns) is not None:
                filtered += 1
                continue
            runs.append(run_to_dict(raw_run))
        data: dict[str, Any] = {"runs": runs, "namespace": ns, **page_fields(response)}
        if filtered:
            data["runs_filtered"] = filtered
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        return pipelines_error(e, "list pipeline runs").model_dump()


def get_pipeline_run(run_id: str, namespace: str | None = None) -> dict[str, Any]:
    """Get a pipeline run: its state, error, parameters and task states.

    Args:
        run_id: Run ID (a UUID) from ``list_pipeline_runs()`` or ``create_pipeline_run()``.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing the run fields plus ``parameters``, ``tasks``
        (``name``, ``state``, ``pod_name``, ``error``) and ``next_steps``.
    """
    id_err = validate_run_id(run_id)
    if id_err is not None:
        return id_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        run = client.get_run(run_id)
        run_ns_err = check_run_namespace(client, run, ns)
        if run_ns_err is not None:
            return run_ns_err.model_dump()
        data = run_to_dict(run, include_details=True)
        state = data.get("state")
        if state == "failed":
            data["next_steps"] = [
                f"get_pipeline_logs(run_id='{run_id}') to read the failed task's output",
                f"retry_pipeline_run(run_id='{run_id}') to preview a retry",
            ]
        elif state in ("pending", "running"):
            data["next_steps"] = [f"get_pipeline_run(run_id='{run_id}') again to follow it"]
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        return pipelines_error(e, f"get pipeline run '{run_id}'").model_dump()


def _decode_pod_log_response(response: Any, limit_bytes: int) -> tuple[str, int]:
    """Decode a K8s pod log response (string mock, stream, or raw HTTPResponse)."""
    if response is None:
        return "", 0
    if isinstance(response, str):
        return response, len(response.encode("utf-8"))
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace"), len(response)
    try:
        read_fn = getattr(response, "read", None)
        if callable(read_fn):
            chunks: list[bytes] = []
            total = 0
            while total < limit_bytes:
                chunk = read_fn(min(8192, limit_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            body = b"".join(chunks)
        else:
            body = response.data or b""
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
        release_conn = getattr(response, "release_conn", None)
        if callable(release_conn):
            release_conn()
    return body.decode("utf-8", errors="replace"), len(body)


def _read_container_log(
    api: Any,
    pod: str,
    namespace: str,
    tail_lines: int,
    limit_bytes: int,
    container: str | None = None,
) -> str:
    """Read the end of one container's log, bounded server-side by ``limit_bytes``."""
    lines = tail_lines
    while True:
        kwargs: dict[str, Any] = {
            "name": pod,
            "namespace": namespace,
            "tail_lines": lines,
            "limit_bytes": limit_bytes,
            "_preload_content": False,
            "_request_timeout": K8S_TIMEOUT,
        }
        if container is not None:
            kwargs["container"] = container
        response = api.read_namespaced_pod_log(**kwargs)
        text, byte_len = _decode_pod_log_response(response, limit_bytes)
        if byte_len < limit_bytes:
            return text
        if lines == 1:
            return (
                f"{text}\n... (incomplete tail: single log line reached "
                f"{limit_bytes} byte read limit; suffix omitted)"
            )
        lines = max(1, lines // 4)


def _read_pod_log(pod: str, namespace: str, tail_lines: int) -> str:
    api = get_core_v1_api()
    try:
        return _read_container_log(
            api, pod, namespace, tail_lines, MAX_LOG_BYTES, container=_STEP_CONTAINER
        )
    except Exception as e:
        # Pods without a "main" container (e.g. KFP driver pods) answer 400; read the
        # pod's default container instead.
        if getattr(e, "status", None) != 400:
            raise
        logger.debug(
            "Pod %s has no %r container (HTTP 400); reading its default container",
            pod,
            _STEP_CONTAINER,
        )
        return _read_container_log(api, pod, namespace, tail_lines, MAX_LOG_BYTES)


def _read_pod_logs(
    pods: list[tuple[str | None, str, bool]], namespace: str, tail_lines: int
) -> list[dict[str, Any]]:
    """Read each pod's log tail, sharing the character budget among the pods read."""
    # Entries already known to be gone take no share of the budget.
    readable = sum(1 for _, _, missing in pods if not missing)
    budget = MAX_LOG_CHARS // max(1, readable)
    results: list[dict[str, Any]] = []
    for name, pod, missing in pods:
        entry: dict[str, Any] = {"task": name, "pod": pod}
        if missing:
            entry["error"] = "pod no longer exists"
        elif validate_k8s_name(pod, "pod") is not None:
            entry["error"] = "invalid pod name"
        else:
            try:
                log = _read_pod_log(pod, namespace, tail_lines) or ""
                entry["logs"] = truncate_log_output(log, budget)
            except Exception as e:
                status = getattr(e, "status", None)
                entry["error"] = (
                    "pod no longer exists" if status == 404 else f"could not read logs: {e}"
                )
                logger.debug("Reading logs of pod %s failed: %s", pod, e)
        results.append(entry)
    return results


def get_pipeline_logs(
    run_id: str,
    task: str | None = None,
    tail_lines: int = 200,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get the logs of a pipeline run's task pods.

    Pods are taken from the run's own task details, so only pods that belong to this
    run are read. Logs of pods that were already cleaned up are reported as missing.

    Args:
        run_id: Run ID (a UUID).
        task: Only pods of the task with this name; ``train`` and ``train(0)`` both
            match. Reads up to 10 pods when omitted.
        tail_lines: Last lines to read per pod, 1 to 1000. Defaults to 200.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing ``pods`` (``task``, ``pod``, ``logs`` or ``error``),
        ``pods_truncated`` and ``tasks`` (task names, for the ``task`` filter).
    """
    id_err = validate_run_id(run_id)
    if id_err is not None:
        return id_err.model_dump()
    if task is not None:
        task_err = validate_display_name(task, "task")
        if task_err is not None:
            return task_err.model_dump()
    if (
        not isinstance(tail_lines, int)
        or isinstance(tail_lines, bool)
        or not 1 <= tail_lines <= MAX_TAIL_LINES
    ):
        return ToolError(
            error=f"tail_lines must be between 1 and {MAX_TAIL_LINES}, got {tail_lines}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        run = client.get_run(run_id)
        run_ns_err = check_run_namespace(client, run, ns)
        if run_ns_err is not None:
            return run_ns_err.model_dump()
    except Exception as e:
        return pipelines_error(e, f"get pipeline run '{run_id}'").model_dump()

    try:
        all_pods = _resolve_run_pods(run_id, run, ns)
    except Exception as e:
        return pipelines_error(e, f"resolve pods for run '{run_id}'").model_dump()
    task_names = sorted({name for name, _, _ in all_pods if name})
    wanted = _strip_index(task) if task is not None else None
    pods = [
        entry
        for entry in all_pods
        if wanted is None or (entry[0] is not None and _strip_index(entry[0]) == wanted)
    ]
    # The same real pod can be recorded under more than one node entry (a task's own
    # entry and its parents child list); read each pod once.
    seen: set[str] = set()
    deduped: list[tuple[str | None, str, bool]] = []
    for entry in pods:
        if entry[1] in seen:
            continue
        seen.add(entry[1])
        deduped.append(entry)
    pods = deduped
    if task is not None and not pods:
        return ToolError(
            error=f"No pods recorded for task '{task}' in run '{run_id}'",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            details={"tasks": task_names},
        ).model_dump()

    return ToolResponse(
        data={
            "run_id": run_id,
            "state": run_to_dict(run).get("state"),
            "pods": _read_pod_logs(pods[:MAX_LOG_PODS], ns, tail_lines),
            "pods_truncated": len(pods) > MAX_LOG_PODS,
            "tasks": task_names,
        }
    ).model_dump()
