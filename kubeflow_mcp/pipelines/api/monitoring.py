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
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api, get_pipelines_client
from kubeflow_mcp.core.security import truncate_log_output, validate_k8s_name
from kubeflow_mcp.pipelines.api._common import (
    RUN_STATES,
    check_run_namespace,
    page_fields,
    pipelines_error,
    resolve_namespace,
    validate_display_name,
    validate_limit,
    validate_run_id,
)
from kubeflow_mcp.pipelines.types import run_pod_names, run_to_dict

logger = logging.getLogger(__name__)

MAX_TAIL_LINES = 1000
MAX_LOG_PODS = 10
# Total characters of log text across all pods in one response.
MAX_LOG_CHARS = 10_000
# Argo, which runs KFP tasks, puts the user's step in the "main" container.
_STEP_CONTAINER = "main"


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
        while ``has_more`` is still true.
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
        response = get_pipelines_client(ns).list_runs(
            pipeline=pipeline,
            experiment=experiment,
            status=normalized_status,
            page_token=page_token or "",
            page_size=limit,
        )
        runs = [run_to_dict(r) for r in getattr(response, "runs", None) or []]
        return ToolResponse(
            data={"runs": runs, "namespace": ns, **page_fields(response)}
        ).model_dump()
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


def _read_pod_log(pod: str, namespace: str, tail_lines: int) -> str:
    api = get_core_v1_api()
    try:
        return api.read_namespaced_pod_log(
            name=pod,
            namespace=namespace,
            container=_STEP_CONTAINER,
            tail_lines=tail_lines,
            _request_timeout=K8S_TIMEOUT,
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
        return api.read_namespaced_pod_log(
            name=pod, namespace=namespace, tail_lines=tail_lines, _request_timeout=K8S_TIMEOUT
        )


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
        task: Only pods of the task with this name. Reads up to 10 pods when omitted.
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

    all_pods = run_pod_names(run)
    task_names = sorted({name for name, _ in all_pods if name})
    pods = [(name, pod) for name, pod in all_pods if task is None or name == task]
    if task is not None and not pods:
        return ToolError(
            error=f"No pods recorded for task '{task}' in run '{run_id}'",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            details={"tasks": task_names},
        ).model_dump()

    budget = MAX_LOG_CHARS // max(1, min(len(pods), MAX_LOG_PODS))
    results: list[dict[str, Any]] = []
    for name, pod in pods[:MAX_LOG_PODS]:
        entry: dict[str, Any] = {"task": name, "pod": pod}
        if validate_k8s_name(pod, "pod") is not None:
            entry["error"] = "invalid pod name"
            results.append(entry)
            continue
        try:
            entry["logs"] = truncate_log_output(_read_pod_log(pod, ns, tail_lines) or "", budget)
        except Exception as e:
            status = getattr(e, "status", None)
            entry["error"] = (
                "pod no longer exists" if status == 404 else f"could not read logs: {e}"
            )
            logger.debug("Reading logs of pod %s failed: %s", pod, e)
        results.append(entry)

    return ToolResponse(
        data={
            "run_id": run_id,
            "state": run_to_dict(run).get("state"),
            "pods": results,
            "pods_truncated": len(pods) > MAX_LOG_PODS,
            "tasks": task_names,
        }
    ).model_dump()
