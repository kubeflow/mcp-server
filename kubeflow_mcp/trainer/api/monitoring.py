# Copyright The Kubeflow Authors
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

"""Monitoring tools for training job logs and events."""

import logging
from collections import deque
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.failures import extract_failure_hint
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import (
    get_core_v1_api,
    get_trainer_client_for_namespace,
    get_trainer_effective_namespace,
)
from kubeflow_mcp.core.security import (
    check_namespace_allowed,
    truncate_log_output,
    validate_k8s_name,
)
from kubeflow_mcp.trainer.api.kueue import get_trainjob_queue_status

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 1000
MAX_LINE_CHARS = 2000
MAX_CURSOR_CHARS = 9000  # Response size cap: max characters per cursor-mode page
MAX_EVENT_LIMIT = 500
MAX_WAIT_TIMEOUT = 3600
MIN_POLLING_INTERVAL = 1
_TARGET_STATUS_ALIASES = {"Succeeded": "Complete"}
_VALID_TARGET_STATUSES = frozenset({"Complete", "Failed", "Running", "Created"})


def _is_pod_for_step(pod: Any, step: str) -> bool:
    """Return whether a JobSet pod corresponds to the requested TrainJob step."""
    labels = getattr(pod.metadata, "labels", None) or {}
    replicated_job = labels.get("jobset.sigs.k8s.io/replicatedjob-name")
    job_index = labels.get("jobset.sigs.k8s.io/job-index")
    if replicated_job == step:
        return True
    return (
        replicated_job is not None
        and job_index is not None
        and f"{replicated_job}-{job_index}" == step
    )


def _collect_log_lines(
    client: Any,
    name: str,
    step: str,
    namespace: str | None,
    cursor_mode: bool,
) -> tuple[list[str], bool]:
    """Fetch log lines and attempt the previous-container fallback if empty.

    Returns (log_lines, fallback_was_used).
    """
    if cursor_mode:
        # No deque cap: cursor mode must be able to address any line index.
        log_lines: list[str] = list(client.get_job_logs(name=name, step=step, follow=False))
    else:
        log_lines = list(
            deque(client.get_job_logs(name=name, step=step, follow=False), maxlen=MAX_LOG_LINES)
        )

    if log_lines:
        return log_lines, False

    # Primary path empty — try previous-container (crash) logs.
    try:
        eff_ns = get_trainer_effective_namespace(namespace)
        v1 = get_core_v1_api()
        pods = v1.list_namespaced_pod(
            namespace=eff_ns,
            label_selector=f"training.kubeflow.org/trainjob-name={name}",
        )
        for pod in pods.items:
            if not _is_pod_for_step(pod, step):
                continue
            try:
                raw = v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=eff_ns,
                    previous=True,
                    tail_lines=MAX_LOG_LINES,
                )
                if raw:
                    log_lines.extend(raw.splitlines())
            except Exception as e:
                logger.debug(
                    "Failed to read previous pod logs for pod %s/%s: %s",
                    eff_ns,
                    pod.metadata.name,
                    e,
                )
    except Exception as e:
        logger.debug(
            "Previous-log fallback failed for job %s (namespace=%s): %s",
            name,
            namespace,
            e,
        )
    return log_lines, True


def _cursor_response(
    name: str,
    step: str,
    log_lines: list[str],
    since_line: int,
    effective_max: int,
) -> dict[str, Any]:
    """Build the cursor-mode response dict. Called only when since_line is set."""
    total = len(log_lines)

    if since_line == total:
        return ToolResponse(
            data={
                "job": name,
                "step": step,
                "logs": "",
                "lines": 0,
                "next_offset": since_line,
                "total_lines": total,
            }
        ).model_dump()

    if since_line > total:
        return ToolResponse(
            data={
                "job": name,
                "step": step,
                "logs": "",
                "lines": 0,
                "next_offset": 0,
                "total_lines": total,
                "log_reset": True,
            }
        ).model_dump()

    page = log_lines[since_line : since_line + effective_max]
    out_lines: list[str] = []
    char_budget = MAX_CURSOR_CHARS
    for line in page:
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "...[truncated]"
        cost = len(line) + (1 if out_lines else 0)
        if out_lines and cost > char_budget:
            break
        out_lines.append(line)
        char_budget -= cost
        if char_budget <= 0:
            break

    # Skip truncate_log_output: per-line cap + char_budget are the backstop;
    # truncate_log_output would eat lines silently while next_offset already advanced.
    data: dict[str, Any] = {
        "job": name,
        "step": step,
        "logs": "\n".join(out_lines),
        "lines": len(out_lines),
        "next_offset": since_line + len(out_lines),
        "total_lines": total,
    }
    hint = extract_failure_hint("\n".join(log_lines[-MAX_LOG_LINES:]))
    if hint:
        data["failure_hint"] = hint
        data["next_steps"] = [
            f"Detected {hint['category']}: {hint['suggestion']}",
            "Read trainer://guides/troubleshooting for detailed fixes",
        ]
    return ToolResponse(data=data).model_dump()


def get_training_logs(
    name: str,
    step: str = "node-0",
    namespace: str | None = None,
    follow: bool = False,
    since_line: int | None = None,
    max_lines: int | None = None,
) -> dict[str, Any]:
    """Get pod logs from a training job.

    Args:
        name: TrainJob name.
        step: Node/worker to get logs from. Defaults to ``node-0``.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        follow: Stream logs continuously (not supported in MCP context).
        since_line: Start of cursor window. When set, enables cursor mode: returns
            lines ``[since_line : since_line + max_lines]`` and adds ``next_offset``
            and ``total_lines`` to the response. Echo ``next_offset`` back as
            ``since_line`` on the next call to poll incrementally. A page may have
            fewer than ``max_lines`` lines due to the char budget; use ``next_offset``
            to advance, not page length — a short page does not mean end of log.
        max_lines: Lines per page. Tail mode default: 1000 (unchanged). Cursor mode
            default: 200. Clamped to [1, 1000].

    Returns:
        dict: Response containing:

        - ``job`` (str): Job name
        - ``step`` (str): Node name
        - ``logs`` (str): Log output
        - ``lines`` (int): Number of log lines in this response
        - ``next_offset`` (int): *Cursor mode only.* Echo as ``since_line`` next call.
        - ``total_lines`` (int): *Cursor mode only.* Total lines collected.
        - ``log_reset`` (bool): *Cursor mode only.* True when ``since_line > total_lines``;
          restart from 0.

    Raises:
        ToolError: If job not found (``RESOURCE_NOT_FOUND``) or cursor used with crash
            logs (``VALIDATION_ERROR``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if since_line is not None and since_line < 0:
        return ToolError(
            error="since_line must be >= 0",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    _effective_max = (
        max(1, min(max_lines, MAX_LOG_LINES))
        if max_lines is not None
        else (200 if since_line is not None else MAX_LOG_LINES)
    )

    try:
        if follow:
            return ToolResponse(
                data={
                    "job": name,
                    "step": step,
                    "logs": "Streaming not supported in MCP context. Use follow=False.",
                    "lines": 1,
                }
            ).model_dump()

        client = get_trainer_client_for_namespace(namespace)
        log_lines, _fallback_was_used = _collect_log_lines(
            client, name, step, namespace, cursor_mode=since_line is not None
        )

        if since_line is not None and _fallback_was_used and log_lines:
            return ToolError(
                error=(
                    "cursor mode unavailable: active container logs empty"
                    " (pod may have restarted); call without since_line to read"
                    " previous-container logs"
                ),
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()

        # ── Cursor mode ──────────────────────────────────────────────────────
        if since_line is not None:
            return _cursor_response(name, step, log_lines, since_line, _effective_max)

        # ── Tail mode (unchanged) ─────────────────────────────────────────────
        if len(log_lines) > _effective_max:
            log_lines = log_lines[-_effective_max:]

        logs = "\n".join(log_lines)
        sanitized = truncate_log_output(logs)

        data = {
            "job": name,
            "step": step,
            "logs": sanitized,
            "lines": len(sanitized.splitlines()) if sanitized else 0,
        }

        hint = extract_failure_hint(logs)
        if hint:
            data["failure_hint"] = hint
            data["next_steps"] = [
                f"Detected {hint['category']}: {hint['suggestion']}",
                "Read trainer://guides/troubleshooting for detailed fixes",
            ]

        return ToolResponse(data=data).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_training_jobs to find available jobs",
                details=exception_details(e),
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            hint="Read trainer://guides/troubleshooting",
            details=exception_details(e),
        ).model_dump()


def get_training_events(
    name: str,
    namespace: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get Kubernetes events for a training job.

    Useful for debugging pending jobs (scheduling issues) or failures.

    Args:
        name: TrainJob name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum events to return. Defaults to 50.

    Returns:
        dict: Response containing:

        - ``job`` (str): Job name
        - ``events`` (list): Events with fields: ``involved_object_kind``,
          ``involved_object_name``, ``reason``, ``message``, ``event_time``
        - ``total`` (int): Total event count
        - ``returned`` (int): Number of events included in the response
        - ``has_more`` (bool): Whether more events were omitted by ``limit``
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        if limit < 1:
            return ToolError(
                error=f"limit must be >= 1, got {limit}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        limit = min(limit, MAX_EVENT_LIMIT)
        client = get_trainer_client_for_namespace(namespace)
        events = client.get_job_events(name=name)

        event_list = []
        for event in events[:limit]:
            et = getattr(event, "event_time", None)
            if et is not None and hasattr(et, "isoformat"):
                event_time_str = et.isoformat()
            else:
                event_time_str = str(et) if et is not None else ""
            event_list.append(
                {
                    "involved_object_kind": getattr(event, "involved_object_kind", "") or "",
                    "involved_object_name": getattr(event, "involved_object_name", "") or "",
                    "reason": event.reason if hasattr(event, "reason") else "",
                    "message": event.message if hasattr(event, "message") else "",
                    "event_time": event_time_str,
                }
            )

        return ToolResponse(
            data={
                "job": name,
                "events": event_list,
                "total": len(events),
                "returned": len(event_list),
                "has_more": len(events) > limit,
            }
        ).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_training_jobs to find available jobs",
                details=exception_details(e),
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            hint="Read trainer://guides/troubleshooting",
            details=exception_details(e),
        ).model_dump()


def _format_queue_timeout_hint(queue_status: dict[str, Any]) -> str | None:
    """Format contextual timeout hint based on Kueue queue status."""
    state = queue_status.get("state")
    q_name = queue_status.get("queue_name")
    q_reason = queue_status.get("reason")
    if state == "queued":
        return (
            f"Job is still queued in '{q_name}' waiting for quota. "
            "Extend timeout_seconds or check queue capacity. Read trainer://guides/queue-states"
        )
    if state == "inadmissible":
        return (
            f"Job cannot be admitted by queue '{q_name}'. "
            "Check queue configuration (see trainer://guides/queue-states)"
        )
    if state == "evicted":
        return f"Job was evicted ({q_reason}) from queue '{q_name}'. Read trainer://guides/queue-states"
    return None


def wait_for_training(
    name: str,
    target_statuses: list[str] | str = "Complete",
    namespace: str | None = None,
    timeout_seconds: int = 600,
    polling_interval: int = 2,
) -> dict[str, Any]:
    """Wait for a job to reach one or more target statuses.

    Blocks until the job reaches any of the expected statuses, or times out.

    Args:
        name: TrainJob name.
        target_statuses: Status string or list of status strings to wait for.
            Valid values: ``Complete``, ``Failed``, ``Running``, ``Created``.
            Pass a list to stop on the first match, e.g.
            ``["Complete", "Failed"]``. Defaults to ``"Complete"``.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        timeout_seconds: Maximum wait time in seconds. Defaults to 600 (10 min).
        polling_interval: Polling interval in seconds. Defaults to 2.

    Returns:
        dict: Response containing:

        - ``job`` (str): Job name
        - ``status`` (str): Final job status
        - ``reached`` (bool): Whether a target status was reached
        - ``message`` (str): Status message or timeout notice
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if isinstance(target_statuses, str):
        raw_statuses = [target_statuses]
    elif isinstance(target_statuses, list) and all(
        isinstance(status, str) for status in target_statuses
    ):
        raw_statuses = target_statuses
    else:
        return ToolError(
            error="target_statuses must be a string or a list of strings",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if not raw_statuses:
        return ToolError(
            error="target_statuses must contain at least one status",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    status_set = {_TARGET_STATUS_ALIASES.get(status, status) for status in raw_statuses}
    invalid_statuses = sorted(status_set - _VALID_TARGET_STATUSES)
    if invalid_statuses:
        return ToolError(
            error=f"Unsupported target status(es): {', '.join(invalid_statuses)}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    try:
        if timeout_seconds < 1:
            return ToolError(
                error=f"timeout_seconds must be >= 1, got {timeout_seconds}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        if polling_interval < MIN_POLLING_INTERVAL:
            return ToolError(
                error=f"polling_interval must be >= {MIN_POLLING_INTERVAL}, got {polling_interval}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        timeout_seconds = min(timeout_seconds, MAX_WAIT_TIMEOUT)
        polling_interval = max(polling_interval, MIN_POLLING_INTERVAL)
        client = get_trainer_client_for_namespace(namespace)

        job = client.wait_for_job_status(
            name=name,
            status=status_set,
            timeout=timeout_seconds,
            polling_interval=polling_interval,
        )

        final_status = job.status if hasattr(job, "status") else "Unknown"
        return ToolResponse(
            data={
                "job": name,
                "status": final_status,
                "reached": True,
                "message": f"Job reached '{final_status}'",
            }
        ).model_dump()

    except TimeoutError:
        effective_ns = namespace or str(
            getattr(getattr(client, "backend", None), "namespace", None) or "default"
        )
        queue_status = None
        try:
            queue_status = get_trainjob_queue_status(name=name, namespace=effective_ns)
        except Exception as e:
            logger.debug("Failed to resolve Kueue status for %s on timeout: %s", name, e)
        data: dict[str, Any] = {
            "job": name,
            "status": "Unknown",
            "reached": False,
            "message": f"Timeout after {timeout_seconds}s",
            "hint": "Use get_training_events to check for scheduling issues",
        }
        if queue_status is not None:
            data["queue_status"] = queue_status
            hint = _format_queue_timeout_hint(queue_status)
            if hint:
                data["hint"] = hint
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_training_jobs to find available jobs",
                details=exception_details(e),
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            hint="Read trainer://guides/troubleshooting",
            details=exception_details(e),
        ).model_dump()
