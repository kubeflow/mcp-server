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
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
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

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 1000
MAX_EVENT_LIMIT = 500
MAX_WAIT_TIMEOUT = 3600
MIN_POLLING_INTERVAL = 1

_FAILURE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"Permission denied.*(/\.local|/home|/\.cache)", re.IGNORECASE),
        "OPENSHIFT_PERMISSION_ERROR",
        "OpenShift random UID cannot write to home directory; set env var HF_HOME=/workspace or mount a writable volume.",
    ),
    (
        re.compile(r"Permission denied.*huggingface|HF_HOME", re.IGNORECASE),
        "HF_CACHE_WRITE_ERROR",
        "Set env var HF_HOME=/workspace to store HuggingFace cache on a writable volume mount.",
    ),
    (
        re.compile(r"CUDA out of memory", re.IGNORECASE),
        "OOM",
        "Reduce batch_size, enable quantization (int8/int4), or request a larger GPU.",
    ),
    (
        re.compile(r"OutOfMemoryError", re.IGNORECASE),
        "OOM",
        "Reduce batch_size, enable quantization (int8/int4), or request a larger GPU.",
    ),
    (
        re.compile(r"RuntimeError: CUDA error", re.IGNORECASE),
        "CUDA_ERROR",
        "Check GPU driver compatibility and CUDA version in the runtime image.",
    ),
    (
        re.compile(r"ModuleNotFoundError: No module named", re.IGNORECASE),
        "MISSING_MODULE",
        "Add the missing package to the 'packages' list.",
    ),
    (
        re.compile(r"ImportError", re.IGNORECASE),
        "IMPORT_ERROR",
        "Verify package versions; add missing package to 'packages'.",
    ),
    (
        re.compile(r"FileNotFoundError", re.IGNORECASE),
        "FILE_NOT_FOUND",
        "Check dataset/model paths and volume mounts.",
    ),
    (
        re.compile(r"PermissionError|Access Denied", re.IGNORECASE),
        "PERMISSION_ERROR",
        "Check service account permissions and storage credentials.",
    ),
    (
        re.compile(r"Connection(Error|Refused|Reset)", re.IGNORECASE),
        "NETWORK_ERROR",
        "Check network policies, DNS, and endpoint reachability.",
    ),
    (
        re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
        "PYTHON_EXCEPTION",
        "Review the traceback above for the root cause.",
    ),
]


def _extract_failure_hint(logs: str) -> dict[str, str] | None:
    """Pattern-match common failure signatures and return an actionable hint."""
    for pattern, category, suggestion in _FAILURE_PATTERNS:
        if pattern.search(logs):
            return {"category": category, "suggestion": suggestion}
    return None


def get_training_logs(
    name: str,
    step: str = "node-0",
    namespace: str | None = None,
    follow: bool = False,
) -> dict[str, Any]:
    """Get pod logs from a training job.

    Args:
        name: TrainJob name.
        step: Node/worker to get logs from. Defaults to ``node-0``.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        follow: Stream logs continuously (not supported in MCP context).

    Returns:
        dict: Response containing:

        - ``job`` (str): Job name
        - ``step`` (str): Node name
        - ``logs`` (str): Sanitized log output
        - ``lines`` (int): Number of log lines

    Raises:
        ToolError: If job not found (``RESOURCE_NOT_FOUND``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

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
        log_lines = list(client.get_job_logs(name=name, step=step, follow=False))
        if not log_lines:
            try:
                eff_ns = get_trainer_effective_namespace(namespace)
                name_err = validate_k8s_name(name)
                if name_err is not None:
                    return name_err.model_dump()
                v1 = get_core_v1_api()
                pods = v1.list_namespaced_pod(
                    namespace=eff_ns,
                    label_selector=f"training.kubeflow.org/trainjob-name={name}",
                )
                for pod in pods.items:
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

        if len(log_lines) > MAX_LOG_LINES:
            log_lines = log_lines[-MAX_LOG_LINES:]

        logs = "\n".join(log_lines)
        sanitized = truncate_log_output(logs)

        data: dict[str, Any] = {
            "job": name,
            "step": step,
            "logs": sanitized,
            "lines": len(sanitized.split("\n")),
        }

        hint = _extract_failure_hint(logs)
        if hint:
            data["failure_hint"] = hint
            data["next_steps"] = [
                f"Detected {hint['category']}: {hint['suggestion']}",
                "Read trainer://workflows/ops for detailed fixes",
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
            hint="Read trainer://workflows/ops",
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
    """
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
            data={"job": name, "events": event_list, "total": len(events)}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            hint="Read trainer://workflows/ops",
            details=exception_details(e),
        ).model_dump()


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
            Valid values: ``Complete``, ``Failed``, ``Running``, ``Created``,
            ``Suspended``. Pass a list to stop on the first match, e.g.
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
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

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

        aliases = {"Succeeded": "Complete"}
        raw = set(target_statuses) if isinstance(target_statuses, list) else {target_statuses}
        status_set = {aliases.get(s, s) for s in raw}

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
        return ToolResponse(
            data={
                "job": name,
                "status": "Unknown",
                "reached": False,
                "message": f"Timeout after {timeout_seconds}s",
                "hint": "Use get_training_events to check for scheduling issues",
            }
        ).model_dump()
    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            hint="Read trainer://workflows/ops",
            details=exception_details(e),
        ).model_dump()
