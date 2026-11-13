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

"""Monitoring tools for SparkConnect sessions and Spark batch jobs."""

import logging
import re
from collections import deque
from collections.abc import Iterable
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import get_spark_client_for_namespace
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name

logger = logging.getLogger(__name__)

DEFAULT_TAIL_LINES = 200
MAX_TAIL_LINES = 2000
MAX_LOG_CHARS = 10_000
_TRUNCATION_MARKER = "... (logs truncated)\n"

# A session whose server pod has not been scheduled yet has no typed SDK error:
# the backend raises a plain ``RuntimeError``. Released kubeflow[spark] 0.4.x
# words it "No server pod for SparkConnect: <ns>/<name>"; SDK ``main`` renamed
# the field to ``driver_pod_name``, so tolerate either noun. Pinned by
# ``sdk_contracts_test.py`` against the installed SDK.
_MISSING_POD_RE = re.compile(r"\bno (?:server|driver) pod\b", re.IGNORECASE)


def _is_missing_server_pod(exc: Exception) -> bool:
    """Return True when *exc* reports a session without a server (driver) pod."""
    return any(
        _MISSING_POD_RE.search(str(e))
        for e in (exc, exc.__cause__, exc.__context__)
        if e is not None
    )


def _bounded_tail(log_iter: Iterable[str], tail_lines: int) -> dict[str, Any]:
    """Keep the last *tail_lines* lines of *log_iter*, capped at ``MAX_LOG_CHARS``.

    The SDK returns an iterator, so a bounded deque keeps memory flat however
    chatty the driver is, while every source line is counted to report truncation.

    Returns:
        dict with ``logs`` (str), ``lines`` (int) and ``truncated`` (bool).
    """
    window: deque[str] = deque()
    window_chars = 0
    total = 0
    truncated = False
    for line in log_iter:
        # Keep memory bounded even if the SDK yields a single enormous line.
        if len(line) > MAX_LOG_CHARS:
            line = line[-MAX_LOG_CHARS:]
            truncated = True
        window.append(line)
        window_chars += len(line)
        if len(window) > tail_lines:
            window_chars -= len(window.popleft())
            truncated = True
        while window and window_chars + max(0, len(window) - 1) > MAX_LOG_CHARS:
            window_chars -= len(window.popleft())
            truncated = True
        # Count each source line, including lines dropped to enforce either bound.
        total += 1
    lines = list(window)
    truncated = truncated or total > len(lines)
    logs = "\n".join(lines)
    if truncated:
        budget = MAX_LOG_CHARS - len(_TRUNCATION_MARKER)
        logs = _TRUNCATION_MARKER + logs[-budget:]
        # The character trim can cut further than the line window did (and
        # can split the oldest surviving line), so recount from the final
        # payload instead of reporting the pre-trim window size.
        lines = logs.splitlines()[1:]
    return {"logs": logs, "lines": len(lines), "truncated": truncated}


def _has_k8s_not_found(exc: BaseException | None) -> bool:
    """Return True when a Kubernetes 404 appears anywhere in *exc*'s cause chain.

    The SDK wraps a failed pod log read twice (``RuntimeError`` from
    ``RuntimeError`` from ``ApiException``), deeper than ``is_k8s_not_found`` looks.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if is_k8s_not_found(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def get_spark_session_logs(
    name: str,
    tail_lines: int = DEFAULT_TAIL_LINES,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get driver-pod logs from a SparkConnect session.

    Streaming (``follow=True``) is intentionally not exposed — a stateless MCP
    tool returns a bounded snapshot. Use ``tail_lines`` to control volume;
    output is also capped at ``MAX_LOG_CHARS`` characters.

    Args:
        name: The SparkConnect session name.
        tail_lines: Number of trailing log lines to return (default 200, max 2000).
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing:

        - ``name`` (str): The session name
        - ``logs`` (str): The captured driver-pod log lines
        - ``lines`` (int): Number of lines returned
        - ``truncated`` (bool): True if log lines or characters were dropped to honor limits

    Raises:
        ToolError: If the session is not found (``RESOURCE_NOT_FOUND``) or has no
        driver pod yet (``VALIDATION_ERROR``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    name_err = validate_k8s_name(name, "session name")
    if name_err is not None:
        return name_err.model_dump()

    if tail_lines < 1:
        return ToolError(
            error=f"tail_lines must be >= 1, got {tail_lines}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    tail_lines = min(tail_lines, MAX_TAIL_LINES)

    try:
        client = get_spark_client_for_namespace(namespace)
        log_iter = client.get_session_logs(name, follow=False)
        return ToolResponse(data={"name": name, **_bounded_tail(log_iter, tail_lines)}).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("get_spark_session_logs(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"SparkConnect session '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        # Surface a not-yet-scheduled server pod as a validation error with a
        # next step, rather than an opaque SDK error.
        if _is_missing_server_pod(e):
            return ToolError(
                error=(
                    f"SparkConnect session '{name}' has no driver pod yet — it is likely still "
                    f"provisioning. Check get_spark_session(name='{name}')."
                ),
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_spark_job_logs(
    name: str,
    tail_lines: int = DEFAULT_TAIL_LINES,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get driver-pod logs from a Spark batch job (SparkApplication).

    Returns a bounded snapshot, like ``get_spark_session_logs``: streaming is not
    exposed, ``tail_lines`` controls volume and output is capped at
    ``MAX_LOG_CHARS`` characters.

    Args:
        name: The Spark job (SparkApplication) name.
        tail_lines: Number of trailing log lines to return (default 200, max 2000).
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``name``, ``logs`` (str), ``lines`` (int) and
        ``truncated`` (bool).

    Raises:
        ToolError: ``RESOURCE_NOT_FOUND`` if the job does not exist, or
        ``VALIDATION_ERROR`` if its driver pod has not started yet or no longer exists.
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    name_err = validate_k8s_name(name, "job name")
    if name_err is not None:
        return name_err.model_dump()

    if tail_lines < 1:
        return ToolError(
            error=f"tail_lines must be >= 1, got {tail_lines}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    tail_lines = min(tail_lines, MAX_TAIL_LINES)

    try:
        client = get_spark_client_for_namespace(namespace)
        # The SDK looks the job up here, so a 404 at this point means the job is missing.
        log_iter = client.get_job_logs(name, follow=False)
    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("get_spark_job_logs(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Spark job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Find jobs with list_spark_jobs()",
            ).model_dump()
        if _is_missing_server_pod(e):
            return ToolError(
                error=(
                    f"Spark job '{name}' has no driver pod yet; it has likely not started. "
                    f"Check get_spark_job(name='{name}')."
                ),
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        return ToolError(
            error=str(e), error_code=ErrorCode.SDK_ERROR, details=exception_details(e)
        ).model_dump()

    try:
        return ToolResponse(data={"name": name, **_bounded_tail(log_iter, tail_lines)}).model_dump()
    except Exception as e:
        logger.warning("get_spark_job_logs(%s) failed reading logs: %s", name, e, exc_info=True)
        # The job exists, so a 404 while reading means its driver pod is gone,
        # which happens once a finished job's pods are cleaned up.
        if _has_k8s_not_found(e):
            return ToolError(
                error=f"The driver pod of Spark job '{name}' no longer exists, so its logs are gone",
                error_code=ErrorCode.VALIDATION_ERROR,
                hint=f"Check the final state with get_spark_job(name='{name}')",
            ).model_dump()
        return ToolError(
            error=str(e), error_code=ErrorCode.SDK_ERROR, details=exception_details(e)
        ).model_dump()
