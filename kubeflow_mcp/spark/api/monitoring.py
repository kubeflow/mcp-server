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

"""Monitoring tools for SparkConnect sessions."""

import logging
import re
from collections import deque
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
        # get_session_logs returns an Iterator[str]. Keep only the last
        # ``tail_lines`` via a bounded deque so a chatty driver can't exhaust
        # memory, while counting the total to report truncation.
        log_iter = client.get_session_logs(name, follow=False)
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

        return ToolResponse(
            data={
                "name": name,
                "logs": logs,
                "lines": len(lines),
                "truncated": truncated,
            }
        ).model_dump()

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
