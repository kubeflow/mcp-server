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
from collections import deque
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import get_spark_client_for_namespace
from kubeflow_mcp.core.security import check_namespace_allowed

logger = logging.getLogger(__name__)

DEFAULT_TAIL_LINES = 200
MAX_TAIL_LINES = 2000


def get_spark_session_logs(
    name: str,
    tail_lines: int = DEFAULT_TAIL_LINES,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get driver-pod logs from a SparkConnect session.

    Streaming (``follow=True``) is intentionally not exposed — a stateless MCP
    tool returns a bounded snapshot. Use ``tail_lines`` to control volume.

    Args:
        name: The SparkConnect session name.
        tail_lines: Number of trailing log lines to return (default 200, max 2000).
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing:

        - ``name`` (str): The session name
        - ``logs`` (str): The captured driver-pod log lines
        - ``lines`` (int): Number of lines returned
        - ``truncated`` (bool): True if older lines were dropped to honor ``tail_lines``

    Raises:
        ToolError: If the session is not found (``RESOURCE_NOT_FOUND``) or has no
        driver pod yet (``VALIDATION_ERROR``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

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
        window: deque[str] = deque(maxlen=tail_lines)
        total = 0
        for line in log_iter:
            window.append(line)
            total += 1
        lines = list(window)
        truncated = total > tail_lines

        return ToolResponse(
            data={
                "name": name,
                "logs": "\n".join(lines),
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
        # The SDK raises a plain RuntimeError with "No driver pod" before the
        # session is Ready — surface it as a validation error with a next step.
        if "no driver pod" in str(e).lower():
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
