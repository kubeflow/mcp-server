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

"""Discovery tools for SparkConnect sessions."""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import get_spark_client_for_namespace
from kubeflow_mcp.core.security import check_namespace_allowed
from kubeflow_mcp.spark.types import session_info_to_dict

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 500


def list_spark_sessions(
    state: str | None = None,
    namespace: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List SparkConnect sessions.

    Args:
        state: Filter by session state: ``Provisioning``, ``Ready``, ``Running``,
            ``NotReady``, ``Failed``. Case-insensitive.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum sessions to return. Defaults to 50.

    Returns:
        dict: Response containing:

        - ``sessions`` (list): Sessions with ``name``, ``state``, ``namespace``,
          ``service_name``, ``driver_pod_name``
        - ``total`` (int): Total matching sessions

    Example:
        >>> list_spark_sessions(state="Ready")
        {"data": {"sessions": [{"name": "spark-connect-ab12", "state": "Ready"}], "total": 1}}
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if limit < 1:
        return ToolError(
            error=f"limit must be >= 1, got {limit}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    limit = min(limit, MAX_LIST_LIMIT)

    try:
        client = get_spark_client_for_namespace(namespace)
        sessions = [session_info_to_dict(s) for s in client.list_sessions()]

        if state:
            want = state.strip().lower()
            sessions = [s for s in sessions if (s.get("state") or "").lower() == want]

        return ToolResponse(
            data={"sessions": sessions[:limit], "total": len(sessions)}
        ).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_spark_session(name: str, namespace: str | None = None) -> dict[str, Any]:
    """Get details of a specific SparkConnect session.

    Args:
        name: The SparkConnect session name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``name``, ``namespace``, ``state``,
        ``service_name``, ``driver_pod_name``, ``pod_ip``, ``creation_timestamp``,
        and (for failed/not-ready sessions) ``next_steps``.

    Raises:
        ToolError: If the session is not found (``RESOURCE_NOT_FOUND``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_spark_client_for_namespace(namespace)
        info = client.get_session(name)
        data = session_info_to_dict(info)

        state = (data.get("state") or "").lower()
        next_steps: list[str] = []
        if state == "failed":
            next_steps = [
                f"get_spark_session_logs(name='{name}') — check driver output for the failure",
                f"delete_spark_session(name='{name}', confirmed=True) — tear down and recreate",
            ]
        elif state in ("notready", "provisioning"):
            next_steps = [
                f"get_spark_session(name='{name}') — poll until state is Ready",
            ]
        if next_steps:
            data["next_steps"] = next_steps

        return ToolResponse(data=data).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"SparkConnect session '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
