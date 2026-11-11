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

"""Discovery tools for SparkConnect sessions and Spark batch jobs."""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import get_spark_client_for_namespace
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.spark.types import job_to_dict, session_info_to_dict

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 500

# SparkJobStatus values in kubeflow[spark] 0.5.0. The SDK maps every
# SparkApplication state onto these four, so a filter outside them can never
# match; it is refused instead of silently returning nothing. Pinned by
# ``sdk_contracts_test.py``.
JOB_STATUSES = ("Created", "Running", "Completed", "Failed")


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
        - ``truncated`` (bool): True when ``limit`` cut the list, so more exist

    Example:
        >>> list_spark_sessions(state="Ready")
        {"data": {"sessions": [{"name": "spark-connect-ab12", "state": "Ready"}],
                  "total": 1, "truncated": False}}
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

        shown = sessions[:limit]
        return ToolResponse(
            data={
                "sessions": shown,
                "total": len(sessions),
                # Signal the cut so an agent knows to raise `limit` rather than
                # assuming it has seen every session.
                "truncated": len(shown) < len(sessions),
            }
        ).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("list_spark_sessions() failed: %s", e, exc_info=True)
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

    name_err = validate_k8s_name(name, "session name")
    if name_err is not None:
        return name_err.model_dump()

    try:
        client = get_spark_client_for_namespace(namespace)
        info = client.get_session(name)
        data = session_info_to_dict(info)

        state = (data.get("state") or "").lower()
        next_steps: list[str] = []
        if state == "failed":
            next_steps = [
                f"get_spark_session_logs(name='{name}') — check driver output for the failure",
                f"delete_spark_session(name='{name}') — preview the teardown, "
                "then repeat with confirmed=True",
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
        logger.warning("get_spark_session(%s) failed: %s", name, e, exc_info=True)
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


def list_spark_jobs(
    status: str | None = None,
    namespace: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List Spark batch jobs (SparkApplications).

    Args:
        status: Filter by job status: ``Created``, ``Running``, ``Completed``,
            ``Failed``. Case-insensitive.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum jobs to return. Defaults to 50.

    Returns:
        dict: Response containing:

        - ``jobs`` (list): Jobs with ``name``, ``namespace``, ``status``,
          ``num_executors``, ``driver_pod_name``, ``creation_timestamp``
        - ``total`` (int): Total matching jobs
        - ``truncated`` (bool): True when ``limit`` cut the list, so more exist

    Example:
        >>> list_spark_jobs(status="Failed")
        {"data": {"jobs": [{"name": "etl-nightly", "status": "Failed"}],
                  "total": 1, "truncated": False}}
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    want = None
    if status:
        want = status.strip().lower()
        if want not in {s.lower() for s in JOB_STATUSES}:
            return ToolError(
                error=f"Invalid status '{status}'. Must be one of: {', '.join(JOB_STATUSES)}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()

    if limit < 1:
        return ToolError(
            error=f"limit must be >= 1, got {limit}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    limit = min(limit, MAX_LIST_LIMIT)

    try:
        client = get_spark_client_for_namespace(namespace)
        jobs = [job_to_dict(j) for j in client.list_jobs()]
        if want:
            jobs = [j for j in jobs if (j.get("status") or "").lower() == want]

        shown = jobs[:limit]
        return ToolResponse(
            data={"jobs": shown, "total": len(jobs), "truncated": len(shown) < len(jobs)}
        ).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("list_spark_jobs() failed: %s", e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_spark_job(name: str, namespace: str | None = None) -> dict[str, Any]:
    """Get details of a Spark batch job (SparkApplication).

    Args:
        name: The Spark job name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``name``, ``namespace``, ``status``,
        ``num_executors``, ``driver_pod_name``, ``creation_timestamp`` and, for
        jobs that failed or have not finished, ``next_steps``.

    Raises:
        ToolError: If the job is not found (``RESOURCE_NOT_FOUND``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    name_err = validate_k8s_name(name, "job name")
    if name_err is not None:
        return name_err.model_dump()

    try:
        client = get_spark_client_for_namespace(namespace)
        data = job_to_dict(client.get_job(name))

        status = (data.get("status") or "").lower()
        if status == "failed":
            data["next_steps"] = [
                f"get_spark_job_logs(name='{name}') to read the driver output for the failure",
            ]
        elif status in ("created", "running"):
            data["next_steps"] = [
                f"get_spark_job(name='{name}') to poll until status is Completed or Failed",
            ]

        return ToolResponse(data=data).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("get_spark_job(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Spark job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Find jobs with list_spark_jobs()",
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
