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

"""Lifecycle tools for SparkConnect sessions (create / delete)."""

import logging
import re
import uuid
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import (
    PreviewResponse,
    ToolError,
    ToolResponse,
    exception_details,
    is_k8s_not_found,
)
from kubeflow_mcp.common.utils import (
    MCP_MANAGED_LABEL,
    MCP_MANAGED_VALUE,
    get_spark_client_for_namespace,
    get_spark_session_ownership,
    get_trainer_effective_namespace,
)
from kubeflow_mcp.core.policy import get_effective_persona
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.spark.types import session_info_to_dict

logger = logging.getLogger(__name__)

# RFC 1123 label — SparkConnect names become Kubernetes object + service names.
_K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_MAX_NAME_LEN = 52  # leave headroom for operator-appended suffixes (-svc, -server)


def _generate_session_name() -> str:
    """Generate a unique, RFC 1123-compliant SparkConnect session name."""
    return f"spark-connect-{uuid.uuid4().hex[:8]}"


def _validate_session_name(name: str) -> str | None:
    """Return an error message if *name* is not a valid session name, else None.

    Applies the shared :func:`validate_k8s_name` check first (conventions require
    it for every resource name), then the stricter SparkConnect-specific limit —
    names become Kubernetes *service* names, so they need more headroom than a
    generic object name.
    """
    shared_err = validate_k8s_name(name)
    if shared_err is not None:
        return shared_err.error
    if len(name) > _MAX_NAME_LEN:
        return f"name too long ({len(name)} chars, max {_MAX_NAME_LEN})"
    if not _K8S_NAME_RE.match(name):
        return (
            f"invalid name '{name}': must be lowercase alphanumeric or '-', "
            "start and end with an alphanumeric character (RFC 1123)"
        )
    return None


def create_spark_session(
    name: str | None = None,
    num_executors: int | None = None,
    executor_resources: dict[str, str] | None = None,
    driver_resources: dict[str, str] | None = None,
    spark_conf: dict[str, str] | None = None,
    timeout: int = 300,
    connect_timeout: int = 120,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a SparkConnect session on the cluster (two-phase confirm).

    Call once with ``confirmed=False`` (the default) to preview the resolved
    configuration, then again with ``confirmed=True`` to provision.

    This wraps ``SparkClient.connect()`` in *create* mode. The SDK provisions the
    SparkConnect CR, waits for it to become Ready, and establishes a driver-side
    session; this tool releases that session immediately and returns the session
    metadata. The data plane (a notebook, job, or agent tool) then attaches with
    PySpark using the returned connect info — the MCP server does not proxy Spark
    RPCs. Requires the ``kubeflow[spark]`` extra on the server host.

    Args:
        name: Session name (RFC 1123). Auto-generated when omitted.
        num_executors: Number of executor pods.
        executor_resources: Per-executor resources, e.g.
            ``{"cpu": "1", "memory": "2Gi"}``. Supports arbitrary K8s resources
            such as ``nvidia.com/gpu``.
        driver_resources: Driver pod resources, e.g. ``{"cpu": "1", "memory": "2Gi"}``.
        spark_conf: Extra Spark configuration properties.
        timeout: Seconds to wait for the session to become Ready.
        connect_timeout: Seconds to wait for the driver connection.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set True to actually create the session.

    Returns:
        dict: Preview config (``confirmed=False``) or the created session's
        metadata (``confirmed=True``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if num_executors is not None and num_executors < 0:
        return ToolError(
            error=f"num_executors must be >= 0, got {num_executors}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    session_name = name or _generate_session_name()
    name_err = _validate_session_name(session_name)
    if name_err is not None:
        return ToolError(error=name_err, error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    config: dict[str, Any] = {
        "name": session_name,
        "namespace": namespace,
        "num_executors": num_executors,
        "executor_resources": executor_resources,
        "driver_resources": driver_resources,
        "spark_conf": spark_conf,
        "timeout": timeout,
        "connect_timeout": connect_timeout,
    }

    if not confirmed:
        return PreviewResponse(
            config=config,
            message=(
                f"Will create SparkConnect session '{session_name}'"
                + (f" in namespace '{namespace}'" if namespace else "")
                + ". Set confirmed=True to execute."
            ),
        ).model_dump()

    try:
        from kubeflow.spark import Driver, Executor, Labels, Name

        client = get_spark_client_for_namespace(namespace)

        executor = None
        if num_executors is not None or executor_resources is not None:
            executor = Executor(
                num_instances=num_executors,
                resources_per_executor=executor_resources,
            )
        driver = Driver(resources=driver_resources) if driver_resources else None

        session = client.connect(
            spark_conf=spark_conf,
            driver=driver,
            executor=executor,
            # Labels marks the SparkConnect CR as MCP-owned so non-admin personas
            # may later mutate it (see delete_spark_session's ownership guard).
            options=[Name(session_name), Labels({MCP_MANAGED_LABEL: MCP_MANAGED_VALUE})],
            timeout=timeout,
            connect_timeout=connect_timeout,
        )
        # The MCP server has no use for a live driver connection — release it so
        # the session's resources aren't tied to this process. The SparkConnect
        # CR keeps running for the data plane to attach to.
        try:
            session.stop()
        except Exception:
            logger.debug("Best-effort stop() of transient SparkSession failed", exc_info=True)

        info = client.get_session(session_name)
        data = session_info_to_dict(info)
        data["message"] = (
            "Session created. Attach with PySpark using the connect info above; "
            "the MCP server released its transient driver connection."
        )
        return ToolResponse(data=data).model_dump()

    except ImportError as e:
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        # connect() waits for readiness and a driver connection, both of which can
        # fail *after* the CR is already created (e.g. the MCP host can't reach the
        # driver). If the session exists, report it as provisioned rather than lost.
        try:
            client = get_spark_client_for_namespace(namespace)
            info = client.get_session(session_name)
            data = session_info_to_dict(info)
            data["warning"] = (
                "Session was provisioned, but the server could not confirm the driver "
                f"connection: {e}. Poll get_spark_session(name='{session_name}') for readiness."
            )
            return ToolResponse(data=data).model_dump()
        except Exception:
            return ToolError(
                error=str(e),
                error_code=ErrorCode.SDK_ERROR,
                details=exception_details(e),
            ).model_dump()


def delete_spark_session(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """[DESTRUCTIVE] Delete a SparkConnect session permanently (two-phase confirm).

    Call once with ``confirmed=False`` to preview, then again with
    ``confirmed=True`` to delete.

    Args:
        name: The SparkConnect session name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set True to actually delete the session.

    Returns:
        dict: Preview (``confirmed=False``) or a deletion confirmation.

    Raises:
        ToolError: If the session is not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    ns = namespace
    if ns is None:
        try:
            ns = get_trainer_effective_namespace(None)
        except Exception:
            ns = None

    # Ownership guard: non-admin personas may only delete sessions MCP created.
    if get_effective_persona() not in ("platform-admin",):
        if ns is None:
            return ToolError(
                error=f"Cannot verify ownership of SparkConnect session '{name}' "
                "(namespace could not be resolved)",
                error_code=ErrorCode.SDK_ERROR,
                details={"hint": "Pass an explicit namespace, or use platform-admin persona."},
            ).model_dump()
        ownership = get_spark_session_ownership(name, ns)
        if ownership == "not_found":
            return ToolError(
                error=f"SparkConnect session '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        if ownership == "unknown":
            return ToolError(
                error=f"Cannot verify ownership of SparkConnect session '{name}' (API error)",
                error_code=ErrorCode.SDK_ERROR,
                details={"hint": "Retry, or use platform-admin persona to bypass."},
            ).model_dump()
        if ownership == "unmanaged":
            return ToolError(
                error=f"SparkConnect session '{name}' was not created by MCP",
                error_code=ErrorCode.VALIDATION_ERROR,
                details={
                    "hint": (
                        "Data scientists can only delete sessions created through MCP tools. "
                        "Use platform-admin persona to delete externally created sessions."
                    ),
                },
            ).model_dump()

    if not confirmed:
        return PreviewResponse(
            config={"name": name, "namespace": ns, "action": "delete"},
            message=f"Will permanently delete SparkConnect session '{name}'. Set confirmed=True.",
        ).model_dump()

    try:
        client = get_spark_client_for_namespace(namespace)
        client.delete_session(name)
        return ToolResponse(
            data={
                "name": name,
                "deleted": True,
                "message": f"Deleted SparkConnect session '{name}'",
            }
        ).model_dump()

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
