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
    get_spark_effective_namespace,
    get_spark_session_ownership,
)
from kubeflow_mcp.core.policy import get_effective_persona
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.spark.types import session_info_to_dict

logger = logging.getLogger(__name__)

# RFC 1123 label — SparkConnect names become Kubernetes object + service names.
_K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_MAX_NAME_LEN = 52  # leave headroom for operator-appended suffixes (-svc, -server)

# A typo (1000 instead of 100) would otherwise ask the cluster for 1000 executor
# pods. The ceiling is deliberately generous; it only stops runaway requests.
_MAX_EXECUTORS = 100

# ``connect()`` blocks the calling thread for up to these many seconds, so an
# unbounded value (timeout=86400) would hold a server thread for a day. One hour
# is far longer than a session needs to become Ready.
_MAX_TIMEOUT = 3600
_MAX_CONNECT_TIMEOUT = 3600

# ``spark_conf`` is forwarded to the SDK verbatim, so these keys would let a
# caller change *what runs and as whom* — a different image, a different service
# account, a namespace outside the allowlist, or host mounts and secrets — none
# of which are visible in the confirm-gate preview. Resources, naming and
# namespace all have dedicated parameters, so refuse these rather than merge
# them silently.
_DENIED_SPARK_CONF_PREFIXES: tuple[str, ...] = (
    "spark.kubernetes.authenticate.",  # service accounts, tokens, OAuth
    "spark.kubernetes.namespace",  # would escape check_namespace_allowed()
    "spark.kubernetes.container.image",  # arbitrary image
    "spark.kubernetes.driver.container.image",
    "spark.kubernetes.executor.container.image",
    "spark.kubernetes.driver.podtemplatefile",  # whole-pod override
    "spark.kubernetes.executor.podtemplatefile",
    "spark.kubernetes.driver.volumes.",  # host mounts
    "spark.kubernetes.executor.volumes.",
    "spark.kubernetes.driver.secrets.",  # secret injection
    "spark.kubernetes.executor.secrets.",
    "spark.kubernetes.driver.secretkeyref.",
    "spark.kubernetes.executor.secretkeyref.",
)


def _validate_spark_conf(spark_conf: dict[str, str] | None) -> str | None:
    """Return an error message if *spark_conf* sets a restricted key, else None."""
    if not spark_conf:
        return None
    denied = sorted(
        key for key in spark_conf if key.strip().lower().startswith(_DENIED_SPARK_CONF_PREFIXES)
    )
    if not denied:
        return None
    return (
        "spark_conf may not set server-managed keys: "
        + ", ".join(denied)
        + ". These control the image, service account, namespace, pod template, "
        "volumes or secrets. Use the namespace, driver_resources and "
        "executor_resources parameters instead."
    )


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


def _provisioned_session_or_none(
    exc: Exception, session_name: str, namespace: str | None
) -> dict[str, Any] | None:
    """Return the session's metadata if it exists despite *exc*, else ``None``.

    ``connect()`` waits for readiness and for a driver connection, either of
    which can fail *after* the SparkConnect CR has already been created (the MCP
    host may not be able to reach the driver, or a client-side import may fail).
    Reporting only the error would leave a running, MCP-labelled session that the
    caller does not know about.
    """
    try:
        client = get_spark_client_for_namespace(namespace)
        info = client.get_session(session_name)
    except Exception:
        return None

    data = session_info_to_dict(info)
    data["warning"] = (
        "Session was provisioned, but the server could not confirm the driver "
        f"connection: {exc}. Poll get_spark_session(name='{session_name}') for readiness."
    )
    return ToolResponse(data=data).model_dump()


def _validate_create_inputs(
    num_executors: int | None,
    timeout: int,
    connect_timeout: int,
    spark_conf: dict[str, str] | None,
) -> ToolError | None:
    """Bound the inputs that reach the cluster or hold a server thread."""
    if num_executors is not None and not 0 <= num_executors <= _MAX_EXECUTORS:
        return ToolError(
            error=f"num_executors must be between 0 and {_MAX_EXECUTORS}, got {num_executors}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    for label, value, ceiling in (
        ("timeout", timeout, _MAX_TIMEOUT),
        ("connect_timeout", connect_timeout, _MAX_CONNECT_TIMEOUT),
    ):
        if not 1 <= value <= ceiling:
            return ToolError(
                error=f"{label} must be between 1 and {ceiling} seconds, got {value}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    conf_err = _validate_spark_conf(spark_conf)
    if conf_err is not None:
        return ToolError(error=conf_err, error_code=ErrorCode.VALIDATION_ERROR)

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

    sizing_err = _validate_create_inputs(num_executors, timeout, connect_timeout, spark_conf)
    if sizing_err is not None:
        return sizing_err.model_dump()

    session_name = name or _generate_session_name()
    name_err = _validate_session_name(session_name)
    if name_err is not None:
        return ToolError(error=name_err, error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    # Show the namespace the session will actually land in, not the bare None the
    # caller passed: the user is confirming a mutation and should see its target.
    effective_ns = namespace
    if effective_ns is None:
        try:
            effective_ns = get_spark_effective_namespace(None)
        except Exception:
            effective_ns = None

    config: dict[str, Any] = {
        "name": session_name,
        "namespace": effective_ns,
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
                + (f" in namespace '{effective_ns}'" if effective_ns else "")
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
        # A missing client-side dependency can surface *after* connect() has already
        # created the CR (pyspark imports pandas only when building the session), so
        # this path needs the same provisioned-session check as any other failure —
        # otherwise a running session is reported as a pure failure and leaks.
        logger.warning("create_spark_session(%s) missing SDK extra: %s", session_name, e)
        provisioned = _provisioned_session_or_none(e, session_name, namespace)
        if provisioned is not None:
            return provisioned
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("create_spark_session(%s) failed: %s", session_name, e, exc_info=True)
        provisioned = _provisioned_session_or_none(e, session_name, namespace)
        if provisioned is not None:
            return provisioned
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
            ns = get_spark_effective_namespace(None)
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
        logger.warning("delete_spark_session(%s) missing SDK extra: %s", name, e)
        return ToolError(error=str(e), error_code=ErrorCode.SDK_ERROR).model_dump()
    except Exception as e:
        logger.warning("delete_spark_session(%s) failed: %s", name, e, exc_info=True)
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
