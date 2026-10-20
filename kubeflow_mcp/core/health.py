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

"""Health check tools for server monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.core.logging import get_log_buffer

logger = logging.getLogger(__name__)

_start_time = datetime.now(timezone.utc)

MAX_SERVER_LOG_LIMIT = 1000


def health_check() -> dict[str, Any]:
    """Check server health and connectivity.

    Returns server status, uptime, and K8s connectivity.
    Call periodically to verify server is operational.

    Returns:
        JSON with {status: str, uptime_seconds: int, kubernetes: bool}
    """
    from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api

    now = datetime.now(timezone.utc)
    uptime = (now - _start_time).total_seconds()

    k8s_ok = False
    try:
        v1 = get_core_v1_api()
        v1.list_namespace(limit=1, _request_timeout=K8S_TIMEOUT)
        k8s_ok = True
    except Exception as e:
        logger.warning(f"K8s health check failed: {e}")

    return ToolResponse(
        data={
            "status": "healthy" if k8s_ok else "degraded",
            "uptime_seconds": int(uptime),
            "kubernetes": k8s_ok,
            "timestamp": now.isoformat(),
        }
    ).model_dump()


def get_server_logs(
    level: str = "INFO",
    limit: int = 100,
) -> dict[str, Any]:
    """Get recent server logs for debugging.

    Retrieves logs from in-memory buffer.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        limit: Maximum number of log entries to return (1-1000)

    Returns:
        JSON with {logs: [{timestamp, level, message}]}
    """
    try:
        if limit < 1:
            return ToolError(
                error=f"limit must be >= 1, got {limit}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        limit = min(limit, MAX_SERVER_LOG_LIMIT)

        level_order = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }
        level_aliases = {
            "WARN": "WARNING",
            "FATAL": "CRITICAL",
        }
        normalized_level = level_aliases.get(level.upper(), level.upper())
        if normalized_level not in level_order:
            return ToolError(
                error=f"Unsupported log level: {level}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        min_level = level_order[normalized_level]

        all_logs = get_log_buffer()
        filtered = [
            log for log in all_logs if level_order.get(log.get("level", "INFO"), 1) >= min_level
        ]

        return ToolResponse(
            data={
                "logs": filtered[-limit:],
                "total": len(filtered),
                "buffer_size": len(all_logs),
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
        ).model_dump()


@dataclass
class ReadinessDiagnostic:
    """A single readiness check result."""

    check: str
    status: str  # "pass", "fail", "warn"
    message: str
    category: str = ""
    resource: str = ""
    verb: str = ""
    namespace: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert diagnostic result to a JSON-serializable dictionary."""
        d: dict[str, str] = {
            "check": self.check,
            "status": self.status,
            "message": self.message,
        }
        if self.category:
            d["category"] = self.category
        if self.resource:
            d["resource"] = self.resource
        if self.verb:
            d["verb"] = self.verb
        if self.namespace:
            d["namespace"] = self.namespace
        return d


def _check_rbac_permission(
    auth_api: Any,
    resource: str,
    verb: str,
    group: str,
    namespace: str,
    timeout: int,
) -> ReadinessDiagnostic:
    """Check a single RBAC permission via SelfSubjectAccessReview."""
    from kubernetes.client import (
        V1ResourceAttributes,
        V1SelfSubjectAccessReview,
        V1SelfSubjectAccessReviewSpec,
    )

    check_name = f"rbac:{verb}:{resource}"
    if namespace:
        check_name += f":{namespace}"

    try:
        review = V1SelfSubjectAccessReview(
            spec=V1SelfSubjectAccessReviewSpec(
                resource_attributes=V1ResourceAttributes(
                    namespace=namespace or "",
                    verb=verb,
                    group=group,
                    resource=resource,
                )
            )
        )
        result = auth_api.create_self_subject_access_review(
            body=review,
            _request_timeout=timeout,
        )
        status_obj = getattr(result, "status", None)
        if getattr(status_obj, "allowed", False):
            scope_desc = f" in namespace '{namespace}'" if namespace else " (cluster-scoped)"
            return ReadinessDiagnostic(
                check=check_name,
                status="pass",
                message=f"Permission granted: {verb} {resource}{scope_desc}",
                resource=resource,
                verb=verb,
                namespace=namespace,
            )
        reason = getattr(status_obj, "reason", None) or "denied by RBAC policy"
        scope_desc = f" in namespace '{namespace}'" if namespace else " (cluster-scoped)"
        return ReadinessDiagnostic(
            check=check_name,
            status="fail",
            message=f"Permission denied: {verb} {resource}{scope_desc} — {reason}",
            category="permission_denied",
            resource=resource,
            verb=verb,
            namespace=namespace,
        )
    except Exception as e:
        logger.warning("RBAC check failed for %s %s: %s", verb, resource, e)
        return ReadinessDiagnostic(
            check=check_name,
            status="warn",
            message=f"Could not verify permission: {verb} {resource}",
            category="unknown",
            resource=resource,
            verb=verb,
            namespace=namespace,
        )


def run_readiness_diagnostics() -> tuple[bool, list[ReadinessDiagnostic]]:
    """Run startup readiness diagnostics.

    Evaluates:
      1. Kubernetes API reachability (list namespaces)
      2. Kubeflow Trainer CRD presence and supported version
      3. Essential RBAC permissions (trainjobs list/create, runtimes list)

    Returns:
        tuple[bool, list[ReadinessDiagnostic]]: (all_passed, diagnostics_list)
    """
    from kubeflow_mcp.common.constants import (
        MIN_TRAINER_CRD_VERSION,
        TRAINER_CRD_GROUP,
        TRAINER_CRD_NAME,
    )
    from kubeflow_mcp.common.utils import (
        K8S_TIMEOUT,
        get_apiextensions_api,
        get_authorization_v1_api,
        get_core_v1_api,
    )

    diagnostics: list[ReadinessDiagnostic] = []

    # 1. Kubernetes API reachability
    try:
        v1 = get_core_v1_api()
        v1.list_namespace(limit=1, _request_timeout=K8S_TIMEOUT)
        diagnostics.append(
            ReadinessDiagnostic(
                check="k8s_api",
                status="pass",
                message="Kubernetes API reachable",
                resource="namespaces",
                verb="list",
            )
        )
    except Exception as e:
        logger.warning("Readiness: K8s API unreachable: %s", e)
        diagnostics.append(
            ReadinessDiagnostic(
                check="k8s_api",
                status="fail",
                message="Kubernetes API unreachable",
                category="api_unreachable",
                resource="namespaces",
                verb="list",
            )
        )
        # Short-circuit: do not attempt downstream CRD or RBAC checks if API is down
        return False, diagnostics

    # 2. Trainer CRD existence and supported version
    try:
        api_ext = get_apiextensions_api()
        crd = api_ext.read_custom_resource_definition(
            TRAINER_CRD_NAME,
            _request_timeout=K8S_TIMEOUT,
        )
        versions = getattr(getattr(crd, "spec", None), "versions", []) or []
        served = [v.name for v in versions if getattr(v, "served", False)]
        if MIN_TRAINER_CRD_VERSION in served:
            diagnostics.append(
                ReadinessDiagnostic(
                    check="trainer_crd",
                    status="pass",
                    message=f"CRD {TRAINER_CRD_NAME} serves {MIN_TRAINER_CRD_VERSION}",
                    resource=TRAINER_CRD_NAME,
                )
            )
        else:
            diagnostics.append(
                ReadinessDiagnostic(
                    check="trainer_crd",
                    status="fail",
                    message=(
                        f"CRD {TRAINER_CRD_NAME} does not serve {MIN_TRAINER_CRD_VERSION}"
                        f" (served: {', '.join(served) if served else 'none'})"
                    ),
                    category="crd_version_unsupported",
                    resource=TRAINER_CRD_NAME,
                )
            )
    except Exception as e:
        logger.warning("Readiness: Trainer CRD check failed: %s", e)
        diagnostics.append(
            ReadinessDiagnostic(
                check="trainer_crd",
                status="fail",
                message=f"CRD {TRAINER_CRD_NAME} not found or not accessible",
                category="crd_missing",
                resource=TRAINER_CRD_NAME,
            )
        )

    # 3. RBAC permission checks
    try:
        auth_api = get_authorization_v1_api()
    except Exception as e:
        logger.warning("Readiness: cannot create AuthorizationV1Api: %s", e)
        diagnostics.append(
            ReadinessDiagnostic(
                check="rbac",
                status="warn",
                message="Could not initialize authorization API for RBAC checks",
                category="unknown",
            )
        )
        all_passed = all(d.status != "fail" for d in diagnostics)
        return all_passed, diagnostics

    # Resolve effective namespace
    try:
        from kubeflow_mcp.common.utils import get_trainer_effective_namespace

        namespace = get_trainer_effective_namespace()
    except Exception:
        namespace = "default"

    rbac_checks = [
        ("trainjobs", "list", TRAINER_CRD_GROUP, namespace),
        ("trainjobs", "create", TRAINER_CRD_GROUP, namespace),
        ("clustertrainingruntimes", "list", TRAINER_CRD_GROUP, ""),
    ]
    for resource, verb, group, ns in rbac_checks:
        diagnostics.append(_check_rbac_permission(auth_api, resource, verb, group, ns, K8S_TIMEOUT))

    all_passed = all(d.status != "fail" for d in diagnostics)
    return all_passed, diagnostics


HEALTH_TOOLS = [health_check, get_server_logs]

# MCP registrations for health tools live here (server-level), not client modules.
HEALTH_TOOL_DESCRIPTIONS: dict[str, str] = {
    "health_check": "Check server health and K8s connectivity.",
    "get_server_logs": "Get recent server logs for debugging. Filter by level.",
}

HEALTH_TOOL_ANNOTATIONS: dict[str, dict] = {
    "health_check": {
        "title": "Health Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "tags": ["health", "monitoring"],
    },
    "get_server_logs": {
        "title": "Get Server Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "tags": ["health", "monitoring", "debug"],
    },
}
