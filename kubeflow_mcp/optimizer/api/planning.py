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

"""Planning tools for Katib pre-flight checks.

Mirrors the trainer planning module's compound-tool pattern:
one call aggregates multiple sub-checks into a single response.
"""

import logging
from typing import Any, NamedTuple

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import (
    ToolError,
    ToolResponse,
    exception_details,
    is_k8s_not_found,
)
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_CRD_NAME,
    KATIB_CONTROLLER_LABELS,
    KATIB_CONTROLLER_NAMESPACE_DEFAULT,
)

logger = logging.getLogger(__name__)


class CheckResult(NamedTuple):
    """What one sub-check learned, plus anything that blocks or merely warns.

    Keeps ``katib_pre_flight`` a flat composition of independent checks rather
    than one long branching function. Named rather than a bare 3-tuple because
    ``blockers`` and ``warnings`` share a type: transposing them at a call site
    would downgrade a blocker to a warning and report a false green, which is
    the exact failure ``_check_controller`` exists to prevent. Construct it with
    keywords so that mistake cannot be made silently.
    """

    info: dict[str, Any]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


_INSTALL_HINT = "https://www.kubeflow.org/docs/components/katib/installation/"


def _check_experiment_crd() -> CheckResult:
    """Confirm the Katib Experiment CRD is registered on the cluster."""
    try:
        crd = mcp_utils.get_apiextensions_api().read_custom_resource_definition(
            name=EXPERIMENT_CRD_NAME,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )
    except Exception as e:
        if is_k8s_not_found(e):
            return CheckResult(
                info={"katib_crd_found": False},
                blockers=(
                    f"Katib Experiment CRD not found ({EXPERIMENT_CRD_NAME}). "
                    f"Install Katib: {_INSTALL_HINT}",
                ),
            )
        return CheckResult(
            info={"katib_crd_found": False},
            blockers=(f"Cannot check Katib CRD: {e}",),
        )

    served = [v.name for v in (crd.spec.versions or []) if getattr(v, "served", True)]
    return CheckResult(info={"katib_crd_found": True, "katib_crd_versions": served})


def _find_controller_pod() -> Any | None:
    """Locate the Katib controller pod, preferring its conventional namespace."""
    core_api = mcp_utils.get_core_v1_api()
    pods = core_api.list_namespaced_pod(
        namespace=KATIB_CONTROLLER_NAMESPACE_DEFAULT,
        label_selector=KATIB_CONTROLLER_LABELS,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )
    if not pods.items:
        pods = core_api.list_pod_for_all_namespaces(
            label_selector=KATIB_CONTROLLER_LABELS,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )
    return pods.items[0] if pods.items else None


def _check_controller() -> CheckResult:
    """Confirm the Katib controller pod is both Running and Ready.

    Phase alone is not enough: a pod can be ``Running`` while its container
    fails readiness, which is exactly what a Katib install missing its
    ClusterRoles looks like. Treating that as healthy reports a false green.
    """
    try:
        pod = _find_controller_pod()
    except Exception as e:
        return CheckResult(
            info={"controller_status": "check_failed"},
            warnings=(f"Cannot check Katib controller pods: {e}",),
        )

    if pod is None:
        return CheckResult(
            info={"controller_status": "not_found"},
            blockers=(
                "Katib controller pod not found. Checked namespace "
                f"'{KATIB_CONTROLLER_NAMESPACE_DEFAULT}' and all namespaces.",
            ),
        )

    phase = pod.status.phase if pod.status else "Unknown"
    where = f"{pod.metadata.namespace}/{pod.metadata.name}"
    info: dict[str, Any] = {
        "controller_status": phase,
        "controller_namespace": pod.metadata.namespace,
        "controller_name": pod.metadata.name,
    }

    if phase != "Running":
        return CheckResult(
            info=info,
            blockers=(
                f"Katib controller pod is in '{phase}' state (expected 'Running'). Pod: {where}",
            ),
        )

    statuses = getattr(pod.status, "container_statuses", None) or []
    not_ready = [cs.name for cs in statuses if not cs.ready]
    if not_ready:
        info["controller_ready"] = False
        return CheckResult(
            info=info,
            blockers=(
                "Katib controller pod is Running but not Ready (containers failing "
                f"readiness: {', '.join(not_ready)}). Pod: {where}. Check controller "
                "logs and verify Katib RBAC (ClusterRole/ClusterRoleBinding) is installed.",
            ),
        )
    if not statuses:
        # Statuses not published yet: indeterminate rather than known-bad.
        return CheckResult(
            info=info,
            warnings=(
                "Katib controller pod reports no container statuses yet; "
                "readiness could not be confirmed.",
            ),
        )

    info["controller_ready"] = True
    return CheckResult(info=info)


def _check_trainer() -> CheckResult:
    """Report whether the trainer client is usable for cross-client workflows.

    Never a blocker: the optimizer is fully usable on its own.
    """
    try:
        mcp_utils.get_trainer_client()
    except Exception:
        return CheckResult(
            info={"trainer_available": False},
            warnings=(
                "Trainer client not available. Cross-client workflows "
                "(optimize → train with best hyperparameters) require "
                "--clients trainer,optimizer",
            ),
        )
    return CheckResult(info={"trainer_available": True})


def katib_pre_flight() -> dict[str, Any]:
    """One-shot readiness check for Katib hyperparameter optimization.

    Validates that the Katib/Optimizer infrastructure is ready:

    1. The Experiment CRD is registered on the cluster
    2. The Katib controller pod is Running **and** passing readiness
    3. Whether the trainer client is loaded, for cross-client workflows

    Call this FIRST before any optimizer operations. It never raises for an
    unhealthy cluster; problems are reported in ``blockers``.

    Returns:
        dict: Response containing ``ready`` (bool), ``blockers`` (issues that
        prevent use), ``warnings`` (non-blocking), ``katib_crd_found``,
        ``katib_crd_versions``, ``controller_status``, ``controller_ready``
        and ``trainer_available``.
    """
    info: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    for check in (_check_experiment_crd, _check_controller, _check_trainer):
        result = check()
        info.update(result.info)
        blockers.extend(result.blockers)
        warnings.extend(result.warnings)

    info["blockers"] = blockers
    info["warnings"] = warnings
    info["ready"] = not blockers

    try:
        return ToolResponse(data=info).model_dump()
    except Exception as e:
        logger.warning("katib_pre_flight failed: %s", e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
