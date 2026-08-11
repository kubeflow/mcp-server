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
from typing import Any

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_CRD_NAME,
    KATIB_CONTROLLER_LABELS,
    KATIB_CONTROLLER_NAMESPACE_DEFAULT,
)

logger = logging.getLogger(__name__)


def katib_pre_flight() -> dict[str, Any]:
    """One-shot readiness check for Katib hyperparameter optimization.

    Validates that the Katib/Optimizer infrastructure is ready:
    1. Experiment CRD exists on the cluster
    2. Katib controller pod is running
    3. Reports trainer cross-client availability

    Call this FIRST before any optimizer operations.

    Returns:
        dict: Response containing ``blockers`` (list of issues preventing use),
              ``warnings`` (non-blocking issues), ``katib_crd_versions``,
              and ``controller_status``.

    Raises:
        ToolError: If cluster is unreachable (``CLUSTER_ERROR``).
    """
    blockers: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    # ── Check 1: Experiment CRD exists ──────────────────────────────
    try:
        api = mcp_utils.get_apiextensions_api()
        crd = api.read_custom_resource_definition(
            name=EXPERIMENT_CRD_NAME,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )
        versions = [v.name for v in (crd.spec.versions or []) if getattr(v, "served", True)]
        info["katib_crd_versions"] = versions
        info["katib_crd_found"] = True
    except Exception as e:
        from kubeflow_mcp.common.types import is_k8s_not_found

        if is_k8s_not_found(e):
            blockers.append(
                f"Katib Experiment CRD not found ({EXPERIMENT_CRD_NAME}). "
                "Install Katib: https://www.kubeflow.org/docs/components/katib/installation/"
            )
            info["katib_crd_found"] = False
        else:
            blockers.append(f"Cannot check Katib CRD: {e}")
            info["katib_crd_found"] = False

    # ── Check 2: Controller pod health ──────────────────────────────
    try:
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

        if not pods.items:
            blockers.append(
                "Katib controller pod not found. "
                f"Checked namespace '{KATIB_CONTROLLER_NAMESPACE_DEFAULT}' and all namespaces."
            )
            info["controller_status"] = "not_found"
        else:
            pod = pods.items[0]
            phase = pod.status.phase if pod.status else "Unknown"
            info["controller_status"] = phase
            info["controller_namespace"] = pod.metadata.namespace
            info["controller_name"] = pod.metadata.name

            if phase != "Running":
                blockers.append(
                    f"Katib controller pod is in '{phase}' state "
                    f"(expected 'Running'). Pod: {pod.metadata.namespace}/{pod.metadata.name}"
                )
            else:
                # Phase "Running" only means the pod was admitted and started —
                # containers can still be failing their readiness probes (e.g. a
                # controller that cannot watch its CRDs because RBAC is missing).
                # Without this check pre-flight reports a false green light.
                statuses = getattr(pod.status, "container_statuses", None) or []
                not_ready = [cs.name for cs in statuses if not cs.ready]
                if not_ready:
                    info["controller_ready"] = False
                    blockers.append(
                        "Katib controller pod is Running but not Ready "
                        f"(containers failing readiness: {', '.join(not_ready)}). "
                        f"Pod: {pod.metadata.namespace}/{pod.metadata.name}. "
                        "Check controller logs and verify Katib RBAC "
                        "(ClusterRole/ClusterRoleBinding) is installed."
                    )
                elif statuses:
                    info["controller_ready"] = True
                else:
                    # Container statuses not published yet — readiness is
                    # indeterminate rather than known-bad, so warn instead.
                    warnings.append(
                        "Katib controller pod reports no container statuses yet; "
                        "readiness could not be confirmed."
                    )
    except Exception as e:
        warnings.append(f"Cannot check Katib controller pods: {e}")
        info["controller_status"] = "check_failed"

    # ── Check 3: Trainer module availability (cross-client hint) ────
    try:
        mcp_utils.get_trainer_client()
        info["trainer_available"] = True
    except Exception:
        info["trainer_available"] = False
        warnings.append(
            "Trainer client not available. Cross-client workflows "
            "(optimize → train with best hyperparameters) require "
            "--clients trainer,optimizer"
        )

    # ── Build response ──────────────────────────────────────────────
    info["blockers"] = blockers
    info["warnings"] = warnings
    info["ready"] = len(blockers) == 0

    try:
        return ToolResponse(data=info).model_dump()
    except Exception as e:
        logger.warning("katib_pre_flight failed: %s", e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
