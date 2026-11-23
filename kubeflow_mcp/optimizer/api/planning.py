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
from kubeflow_mcp.optimizer.api._common import api_message, check_optimizer_namespace
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_CRD_NAME,
    EXPERIMENT_KIND,
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    KATIB_CONTROLLER_LABELS,
    KATIB_CONTROLLER_NAMESPACE_DEFAULT,
)

logger = logging.getLogger(__name__)

# Name for the dry-run probe in _check_webhook. Never persisted, but a distinct
# name keeps it obvious in an audit log that this was a readiness check.
_PROBE_NAME = "katib-pre-flight-probe"

# Katib's webhooks are named validator/defaulter.experiment.katib.kubeflow.org,
# and the API server quotes the name in every denial they issue, so its presence
# proves the probe reached Katib rather than being stopped by another webhook.
_KATIB_WEBHOOK_MARK = "experiment.katib.kubeflow.org"
# Katib's validating webhook rejects every Experiment in a namespace without
# this label, before looking at the Experiment itself.
_INJECTION_LABEL = "katib.kubeflow.org/metrics-collector-injection"


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


def _readiness_rank(pod: Any) -> int:
    """0 for a Running and Ready pod, 1 for Running only, 2 otherwise."""
    status = pod.status
    if not status or status.phase != "Running":
        return 2
    statuses = getattr(status, "container_statuses", None) or []
    return 0 if statuses and all(cs.ready for cs in statuses) else 1


def _find_controller_pod() -> Any | None:
    """Locate the Katib controller pod, preferring its conventional namespace.

    During a rollout two controller pods can coexist, so the healthiest one is
    reported rather than whichever the API listed first.
    """
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
    return min(pods.items, key=_readiness_rank) if pods.items else None


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


def _check_webhook() -> CheckResult:
    """Confirm the API server can actually reach Katib's admission webhooks.

    A Ready controller is not sufficient. Katib registers its webhooks with
    ``failurePolicy: Fail``, so if the API server cannot reach the controller
    pod every create is rejected even though the CRD is installed and the pod is
    healthy. That combination reports a green pre-flight while no experiment can
    be created, which is the failure this check exists to catch.

    Uses a server-side dry run: admission runs in full, nothing is persisted.
    The probe targets the default namespace, so it is skipped, with a warning,
    when policy does not allow that namespace.
    """
    ns_err = check_optimizer_namespace(None)
    if ns_err is not None:
        return CheckResult(
            info={"webhook_reachable": None},
            warnings=(
                "Skipped the Katib admission webhook check: it dry-runs a create in the "
                f"default namespace, which policy does not allow ({ns_err.error}).",
            ),
        )

    minimal = {
        "apiVersion": f"{KATIB_API_GROUP}/{KATIB_API_VERSION}",
        "kind": EXPERIMENT_KIND,
        "metadata": {"name": _PROBE_NAME},
        "spec": {
            "objective": {"type": "maximize", "objectiveMetricName": "probe"},
            "algorithm": {"algorithmName": "random"},
            "maxTrialCount": 1,
            "parallelTrialCount": 1,
            "parameters": [
                {
                    "name": "probe",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.1", "max": "0.2"},
                }
            ],
            "trialTemplate": {"primaryContainerName": "probe", "trialSpec": {}},
        },
    }

    try:
        namespace = mcp_utils.get_optimizer_effective_namespace(None)
        mcp_utils.get_custom_objects_api().create_namespaced_custom_object(
            group=KATIB_API_GROUP,
            version=KATIB_API_VERSION,
            namespace=namespace,
            plural=EXPERIMENT_PLURAL,
            body=minimal,
            dry_run="All",
            _request_timeout=mcp_utils.K8S_WRITE_TIMEOUT,
        )
    except Exception as e:
        status = getattr(e, "status", None)
        message = api_message(e)
        if status in (400, 422):
            return _classify_probe_rejection(message)
        if status == 403:
            return CheckResult(
                info={"webhook_reachable": None},
                warnings=(
                    "Cannot verify Katib admission webhooks: no permission to dry-run "
                    "an Experiment create. Creation may still fail at runtime.",
                ),
            )
        return CheckResult(
            info={"webhook_reachable": False},
            blockers=(
                "Katib's admission webhooks are not reachable by the API server, so no "
                f"experiment can be created: {api_message(e)} A Ready controller pod is "
                "not sufficient; check that the API server can reach the katib-controller "
                "Service on its webhook port.",
            ),
        )
    return CheckResult(info={"webhook_reachable": True})


def _classify_probe_rejection(message: str) -> CheckResult:
    """Interpret the probe being refused at admission.

    The probe spec is deliberately incomplete, so Katib's validating webhook
    refusing it on content proves the webhook is reachable. A refusal Katib did
    not issue (another admission webhook ran first) proves nothing either way.
    """
    if _KATIB_WEBHOOK_MARK not in message:
        return CheckResult(
            info={"webhook_reachable": None},
            warnings=(
                "Could not confirm Katib's admission webhooks: the dry-run probe was "
                f"rejected before Katib answered ({message}).",
            ),
        )
    if _INJECTION_LABEL in message:
        return CheckResult(
            info={"webhook_reachable": True},
            blockers=(
                f"Katib rejects every Experiment in the default namespace: {message} "
                f"Label it with {_INJECTION_LABEL}=enabled, or pass a labelled namespace "
                "to the creation tools.",
            ),
        )
    return CheckResult(info={"webhook_reachable": True})


def _check_trainer() -> CheckResult:
    """Report whether the Trainer SDK client can be initialized.

    This says whether the cluster and kubeconfig support the Trainer client, not
    whether this server registered the trainer tools: that depends on
    ``--clients``, which the agent can see from its own tool list. Never a
    blocker: the optimizer is fully usable on its own.
    """
    try:
        mcp_utils.get_trainer_client()
    except Exception:
        return CheckResult(
            info={"trainer_sdk_available": False},
            warnings=(
                "The Trainer SDK client could not be initialized, so cross-client "
                "workflows (optimize, then train with the best hyperparameters) are "
                "unavailable.",
            ),
        )
    return CheckResult(info={"trainer_sdk_available": True})


def katib_pre_flight() -> dict[str, Any]:
    """One-shot readiness check for Katib hyperparameter optimization.

    Validates that the Katib/Optimizer infrastructure is ready:

    1. The Experiment CRD is registered on the cluster
    2. The Katib controller pod is Running **and** passing readiness
    3. The API server can actually reach Katib's admission webhooks, without
       which every create is rejected however healthy the controller looks
    4. Whether the Trainer SDK client can be initialized, for cross-client
       workflows

    Call this FIRST before any optimizer operations. It never raises for an
    unhealthy cluster; problems are reported in ``blockers``.

    Returns:
        dict: Response containing ``ready`` (bool), ``blockers`` (issues that
        prevent use), ``warnings`` (non-blocking), ``katib_crd_found``,
        ``katib_crd_versions``, ``controller_status``, ``controller_ready``,
        ``webhook_reachable`` and ``trainer_sdk_available``.
    """
    info: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    for check in (_check_experiment_crd, _check_controller, _check_webhook, _check_trainer):
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
