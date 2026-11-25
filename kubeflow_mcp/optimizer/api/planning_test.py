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

"""Unit tests for katib_pre_flight (CRD + controller + trainer detection)."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.optimizer.api.planning import katib_pre_flight

_UTILS = "kubeflow_mcp.common.utils"


def _crd(versions=("v1beta1",)):
    return SimpleNamespace(
        spec=SimpleNamespace(versions=[SimpleNamespace(name=v, served=True) for v in versions])
    )


def _pod(phase="Running", namespace="kubeflow", name="katib-controller-abc", ready=True):
    """Build a controller pod. Real Running pods publish container statuses."""
    statuses = None if ready is None else [SimpleNamespace(name="katib-controller", ready=ready)]
    return SimpleNamespace(
        status=SimpleNamespace(phase=phase, container_statuses=statuses),
        metadata=SimpleNamespace(namespace=namespace, name=name),
    )


def _denied(status, message):
    """An admission denial as the API server returns it."""
    err = ApiException(status=status, reason="Bad Request")
    err.body = json.dumps({"message": message})
    return err


_KATIB_DENIAL = (
    'admission webhook "validator.experiment.katib.kubeflow.org" denied the request: '
    "spec.trialTemplate: Required value"
)


def _katib_webhook():
    """The probe's API as a healthy Katib answers it: refusing the incomplete probe."""
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(400, _KATIB_DENIAL)
    return webhook


@contextmanager
def _patch(apiext, core, trainer_ok=True, webhook=None):
    """Patch every cluster call pre-flight makes.

    ``webhook`` is the CustomObjects API backing the dry-run admission probe,
    healthy Katib by default. It is always patched: leaving it out lets the
    check reach a real cluster, which both slows the suite and makes results
    depend on the developer's kubeconfig.
    """
    trainer_ctx = (
        patch(f"{_UTILS}.get_trainer_client", return_value=MagicMock())
        if trainer_ok
        else patch(f"{_UTILS}.get_trainer_client", side_effect=RuntimeError("no trainer"))
    )
    with (
        patch(f"{_UTILS}.get_apiextensions_api", return_value=apiext),
        patch(f"{_UTILS}.get_core_v1_api", return_value=core),
        trainer_ctx,
        patch(f"{_UTILS}.get_custom_objects_api", return_value=webhook or _katib_webhook()),
        patch(f"{_UTILS}.get_optimizer_effective_namespace", return_value="default"),
    ):
        yield


def test_pre_flight_all_healthy():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod()])
    with _patch(apiext, core, trainer_ok=True):
        result = katib_pre_flight()
    assert result["success"] is True
    data = result["data"]
    assert data["ready"] is True
    assert data["blockers"] == []
    assert data["katib_crd_found"] is True
    assert data["katib_crd_versions"] == ["v1beta1"]
    assert data["controller_status"] == "Running"
    assert data["trainer_sdk_available"] is True
    assert data["webhook_reachable"] is True
    assert data["warnings"] == []


def test_pre_flight_crd_missing():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.side_effect = ApiException(
        status=404, reason="Not Found"
    )
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod()])
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["katib_crd_found"] is False
    assert any("CRD not found" in b for b in data["blockers"])


def test_pre_flight_controller_not_running():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod(phase="Pending")])
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["controller_status"] == "Pending"
    assert any("Pending" in b for b in data["blockers"])


def test_pre_flight_no_controller_pods():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    core.list_pod_for_all_namespaces.return_value = SimpleNamespace(items=[])
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["controller_status"] == "not_found"


def test_pre_flight_controller_running_but_not_ready():
    """Regression: a Running pod whose container fails readiness (e.g. missing
    Katib RBAC) must not be reported as ready."""
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod(ready=False)])
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["controller_ready"] is False
    assert any("not Ready" in b for b in data["blockers"])
    assert any("RBAC" in b for b in data["blockers"])


def test_pre_flight_ready_controller_sets_controller_ready():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod(ready=True)])
    with _patch(apiext, core):
        result = katib_pre_flight()
    assert result["data"]["controller_ready"] is True


def test_pre_flight_missing_container_statuses_warns_only():
    """Statuses not published yet is indeterminate, not a hard blocker."""
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod(ready=None)])
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is True
    assert any("readiness could not be confirmed" in w for w in data["warnings"])


def test_pre_flight_trainer_unavailable_is_warning_not_blocker():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod()])
    with _patch(apiext, core, trainer_ok=False):
        result = katib_pre_flight()
    data = result["data"]
    # Trainer missing must not block optimizer-only usage.
    assert data["ready"] is True
    assert data["trainer_sdk_available"] is False
    assert any("Trainer SDK client could not be initialized" in w for w in data["warnings"])


def _healthy_cluster():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod()])
    return apiext, core


def test_pre_flight_blocks_when_webhook_is_unreachable():
    """Regression: a Ready controller whose webhook the API server cannot reach.

    Katib registers its webhooks with failurePolicy=Fail, so every create is
    rejected while the CRD is installed and the pod looks perfectly healthy.
    Reporting ready here is a false green.
    """
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(500, _KATIB_UNREACHABLE)
    with _patch(apiext, core, webhook=webhook):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["webhook_reachable"] is False
    assert data["controller_ready"] is True, "the pod itself is healthy; only admission is broken"
    assert any("cannot call an admission webhook" in b for b in data["blockers"])
    assert any("katib-controller Service" in b for b in data["blockers"])


# What the API server answered on the live test cluster when it could not reach Katib.
_KATIB_UNREACHABLE = (
    'Internal error occurred: failed calling webhook "defaulter.experiment.katib.kubeflow.org": '
    'failed to call webhook: Post "https://katib-controller.kubeflow.svc:443/mutate-experiment'
    '?timeout=10s": context deadline exceeded'
)


def test_pre_flight_webhook_probe_is_a_dry_run():
    """The probe must never persist an Experiment."""
    apiext, core = _healthy_cluster()
    webhook = _katib_webhook()
    with _patch(apiext, core, webhook=webhook):
        katib_pre_flight()
    kwargs = webhook.create_namespaced_custom_object.call_args.kwargs
    assert kwargs["dry_run"] == "All"
    assert kwargs["plural"] == "experiments"


@pytest.mark.parametrize("status", [400, 422])
def test_pre_flight_katib_rejecting_the_probe_counts_as_reachable(status):
    """Katib refusing the deliberately incomplete probe proves its webhook answered."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(status, _KATIB_DENIAL)
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["ready"] is True
    assert data["webhook_reachable"] is True


def test_pre_flight_a_rejection_before_katib_proves_nothing():
    """Another mutating webhook can refuse the probe before Katib ever sees it."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(
        400, 'admission webhook "policy.example.com" denied the request: labels required'
    )
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["webhook_reachable"] is None
    assert any("rejected before Katib answered" in w for w in data["warnings"])


def test_pre_flight_blocks_on_a_namespace_katib_will_not_accept():
    """Katib rejects every Experiment in a namespace without its injection label,
    so creation cannot work there however healthy everything else is."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(
        400,
        'admission webhook "validator.experiment.katib.kubeflow.org" denied the request: '
        'Cannot create the Experiment "katib-pre-flight-probe" in namespace "default": '
        'the namespace lacks label "katib.kubeflow.org/metrics-collector-injection: enabled"',
    )
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["webhook_reachable"] is True
    assert data["ready"] is False
    assert any("metrics-collector-injection=enabled" in b for b in data["blockers"])


def test_pre_flight_webhook_forbidden_warns_rather_than_blocks():
    """Without create permission the probe cannot conclude anything."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = ApiException(
        status=403, reason="Forbidden"
    )
    with _patch(apiext, core, webhook=webhook):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is True
    assert data["webhook_reachable"] is None
    assert any("Cannot verify Katib admission webhooks" in w for w in data["warnings"])


def test_pre_flight_an_admitted_probe_means_katib_never_answered():
    """A healthy Katib always refuses the incomplete probe, so admission proves
    only that its webhook registrations are missing."""
    apiext, core = _healthy_cluster()
    with _patch(apiext, core, webhook=MagicMock()):
        data = katib_pre_flight()["data"]
    assert data["webhook_reachable"] is None
    assert any("validating webhook did not answer" in w for w in data["warnings"])


def test_pre_flight_rejected_credentials_are_not_a_webhook_outage():
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = ApiException(
        status=401, reason="Unauthorized"
    )
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["webhook_reachable"] is None
    assert not any("webhook" in b for b in data["blockers"])
    assert any("rejected this server's credentials" in w for w in data["warnings"])


def test_pre_flight_another_failing_webhook_blocks_without_blaming_katib():
    """Any webhook the API server cannot call stops every create, but only
    Katib's own points at the katib-controller Service."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(
        500,
        'Internal error occurred: failed calling webhook "policy.example.com": connection refused',
    )
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["ready"] is False
    assert data["webhook_reachable"] is None
    assert any("policy.example.com" in b for b in data["blockers"])
    assert not any("katib-controller" in b for b in data["blockers"])


def test_pre_flight_an_unexplained_probe_failure_blocks_without_a_diagnosis():
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = _denied(503, "etcd is unavailable")
    with _patch(apiext, core, webhook=webhook):
        data = katib_pre_flight()["data"]
    assert data["ready"] is False
    assert data["webhook_reachable"] is None
    assert data["blockers"] == [
        "A dry-run Experiment create failed, so creation is likely to fail too: etcd is unavailable"
    ]


def test_pre_flight_skips_the_probe_when_the_namespace_cannot_be_resolved():
    apiext, core = _healthy_cluster()
    webhook = _katib_webhook()
    with (
        _patch(apiext, core, webhook=webhook),
        patch(
            f"{_UTILS}.get_optimizer_effective_namespace",
            side_effect=RuntimeError("no kubeconfig found"),
        ),
    ):
        data = katib_pre_flight()["data"]
    webhook.create_namespaced_custom_object.assert_not_called()
    assert data["webhook_reachable"] is None
    assert data["ready"] is True
    assert any("default namespace could not be resolved" in w for w in data["warnings"])


def test_pre_flight_logs_why_a_sub_check_failed(caplog):
    """A sub-check's failure is reported as a warning; its cause goes to the log."""
    apiext, core = _healthy_cluster()
    with caplog.at_level("WARNING"), _patch(apiext, core, trainer_ok=False):
        katib_pre_flight()
    (record,) = [r for r in caplog.records if "Trainer SDK client" in r.getMessage()]
    assert record.exc_info[1].args == ("no trainer",)


def test_pre_flight_controller_falls_back_to_all_namespaces():
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    # Not in the default kubeflow namespace, but found cluster-wide.
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    core.list_pod_for_all_namespaces.return_value = SimpleNamespace(
        items=[_pod(namespace="katib-system")]
    )
    with _patch(apiext, core):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is True
    assert data["controller_namespace"] == "katib-system"


@pytest.mark.parametrize("order", ["pending_first", "healthy_first"])
def test_pre_flight_reports_the_healthy_controller_whatever_the_list_order(order):
    """During a rollout an old Pending pod can be listed next to a Ready one."""
    pending = _pod(phase="Pending", name="katib-controller-new", ready=None)
    healthy = _pod(name="katib-controller-old")
    items = [pending, healthy] if order == "pending_first" else [healthy, pending]
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=items)
    with _patch(apiext, core):
        data = katib_pre_flight()["data"]
    assert data["controller_name"] == "katib-controller-old"
    assert data["controller_ready"] is True
    assert data["ready"] is True


def test_pre_flight_skips_the_webhook_probe_outside_the_allowlist():
    """The probe dry-runs a create in the default namespace, which policy may forbid."""
    apiext = MagicMock()
    apiext.read_custom_resource_definition.return_value = _crd()
    core = MagicMock()
    core.list_namespaced_pod.return_value = SimpleNamespace(items=[_pod()])
    webhook = MagicMock()
    with (
        _patch(apiext, core, webhook=webhook),
        patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["team-a"]),
    ):
        data = katib_pre_flight()["data"]
    webhook.create_namespaced_custom_object.assert_not_called()
    assert data["webhook_reachable"] is None
    assert any("Skipped the Katib admission webhook check" in w for w in data["warnings"])
    assert data["ready"] is True
