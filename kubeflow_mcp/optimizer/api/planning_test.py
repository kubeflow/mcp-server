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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


@contextmanager
def _patch(apiext, core, trainer_ok=True, webhook=None):
    """Patch every cluster call pre-flight makes.

    ``webhook`` is the CustomObjects API backing the dry-run admission probe. It
    is always patched: leaving it out lets the check reach a real cluster, which
    both slows the suite and makes results depend on the developer's kubeconfig.
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
        patch(f"{_UTILS}.get_custom_objects_api", return_value=webhook or MagicMock()),
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
    assert data["trainer_available"] is True


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
    assert data["trainer_available"] is False
    assert any("Trainer client not available" in w for w in data["warnings"])


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
    webhook.create_namespaced_custom_object.side_effect = ApiException(
        status=500, reason="Internal Server Error"
    )
    with _patch(apiext, core, webhook=webhook):
        result = katib_pre_flight()
    data = result["data"]
    assert data["ready"] is False
    assert data["webhook_reachable"] is False
    assert data["controller_ready"] is True, "the pod itself is healthy; only admission is broken"
    assert any("admission webhooks are not reachable" in b for b in data["blockers"])


def test_pre_flight_webhook_probe_is_a_dry_run():
    """The probe must never persist an Experiment."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    with _patch(apiext, core, webhook=webhook):
        katib_pre_flight()
    kwargs = webhook.create_namespaced_custom_object.call_args.kwargs
    assert kwargs["dry_run"] == "All"
    assert kwargs["plural"] == "experiments"


def test_pre_flight_webhook_rejection_still_counts_as_reachable():
    """A 400/422 means the webhook answered, which is all the probe asks."""
    apiext, core = _healthy_cluster()
    webhook = MagicMock()
    webhook.create_namespaced_custom_object.side_effect = ApiException(
        status=422, reason="Unprocessable Entity"
    )
    with _patch(apiext, core, webhook=webhook):
        result = katib_pre_flight()
    assert result["data"]["ready"] is True
    assert result["data"]["webhook_reachable"] is True


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
