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

"""Tests for trainer/api/platform.py — CRD inspection and runtime CRUD."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client.exceptions import ApiException
from tests.common import (
    FAILED,
    KUBERNETES_ERROR,
    PERMISSION_DENIED,
    RESOURCE_NOT_FOUND,
    SUCCESS,
    VALIDATION_ERROR,
    TestCase,
    assert_test_case,
)

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.conftest import create_mock_trainjob, verify_tool_error, verify_tool_success
from kubeflow_mcp.core.policy import get_allowed_tools
from kubeflow_mcp.trainer.api.platform import (
    _get_controller_namespace,
    create_runtime,
    delete_runtime,
    inspect_controller,
    inspect_crd,
    patch_runtime,
)

# ─── Validation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="rejects invalid top-level patch keys",
            expected_status=FAILED,
            config={
                "name": "torchtune-llama",
                "patch": {"status": {"phase": "Ready"}},
                "confirmed": False,
            },
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="preview accepts valid patch keys",
            expected_status=SUCCESS,
            config={
                "name": "torchtune-llama",
                "patch": {"spec": {"template": {}}},
                "confirmed": False,
            },
        ),
    ],
)
def test_patch_runtime_validation(test_case):
    assert_test_case(test_case, patch_runtime)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="rejects invalid top-level spec keys",
            expected_status=FAILED,
            config={
                "name": "torchtune-llama",
                "spec": {"replicas": 1},
                "confirmed": False,
            },
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="preview accepts valid spec keys",
            expected_status=SUCCESS,
            config={
                "name": "torchtune-llama",
                "spec": {"template": {"spec": {"numNodes": 1}}},
                "confirmed": False,
            },
        ),
    ],
)
def test_create_runtime_validation(test_case):
    assert_test_case(test_case, create_runtime)


_BAD_RUNTIME_NAMES = [
    "",
    "   ",
    "UPPER-Case",
    "bad_underscore",
    "name with space",
    "../escape",
    "name/slash",
    "-leading",
    "trailing-",
    "a" * 300,
]


@pytest.mark.parametrize("bad_name", _BAD_RUNTIME_NAMES)
def test_runtime_tools_reject_invalid_names_before_api_call(bad_name, mock_k8s_apis):
    """Runtime CRUD must validate names like every other trainer module.

    The K8s API would reject most of these too, but the tools reported success
    and issued the call, which is inconsistent with discovery/monitoring and
    lets an empty name reach a cluster-scoped delete.
    """
    api = mock_k8s_apis["custom"]

    for result in (
        delete_runtime(bad_name, confirmed=True),
        patch_runtime(bad_name, patch={"spec": {"template": {}}}, confirmed=True),
        create_runtime(bad_name, spec={"template": {}}, confirmed=True),
    ):
        verify_tool_error(result, error_code=VALIDATION_ERROR)

    assert not api.delete_cluster_custom_object.called
    assert not api.patch_cluster_custom_object.called
    assert not api.create_cluster_custom_object.called


@pytest.mark.parametrize("bad_namespace", ["my namespace", "INVALID_NS", "namespace/name"])
def test_inspect_controller_rejects_invalid_namespace_before_api_call(bad_namespace, monkeypatch):
    core_api_requested = False

    def fail_if_core_api_requested():
        nonlocal core_api_requested
        core_api_requested = True
        raise AssertionError("Kubernetes API must not be initialized for invalid namespace")

    monkeypatch.setattr(mcp_utils, "get_core_v1_api", fail_if_core_api_requested)
    result = inspect_controller(namespace=bad_namespace)

    verify_tool_error(result, error_code=VALIDATION_ERROR)
    assert not core_api_requested


@pytest.mark.parametrize(
    "good_name",
    ["torchtune-llama3.2-1b", "torchtune-qwen2.5-1.5b", "torch-distributed", "r1"],
)
def test_runtime_tools_accept_dotted_runtime_names(good_name, mock_k8s_apis):
    """Runtime names are K8s object names, so the torchtune runtimes Trainer
    ships (which contain dots) must keep working."""
    result = delete_runtime(good_name, confirmed=False)

    data = verify_tool_success(result)
    assert data["runtime"] == good_name


# ─── Runtime CRUD ───────────────────────────────────────────────────────────


def test_patch_runtime_confirmed_applies_strategic_patch(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    api.patch_cluster_custom_object.return_value = {
        "metadata": {"resourceVersion": "123"},
    }
    patch_body = {"spec": {"template": {"spec": {"numNodes": 2}}}}

    result = patch_runtime("torchtune-llama", patch=patch_body, confirmed=True)

    data = verify_tool_success(result)
    assert data["runtime"] == "torchtune-llama"
    assert data["patched"] is True
    assert data["resource_version"] == "123"
    api.patch_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        name="torchtune-llama",
        body=patch_body,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_patch_runtime_invalid_runtime_name_returns_not_found(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    api.patch_cluster_custom_object.side_effect = ApiException(status=404, reason="Not Found")

    result = patch_runtime(
        "missing-runtime",
        patch={"spec": {"labels": {"env": "test"}}},
        confirmed=True,
    )

    error = verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)
    assert "missing-runtime" in error["error"]


def test_create_runtime_confirmed_creates_runtime(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    api.create_cluster_custom_object.return_value = {
        "metadata": {"resourceVersion": "9"},
    }
    spec = {"template": {"spec": {"numNodes": 1}}}

    result = create_runtime("new-runtime", spec=spec, confirmed=True)

    data = verify_tool_success(result)
    assert data["runtime"] == "new-runtime"
    assert data["created"] is True
    assert data["resource_version"] == "9"
    api.create_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        body={
            "apiVersion": f"{trainer_constants.GROUP}/{trainer_constants.VERSION}",
            "kind": "ClusterTrainingRuntime",
            "metadata": {"name": "new-runtime"},
            "spec": spec,
        },
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_create_runtime_name_collision_returns_kubernetes_error(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    collision = ApiException(status=409, reason="Already Exists")
    api.create_cluster_custom_object.side_effect = collision

    result = create_runtime(
        "existing-runtime",
        spec={"labels": {"team": "ml"}},
        confirmed=True,
    )

    error = verify_tool_error(result, error_code=KUBERNETES_ERROR)
    assert error["details"]["message"] == str(collision)


def test_delete_runtime_preview_lists_dependent_trainjobs(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    api.list_cluster_custom_object.return_value = {
        "items": [
            create_mock_trainjob(name="job-a", namespace="ml"),
            create_mock_trainjob(name="job-b", namespace="ml"),
        ],
    }

    result = delete_runtime("torchtune-llama", confirmed=False)

    data = verify_tool_success(result)
    assert data["dependent_count"] == 2
    assert data["dependent_jobs"] == [
        {"name": "job-a", "namespace": "ml"},
        {"name": "job-b", "namespace": "ml"},
    ]
    assert "2 TrainJob(s)" in data["warning"]
    api.delete_cluster_custom_object.assert_not_called()


def test_delete_runtime_confirmed_removes_runtime(mock_k8s_apis):
    api = mock_k8s_apis["custom"]
    api.list_cluster_custom_object.return_value = {
        "items": [create_mock_trainjob(name="job-a", namespace="ml")],
    }

    result = delete_runtime("torchtune-llama", confirmed=True)

    data = verify_tool_success(result)
    assert data["runtime"] == "torchtune-llama"
    assert data["deleted"] is True
    assert data["dependent_jobs_affected"] == 1
    api.delete_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        name="torchtune-llama",
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_non_admin_persona_cannot_manage_runtimes():
    allowed_tools = get_allowed_tools("ml-engineer")

    assert allowed_tools is not None
    assert "patch_runtime" not in allowed_tools
    assert "create_runtime" not in allowed_tools
    assert "delete_runtime" not in allowed_tools


# ─── inspect_crd ─────────────────────────────────────────────────────────────


def test_inspect_crd_not_found(mock_k8s_apis):
    api = mock_k8s_apis["apiextensions"]
    api.read_custom_resource_definition.side_effect = ApiException(status=404, reason="Not Found")

    result = inspect_crd("missing.trainer.kubeflow.org")

    verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)


# ─── inspect_controller ─────────────────────────────────────────────────────


@pytest.fixture
def scan_default_namespaces(monkeypatch):
    monkeypatch.setattr("kubeflow_mcp.trainer.api.platform._get_controller_namespace", lambda: None)


@pytest.mark.parametrize(
    ("lookup_error", "error_code"),
    [
        (ApiException(status=403, reason="Forbidden"), PERMISSION_DENIED),
        (ApiException(status=500, reason="Internal Server Error"), KUBERNETES_ERROR),
        (ConnectionError("Connection refused"), KUBERNETES_ERROR),
    ],
)
def test_inspect_controller_reports_failed_lookup_instead_of_missing_pod(
    mock_k8s_apis, scan_default_namespaces, lookup_error, error_code
):
    mock_k8s_apis["core_v1"].list_namespaced_pod.side_effect = lookup_error

    result = inspect_controller()

    error = verify_tool_error(result, error_code=error_code)
    assert "No controller pod found" not in error["error"]


def test_inspect_controller_missing_pod_is_still_not_found(mock_k8s_apis, scan_default_namespaces):
    mock_k8s_apis["core_v1"].list_namespaced_pod.return_value = MagicMock(items=[])

    result = inspect_controller()

    verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)


def test_inspect_controller_finds_pod_despite_forbidden_namespace(
    mock_k8s_apis, scan_default_namespaces
):
    core = mock_k8s_apis["core_v1"]
    pod = MagicMock()
    pod.metadata.name = "trainer-controller-manager-0"
    pod.metadata.namespace = "kubeflow-system"
    pod.status.phase = "Running"

    def list_pods(namespace, **_kwargs):
        if namespace == "kubeflow":
            raise ApiException(status=403, reason="Forbidden")
        return MagicMock(items=[pod])

    core.list_namespaced_pod.side_effect = list_pods
    core.read_namespaced_pod_log.return_value = "controller started"

    result = inspect_controller()

    data = verify_tool_success(result)
    assert data["pod"] == "trainer-controller-manager-0"
    assert data["namespace"] == "kubeflow-system"


def _controller_pod(namespace: str) -> MagicMock:
    pod = MagicMock()
    pod.metadata.name = "trainer-controller-manager-0"
    pod.metadata.namespace = namespace
    pod.status.phase = "Running"
    return pod


def test_inspect_controller_events_response(mock_k8s_apis, scan_default_namespaces):
    core = mock_k8s_apis["core_v1"]
    core.list_namespaced_pod.return_value = MagicMock(items=[_controller_pod("kubeflow-system")])
    event = MagicMock()
    event.type = "Warning"
    event.reason = "Failed"
    event.message = "Training job failed"
    event.count = 2
    event.first_timestamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    event.last_timestamp = None
    core.list_namespaced_event.return_value = MagicMock(items=[event])
    result = inspect_controller(view="events")
    data = verify_tool_success(result)
    assert data["pod"] == "trainer-controller-manager-0"
    assert data["namespace"] == "kubeflow-system"
    assert data["events"] == [
        {
            "type": "Warning",
            "reason": "Failed",
            "message": "Training job failed",
            "count": 2,
            "first_seen": "2024-01-02T03:04:05+00:00",
            "last_seen": None,
        }
    ]
    assert data["count"] == 1
    core.list_namespaced_event.assert_called_once_with(
        namespace="kubeflow-system",
        field_selector="involvedObject.name=trainer-controller-manager-0",
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_inspect_controller_rejects_namespace_outside_policy(mock_k8s_apis, tmp_policy_file):
    tmp_policy_file({"policy": {"namespaces": ["team-a"]}})

    result = inspect_controller(namespace="team-b")

    verify_tool_error(result, error_code=PERMISSION_DENIED)
    assert not mock_k8s_apis["core_v1"].list_namespaced_pod.called


def test_inspect_controller_allows_namespace_in_policy(mock_k8s_apis, tmp_policy_file):
    tmp_policy_file({"policy": {"namespaces": ["kubeflow-system"]}})
    core = mock_k8s_apis["core_v1"]
    core.list_namespaced_pod.return_value = MagicMock(items=[_controller_pod("kubeflow-system")])
    core.read_namespaced_pod_log.return_value = "controller started"

    result = inspect_controller(namespace="kubeflow-system")

    assert verify_tool_success(result)["namespace"] == "kubeflow-system"


def test_inspect_controller_auto_discovery_is_exempt_from_policy(
    mock_k8s_apis, scan_default_namespaces, tmp_policy_file
):
    """Omitting namespace targets the admin-configured controller namespace, not the caller's."""
    tmp_policy_file({"policy": {"namespaces": ["team-a"]}})
    core = mock_k8s_apis["core_v1"]
    core.list_namespaced_pod.return_value = MagicMock(items=[_controller_pod("kubeflow")])
    core.read_namespaced_pod_log.return_value = "controller started"

    result = inspect_controller()

    assert verify_tool_success(result)["namespace"] == "kubeflow"


# ─── _get_controller_namespace ───────────────────────────────────────────────


def test_get_controller_namespace_from_env(monkeypatch):
    monkeypatch.setenv("KUBEFLOW_MCP_CONTROLLER_NAMESPACE", "custom-controller-ns")
    assert _get_controller_namespace() == "custom-controller-ns"


def test_get_controller_namespace_from_config(monkeypatch):
    monkeypatch.delenv("KUBEFLOW_MCP_CONTROLLER_NAMESPACE", raising=False)
    mock_cfg = MagicMock()
    mock_cfg.trainer.controller_namespace = "config-controller-ns"
    monkeypatch.setattr("kubeflow_mcp.core.config.load_config", lambda: mock_cfg)
    assert _get_controller_namespace() == "config-controller-ns"


def test_get_controller_namespace_handles_config_exception(monkeypatch):
    monkeypatch.delenv("KUBEFLOW_MCP_CONTROLLER_NAMESPACE", raising=False)

    def raise_err():
        raise RuntimeError("config broken")

    monkeypatch.setattr("kubeflow_mcp.core.config.load_config", raise_err)
    assert _get_controller_namespace() is None


# ─── inspect_crd ─────────────────────────────────────────────────────────────


def test_inspect_crd_lists_all_trainer_crds(mock_k8s_apis):
    apiext = mock_k8s_apis["apiextensions"]
    trainer_crd = MagicMock()
    trainer_crd.metadata.name = "trainjobs.trainer.kubeflow.org"
    trainer_crd.spec.group = "trainer.kubeflow.org"
    trainer_crd.spec.scope = "Namespaced"
    v1 = MagicMock(served=True)
    v1.name = "v1alpha1"
    v1.served = True
    trainer_crd.spec.versions = [v1]

    other_crd = MagicMock()
    other_crd.metadata.name = "virtualservices.networking.istio.io"
    other_crd.spec.group = "networking.istio.io"

    apiext.list_custom_resource_definition.return_value = MagicMock(items=[trainer_crd, other_crd])

    result = inspect_crd()

    data = verify_tool_success(result)
    assert data["count"] == 1
    assert data["crds"][0]["name"] == "trainjobs.trainer.kubeflow.org"
    assert data["crds"][0]["group"] == "trainer.kubeflow.org"
    assert data["crds"][0]["versions"] == ["v1alpha1"]
    assert data["crds"][0]["served_versions"] == ["v1alpha1"]
    assert data["crds"][0]["scope"] == "Namespaced"


def test_inspect_crd_returns_schema_and_conditions(mock_k8s_apis):
    apiext = mock_k8s_apis["apiextensions"]
    crd = MagicMock()
    crd.metadata.name = "trainjobs.trainer.kubeflow.org"
    crd.spec.group = "trainer.kubeflow.org"
    crd.spec.scope = "Namespaced"

    v1 = MagicMock()
    v1.name = "v1alpha1"
    v1.served = True
    v1.storage = True
    spec_prop = MagicMock()
    spec_prop.properties = {"trainer": MagicMock(), "datasetConfig": MagicMock()}
    v1.schema.open_apiv3_schema.properties = {"spec": spec_prop}
    crd.spec.versions = [v1]

    cond = MagicMock()
    cond.type = "Established"
    cond.status = "True"
    crd.status.conditions = [cond]

    apiext.read_custom_resource_definition.return_value = crd

    result = inspect_crd(name="trainjobs.trainer.kubeflow.org")

    data = verify_tool_success(result)
    assert data["name"] == "trainjobs.trainer.kubeflow.org"
    assert data["group"] == "trainer.kubeflow.org"
    assert data["scope"] == "Namespaced"
    assert len(data["versions"]) == 1
    assert data["versions"][0]["name"] == "v1alpha1"
    assert data["versions"][0]["served"] is True
    assert data["versions"][0]["storage"] is True
    assert set(data["versions"][0]["spec_fields"]) == {"trainer", "datasetConfig"}
    assert data["conditions"] == [{"type": "Established", "status": "True"}]


def test_inspect_crd_generic_exception(mock_k8s_apis):
    apiext = mock_k8s_apis["apiextensions"]
    apiext.read_custom_resource_definition.side_effect = RuntimeError("API server down")

    result = inspect_crd(name="trainjobs.trainer.kubeflow.org")

    verify_tool_error(result, error_code=KUBERNETES_ERROR)


# ─── inspect_controller validation and edge cases ────────────────────────────


def test_inspect_controller_invalid_view():
    result = inspect_controller(view="metrics")
    error = verify_tool_error(result, error_code=VALIDATION_ERROR)
    assert "Invalid view 'metrics'" in error["error"]


def test_inspect_controller_invalid_tail_lines():
    result = inspect_controller(tail_lines=0)
    error = verify_tool_error(result, error_code=VALIDATION_ERROR)
    assert "tail_lines must be >= 1" in error["error"]


def test_inspect_controller_view_logs_success(mock_k8s_apis, scan_default_namespaces):
    core = mock_k8s_apis["core_v1"]
    core.list_namespaced_pod.return_value = MagicMock(items=[_controller_pod("kubeflow")])
    core.read_namespaced_pod_log.return_value = "line 1\nline 2"

    result = inspect_controller(view="logs", tail_lines=50)

    data = verify_tool_success(result)
    assert data["pod"] == "trainer-controller-manager-0"
    assert data["namespace"] == "kubeflow"
    assert data["logs"] == "line 1\nline 2"
    assert data["tail_lines"] == 50
    assert data["phase"] == "Running"


def test_inspect_controller_generic_read_log_exception(mock_k8s_apis, scan_default_namespaces):
    core = mock_k8s_apis["core_v1"]
    core.list_namespaced_pod.return_value = MagicMock(items=[_controller_pod("kubeflow")])
    core.read_namespaced_pod_log.side_effect = RuntimeError("Socket read timeout")

    result = inspect_controller(view="logs")

    verify_tool_error(result, error_code=KUBERNETES_ERROR)


# ─── patch_runtime edge cases ───────────────────────────────────────────────


def test_patch_runtime_missing_patch():
    result = patch_runtime(name="torch-distributed", patch=None)
    error = verify_tool_error(result, error_code=VALIDATION_ERROR)
    assert "patch parameter is required" in error["error"]


def test_patch_runtime_generic_exception(mock_k8s_apis):
    custom = mock_k8s_apis["custom"]
    custom.patch_cluster_custom_object.side_effect = ApiException(
        status=500, reason="Internal Error"
    )

    result = patch_runtime(
        name="torch-distributed",
        patch={"spec": {"template": {}}},
        confirmed=True,
    )

    verify_tool_error(result, error_code=KUBERNETES_ERROR)


# ─── create_runtime edge cases ──────────────────────────────────────────────


def test_create_runtime_missing_spec():
    result = create_runtime(name="new-runtime", spec=None)
    error = verify_tool_error(result, error_code=VALIDATION_ERROR)
    assert "spec parameter is required" in error["error"]


def test_create_runtime_generic_exception(mock_k8s_apis):
    custom = mock_k8s_apis["custom"]
    custom.create_cluster_custom_object.side_effect = ApiException(
        status=503, reason="Service Unavailable"
    )

    result = create_runtime(
        name="new-runtime",
        spec={"template": {}},
        confirmed=True,
    )

    verify_tool_error(result, error_code=KUBERNETES_ERROR)


# ─── delete_runtime edge cases ──────────────────────────────────────────────


def test_delete_runtime_not_found(mock_k8s_apis):
    custom = mock_k8s_apis["custom"]
    custom.list_cluster_custom_object.return_value = {"items": []}
    custom.delete_cluster_custom_object.side_effect = ApiException(status=404, reason="Not Found")

    result = delete_runtime(name="nonexistent-runtime", confirmed=True)

    error = verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)
    assert "not found" in error["error"]


def test_delete_runtime_generic_exception(mock_k8s_apis):
    custom = mock_k8s_apis["custom"]
    custom.list_cluster_custom_object.return_value = {"items": []}
    custom.delete_cluster_custom_object.side_effect = ApiException(
        status=500, reason="Internal Error"
    )

    result = delete_runtime(name="torch-distributed", confirmed=True)

    verify_tool_error(result, error_code=KUBERNETES_ERROR)


def test_delete_runtime_preview_handles_dependent_jobs_listing_exception(mock_k8s_apis):
    custom = mock_k8s_apis["custom"]
    custom.list_cluster_custom_object.side_effect = RuntimeError("RBAC list denied")

    result = delete_runtime(name="torch-distributed", confirmed=False)

    data = verify_tool_success(result)
    assert data["action"] == "preview"
    assert data["runtime"] == "torch-distributed"
    assert data["dependent_jobs"] == []
    assert data["dependent_count"] == 0
    assert "No dependent TrainJobs found." in data["warning"]
