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
    create_runtime,
    delete_runtime,
    inspect_controller,
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


# Remaining TODOs are outside this PR's runtime CRUD slice.
# TODO(test): test inspect_crd — lists all Trainer CRDs
# TODO(test): test inspect_crd(name) — returns CRD schema and conditions
# TODO(test): test inspect_crd — invalid CRD name
# TODO(test): test inspect_controller(view="logs") — returns controller logs
# TODO(test): test inspect_controller(view="events") — returns controller events
