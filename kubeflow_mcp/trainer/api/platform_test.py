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

import pytest
from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client.exceptions import ApiException
from tests.common import (
    FAILED,
    KUBERNETES_ERROR,
    RESOURCE_NOT_FOUND,
    SUCCESS,
    VALIDATION_ERROR,
    TestCase,
    assert_test_case,
)

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.conftest import create_mock_trainjob, verify_tool_error, verify_tool_success
from kubeflow_mcp.core.policy import get_allowed_tools
from kubeflow_mcp.trainer.api.platform import create_runtime, delete_runtime, patch_runtime

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


# Remaining TODOs are outside this PR's runtime CRUD slice.
# TODO(test): test inspect_crd — lists all Trainer CRDs
# TODO(test): test inspect_crd(name) — returns CRD schema and conditions
# TODO(test): test inspect_crd — invalid CRD name
# TODO(test): test inspect_controller(view="logs") — returns controller logs
# TODO(test): test inspect_controller(view="events") — returns controller events
