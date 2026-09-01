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

"""Tests for trainer/api/discovery.py.

Covers discovery behavior, input validation, and status filter aliasing.
K8s API interaction tests require mocking the SDK and are marked as TODOs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from tests.common import RESOURCE_NOT_FOUND

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.conftest import (
    NOT_FOUND_NAME,
    VALID_JOB_NAME,
    verify_tool_error,
    verify_tool_success,
)
from kubeflow_mcp.trainer.api.discovery import (
    _JOB_STATUS_FILTER_ALIASES,
    _get_runtime_image,
    _trainjob_runtime_to_mcp,
    get_runtime,
    get_training_job,
    list_training_jobs,
)


class TestJobStatusFilterAliases:
    def test_succeeded_maps_to_complete(self):
        assert _JOB_STATUS_FILTER_ALIASES["Succeeded"] == "Complete"


class TestTrainjobRuntimeToMcp:
    def test_none_returns_none(self):
        assert _trainjob_runtime_to_mcp(None) is None

    def test_serializes_name(self):
        class FakeRuntime:
            name = "torchtune-llama"

        assert _trainjob_runtime_to_mcp(FakeRuntime()) == {"name": "torchtune-llama"}

    def test_empty_name_returns_none(self):
        class FakeRuntime:
            name = ""

        assert _trainjob_runtime_to_mcp(FakeRuntime()) is None


def test_get_training_job_returns_details(mock_trainer_client):
    result = get_training_job(name=VALID_JOB_NAME)
    data = verify_tool_success(result)
    assert data["name"] == VALID_JOB_NAME
    assert data["status"] == "Running"


def test_get_training_job_not_found(mock_trainer_client):
    result = get_training_job(name=NOT_FOUND_NAME)
    verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)


@pytest.mark.parametrize(
    ("tool", "kwargs", "client_path"),
    [
        (
            get_training_job,
            {"name": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        ),
        (
            get_runtime,
            {"name": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client",
        ),
        (
            list_training_jobs,
            {"runtime": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        ),
    ],
)
def test_rejects_invalid_resource_name_before_calling_sdk(tool, kwargs, client_path):
    with patch(client_path) as mock_client:
        result = tool(**kwargs)

    assert result["success"] is False
    assert result["error_code"] == ErrorCode.VALIDATION_ERROR
    mock_client.assert_not_called()


class TestGetRuntime:
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_extracts_sdk_trainer_metadata(self, mock_client_fn):
        mock_trainer = MagicMock()
        mock_trainer.framework = "torch"
        mock_trainer.image = "pytorch/pytorch:2.0"
        mock_trainer.num_nodes = 2
        mock_trainer.device = "gpu"
        mock_trainer.device_count = "4"
        mock_trainer.trainer_type.value = "CustomTrainer"

        mock_rt = MagicMock()
        mock_rt.name = "torch-distributed"
        mock_rt.trainer = mock_trainer
        mock_rt.pretrained_model = "meta-llama/Llama-3.2-1B"
        mock_rt.spec = None

        mock_client = MagicMock()
        mock_client.get_runtime.return_value = mock_rt
        mock_client_fn.return_value = mock_client

        result = get_runtime("torch-distributed")
        assert result["success"] is True
        data = result["data"]
        assert data["name"] == "torch-distributed"
        assert data["framework"] == "torch"
        assert data["image"] == "pytorch/pytorch:2.0"
        assert data["num_nodes"] == 2
        assert data["device"] == "gpu"
        assert data["device_count"] == "4"
        assert data["trainer_type"] == "CustomTrainer"
        assert data["pretrained_model"] == "meta-llama/Llama-3.2-1B"

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_with_real_sdk_runtime_objects(self, mock_client_fn):
        from kubeflow.trainer.types.types import Runtime, RuntimeTrainer, TrainerType

        real_trainer = RuntimeTrainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="pytorch/pytorch:2.0-cuda11.8",
            num_nodes=0,
            device="gpu",
            device_count="4",
        )
        real_rt = Runtime(
            name="torch-distributed",
            trainer=real_trainer,
            pretrained_model="meta-llama/Llama-3.2-1B",
        )

        mock_client = MagicMock()
        mock_client.get_runtime.return_value = real_rt
        mock_client_fn.return_value = mock_client

        result = get_runtime("torch-distributed")
        assert result["success"] is True
        data = result["data"]
        assert data["name"] == "torch-distributed"
        assert data["framework"] == "torch"
        assert data["image"] == "pytorch/pytorch:2.0-cuda11.8"
        assert data["num_nodes"] == 0
        assert data["device"] == "gpu"
        assert data["device_count"] == "4"
        assert data["trainer_type"] == "CustomTrainer"
        assert data["pretrained_model"] == "meta-llama/Llama-3.2-1B"

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_spec_fallback(self, mock_client_fn):
        mock_rt = MagicMock()
        mock_rt.name = "custom-runtime"
        mock_rt.trainer = None
        mock_rt.pretrained_model = None

        mock_ml_policy = MagicMock()
        mock_ml_policy.torch = "torch"
        mock_ml_policy.mpi = None

        mock_replicated_job = MagicMock()
        mock_replicated_job.name = "worker"
        mock_replicated_job.template.spec.completions = 4

        mock_spec = MagicMock()
        mock_spec.ml_policy = mock_ml_policy
        mock_spec.template.spec.replicated_jobs = [mock_replicated_job]
        mock_rt.spec = mock_spec

        mock_client = MagicMock()
        mock_client.get_runtime.return_value = mock_rt
        mock_client_fn.return_value = mock_client

        result = get_runtime("custom-runtime")
        assert result["success"] is True
        data = result["data"]
        assert data["name"] == "custom-runtime"
        assert data["framework"] == "torch"
        assert data["replicated_jobs"] == [{"name": "worker", "replicas": 4}]

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_not_found(self, mock_client_fn):
        from kubernetes.client.exceptions import ApiException

        mock_client = MagicMock()
        mock_client.get_runtime.side_effect = ApiException(status=404, reason="Not Found")
        mock_client_fn.return_value = mock_client

        result = get_runtime("non-existent-runtime")
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.RESOURCE_NOT_FOUND

    @patch("kubeflow_mcp.trainer.api.discovery._fetch_packages_via_pod")
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_include_packages_passes_runtime_obj(
        self, mock_client_fn, mock_fetch_packages
    ):
        mock_rt = MagicMock()
        mock_rt.name = "torch-distributed"
        mock_rt.trainer.image = "pytorch/pytorch:2.0"
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = mock_rt
        mock_client_fn.return_value = mock_client
        mock_fetch_packages.return_value = {"packages": [{"name": "torch", "version": "2.0"}]}

        result = get_runtime("torch-distributed", include_packages=True)
        assert result["success"] is True
        mock_fetch_packages.assert_called_once_with("torch-distributed", runtime_obj=mock_rt)
        assert result["data"]["packages"] == [{"name": "torch", "version": "2.0"}]


class TestGetRuntimeImage:
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_extracts_image_from_passed_runtime_obj(self, mock_client_fn):
        mock_rt = MagicMock()
        mock_rt.trainer.image = "docker.io/kubeflow/passed:v1"

        image = _get_runtime_image("torchtune-llama", runtime_obj=mock_rt)
        assert image == "docker.io/kubeflow/passed:v1"
        mock_client_fn.assert_not_called()

    def test_extracts_image_from_real_sdk_runtime_obj(self):
        from kubeflow.trainer.types.types import Runtime, RuntimeTrainer, TrainerType

        real_trainer = RuntimeTrainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="docker.io/kubeflow/real-trainer:v1",
        )
        real_rt = Runtime(name="real-runtime", trainer=real_trainer)

        image = _get_runtime_image("real-runtime", runtime_obj=real_rt)
        assert image == "docker.io/kubeflow/real-trainer:v1"

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_extracts_image_from_sdk_runtime(self, mock_client_fn):
        mock_rt = MagicMock()
        mock_rt.trainer.image = "docker.io/kubeflow/trainer:v2"
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = mock_rt
        mock_client_fn.return_value = mock_client

        image = _get_runtime_image("torchtune-llama")
        assert image == "docker.io/kubeflow/trainer:v2"

    @patch("kubeflow_mcp.trainer.api.discovery.get_custom_objects_api")
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_fallback_to_custom_objects_api(self, mock_client_fn, mock_custom_api_fn):
        mock_client_fn.side_effect = RuntimeError("SDK runtime lookup failed")

        mock_custom_api = MagicMock()
        mock_custom_api.get_cluster_custom_object.return_value = {
            "spec": {
                "template": {
                    "spec": {
                        "replicatedJobs": [
                            {
                                "template": {
                                    "spec": {
                                        "template": {
                                            "spec": {
                                                "containers": [
                                                    {"image": "docker.io/kubeflow/fallback:v1"}
                                                ]
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        mock_custom_api_fn.return_value = mock_custom_api

        image = _get_runtime_image("cluster-runtime")
        assert image == "docker.io/kubeflow/fallback:v1"
