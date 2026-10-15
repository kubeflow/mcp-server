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
"""

from __future__ import annotations

from types import SimpleNamespace
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
    _fetch_packages_via_pod,
    _get_runtime_image,
    _trainjob_runtime_to_mcp,
    get_runtime,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)


class TestJobStatusFilterAliases:
    def test_succeeded_maps_to_complete(self):
        assert _JOB_STATUS_FILTER_ALIASES["Succeeded"] == "Complete"


@pytest.mark.parametrize("status", ["", "INVALID_STATUS", "completed", "Suspended"])
def test_list_training_jobs_rejects_invalid_status_before_sdk_call(status):
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace"
    ) as mock_client:
        result = list_training_jobs(status=status)

    verify_tool_error(result, error_code=ErrorCode.VALIDATION_ERROR)
    mock_client.assert_not_called()


def test_list_training_jobs_invalid_status_lists_supported_values():
    result = list_training_jobs(status="INVALID_STATUS")

    error = verify_tool_error(result, error_code=ErrorCode.VALIDATION_ERROR)
    assert error["details"] == {
        "supported_statuses": ["Complete", "Created", "Failed", "Running"],
        "aliases": {"Succeeded": "Complete"},
    }


@pytest.mark.parametrize("status", ["Created", "Running", "Complete", "Failed", "Succeeded"])
def test_list_training_jobs_accepts_supported_statuses(status):
    with (
        patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None),
        patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace") as mock_client,
    ):
        mock_client.return_value.list_jobs.return_value = []
        result = list_training_jobs(status=status)

    verify_tool_success(result)


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


@pytest.mark.parametrize(
    ("tool", "kwargs", "client_path"),
    [
        (
            get_runtime,
            {"name": "torchtune-llama3.2-1b"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client",
        ),
        (
            list_training_jobs,
            {"runtime": "torchtune-qwen2.5-1.5b"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        ),
    ],
)
def test_accepts_dotted_runtime_names(tool, kwargs, client_path):
    with patch(client_path) as mock_client:
        result = tool(**kwargs)

    assert result.get("error_code") != ErrorCode.VALIDATION_ERROR
    mock_client.assert_called_once()


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
        from kubeflow.trainer.types.types import (
            Runtime,
            RuntimeKind,
            RuntimeTrainer,
            TrainerType,
        )

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
            kind=RuntimeKind.CLUSTER_TRAINING_RUNTIME,
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

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_sdk_error_wrapped(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.get_runtime.side_effect = RuntimeError("connection refused")
        mock_client_fn.return_value = mock_client

        result = get_runtime("any-runtime")
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.SDK_ERROR


class TestGetRuntimeImage:
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_extracts_image_from_passed_runtime_obj(self, mock_client_fn):
        mock_rt = MagicMock()
        mock_rt.trainer.image = "docker.io/kubeflow/passed:v1"

        image = _get_runtime_image("torchtune-llama", runtime_obj=mock_rt)
        assert image == "docker.io/kubeflow/passed:v1"
        mock_client_fn.assert_not_called()

    def test_extracts_image_from_real_sdk_runtime_obj(self):
        from kubeflow.trainer.types.types import Runtime, RuntimeKind, RuntimeTrainer, TrainerType

        real_trainer = RuntimeTrainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework="torch",
            image="docker.io/kubeflow/real-trainer:v1",
        )
        real_rt = Runtime(
            name="real-runtime",
            trainer=real_trainer,
            kind=RuntimeKind.CLUSTER_TRAINING_RUNTIME,
        )

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

    @patch("kubeflow_mcp.trainer.api.discovery.get_custom_objects_api")
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_fallback_custom_objects_exception_returns_none(
        self, mock_client_fn, mock_custom_api_fn
    ):
        mock_client_fn.side_effect = RuntimeError("SDK lookup failed")
        mock_custom_api = MagicMock()
        mock_custom_api.get_cluster_custom_object.side_effect = Exception("CRD access denied")
        mock_custom_api_fn.return_value = mock_custom_api

        assert _get_runtime_image("missing-cr") is None


class TestFetchPackagesViaPod:
    @patch("kubeflow_mcp.trainer.api.discovery._get_runtime_image", return_value=None)
    def test_image_not_found_returns_error(self, mock_get_image):
        res = _fetch_packages_via_pod("unknown-runtime")
        assert "packages_error" in res
        assert "Could not extract container image" in res["packages_error"]

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="restricted-ns",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.check_namespace_allowed",
        return_value=MagicMock(error="Namespace 'restricted-ns' not allowed"),
    )
    def test_namespace_policy_denied(self, mock_policy, mock_ns, mock_get_image):
        res = _fetch_packages_via_pod("my-runtime")
        assert "packages_error" in res
        assert "not allowed by policy" in res["packages_error"]

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="default",
    )
    @patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None)
    @patch("kubeflow_mcp.trainer.api.discovery.get_core_v1_api")
    def test_pod_creation_error_returns_error(
        self, mock_core_api, mock_policy, mock_ns, mock_get_image
    ):
        mock_core = MagicMock()
        mock_core.create_namespaced_pod.side_effect = RuntimeError("cluster quota exceeded")
        mock_core_api.return_value = mock_core

        res = _fetch_packages_via_pod("my-runtime")
        assert "packages_error" in res
        assert "Failed to create pip-list pod: cluster quota exceeded" in res["packages_error"]
        mock_core.delete_namespaced_pod.assert_not_called()

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="default",
    )
    @patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None)
    @patch("kubeflow_mcp.trainer.api.discovery.get_core_v1_api")
    @patch("kubeflow_mcp.trainer.api.discovery._PACKAGES_POD_TIMEOUT", 0.05)
    @patch("kubeflow_mcp.trainer.api.discovery._PACKAGES_POLL_INTERVAL", 0.01)
    def test_pod_timeout_returns_error_and_deletes_pod(
        self, mock_core_api, mock_policy, mock_ns, mock_get_image
    ):
        mock_core = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.phase = "Pending"
        mock_core.read_namespaced_pod.return_value = mock_pod
        mock_core_api.return_value = mock_core

        res = _fetch_packages_via_pod("my-runtime")
        assert "packages_error" in res
        assert "did not complete within" in res["packages_error"]
        mock_core.delete_namespaced_pod.assert_called_once()
        _, delete_kwargs = mock_core.delete_namespaced_pod.call_args
        assert delete_kwargs.get("grace_period_seconds") == 0

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="default",
    )
    @patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None)
    @patch("kubeflow_mcp.trainer.api.discovery.get_core_v1_api")
    def test_pod_failed_phase_returns_error_with_logs(
        self, mock_core_api, mock_policy, mock_ns, mock_get_image
    ):
        mock_core = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.phase = "Failed"
        mock_core.read_namespaced_pod.return_value = mock_pod
        mock_core.read_namespaced_pod_log.return_value = "OOMKilled: exit code 137"
        mock_core_api.return_value = mock_core

        res = _fetch_packages_via_pod("my-runtime")
        assert res["packages_error"] == "pip-list pod failed"
        assert "OOMKilled" in res["logs"]
        mock_core.delete_namespaced_pod.assert_called_once()

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="default",
    )
    @patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None)
    @patch("kubeflow_mcp.trainer.api.discovery.get_core_v1_api")
    def test_pod_succeeded_parses_packages(
        self, mock_core_api, mock_policy, mock_ns, mock_get_image
    ):
        mock_core = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.phase = "Succeeded"
        mock_core.read_namespaced_pod.return_value = mock_pod
        mock_core.read_namespaced_pod_log.return_value = (
            "Package    Version\n"
            "---        ---\n"
            "pip        23.2.1\n"
            "torch      2.1.0\n"
            "torchtune  0.1.0\n"
        )
        mock_core_api.return_value = mock_core

        res = _fetch_packages_via_pod("my-runtime")
        assert "packages" in res
        assert res["total"] == 3
        assert res["packages"] == [
            {"name": "pip", "version": "23.2.1"},
            {"name": "torch", "version": "2.1.0"},
            {"name": "torchtune", "version": "0.1.0"},
        ]
        mock_core.delete_namespaced_pod.assert_called_once()

    @patch(
        "kubeflow_mcp.trainer.api.discovery._get_runtime_image",
        return_value="pytorch/pytorch:latest",
    )
    @patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_effective_namespace",
        return_value="default",
    )
    @patch("kubeflow_mcp.trainer.api.discovery.check_namespace_allowed", return_value=None)
    @patch("kubeflow_mcp.trainer.api.discovery.get_core_v1_api")
    def test_pod_cleanup_exception_handled_gracefully(
        self, mock_core_api, mock_policy, mock_ns, mock_get_image
    ):
        mock_core = MagicMock()
        mock_pod = MagicMock()
        mock_pod.status.phase = "Succeeded"
        mock_core.read_namespaced_pod.return_value = mock_pod
        mock_core.read_namespaced_pod_log.return_value = "torch 2.1.0"
        mock_core.delete_namespaced_pod.side_effect = RuntimeError("API server transient error")
        mock_core_api.return_value = mock_core

        res = _fetch_packages_via_pod("my-runtime")
        assert "packages" in res
        assert res["packages"] == [{"name": "torch", "version": "2.1.0"}]


# ---------------------------------------------------------------------------
# list_training_jobs
# ---------------------------------------------------------------------------


def _fake_job(name, status="Running", runtime_name="torch-tune"):
    runtime = SimpleNamespace(name=runtime_name) if runtime_name else None
    return SimpleNamespace(name=name, status=status, runtime=runtime)


def _fake_runtime(name):
    return SimpleNamespace(name=name)


def test_list_training_jobs_returns_all_when_no_filters():
    client = MagicMock()
    client.list_jobs.return_value = [_fake_job("job-a"), _fake_job("job-b")]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs()
    assert result["success"] is True
    assert result["data"]["total"] == 2
    assert result["data"]["returned"] == 2
    assert result["data"]["has_more"] is False


def test_list_training_jobs_filters_by_status():
    client = MagicMock()
    client.list_jobs.return_value = [
        _fake_job("job-running", status="Running"),
        _fake_job("job-complete", status="Complete"),
    ]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(status="Running")
    assert result["data"]["total"] == 1
    assert result["data"]["jobs"][0]["name"] == "job-running"


def test_list_training_jobs_succeeded_alias_maps_to_complete():
    client = MagicMock()
    client.list_jobs.return_value = [_fake_job("job-done", status="Complete")]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(status="Succeeded")
    assert result["data"]["total"] == 1


def test_list_training_jobs_empty_list_on_fresh_cluster():
    client = MagicMock()
    client.list_jobs.return_value = []
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs()
    assert result["data"] == {
        "jobs": [],
        "total": 0,
        "returned": 0,
        "has_more": False,
    }


def test_list_training_jobs_reports_truncation_metadata():
    client = MagicMock()
    client.list_jobs.return_value = [
        _fake_job("job-a"),
        _fake_job("job-b"),
        _fake_job("job-c"),
    ]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(limit=2)

    assert result["data"]["total"] == 3
    assert result["data"]["returned"] == 2
    assert result["data"]["has_more"] is True
    assert [job["name"] for job in result["data"]["jobs"]] == ["job-a", "job-b"]


def test_list_training_jobs_sdk_error_wrapped_as_tool_error():
    client = MagicMock()
    client.list_jobs.side_effect = RuntimeError("cluster unreachable")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs()
    assert result["success"] is False
    assert result["error_code"] == ErrorCode.SDK_ERROR


# ---------------------------------------------------------------------------
# list_runtimes
# ---------------------------------------------------------------------------


def test_list_runtimes_success():
    client = MagicMock()
    client.list_runtimes.return_value = [
        _fake_runtime("torch-tune"),
        _fake_runtime("torch-distributed"),
    ]
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()
    assert result["success"] is True
    assert result["data"]["total"] == 2


def test_list_runtimes_empty_when_none_installed():
    client = MagicMock()
    client.list_runtimes.return_value = []
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()
    assert result["data"] == {"runtimes": [], "total": 0}


def test_list_runtimes_sdk_error_wrapped():
    client = MagicMock()
    client.list_runtimes.side_effect = RuntimeError("api server down")
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()
    assert result["success"] is False
    assert result["error_code"] == ErrorCode.SDK_ERROR


# ---------------------------------------------------------------------------
# get_training_job — net-new status/next-steps coverage
# ---------------------------------------------------------------------------


def test_get_training_job_failed_status_suggests_diagnostics():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-a", status="Failed")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-a")
    next_steps = result["data"]["next_steps"]
    assert any("get_training_events" in step for step in next_steps)
    assert any("get_training_logs" in step for step in next_steps)


def test_get_training_job_complete_status_has_no_next_steps():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-a", status="Complete")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-a")
    assert "next_steps" not in result["data"]


def test_get_training_job_namespace_not_allowed_short_circuits():
    from kubeflow_mcp.common.types import ToolError as ToolErrorModel

    with patch(
        "kubeflow_mcp.trainer.api.discovery.check_namespace_allowed",
        return_value=ToolErrorModel(error="namespace blocked", error_code="PERMISSION_DENIED"),
    ):
        result = get_training_job("job-a", namespace="restricted")

    assert result["success"] is False
    assert result["error_code"] == ErrorCode.PERMISSION_DENIED


def test_list_training_jobs_rejects_limit_below_one():
    result = list_training_jobs(limit=0)
    assert result["success"] is False
    assert result["error_code"] == ErrorCode.VALIDATION_ERROR
    assert "limit must be >= 1" in result["error"]


def test_list_training_jobs_namespace_not_allowed_short_circuits():
    from kubeflow_mcp.common.types import ToolError as ToolErrorModel

    with patch(
        "kubeflow_mcp.trainer.api.discovery.check_namespace_allowed",
        return_value=ToolErrorModel(error="namespace blocked", error_code="PERMISSION_DENIED"),
    ):
        result = list_training_jobs(namespace="restricted")
    assert result["success"] is False
    assert result["error_code"] == ErrorCode.PERMISSION_DENIED


def test_get_training_job_created_status_suggests_events():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-created", status="Created")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-created")
    next_steps = result["data"]["next_steps"]
    assert any("get_training_events" in step for step in next_steps)


def test_get_training_job_sdk_error_wrapped():
    client = MagicMock()
    client.get_job.side_effect = RuntimeError("backend unreachable")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-err")
    assert result["success"] is False
    assert result["error_code"] == ErrorCode.SDK_ERROR
