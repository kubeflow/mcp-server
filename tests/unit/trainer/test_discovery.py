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

"""Tests for discovery tools: list_training_jobs, get_training_job,
list_runtimes, get_runtime.

Part of #68 (kubeflow/mcp-server), discovery slice. Real cluster calls
and packages-via-pod polling (get_runtime include_packages=True) are out
of scope per the issue; _fetch_packages_via_pod is mocked directly where
exercised.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)


def _fake_job(name, status="Running", runtime_name="torch-tune"):
    runtime = SimpleNamespace(name=runtime_name) if runtime_name else None
    return SimpleNamespace(name=name, status=status, runtime=runtime)


def _fake_runtime(name):
    return SimpleNamespace(name=name)


# ---------------------------------------------------------------------------
# list_training_jobs
# ---------------------------------------------------------------------------


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
    names = {j["name"] for j in result["data"]["jobs"]}
    assert names == {"job-a", "job-b"}


def test_list_training_jobs_filters_by_runtime():
    client = MagicMock()
    client.get_runtime.return_value = _fake_runtime("torch-tune")
    client.list_jobs.return_value = [_fake_job("job-a")]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(runtime="torch-tune")

    client.get_runtime.assert_called_once_with(name="torch-tune")
    client.list_jobs.assert_called_once_with(runtime=client.get_runtime.return_value)
    assert result["data"]["total"] == 1


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
    # Legacy filter/docs use "Succeeded"; the live API uses "Complete".
    client = MagicMock()
    client.list_jobs.return_value = [_fake_job("job-done", status="Complete")]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(status="Succeeded")

    assert result["data"]["total"] == 1
    assert result["data"]["jobs"][0]["name"] == "job-done"


def test_list_training_jobs_empty_list_on_fresh_cluster():
    client = MagicMock()
    client.list_jobs.return_value = []
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs()

    assert result["data"] == {"jobs": [], "total": 0}


def test_list_training_jobs_rejects_limit_below_one():
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace"):
        result = list_training_jobs(limit=0)

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


def test_list_training_jobs_caps_limit_at_max():
    client = MagicMock()
    client.list_jobs.return_value = [_fake_job(f"job-{i}") for i in range(5)]
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs(limit=10_000)

    # All 5 fake jobs returned; the cap only affects slicing, not a crash,
    # so this mainly documents that an oversized limit doesn't error.
    assert result["success"] is True
    assert result["data"]["total"] == 5


def test_list_training_jobs_sdk_error_wrapped_as_tool_error():
    client = MagicMock()
    client.list_jobs.side_effect = RuntimeError("cluster unreachable")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = list_training_jobs()

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "cluster unreachable" in result["error"]


def test_list_training_jobs_namespace_not_allowed_short_circuits():
    with patch(
        "kubeflow_mcp.trainer.api.discovery.check_namespace_allowed",
        return_value=SimpleNamespace(
            model_dump=lambda: {
                "success": False,
                "error": "Namespace 'restricted' not allowed by policy",
                "error_code": "PERMISSION_DENIED",
            }
        ),
    ):
        result = list_training_jobs(namespace="restricted")

    assert result["success"] is False
    assert result["error_code"] == "PERMISSION_DENIED"


# ---------------------------------------------------------------------------
# get_training_job
# ---------------------------------------------------------------------------


def test_get_training_job_success():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-a", status="Running")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-a")

    assert result["success"] is True
    assert result["data"]["name"] == "job-a"
    assert result["data"]["status"] == "Running"


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


def test_get_training_job_running_status_suggests_logs_only():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-a", status="Running")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-a")

    assert result["data"]["next_steps"] == ["get_training_logs(name='job-a') — check progress"]


def test_get_training_job_complete_status_has_no_next_steps():
    client = MagicMock()
    client.get_job.return_value = _fake_job("job-a", status="Complete")
    with patch(
        "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        return_value=client,
    ):
        result = get_training_job("job-a")

    assert "next_steps" not in result["data"]


def test_get_training_job_not_found_returns_resource_not_found():
    client = MagicMock()
    client.get_job.side_effect = Exception("404 not found")
    with (
        patch(
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.trainer.api.discovery.is_k8s_not_found", return_value=True),
    ):
        result = get_training_job("missing-job")

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-job" in result["error"]


def test_get_training_job_generic_error_returns_sdk_error():
    client = MagicMock()
    client.get_job.side_effect = RuntimeError("connection reset")
    with (
        patch(
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.trainer.api.discovery.is_k8s_not_found", return_value=False),
    ):
        result = get_training_job("job-a")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


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
    names = {rt["name"] for rt in result["data"]["runtimes"]}
    assert names == {"torch-tune", "torch-distributed"}


def test_list_runtimes_empty_when_none_installed():
    client = MagicMock()
    client.list_runtimes.return_value = []
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()

    assert result["data"] == {"runtimes": [], "total": 0}


def test_list_runtimes_falls_back_to_str_when_no_name_attr():
    client = MagicMock()
    client.list_runtimes.return_value = ["raw-runtime-string"]
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()

    assert result["data"]["runtimes"] == [{"name": "raw-runtime-string"}]


def test_list_runtimes_sdk_error_wrapped():
    client = MagicMock()
    client.list_runtimes.side_effect = RuntimeError("api server down")
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = list_runtimes()

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


# ---------------------------------------------------------------------------
# get_runtime
# ---------------------------------------------------------------------------


def test_get_runtime_success_without_packages():
    client = MagicMock()
    client.get_runtime.return_value = _fake_runtime("torch-tune")
    with patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client):
        result = get_runtime("torch-tune")

    assert result["success"] is True
    assert result["data"]["name"] == "torch-tune"
    assert "packages" not in result["data"]


def test_get_runtime_includes_packages_when_requested():
    client = MagicMock()
    client.get_runtime.return_value = _fake_runtime("torch-tune")
    fake_packages = {"packages": [{"name": "torch", "version": "2.1.0"}], "total": 1}
    with (
        patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client),
        patch(
            "kubeflow_mcp.trainer.api.discovery._fetch_packages_via_pod",
            return_value=fake_packages,
        ),
    ):
        result = get_runtime("torch-tune", include_packages=True)

    assert result["data"]["packages"] == [{"name": "torch", "version": "2.1.0"}]
    assert result["data"]["total"] == 1


def test_get_runtime_not_found_returns_resource_not_found():
    client = MagicMock()
    client.get_runtime.side_effect = Exception("404")
    with (
        patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client),
        patch("kubeflow_mcp.trainer.api.discovery.is_k8s_not_found", return_value=True),
    ):
        result = get_runtime("missing-runtime")

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-runtime" in result["error"]


def test_get_runtime_generic_error_returns_sdk_error():
    client = MagicMock()
    client.get_runtime.side_effect = RuntimeError("timeout")
    with (
        patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client", return_value=client),
        patch("kubeflow_mcp.trainer.api.discovery.is_k8s_not_found", return_value=False),
    ):
        result = get_runtime("torch-tune")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
