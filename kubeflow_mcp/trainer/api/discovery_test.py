"""Tests for discovery tools (SDK-based).

Tests list_training_jobs, get_training_job, list_runtimes, get_runtime.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)


@dataclass
class MockRuntimeTrainer:
    """Mock RuntimeTrainer."""

    trainer_type: str = "BuiltinTrainer"
    framework: str = "torch"
    image: str = "pytorch/pytorch:2.2.0"


@dataclass
class MockRuntime:
    """Mock Runtime matching SDK's types.Runtime."""

    name: str
    trainer: MockRuntimeTrainer | None = None

    def __post_init__(self):
        if self.trainer is None:
            self.trainer = MockRuntimeTrainer()


@dataclass
class MockTrainJob:
    """Mock TrainJob matching SDK's types.TrainJob."""

    name: str
    status: str = "Running"
    runtime: MockRuntime | None = None

    def __post_init__(self):
        if self.runtime is None:
            self.runtime = MockRuntime(name="torch-distributed")


class TestListTrainingJobs:
    """Tests for list_training_jobs()."""

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_success(self, mock_get_client):
        """Test listing jobs returns formatted response."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            MockTrainJob(name="job-1", status="Running"),
            MockTrainJob(name="job-2", status="Complete"),
            MockTrainJob(name="job-3", status="Failed"),
        ]
        mock_get_client.return_value = mock_client

        result = list_training_jobs()

        assert result["success"] is True
        assert len(result["data"]["jobs"]) == 3
        assert result["data"]["total"] == 3
        assert result["data"]["jobs"][0]["name"] == "job-1"
        assert result["data"]["jobs"][0]["status"] == "Running"
        assert result["data"]["jobs"][0]["runtime"] == {"name": "torch-distributed"}

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_with_status_filter(self, mock_get_client):
        """Test filtering jobs by status."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            MockTrainJob(name="job-1", status="Running"),
            MockTrainJob(name="job-2", status="Complete"),
            MockTrainJob(name="job-3", status="Running"),
        ]
        mock_get_client.return_value = mock_client

        result = list_training_jobs(status="Running")

        assert result["success"] is True
        assert len(result["data"]["jobs"]) == 2
        assert all(j["status"] == "Running" for j in result["data"]["jobs"])

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_status_alias_succeeded_to_complete(self, mock_get_client):
        """Legacy 'Succeeded' filter matches SDK Complete status."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            MockTrainJob(name="job-1", status="Complete"),
        ]
        mock_get_client.return_value = mock_client

        result = list_training_jobs(status="Succeeded")

        assert result["success"] is True
        assert len(result["data"]["jobs"]) == 1
        assert result["data"]["jobs"][0]["name"] == "job-1"

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_with_limit(self, mock_get_client):
        """Test limiting job count."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [MockTrainJob(name=f"job-{i}") for i in range(100)]
        mock_get_client.return_value = mock_client

        result = list_training_jobs(limit=10)

        assert result["success"] is True
        assert len(result["data"]["jobs"]) == 10
        assert result["data"]["total"] == 100

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_with_runtime_filter(self, mock_get_client):
        """Test filtering by runtime resolves name to Runtime for list_jobs."""
        mock_client = MagicMock()
        rt = MockRuntime(name="torch-tune")
        mock_client.get_runtime.return_value = rt
        mock_client.list_jobs.return_value = []
        mock_get_client.return_value = mock_client

        list_training_jobs(runtime="torch-tune")

        mock_client.get_runtime.assert_called_once_with(name="torch-tune")
        mock_client.list_jobs.assert_called_once_with(runtime=rt)

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_empty(self, mock_get_client):
        """Test empty job list."""
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = []
        mock_get_client.return_value = mock_client

        result = list_training_jobs()

        assert result["success"] is True
        assert result["data"]["jobs"] == []
        assert result["data"]["total"] == 0

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_list_jobs_sdk_error(self, mock_get_client):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.list_jobs.side_effect = RuntimeError("Connection refused")
        mock_get_client.return_value = mock_client

        result = list_training_jobs()

        assert result["success"] is False
        assert "Connection refused" in result["error"]


class TestGetTrainingJob:
    """Tests for get_training_job()."""

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_get_job_success(self, mock_get_client):
        """Test getting a job by name."""
        mock_client = MagicMock()
        mock_client.get_job.return_value = MockTrainJob(
            name="my-job",
            status="Running",
            runtime=MockRuntime(name="torch-tune"),
        )
        mock_get_client.return_value = mock_client

        result = get_training_job(name="my-job")

        assert result["success"] is True
        assert result["data"]["name"] == "my-job"
        assert result["data"]["status"] == "Running"
        assert result["data"]["runtime"] == {"name": "torch-tune"}
        mock_client.get_job.assert_called_once_with(name="my-job")

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_get_job_not_found(self, mock_get_client):
        """Test job not found error."""
        mock_client = MagicMock()
        mock_client.get_job.side_effect = RuntimeError("TrainJob 'missing' not found")
        mock_get_client.return_value = mock_client

        result = get_training_job(name="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["error_code"] in ("RESOURCE_NOT_FOUND", "SDK_ERROR")

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace")
    def test_get_job_sdk_error(self, mock_get_client):
        """Test generic SDK error."""
        mock_client = MagicMock()
        mock_client.get_job.side_effect = RuntimeError("API timeout")
        mock_get_client.return_value = mock_client

        result = get_training_job(name="my-job")

        assert result["success"] is False
        assert "API timeout" in result["error"]


class TestListRuntimes:
    """Tests for list_runtimes()."""

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_list_runtimes_success(self, mock_get_client):
        """Test listing available runtimes."""
        mock_client = MagicMock()
        mock_client.list_runtimes.return_value = [
            MockRuntime(name="torch-distributed"),
            MockRuntime(name="torch-tune"),
            MockRuntime(name="jax-distributed"),
        ]
        mock_get_client.return_value = mock_client

        result = list_runtimes()

        assert result["success"] is True
        assert len(result["data"]["runtimes"]) == 3
        assert result["data"]["total"] == 3
        runtime_names = [r["name"] for r in result["data"]["runtimes"]]
        assert "torch-tune" in runtime_names

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_list_runtimes_empty(self, mock_get_client):
        """Test empty runtime list."""
        mock_client = MagicMock()
        mock_client.list_runtimes.return_value = []
        mock_get_client.return_value = mock_client

        result = list_runtimes()

        assert result["success"] is True
        assert result["data"]["runtimes"] == []

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_list_runtimes_sdk_error(self, mock_get_client):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.list_runtimes.side_effect = RuntimeError("No kubeconfig")
        mock_get_client.return_value = mock_client

        result = list_runtimes()

        assert result["success"] is False
        assert "kubeconfig" in result["error"].lower()


class TestGetRuntime:
    """Tests for get_runtime()."""

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_success(self, mock_get_client):
        """Test getting runtime details."""
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = MockRuntime(name="torch-tune")
        mock_get_client.return_value = mock_client

        result = get_runtime(name="torch-tune")

        assert result["success"] is True
        assert result["data"]["name"] == "torch-tune"
        mock_client.get_runtime.assert_called_once_with(name="torch-tune")

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_not_found(self, mock_get_client):
        """Test runtime not found error."""
        mock_client = MagicMock()
        mock_client.get_runtime.side_effect = RuntimeError("Runtime 'bad' not found")
        mock_get_client.return_value = mock_client

        result = get_runtime(name="bad")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["error_code"] in ("RESOURCE_NOT_FOUND", "SDK_ERROR")


# --- get_runtime: spec serialisation ---


class TestGetRuntimeSpec:
    """Tests that get_runtime() returns framework and replicated_jobs from spec."""

    def _make_runtime(self, name: str, torch_framework=None, replicated_jobs=None):
        """Build a SimpleNamespace tree matching the SDK runtime structure."""
        rj_list = []
        for rj_name, completions in replicated_jobs or []:
            rj = SimpleNamespace(
                name=rj_name,
                template=SimpleNamespace(spec=SimpleNamespace(completions=completions)),
            )
            rj_list.append(rj)

        ml_policy = SimpleNamespace(torch=torch_framework, mpi=None)
        spec = SimpleNamespace(
            ml_policy=ml_policy,
            template=SimpleNamespace(
                spec=SimpleNamespace(replicated_jobs=rj_list if rj_list else None)
            ),
        )
        return SimpleNamespace(name=name, spec=spec)

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_returns_framework(self, mock_get_client):
        rt = self._make_runtime("torch-tune", torch_framework="torch")
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = rt
        mock_get_client.return_value = mock_client

        result = get_runtime(name="torch-tune")

        assert result["success"] is True
        assert result["data"]["framework"] == "torch"

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_returns_replicated_jobs(self, mock_get_client):
        rt = self._make_runtime(
            "torch-tune",
            torch_framework="torch",
            replicated_jobs=[("node", 4), ("model-initializer", 1)],
        )
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = rt
        mock_get_client.return_value = mock_client

        result = get_runtime(name="torch-tune")

        assert result["success"] is True
        jobs = result["data"]["replicated_jobs"]
        assert len(jobs) == 2
        assert jobs[0] == {"name": "node", "replicas": 4}
        assert jobs[1] == {"name": "model-initializer", "replicas": 1}

    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    def test_get_runtime_no_spec_returns_name_only(self, mock_get_client):
        """Runtime objects without spec (e.g. older SDK) still return name."""
        rt = SimpleNamespace(name="torch-distributed")  # no spec attribute
        mock_client = MagicMock()
        mock_client.get_runtime.return_value = rt
        mock_get_client.return_value = mock_client

        result = get_runtime(name="torch-distributed")

        assert result["success"] is True
        assert result["data"]["name"] == "torch-distributed"
        assert "replicated_jobs" not in result["data"]
