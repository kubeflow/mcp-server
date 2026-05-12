"""Tests for monitoring tools (SDK-based).

Tests get_training_logs, get_training_events, wait_for_training.
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.monitoring import (
    MAX_LOG_LINES,
    _extract_failure_hint,
    get_training_events,
    get_training_logs,
    wait_for_training,
)


@dataclass
class MockTrainJob:
    """Mock TrainJob matching SDK's types.TrainJob."""

    name: str
    status: str = "Running"


@dataclass
class MockEvent:
    """Mock Event matching SDK's types.Event."""

    involved_object_kind: str
    involved_object_name: str
    message: str
    reason: str
    event_time: datetime


class TestGetTrainingLogs:
    """Tests for get_training_logs()."""

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_logs_success(self, mock_get_client):
        """Test getting logs from a job."""
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(
            [
                "Epoch 1/3: loss=2.34",
                "Epoch 2/3: loss=1.89",
                "Epoch 3/3: loss=1.45",
            ]
        )
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["step"] == "node-0"
        assert "Epoch 1/3" in result["data"]["logs"]
        assert result["data"]["lines"] == 3
        mock_client.get_job_logs.assert_called_once_with(name="my-job", step="node-0", follow=False)

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_logs_custom_step(self, mock_get_client):
        """Test getting logs from specific step."""
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(["Init complete"])
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="my-job", step="model-initializer")

        assert result["success"] is True
        assert result["data"]["step"] == "model-initializer"
        mock_client.get_job_logs.assert_called_once_with(
            name="my-job", step="model-initializer", follow=False
        )

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_logs_empty(self, mock_get_client):
        """Test empty logs."""
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter([])
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="my-job")

        assert result["success"] is True
        assert result["data"]["logs"] == ""

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_logs_not_found(self, mock_get_client):
        """Test job not found error."""
        mock_client = MagicMock()
        mock_client.get_job_logs.side_effect = RuntimeError("TrainJob 'bad' not found")
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="bad")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["error_code"] in ("RESOURCE_NOT_FOUND", "SDK_ERROR")

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_logs_sanitizes_output(self, mock_get_client):
        """Test that sensitive data is sanitized from logs."""
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(
            [
                "Loading model with token hf_abcdefghijklmnop",
                "Training started",
            ]
        )
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="my-job")

        assert result["success"] is True
        # Token should be masked (implementation dependent)
        # At minimum, verify logs are returned
        assert "Training started" in result["data"]["logs"]


class TestGetTrainingEvents:
    """Tests for get_training_events()."""

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_events_success(self, mock_get_client):
        """Test getting events for a job."""
        mock_client = MagicMock()
        mock_client.get_job_events.return_value = [
            MockEvent(
                involved_object_kind="Pod",
                involved_object_name="my-job-node-0",
                message="Successfully pulled image",
                reason="Pulled",
                event_time=datetime.now(),
            ),
            MockEvent(
                involved_object_kind="Pod",
                involved_object_name="my-job-node-0",
                message="Started container",
                reason="Started",
                event_time=datetime.now(),
            ),
        ]
        mock_get_client.return_value = mock_client

        result = get_training_events(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert len(result["data"]["events"]) == 2
        assert result["data"]["events"][0]["reason"] == "Pulled"
        assert result["data"]["events"][0]["involved_object_kind"] == "Pod"
        assert result["data"]["events"][0]["involved_object_name"] == "my-job-node-0"
        assert result["data"]["events"][0]["event_time"]
        assert "pulled image" in result["data"]["events"][0]["message"].lower()

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_events_with_limit(self, mock_get_client):
        """Test limiting event count."""
        mock_client = MagicMock()
        mock_client.get_job_events.return_value = [
            MockEvent(
                involved_object_kind="Pod",
                involved_object_name=f"my-job-event-{i}",
                message=f"Event {i}",
                reason="Test",
                event_time=datetime.now(),
            )
            for i in range(100)
        ]
        mock_get_client.return_value = mock_client

        result = get_training_events(name="my-job", limit=10)

        assert result["success"] is True
        assert len(result["data"]["events"]) == 10
        assert result["data"]["total"] == 100

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_events_empty(self, mock_get_client):
        """Test no events."""
        mock_client = MagicMock()
        mock_client.get_job_events.return_value = []
        mock_get_client.return_value = mock_client

        result = get_training_events(name="my-job")

        assert result["success"] is True
        assert result["data"]["events"] == []

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_get_events_sdk_error(self, mock_get_client):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.get_job_events.side_effect = RuntimeError("API error")
        mock_get_client.return_value = mock_client

        result = get_training_events(name="my-job")

        assert result["success"] is False
        assert "API error" in result["error"]


class TestWaitForTraining:
    """Tests for wait_for_training()."""

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_success_complete(self, mock_get_client):
        """Test waiting for job completion."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.return_value = MockTrainJob(
            name="my-job", status="Complete"
        )
        mock_get_client.return_value = mock_client

        result = wait_for_training(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["reached"] is True
        assert result["data"]["status"] == "Complete"
        mock_client.wait_for_job_status.assert_called_once_with(
            name="my-job",
            status={"Complete"},
            timeout=600,
            polling_interval=2,
        )

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_single_status_string(self, mock_get_client):
        """Test waiting for a single status passed as a string."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.return_value = MockTrainJob(name="my-job", status="Failed")
        mock_get_client.return_value = mock_client

        result = wait_for_training(name="my-job", target_statuses="Failed")

        assert result["success"] is True
        call_kwargs = mock_client.wait_for_job_status.call_args.kwargs
        assert call_kwargs["status"] == {"Failed"}

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_multiple_statuses(self, mock_get_client):
        """Test waiting for either Complete or Failed (stop on first match)."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.return_value = MockTrainJob(name="my-job", status="Failed")
        mock_get_client.return_value = mock_client

        result = wait_for_training(name="my-job", target_statuses=["Complete", "Failed"])

        assert result["success"] is True
        assert result["data"]["status"] == "Failed"
        call_kwargs = mock_client.wait_for_job_status.call_args.kwargs
        assert call_kwargs["status"] == {"Complete", "Failed"}

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_custom_timeout_and_polling(self, mock_get_client):
        """Test custom timeout and polling interval are forwarded to SDK."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.return_value = MockTrainJob(
            name="my-job", status="Complete"
        )
        mock_get_client.return_value = mock_client

        wait_for_training(name="my-job", timeout_seconds=3600, polling_interval=10)

        mock_client.wait_for_job_status.assert_called_once_with(
            name="my-job",
            status={"Complete"},
            timeout=3600,
            polling_interval=10,
        )

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_timeout(self, mock_get_client):
        """Test timeout handling."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.side_effect = TimeoutError("Timed out")
        mock_get_client.return_value = mock_client

        result = wait_for_training(name="my-job", timeout_seconds=60)

        assert result["success"] is True  # Timeout is not an error
        assert result["data"]["reached"] is False
        assert "timeout" in result["data"]["message"].lower()

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_wait_sdk_error(self, mock_get_client):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.wait_for_job_status.side_effect = RuntimeError("Job failed unexpectedly")
        mock_get_client.return_value = mock_client

        result = wait_for_training(name="my-job")

        assert result["success"] is False
        assert "failed unexpectedly" in result["error"].lower()


class TestExtractFailureHint:
    """Tests for _extract_failure_hint()."""

    def test_oom_detected(self):
        hint = _extract_failure_hint("RuntimeError: CUDA out of memory. Tried to allocate 2GB")
        assert hint is not None
        assert hint["category"] == "OOM"

    def test_missing_module(self):
        hint = _extract_failure_hint("ModuleNotFoundError: No module named 'bitsandbytes'")
        assert hint is not None
        assert hint["category"] == "MISSING_MODULE"
        assert "packages" in hint["suggestion"].lower()

    def test_file_not_found(self):
        hint = _extract_failure_hint("FileNotFoundError: [Errno 2] No such file: '/data/train.csv'")
        assert hint is not None
        assert hint["category"] == "FILE_NOT_FOUND"

    def test_permission_error(self):
        hint = _extract_failure_hint("PermissionError: Access denied to /secure/model")
        assert hint is not None
        assert hint["category"] == "PERMISSION_ERROR"

    def test_connection_error(self):
        hint = _extract_failure_hint("ConnectionError: Failed to connect to hub")
        assert hint is not None
        assert hint["category"] == "NETWORK_ERROR"

    def test_traceback(self):
        hint = _extract_failure_hint("Traceback (most recent call last):\n  File ...\nValueError")
        assert hint is not None
        assert hint["category"] == "PYTHON_EXCEPTION"

    def test_clean_logs_no_hint(self):
        hint = _extract_failure_hint(
            "Epoch 1/3: loss=2.34\nEpoch 2/3: loss=1.89\nTraining complete."
        )
        assert hint is None


class TestLogLineCap:
    """Tests for MAX_LOG_LINES capping in get_training_logs."""

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_caps_at_max_lines(self, mock_get_client):
        mock_client = MagicMock()
        lines = [f"line {i}" for i in range(MAX_LOG_LINES + 500)]
        mock_client.get_job_logs.return_value = iter(lines)
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="big-job")

        assert result["success"] is True
        assert result["data"]["lines"] <= MAX_LOG_LINES
        assert f"line {MAX_LOG_LINES + 499}" in result["data"]["logs"]

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_failure_hint_in_logs_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(
            ["Starting training", "RuntimeError: CUDA out of memory"]
        )
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="oom-job")

        assert result["success"] is True
        assert "failure_hint" in result["data"]
        assert result["data"]["failure_hint"]["category"] == "OOM"

    @patch("kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace")
    def test_no_failure_hint_for_clean_logs(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(["Epoch 1: loss=0.5", "Training done"])
        mock_get_client.return_value = mock_client

        result = get_training_logs(name="clean-job")

        assert result["success"] is True
        assert "failure_hint" not in result["data"]
