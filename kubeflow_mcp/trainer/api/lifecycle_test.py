"""Tests for lifecycle tools (SDK + K8s direct).

Tests delete_training_job (SDK), suspend_training_job (K8s), resume_training_job (K8s).
"""

from unittest.mock import MagicMock, patch

from kubeflow.trainer.constants import constants as trainer_constants

from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
    resume_training_job,
    suspend_training_job,
)


class TestDeleteTrainingJob:
    """Tests for delete_training_job() - uses SDK."""

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_delete_success(self, mock_get_client):
        """Test successful job deletion."""
        mock_client = MagicMock()
        mock_client.delete_job.return_value = None
        mock_get_client.return_value = mock_client

        result = delete_training_job(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["deleted"] is True
        mock_client.delete_job.assert_called_once_with(name="my-job")

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_delete_not_found(self, mock_get_client):
        """Test deleting non-existent job."""
        mock_client = MagicMock()
        mock_client.delete_job.side_effect = RuntimeError("TrainJob 'missing' not found")
        mock_get_client.return_value = mock_client

        result = delete_training_job(name="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["error_code"] == "RESOURCE_NOT_FOUND"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_delete_sdk_error(self, mock_get_client):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.delete_job.side_effect = RuntimeError("Permission denied")
        mock_get_client.return_value = mock_client

        result = delete_training_job(name="my-job")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


def _make_k8s_backend_mock(namespace: str = "default"):
    mock_api = MagicMock()
    mock_backend = MagicMock()
    mock_backend.namespace = namespace
    mock_backend.custom_api = mock_api
    mock_client = MagicMock()
    mock_client.backend = mock_backend
    return mock_client, mock_api


class TestSuspendTrainingJob:
    """Tests for suspend_training_job() - uses SDK Kubernetes CustomObjects API."""

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_suspend_success(self, mock_get_client):
        """Test successful job suspension."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_api.patch_namespaced_custom_object.return_value = {"status": "patched"}
        mock_get_client.return_value = mock_client

        result = suspend_training_job(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["suspended"] is True
        assert result["data"]["namespace"] == "default"

        mock_api.patch_namespaced_custom_object.assert_called_once_with(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace="default",
            plural=trainer_constants.TRAINJOB_PLURAL,
            name="my-job",
            body={"spec": {"suspend": True}},
            _request_timeout=5,
        )

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_suspend_custom_namespace(self, mock_get_client):
        """Test suspension in custom namespace."""
        mock_client, mock_api = _make_k8s_backend_mock(namespace="sandbox")
        mock_get_client.return_value = mock_client

        result = suspend_training_job(name="my-job", namespace="ml-team")

        assert result["success"] is True
        assert result["data"]["namespace"] == "ml-team"
        mock_api.patch_namespaced_custom_object.assert_called_once()
        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["namespace"] == "ml-team"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_suspend_not_found(self, mock_get_client):
        """Test suspending non-existent job."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_api.patch_namespaced_custom_object.side_effect = Exception(
            "trainjobs.trainer.kubeflow.org 'missing' not found"
        )
        mock_get_client.return_value = mock_client

        result = suspend_training_job(name="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["error_code"] == "RESOURCE_NOT_FOUND"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_suspend_k8s_error(self, mock_get_client):
        """Test K8s API error handling."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_api.patch_namespaced_custom_object.side_effect = Exception("Forbidden")
        mock_get_client.return_value = mock_client

        result = suspend_training_job(name="my-job")

        assert result["success"] is False
        assert "Forbidden" in result["error"]

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_suspend_verifies_patch_body(self, mock_get_client):
        """Test that suspend patch body is correct."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        suspend_training_job(name="my-job")

        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["body"] == {"spec": {"suspend": True}}
        assert call_kwargs["group"] == trainer_constants.GROUP
        assert call_kwargs["version"] == trainer_constants.VERSION
        assert call_kwargs["plural"] == trainer_constants.TRAINJOB_PLURAL


class TestResumeTrainingJob:
    """Tests for resume_training_job() - uses SDK Kubernetes CustomObjects API."""

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_resume_success(self, mock_get_client):
        """Test successful job resumption."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_api.patch_namespaced_custom_object.return_value = {"status": "patched"}
        mock_get_client.return_value = mock_client

        result = resume_training_job(name="my-job")

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["resumed"] is True

        mock_api.patch_namespaced_custom_object.assert_called_once_with(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace="default",
            plural=trainer_constants.TRAINJOB_PLURAL,
            name="my-job",
            body={"spec": {"suspend": False}},
            _request_timeout=5,
        )

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_resume_custom_namespace(self, mock_get_client):
        """Test resumption in custom namespace."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        result = resume_training_job(name="my-job", namespace="prod")

        assert result["success"] is True
        assert result["data"]["namespace"] == "prod"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_resume_not_found(self, mock_get_client):
        """Test resuming non-existent job."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_api.patch_namespaced_custom_object.side_effect = Exception(
            "trainjobs.trainer.kubeflow.org 'missing' not found"
        )
        mock_get_client.return_value = mock_client

        result = resume_training_job(name="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_resume_verifies_patch_body(self, mock_get_client):
        """Test that resume patch body is correct (suspend: False)."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        resume_training_job(name="my-job")

        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["body"] == {"spec": {"suspend": False}}


class TestK8sApiContract:
    """Tests that verify K8s API contract for TrainJob CRD (Trainer v2)."""

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_trainjob_crd_group(self, mock_get_client):
        """Verify TrainJob CRD group matches Kubeflow Trainer SDK."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        suspend_training_job(name="test")

        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["group"] == "trainer.kubeflow.org"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_trainjob_crd_version(self, mock_get_client):
        """Verify TrainJob CRD version is v1alpha1."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        suspend_training_job(name="test")

        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["version"] == "v1alpha1"

    @patch("kubeflow_mcp.common.utils.get_trainer_client")
    def test_trainjob_crd_plural(self, mock_get_client):
        """Verify TrainJob CRD plural is trainjobs."""
        mock_client, mock_api = _make_k8s_backend_mock()
        mock_get_client.return_value = mock_client

        suspend_training_job(name="test")

        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["plural"] == "trainjobs"
