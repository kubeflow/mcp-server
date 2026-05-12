"""Tests for lifecycle tools (SDK-based).

Tests delete_training_job and update_training_job.
"""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
)


def _patch_lifecycle_deps(namespace="default"):
    """Context managers to mock out K8s-dependent helpers in lifecycle."""
    return [
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value=namespace,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed", return_value=True),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
    ]


class TestDeleteTrainingJob:
    """Tests for delete_training_job() - uses SDK."""

    def test_delete_returns_preview_without_confirmed(self):
        """Without confirmed=True, should return preview."""
        with (
            patch(
                "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
                return_value="default",
            ),
            patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
            patch(
                "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
                return_value="platform-admin",
            ),
        ):
            result = delete_training_job(name="my-job")

        assert (
            "preview" in result
            or result.get("confirmed") is False
            or "confirmed=True" in str(result)
        )

    def test_delete_success(self):
        """Test successful job deletion with confirmed=True."""
        mock_client = MagicMock()
        mock_client.delete_job.return_value = None

        patches = _patch_lifecycle_deps()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
                return_value=mock_client,
            ),
        ):
            result = delete_training_job(name="my-job", confirmed=True)

        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["deleted"] is True
        mock_client.delete_job.assert_called_once_with(name="my-job")

    def test_delete_not_found(self):
        """Test deleting non-existent job."""
        mock_client = MagicMock()
        mock_client.delete_job.side_effect = RuntimeError("TrainJob 'missing' not found")

        patches = _patch_lifecycle_deps()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
                return_value=mock_client,
            ),
        ):
            result = delete_training_job(name="missing", confirmed=True)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_delete_sdk_error(self):
        """Test SDK error handling."""
        mock_client = MagicMock()
        mock_client.delete_job.side_effect = RuntimeError("Permission denied")

        patches = _patch_lifecycle_deps()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
                return_value=mock_client,
            ),
        ):
            result = delete_training_job(name="my-job", confirmed=True)

        assert result["success"] is False
        assert "Permission denied" in result["error"]
