"""Tests for lifecycle tools: delete_training_job and update_training_job."""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
    update_training_job,
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


def _patch_update_deps(namespace="default"):
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
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=MagicMock(),
        ),
    ]


class TestUpdateTrainingJob:
    """Tests for update_training_job() — suspend/resume via K8s patch."""

    def test_invalid_action_rejected(self):
        result = update_training_job(name="my-job", action="restart")

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "restart" in result["error"]

    def test_invalid_name_rejected(self):
        result = update_training_job(name="INVALID NAME!", action="suspend")

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    def test_suspend_success(self):
        mock_api = MagicMock()
        mock_api.patch_namespaced_custom_object.return_value = {}

        patches = _patch_update_deps()
        patches[4] = patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=mock_api,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = update_training_job(name="my-job", action="suspend")

        assert result["success"] is True
        assert result["data"]["action"] == "suspend"
        assert result["data"]["job"] == "my-job"
        body = mock_api.patch_namespaced_custom_object.call_args.kwargs["body"]
        assert body["spec"]["suspend"] is True

    def test_resume_success(self):
        mock_api = MagicMock()
        mock_api.patch_namespaced_custom_object.return_value = {}

        patches = _patch_update_deps()
        patches[4] = patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=mock_api,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = update_training_job(name="my-job", action="resume")

        assert result["success"] is True
        body = mock_api.patch_namespaced_custom_object.call_args.kwargs["body"]
        assert body["spec"]["suspend"] is False

    def test_suspend_patch_body_is_correct(self):
        """Suspend must send spec.suspend=True, resume must send spec.suspend=False."""
        mock_api = MagicMock()
        patches = _patch_update_deps()
        patches[4] = patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=mock_api,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            update_training_job(name="job", action="suspend")
            suspend_body = mock_api.patch_namespaced_custom_object.call_args.kwargs["body"]
            update_training_job(name="job", action="resume")
            resume_body = mock_api.patch_namespaced_custom_object.call_args.kwargs["body"]

        assert suspend_body == {"spec": {"suspend": True}}
        assert resume_body == {"spec": {"suspend": False}}

    def test_job_not_found(self):
        mock_api = MagicMock()
        mock_api.patch_namespaced_custom_object.side_effect = Exception(
            '{"code": 404, "reason": "NotFound"}'
        )
        patches = _patch_update_deps()
        patches[4] = patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=mock_api,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = update_training_job(name="missing-job", action="suspend")

        assert result["success"] is False

    def test_namespace_propagated_to_patch_call(self):
        mock_api = MagicMock()
        patches = _patch_update_deps(namespace="ml-team")
        patches[4] = patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=mock_api,
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = update_training_job(name="my-job", action="suspend", namespace="ml-team")

        assert result["success"] is True
        call_kwargs = mock_api.patch_namespaced_custom_object.call_args.kwargs
        assert call_kwargs["namespace"] == "ml-team"
