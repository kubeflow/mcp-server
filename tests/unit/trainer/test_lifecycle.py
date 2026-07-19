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

"""Tests for lifecycle tools: delete_training_job, update_training_job.

Part of #68 (kubeflow/mcp-server), lifecycle slice.

Covers suspend/resume/delete paths with mocked SDK / CustomObjects API:
- confirmed=False preview gate for delete
- confirmed=True success path
- MCP ownership checks for non-admin personas
- platform-admin bypass
- invalid name / invalid action / namespace policy
- not-found and generic SDK errors
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.lifecycle import delete_training_job, update_training_job


def _permission_denied(ns: str = "restricted"):
    return SimpleNamespace(
        model_dump=lambda: {
            "success": False,
            "error": f"Namespace '{ns}' not allowed by policy",
            "error_code": "PERMISSION_DENIED",
        }
    )


def _validation_error(msg: str):
    return SimpleNamespace(
        model_dump=lambda: {
            "success": False,
            "error": msg,
            "error_code": "VALIDATION_ERROR",
        }
    )


# ---------------------------------------------------------------------------
# delete_training_job
# ---------------------------------------------------------------------------


def test_delete_training_job_preview_when_not_confirmed():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace"
        ) as mock_client_factory,
    ):
        result = delete_training_job("job-a", confirmed=False)

    assert result["success"] is True
    assert result["status"] == "preview"
    assert result["config"] == {"job": "job-a", "namespace": "default"}
    assert "confirmed=True" in result["message"]
    mock_client_factory.assert_not_called()


def test_delete_training_job_success_when_confirmed_as_admin():
    client = MagicMock()
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="ml",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed") as mock_managed,
    ):
        result = delete_training_job("job-a", namespace="ml", confirmed=True)

    assert result["success"] is True
    assert result["data"]["deleted"] is True
    assert result["data"]["job"] == "job-a"
    assert result["data"]["namespace"] == "ml"
    client.delete_job.assert_called_once_with(name="job-a")
    # platform-admin bypasses ownership checks
    mock_managed.assert_not_called()


def test_delete_training_job_rejects_unmanaged_job_for_non_admin():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="data-scientist",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=False,
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace"
        ) as mock_client_factory,
    ):
        result = delete_training_job("job-a", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "not created by MCP" in result["error"]
    mock_client_factory.assert_not_called()


def test_delete_training_job_ownership_api_error_for_non_admin():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="ml-engineer",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=None,
        ),
    ):
        result = delete_training_job("job-a", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "Cannot verify ownership" in result["error"]


def test_delete_training_job_allows_managed_job_for_non_admin():
    client = MagicMock()
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="data-scientist",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=True,
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
            return_value=client,
        ),
    ):
        result = delete_training_job("job-a", confirmed=True)

    assert result["success"] is True
    assert result["data"]["deleted"] is True
    client.delete_job.assert_called_once_with(name="job-a")


def test_delete_training_job_invalid_name_short_circuits():
    with patch(
        "kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name",
        return_value=_validation_error("name must be lowercase alphanumeric with hyphens"),
    ):
        result = delete_training_job("BAD_NAME", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


def test_delete_training_job_namespace_not_allowed_short_circuits():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed",
            return_value=_permission_denied("restricted"),
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace"
        ) as mock_client_factory,
    ):
        result = delete_training_job("job-a", namespace="restricted", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "PERMISSION_DENIED"
    mock_client_factory.assert_not_called()


def test_delete_training_job_not_found_returns_resource_not_found():
    client = MagicMock()
    client.delete_job.side_effect = Exception("404 not found")
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.is_k8s_not_found", return_value=True),
    ):
        result = delete_training_job("missing-job", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-job" in result["error"]


def test_delete_training_job_generic_error_returns_sdk_error():
    client = MagicMock()
    client.delete_job.side_effect = RuntimeError("cluster unreachable")
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.is_k8s_not_found", return_value=False),
    ):
        result = delete_training_job("job-a", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "cluster unreachable" in result["error"]


# ---------------------------------------------------------------------------
# update_training_job (suspend / resume)
# ---------------------------------------------------------------------------


def test_update_training_job_suspend_success_as_admin():
    api = MagicMock()
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed") as mock_managed,
    ):
        result = update_training_job("job-a", action="suspend")

    assert result["success"] is True
    assert result["data"]["action"] == "suspend"
    assert result["data"]["job"] == "job-a"
    assert "suspended" in result["data"]["message"]
    api.patch_namespaced_custom_object.assert_called_once()
    kwargs = api.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs["name"] == "job-a"
    assert kwargs["namespace"] == "default"
    assert kwargs["body"] == {"spec": {"suspend": True}}
    mock_managed.assert_not_called()


def test_update_training_job_resume_success_as_admin():
    api = MagicMock()
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="ml",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=api,
        ),
    ):
        result = update_training_job("job-a", action="resume", namespace="ml")

    assert result["success"] is True
    assert result["data"]["action"] == "resume"
    assert "resumed" in result["data"]["message"]
    kwargs = api.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs["body"] == {"spec": {"suspend": False}}
    assert kwargs["namespace"] == "ml"


def test_update_training_job_rejects_invalid_action():
    with patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None):
        result = update_training_job("job-a", action="pause")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "Invalid action" in result["error"]


def test_update_training_job_rejects_unmanaged_job_for_non_admin():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="data-scientist",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=False,
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api"
        ) as mock_api,
    ):
        result = update_training_job("job-a", action="suspend")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "not created by MCP" in result["error"]
    mock_api.assert_not_called()


def test_update_training_job_ownership_api_error_for_non_admin():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="ml-engineer",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=None,
        ),
    ):
        result = update_training_job("job-a", action="resume")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "Cannot verify ownership" in result["error"]


def test_update_training_job_allows_managed_job_for_non_admin():
    api = MagicMock()
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="data-scientist",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed",
            return_value=True,
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=api,
        ),
    ):
        result = update_training_job("job-a", action="suspend")

    assert result["success"] is True
    assert result["data"]["action"] == "suspend"
    api.patch_namespaced_custom_object.assert_called_once()


def test_update_training_job_invalid_name_short_circuits():
    with patch(
        "kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name",
        return_value=_validation_error("name cannot be empty"),
    ):
        result = update_training_job("", action="suspend")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


def test_update_training_job_namespace_not_allowed_short_circuits():
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed",
            return_value=_permission_denied("restricted"),
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api"
        ) as mock_api,
    ):
        result = update_training_job("job-a", action="suspend", namespace="restricted")

    assert result["success"] is False
    assert result["error_code"] == "PERMISSION_DENIED"
    mock_api.assert_not_called()


def test_update_training_job_not_found_returns_resource_not_found():
    api = MagicMock()
    api.patch_namespaced_custom_object.side_effect = Exception("404")
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.is_k8s_not_found", return_value=True),
    ):
        result = update_training_job("missing-job", action="suspend")

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-job" in result["error"]


def test_update_training_job_generic_error_returns_sdk_error():
    api = MagicMock()
    api.patch_namespaced_custom_object.side_effect = RuntimeError("api server down")
    with (
        patch("kubeflow_mcp.trainer.api.lifecycle.validate_k8s_name", return_value=None),
        patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed", return_value=None),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
            return_value="default",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona",
            return_value="platform-admin",
        ),
        patch(
            "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.lifecycle.is_k8s_not_found", return_value=False),
    ):
        result = update_training_job("job-a", action="resume")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "api server down" in result["error"]
