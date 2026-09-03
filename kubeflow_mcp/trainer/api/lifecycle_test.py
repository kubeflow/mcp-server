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

"""Tests for trainer/api/lifecycle.py — delete, suspend, resume.

Covers input validation. K8s API interaction tests require mocking and are
marked as TODOs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client.exceptions import ApiException
from tests.common import (
    FAILED,
    PERMISSION_DENIED,
    RESOURCE_NOT_FOUND,
    VALIDATION_ERROR,
    TestCase,
    assert_test_case,
)

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.trainer.api.lifecycle import delete_training_job, update_training_job


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"name": "INVALID"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="empty name rejected",
            expected_status=FAILED,
            config={"name": ""},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_delete_training_job_validation(test_case):
    assert_test_case(test_case, delete_training_job)


class TestDeleteTrainingJob:
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_preview_delete_returns_preview_response(self, _ns, _persona):
        result = delete_training_job(name="test-job", confirmed=False)
        assert result["status"] == "preview"
        assert "Will permanently delete" in result["message"]
        assert result["config"] == {"job": "test-job", "namespace": "default"}

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace")
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_confirmed_delete_removes_job(self, _ns, mock_client_fn, _persona):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        result = delete_training_job(name="test-job", confirmed=True)
        assert result["success"] is True
        assert result["data"]["deleted"] is True
        mock_client.delete_job.assert_called_once_with(name="test-job")

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="data-scientist"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed", return_value=False)
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_non_admin_cannot_delete_non_mcp_job(self, _ns, _managed, _persona):
        result = delete_training_job(name="external-job", confirmed=False)
        assert result["success"] is False
        assert result["error_code"] == VALIDATION_ERROR
        assert "was not created by MCP" in result["error"]

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_client_for_namespace")
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_confirmed_not_found_returns_resource_not_found(self, _ns, mock_client_fn, _persona):
        mock_client = MagicMock()
        mock_client.delete_job.side_effect = ApiException(status=404, reason="Not Found")
        mock_client_fn.return_value = mock_client
        result = delete_training_job(name="test-job", confirmed=True)
        assert result["success"] is False
        assert result["error_code"] == RESOURCE_NOT_FOUND

    @patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed")
    def test_namespace_policy_enforcement(self, mock_ns_check):
        mock_ns_check.return_value = ToolError(
            error="Namespace 'restricted-ns' is not allowed",
            error_code=ErrorCode.PERMISSION_DENIED,
        )
        result = delete_training_job(name="test-job", namespace="restricted-ns", confirmed=True)
        assert result["success"] is False
        assert result["error_code"] == PERMISSION_DENIED


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"name": "INVALID", "action": "suspend"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="empty name rejected",
            expected_status=FAILED,
            config={"name": "", "action": "suspend"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="invalid action rejected",
            expected_status=FAILED,
            config={"name": "valid-job", "action": "restart"},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_update_training_job_validation(test_case):
    assert_test_case(test_case, update_training_job)


class TestUpdateTrainingJob:
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_preview_suspend_returns_preview_response(self, _ns, _persona):
        result = update_training_job(name="test-job", action="suspend", confirmed=False)
        assert result["status"] == "preview"
        assert "Will suspend training job 'test-job'" in result["message"]
        assert result["config"] == {"job": "test-job", "namespace": "default", "action": "suspend"}

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_preview_resume_returns_preview_response(self, _ns, _persona):
        result = update_training_job(name="test-job", action="resume", confirmed=False)
        assert result["status"] == "preview"
        assert "Will resume training job 'test-job'" in result["message"]
        assert result["config"] == {"job": "test-job", "namespace": "default", "action": "resume"}

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="data-scientist"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.is_mcp_managed", return_value=False)
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_non_admin_cannot_preview_non_mcp_job(self, _ns, _managed, _persona):
        result = update_training_job(name="external-job", action="suspend", confirmed=False)
        assert result["success"] is False
        assert result["error_code"] == VALIDATION_ERROR
        assert "was not created by MCP" in result["error"]

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api")
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_confirmed_suspend_patches_job(self, _ns, mock_api_fn, _persona):
        mock_api = MagicMock()
        mock_api_fn.return_value = mock_api
        result = update_training_job(name="test-job", action="suspend", confirmed=True)
        assert result["success"] is True
        assert result["data"]["action"] == "suspend"
        assert "suspended" in result["data"]["message"]
        mock_api.patch_namespaced_custom_object.assert_called_once_with(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace="default",
            plural=trainer_constants.TRAINJOB_PLURAL,
            name="test-job",
            body={"spec": {"suspend": True}},
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api")
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_confirmed_resume_patches_job(self, _ns, mock_api_fn, _persona):
        mock_api = MagicMock()
        mock_api_fn.return_value = mock_api
        result = update_training_job(name="test-job", action="resume", confirmed=True)
        assert result["success"] is True
        assert result["data"]["action"] == "resume"
        assert "resumed" in result["data"]["message"]
        mock_api.patch_namespaced_custom_object.assert_called_once_with(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace="default",
            plural=trainer_constants.TRAINJOB_PLURAL,
            name="test-job",
            body={"spec": {"suspend": False}},
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.get_effective_persona", return_value="platform-admin"
    )
    @patch("kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_custom_objects_api")
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_confirmed_not_found_returns_resource_not_found(self, _ns, mock_api_fn, _persona):
        mock_api = MagicMock()
        mock_api.patch_namespaced_custom_object.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        mock_api_fn.return_value = mock_api
        result = update_training_job(name="test-job", action="suspend", confirmed=True)
        assert result["success"] is False
        assert result["error_code"] == RESOURCE_NOT_FOUND

    @patch("kubeflow_mcp.trainer.api.lifecycle.check_namespace_allowed")
    def test_namespace_policy_enforcement(self, mock_ns_check):
        mock_ns_check.return_value = ToolError(
            error="Namespace 'restricted-ns' is not allowed",
            error_code=ErrorCode.PERMISSION_DENIED,
        )
        result = update_training_job(
            name="test-job", action="suspend", namespace="restricted-ns", confirmed=True
        )
        assert result["success"] is False
        assert result["error_code"] == PERMISSION_DENIED
