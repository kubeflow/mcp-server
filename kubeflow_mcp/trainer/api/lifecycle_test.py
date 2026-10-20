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

Covers input validation, confirmation previews, the MCP ownership gate,
and mocked SDK/K8s API success and error paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client.exceptions import ApiException
from tests.common import (
    FAILED,
    PERMISSION_DENIED,
    RESOURCE_NOT_FOUND,
    SDK_ERROR,
    VALIDATION_ERROR,
    TestCase,
    assert_test_case,
)

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.conftest import verify_tool_error, verify_tool_success
from kubeflow_mcp.trainer.api.lifecycle import delete_training_job, update_training_job

_UTILS = "kubeflow_mcp.common.utils"
_LIFECYCLE = "kubeflow_mcp.trainer.api.lifecycle"


@contextmanager
def cluster(ownership=mcp_utils.OWNERSHIP_MANAGED, persona="data-scientist"):
    """Patch namespace resolution, the ownership lookup and the K8s APIs."""
    api = MagicMock()
    client = MagicMock()
    with (
        patch(f"{_UTILS}.get_trainer_effective_namespace", return_value="default"),
        patch(f"{_UTILS}.get_trainer_custom_objects_api", return_value=api),
        patch(f"{_UTILS}.get_trainer_client_for_namespace", return_value=client),
        patch(f"{_UTILS}.get_trainer_ownership", return_value=ownership) as lookup,
        patch(f"{_LIFECYCLE}.get_effective_persona", return_value=persona),
        patch(f"{_LIFECYCLE}.check_namespace_allowed", return_value=None),
    ):
        yield api, client, lookup


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
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_ownership",
        return_value=mcp_utils.OWNERSHIP_UNMANAGED,
    )
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_non_admin_cannot_delete_non_mcp_job(self, _ns, _ownership, _persona):
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

    def test_generic_error_returns_sdk_error(self, mock_trainer_client: MagicMock) -> None:
        mock_trainer_client.delete_job.side_effect = RuntimeError("cluster unreachable")
        with patch(f"{_LIFECYCLE}.get_effective_persona", return_value="platform-admin"):
            result = delete_training_job(name="test-job", namespace="default", confirmed=True)

        verify_tool_error(result, error_code=SDK_ERROR)
        assert "cluster unreachable" in result["error"]
        mock_trainer_client.delete_job.assert_called_once_with(name="test-job")


# ─── MCP ownership enforcement ─────────────────────────────────────────────


@pytest.mark.parametrize("persona", ["data-scientist", "ml-engineer", "platform-admin"])
@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ghost-job", confirmed=True)),
        ("update", lambda: update_training_job("ghost-job", "suspend", confirmed=True)),
    ],
)
def test_missing_job_reports_not_found_for_every_persona(persona, tool, call):
    """A name that does not exist is a not-found, whoever asks.

    The ownership gate used to collapse "missing" into "not created by MCP",
    so a typo sent non-admins after RBAC that was never the problem, and the
    same job answered differently depending on the caller's persona.
    """
    with cluster(ownership=mcp_utils.OWNERSHIP_MISSING, persona=persona) as (api, client, _):
        # platform-admin skips the ownership lookup and reaches the cluster,
        # which returns its own 404.
        client.delete_job.side_effect = ApiException(status=404, reason="Not Found")
        api.patch_namespaced_custom_object.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        result = call()

    assert result["success"] is False
    assert result["error_code"] == RESOURCE_NOT_FOUND
    assert "not created by MCP" not in result["error"]


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("external", confirmed=True)),
        ("update", lambda: update_training_job("external", "suspend", confirmed=True)),
    ],
)
def test_non_admin_blocked_from_mutating_external_job(tool, call):
    with cluster(ownership=mcp_utils.OWNERSHIP_UNMANAGED) as (api, client, _):
        result = call()
    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    assert "was not created by MCP" in result["error"]
    client.delete_job.assert_not_called()
    api.patch_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ours", confirmed=True)),
        ("update", lambda: update_training_job("ours", "suspend", confirmed=True)),
    ],
)
def test_non_admin_allowed_for_mcp_created_job(tool, call):
    with cluster(ownership=mcp_utils.OWNERSHIP_MANAGED) as (api, client, _lookup):
        result = call()
    assert result["success"] is True
    assert client.delete_job.call_count == (1 if tool == "delete" else 0)
    assert api.patch_namespaced_custom_object.call_count == (1 if tool == "update" else 0)


def test_admin_bypasses_ownership_check_entirely():
    """platform-admin must not even perform the lookup."""
    with cluster(ownership=mcp_utils.OWNERSHIP_UNMANAGED, persona="platform-admin") as (
        _,
        client,
        lookup,
    ):
        result = delete_training_job("external", confirmed=True)
    assert result["success"] is True
    client.delete_job.assert_called_once()
    lookup.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ghost-job", confirmed=True)),
        ("update", lambda: update_training_job("ghost-job", "suspend", confirmed=True)),
    ],
)
def test_missing_job_end_to_end_from_a_real_404(tool, call):
    """End to end through the real ownership helper, mocking only Kubernetes.

    The tests above stub ``get_trainer_ownership``, so they pin the gate's
    branching but cannot see how a genuine 404 is classified. This one closes
    that gap: it is the test that fails if the 404 is ever folded back into
    "unmanaged".
    """
    k8s = MagicMock()
    k8s.get_namespaced_custom_object.side_effect = ApiException(status=404, reason="Not Found")
    client = MagicMock()

    with (
        patch(f"{_UTILS}.get_trainer_effective_namespace", return_value="default"),
        patch(f"{_UTILS}.get_custom_objects_api", return_value=k8s),
        patch(f"{_UTILS}.get_trainer_custom_objects_api", return_value=k8s),
        patch(f"{_UTILS}.get_trainer_client_for_namespace", return_value=client),
        patch(f"{_LIFECYCLE}.get_effective_persona", return_value="data-scientist"),
        patch(f"{_LIFECYCLE}.check_namespace_allowed", return_value=None),
    ):
        result = call()

    assert result["error_code"] == RESOURCE_NOT_FOUND
    assert "not created by MCP" not in result["error"]
    client.delete_job.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("unknown", confirmed=True)),
        ("update", lambda: update_training_job("unknown", "suspend", confirmed=True)),
    ],
)
def test_ownership_lookup_failure_does_not_mutate(tool, call):
    with cluster(ownership=None) as (api, client, _lookup):
        result = call()
    assert result["error_code"] == SDK_ERROR
    client.delete_job.assert_not_called()
    api.patch_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ours", confirmed=False)),
        ("update", lambda: update_training_job("ours", "suspend", confirmed=False)),
    ],
)
def test_non_admin_preview_checks_ownership_without_mutating(tool, call):
    """The preview runs after the ownership gate and never touches the job."""
    with cluster(ownership=mcp_utils.OWNERSHIP_MANAGED) as (api, client, lookup):
        result = call()
    assert result["status"] == "preview"
    assert result["config"]["job"] == "ours"
    lookup.assert_called_once_with("ours", "default")
    client.delete_job.assert_not_called()
    api.patch_namespaced_custom_object.assert_not_called()


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
    def test_ownership_api_error_for_non_admin(self, mock_k8s_apis: dict[str, MagicMock]) -> None:
        with (
            patch(f"{_LIFECYCLE}.get_effective_persona", return_value="ml-engineer"),
            patch(f"{_UTILS}.get_trainer_ownership", return_value=None) as lookup,
        ):
            result = update_training_job(
                name="test-job", action="suspend", namespace="default", confirmed=True
            )

        verify_tool_error(result, error_code=SDK_ERROR)
        assert "Cannot verify ownership" in result["error"]
        lookup.assert_called_once_with("test-job", "default")
        mock_k8s_apis["custom"].patch_namespaced_custom_object.assert_not_called()

    def test_managed_job_can_be_updated_by_non_admin(
        self, mock_k8s_apis: dict[str, MagicMock]
    ) -> None:
        with (
            patch(f"{_LIFECYCLE}.get_effective_persona", return_value="data-scientist"),
            patch(
                f"{_UTILS}.get_trainer_ownership", return_value=mcp_utils.OWNERSHIP_MANAGED
            ) as lookup,
        ):
            result = update_training_job(
                name="test-job", action="suspend", namespace="default", confirmed=True
            )

        data = verify_tool_success(result)
        assert data["job"] == "test-job"
        assert data["namespace"] == "default"
        assert data["action"] == "suspend"
        lookup.assert_called_once_with("test-job", "default")
        mock_k8s_apis["custom"].patch_namespaced_custom_object.assert_called_once_with(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace="default",
            plural=trainer_constants.TRAINJOB_PLURAL,
            name="test-job",
            body={"spec": {"suspend": True}},
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

    def test_generic_error_returns_sdk_error(self, mock_k8s_apis: dict[str, MagicMock]) -> None:
        api = mock_k8s_apis["custom"]
        api.patch_namespaced_custom_object.side_effect = RuntimeError("api server down")
        with patch(f"{_LIFECYCLE}.get_effective_persona", return_value="platform-admin"):
            result = update_training_job(
                name="test-job", action="resume", namespace="default", confirmed=True
            )

        verify_tool_error(result, error_code=SDK_ERROR)
        assert "api server down" in result["error"]
        api.patch_namespaced_custom_object.assert_called_once_with(
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
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_ownership",
        return_value=mcp_utils.OWNERSHIP_UNMANAGED,
    )
    @patch(
        "kubeflow_mcp.trainer.api.lifecycle.mcp_utils.get_trainer_effective_namespace",
        return_value="default",
    )
    def test_non_admin_cannot_preview_non_mcp_job(self, _ns, _ownership, _persona):
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
