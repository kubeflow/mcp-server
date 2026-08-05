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

"""Tests for platform runtime tools: patch_runtime, create_runtime, delete_runtime.

Part of #68 (kubeflow/mcp-server), platform slice.

Covers runtime CRUD preview/confirmed paths with mocked CustomObjects API:
- missing required params / invalid keys
- confirmed=False preview responses
- confirmed=True success paths
- not-found and generic Kubernetes errors
- delete_runtime dependent job listing in preview
"""

from unittest.mock import MagicMock, patch

from kubeflow.trainer.constants import constants as trainer_constants

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.trainer.api.platform import create_runtime, delete_runtime, patch_runtime

# ---------------------------------------------------------------------------
# patch_runtime
# ---------------------------------------------------------------------------


def test_patch_runtime_requires_patch():
    result = patch_runtime("torch-tune", patch=None, confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "patch parameter is required" in result["error"]


def test_patch_runtime_rejects_invalid_top_level_keys():
    result = patch_runtime("torch-tune", patch={"status": {"foo": 1}}, confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "Invalid top-level patch keys" in result["error"]


def test_patch_runtime_preview_when_not_confirmed():
    patch_body = {"spec": {"template": {"spec": {"containers": []}}}}
    with patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api") as mock_api:
        result = patch_runtime("torch-tune", patch=patch_body, confirmed=False)

    assert result["success"] is True
    assert result["data"]["action"] == "preview"
    assert result["data"]["runtime"] == "torch-tune"
    assert result["data"]["patch"] == patch_body
    assert "confirmed=True" in result["data"]["message"]
    mock_api.assert_not_called()


def test_patch_runtime_success_when_confirmed():
    api = MagicMock()
    api.patch_cluster_custom_object.return_value = {"metadata": {"resourceVersion": "123"}}
    patch_body = {"spec": {"labels": {"env": "dev"}}}
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = patch_runtime("torch-tune", patch=patch_body, confirmed=True)

    assert result["success"] is True
    assert result["data"]["patched"] is True
    assert result["data"]["runtime"] == "torch-tune"
    assert result["data"]["resource_version"] == "123"
    api.patch_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        name="torch-tune",
        body=patch_body,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_patch_runtime_not_found():
    api = MagicMock()
    api.patch_cluster_custom_object.side_effect = Exception("404")
    with (
        patch(
            "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.platform.is_k8s_not_found", return_value=True),
    ):
        result = patch_runtime("missing", patch={"spec": {"labels": {}}}, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing" in result["error"]


def test_patch_runtime_generic_error():
    api = MagicMock()
    api.patch_cluster_custom_object.side_effect = RuntimeError("api down")
    with (
        patch(
            "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.platform.is_k8s_not_found", return_value=False),
    ):
        result = patch_runtime("torch-tune", patch={"metadata": {"labels": {}}}, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "KUBERNETES_ERROR"
    assert "api down" in result["error"]


# ---------------------------------------------------------------------------
# create_runtime
# ---------------------------------------------------------------------------


def test_create_runtime_requires_spec():
    result = create_runtime("my-runtime", spec=None, confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "spec parameter is required" in result["error"]


def test_create_runtime_rejects_invalid_top_level_keys():
    result = create_runtime("my-runtime", spec={"foo": 1}, confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "Invalid top-level spec keys" in result["error"]


def test_create_runtime_preview_when_not_confirmed():
    spec = {"template": {"spec": {"containers": [{"name": "main"}]}}}
    with patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api") as mock_api:
        result = create_runtime("my-runtime", spec=spec, confirmed=False)

    assert result["success"] is True
    assert result["data"]["action"] == "preview"
    assert result["data"]["runtime"] == "my-runtime"
    assert result["data"]["body"]["kind"] == "ClusterTrainingRuntime"
    assert result["data"]["body"]["metadata"]["name"] == "my-runtime"
    assert result["data"]["body"]["spec"] == spec
    mock_api.assert_not_called()


def test_create_runtime_success_when_confirmed():
    api = MagicMock()
    api.create_cluster_custom_object.return_value = {"metadata": {"resourceVersion": "9"}}
    spec = {"labels": {"team": "ml"}}
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = create_runtime("my-runtime", spec=spec, confirmed=True)

    assert result["success"] is True
    assert result["data"]["created"] is True
    assert result["data"]["runtime"] == "my-runtime"
    assert result["data"]["resource_version"] == "9"
    api.create_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        body={
            "apiVersion": f"{trainer_constants.GROUP}/{trainer_constants.VERSION}",
            "kind": "ClusterTrainingRuntime",
            "metadata": {"name": "my-runtime"},
            "spec": spec,
        },
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_create_runtime_generic_error():
    api = MagicMock()
    api.create_cluster_custom_object.side_effect = RuntimeError("quota exceeded")
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = create_runtime("my-runtime", spec={"annotations": {"a": "b"}}, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "KUBERNETES_ERROR"
    assert "quota exceeded" in result["error"]


# ---------------------------------------------------------------------------
# delete_runtime
# ---------------------------------------------------------------------------


def test_delete_runtime_preview_lists_dependent_jobs():
    api = MagicMock()
    api.list_cluster_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": "job-a", "namespace": "ml"},
                "spec": {"runtimeRef": {"name": "torch-tune"}},
            },
            {
                "metadata": {"name": "job-b", "namespace": "ml"},
                "spec": {"runtimeRef": {"name": "other"}},
            },
        ]
    }
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = delete_runtime("torch-tune", confirmed=False)

    assert result["success"] is True
    assert result["data"]["action"] == "preview"
    assert result["data"]["runtime"] == "torch-tune"
    assert result["data"]["dependent_count"] == 1
    assert result["data"]["dependent_jobs"] == [{"name": "job-a", "namespace": "ml"}]
    assert "1 TrainJob(s)" in result["data"]["warning"]
    api.delete_cluster_custom_object.assert_not_called()


def test_delete_runtime_preview_with_no_dependents():
    api = MagicMock()
    api.list_cluster_custom_object.return_value = {"items": []}
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = delete_runtime("torch-tune", confirmed=False)

    assert result["data"]["dependent_count"] == 0
    assert "No dependent TrainJobs found" in result["data"]["warning"]


def test_delete_runtime_success_when_confirmed():
    api = MagicMock()
    api.list_cluster_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": "job-a", "namespace": "ml"},
                "spec": {"runtimeRef": {"name": "torch-tune"}},
            }
        ]
    }
    with patch(
        "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
        return_value=api,
    ):
        result = delete_runtime("torch-tune", confirmed=True)

    assert result["success"] is True
    assert result["data"]["deleted"] is True
    assert result["data"]["runtime"] == "torch-tune"
    assert result["data"]["dependent_jobs_affected"] == 1
    api.list_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural=trainer_constants.TRAINJOB_PLURAL,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )
    api.delete_cluster_custom_object.assert_called_once_with(
        group=trainer_constants.GROUP,
        version=trainer_constants.VERSION,
        plural="clustertrainingruntimes",
        name="torch-tune",
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def test_delete_runtime_not_found():
    api = MagicMock()
    api.list_cluster_custom_object.return_value = {"items": []}
    api.delete_cluster_custom_object.side_effect = Exception("404")
    with (
        patch(
            "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.platform.is_k8s_not_found", return_value=True),
    ):
        result = delete_runtime("missing", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing" in result["error"]


def test_delete_runtime_generic_error():
    api = MagicMock()
    api.list_cluster_custom_object.side_effect = RuntimeError("list failed")
    # listing dependents is best-effort; failure continues into delete path
    api.delete_cluster_custom_object.side_effect = RuntimeError("delete failed")
    with (
        patch(
            "kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api",
            return_value=api,
        ),
        patch("kubeflow_mcp.trainer.api.platform.is_k8s_not_found", return_value=False),
    ):
        result = delete_runtime("torch-tune", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "KUBERNETES_ERROR"
    assert "delete failed" in result["error"]
