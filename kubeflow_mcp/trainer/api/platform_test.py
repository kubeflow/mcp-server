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
"""Tests for platform administration tools (inspect_crd, inspect_controller,
patch/create/delete_runtime).
"""

from unittest.mock import MagicMock, patch  # noqa: I001

from kubeflow_mcp.trainer.api.platform import (
    create_runtime,
    delete_runtime,
    inspect_crd,
    inspect_controller,
    patch_runtime,
)


def _make_mock_crd(name="trainjobs.trainer.kubeflow.org", group="trainer.kubeflow.org"):
    crd = MagicMock()
    crd.metadata.name = name
    crd.spec.group = group
    crd.spec.scope = "Cluster"
    v = MagicMock()
    v.name = "v1alpha1"
    v.served = True
    v.storage = True
    v.schema = None
    crd.spec.versions = [v]
    crd.status.conditions = []
    return crd


class TestInspectCrd:
    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_apiextensions_api")
    def test_list_all_trainer_crds(self, mock_api_fn):
        crd = _make_mock_crd()
        mock_api = MagicMock()
        mock_api.list_custom_resource_definition.return_value.items = [crd]
        mock_api_fn.return_value = mock_api

        result = inspect_crd()

        assert result["success"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["crds"][0]["name"] == "trainjobs.trainer.kubeflow.org"

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_apiextensions_api")
    def test_list_filters_to_trainer_group_only(self, mock_api_fn):
        trainer_crd = _make_mock_crd()
        other_crd = _make_mock_crd(name="foos.other.io", group="other.io")
        mock_api = MagicMock()
        mock_api.list_custom_resource_definition.return_value.items = [trainer_crd, other_crd]
        mock_api_fn.return_value = mock_api

        result = inspect_crd()

        assert result["data"]["count"] == 1

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_apiextensions_api")
    def test_get_specific_crd(self, mock_api_fn):
        crd = _make_mock_crd()
        mock_api = MagicMock()
        mock_api.read_custom_resource_definition.return_value = crd
        mock_api_fn.return_value = mock_api

        result = inspect_crd(name="trainjobs.trainer.kubeflow.org")

        assert result["success"] is True
        assert result["data"]["name"] == "trainjobs.trainer.kubeflow.org"
        assert result["data"]["group"] == "trainer.kubeflow.org"

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_apiextensions_api")
    def test_get_crd_not_found(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.read_custom_resource_definition.side_effect = Exception(
            '{"code": 404, "reason": "NotFound"}'
        )
        mock_api_fn.return_value = mock_api

        result = inspect_crd(name="missing.crd")

        assert result["success"] is False

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_apiextensions_api")
    def test_api_error_returns_failure(self, mock_api_fn):
        mock_api_fn.side_effect = Exception("connection refused")

        result = inspect_crd()

        assert result["success"] is False
        assert result["error_code"] == "KUBERNETES_ERROR"


class TestInspectController:
    def test_invalid_view_rejected(self):
        result = inspect_controller(view="invalid")

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "Invalid view" in result["error"]

    def test_tail_lines_too_small_rejected(self):
        result = inspect_controller(tail_lines=0)

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch("kubeflow_mcp.trainer.api.platform._find_controller_pod")
    def test_no_controller_pod_returns_not_found(self, mock_find):
        mock_find.return_value = (None, "kubeflow", MagicMock())

        result = inspect_controller()

        assert result["success"] is False
        assert result["error_code"] == "RESOURCE_NOT_FOUND"
        assert "kubeflow" in result["error"]

    @patch("kubeflow_mcp.trainer.api.platform._find_controller_pod")
    def test_logs_view_success(self, mock_find):
        mock_pod = MagicMock()
        mock_pod.metadata.name = "trainer-controller-abc"
        mock_pod.metadata.namespace = "kubeflow"
        mock_pod.status.phase = "Running"
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log.return_value = "2026-01-01 INFO starting\n"
        mock_find.return_value = (mock_pod, "kubeflow", mock_core)

        result = inspect_controller(view="logs")

        assert result["success"] is True
        assert result["data"]["pod"] == "trainer-controller-abc"
        assert result["data"]["namespace"] == "kubeflow"
        assert "2026" in result["data"]["logs"]

    @patch("kubeflow_mcp.trainer.api.platform._find_controller_pod")
    def test_events_view_success(self, mock_find):
        mock_pod = MagicMock()
        mock_pod.metadata.name = "trainer-controller-abc"
        mock_pod.metadata.namespace = "kubeflow"
        mock_core = MagicMock()
        ev = MagicMock()
        ev.type = "Normal"
        ev.reason = "Pulled"
        ev.message = "Image pulled"
        ev.count = 1
        ev.first_timestamp = None
        ev.last_timestamp = None
        mock_core.list_namespaced_event.return_value.items = [ev]
        mock_find.return_value = (mock_pod, "kubeflow", mock_core)

        result = inspect_controller(view="events")

        assert result["success"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["events"][0]["reason"] == "Pulled"

    @patch("kubeflow_mcp.trainer.api.platform._find_controller_pod")
    def test_tail_lines_capped_at_maximum(self, mock_find):
        mock_pod = MagicMock()
        mock_pod.metadata.name = "pod"
        mock_pod.metadata.namespace = "ns"
        mock_pod.status.phase = "Running"
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log.return_value = ""
        mock_find.return_value = (mock_pod, "ns", mock_core)

        inspect_controller(tail_lines=99999)

        call_kwargs = mock_core.read_namespaced_pod_log.call_args.kwargs
        assert call_kwargs["tail_lines"] <= 1000


class TestPatchRuntime:
    def test_preview_without_confirmed(self):
        result = patch_runtime(name="torch-tune", patch={"spec": {"template": {}}})

        assert result["success"] is True
        assert result["data"]["action"] == "preview"
        assert result["data"]["runtime"] == "torch-tune"
        assert "confirmed=True" in result["data"]["message"]

    def test_missing_patch_rejected(self):
        result = patch_runtime(name="torch-tune")

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    def test_invalid_top_level_key_rejected(self):
        result = patch_runtime(name="torch-tune", patch={"status": {}})

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "status" in result["error"]

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_patch_applied_successfully(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.patch_cluster_custom_object.return_value = {"metadata": {"resourceVersion": "42"}}
        mock_api_fn.return_value = mock_api

        result = patch_runtime(
            name="torch-tune",
            patch={"spec": {"template": {"spec": {}}}},
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["patched"] is True
        assert result["data"]["resource_version"] == "42"

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_patch_not_found(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.patch_cluster_custom_object.side_effect = Exception(
            '{"code": 404, "reason": "NotFound"}'
        )
        mock_api_fn.return_value = mock_api

        result = patch_runtime(name="missing", patch={"spec": {}}, confirmed=True)

        assert result["success"] is False


class TestCreateRuntime:
    def test_preview_without_confirmed(self):
        result = create_runtime(name="my-runtime", spec={"template": {}})

        assert result["success"] is True
        assert result["data"]["action"] == "preview"
        assert "apiVersion" in str(result["data"]["body"])

    def test_missing_spec_rejected(self):
        result = create_runtime(name="my-runtime")

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    def test_invalid_spec_key_rejected(self):
        result = create_runtime(name="my-runtime", spec={"status": {}})

        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "status" in result["error"]

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_create_success(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.create_cluster_custom_object.return_value = {"metadata": {"resourceVersion": "1"}}
        mock_api_fn.return_value = mock_api

        result = create_runtime(
            name="my-runtime",
            spec={"template": {"spec": {}}},
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["created"] is True

    def test_preview_body_contains_correct_kind(self):
        result = create_runtime(name="new-rt", spec={"template": {}})

        body = result["data"]["body"]
        assert body["kind"] == "ClusterTrainingRuntime"
        assert body["metadata"]["name"] == "new-rt"


class TestDeleteRuntime:
    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_preview_shows_no_dependents(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {"items": []}
        mock_api_fn.return_value = mock_api

        result = delete_runtime(name="old-runtime")

        assert result["success"] is True
        assert result["data"]["action"] == "preview"
        assert result["data"]["dependent_count"] == 0

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_preview_lists_dependent_jobs(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {
            "items": [
                {
                    "metadata": {"name": "job-1", "namespace": "default"},
                    "spec": {"runtimeRef": {"name": "old-runtime"}},
                },
                {
                    "metadata": {"name": "job-2", "namespace": "ml"},
                    "spec": {"runtimeRef": {"name": "other-runtime"}},
                },
            ]
        }
        mock_api_fn.return_value = mock_api

        result = delete_runtime(name="old-runtime")

        assert result["data"]["dependent_count"] == 1
        assert result["data"]["dependent_jobs"][0]["name"] == "job-1"

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_delete_success(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {"items": []}
        mock_api.delete_cluster_custom_object.return_value = None
        mock_api_fn.return_value = mock_api

        result = delete_runtime(name="old-runtime", confirmed=True)

        assert result["success"] is True
        assert result["data"]["deleted"] is True

    @patch("kubeflow_mcp.trainer.api.platform.mcp_utils.get_custom_objects_api")
    def test_delete_not_found(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {"items": []}
        mock_api.delete_cluster_custom_object.side_effect = Exception(
            '{"code": 404, "reason": "NotFound"}'
        )
        mock_api_fn.return_value = mock_api

        result = delete_runtime(name="missing", confirmed=True)

        assert result["success"] is False
