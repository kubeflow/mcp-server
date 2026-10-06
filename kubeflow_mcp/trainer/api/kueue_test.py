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

"""Unit tests for Kueue discovery, Workload parsing, and queue status resolution."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest
import yaml
from kubernetes.client.exceptions import ApiException
from kubernetes.client.models import (
    V1APIGroup,
    V1APIGroupList,
    V1GroupVersionForDiscovery,
)

from kubeflow_mcp.trainer.api.kueue import (
    MAX_QUEUE_MESSAGE_LENGTH,
    discover_kueue_version,
    get_trainjob_queue_status,
    reset_kueue_cache,
)

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "kueue"


def _load_fixture(name: str) -> dict:
    path = _FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_kueue_cache()
    yield
    reset_kueue_cache()


# ─── discover_kueue_version tests ──────────────────────────────────────────


class TestDiscoverKueueVersion:
    @patch("kubernetes.client.ApisApi")
    def test_discovery_success_with_preferred_version(self, mock_apis_cls):
        gv = V1GroupVersionForDiscovery(group_version="kueue.x-k8s.io/v1beta1", version="v1beta1")
        group = V1APIGroup(name="kueue.x-k8s.io", versions=[gv], preferred_version=gv)
        mock_apis = MagicMock()
        mock_apis.get_api_versions.return_value = V1APIGroupList(groups=[group])
        mock_apis_cls.return_value = mock_apis

        version = discover_kueue_version()
        assert version == "v1beta1"

        # Subsequent call should hit cache without calling API
        version2 = discover_kueue_version()
        assert version2 == "v1beta1"
        assert mock_apis.get_api_versions.call_count == 1

    @patch("kubernetes.client.ApisApi")
    def test_discovery_fallback_to_versions_list(self, mock_apis_cls):
        gv = V1GroupVersionForDiscovery(group_version="kueue.x-k8s.io/v1alpha2", version="v1alpha2")
        group = V1APIGroup(name="kueue.x-k8s.io", versions=[gv])
        group.preferred_version = None
        mock_apis = MagicMock()
        mock_apis.get_api_versions.return_value = V1APIGroupList(groups=[group])
        mock_apis_cls.return_value = mock_apis

        version = discover_kueue_version()
        assert version == "v1alpha2"

    @patch("kubernetes.client.ApisApi")
    def test_discovery_absent_group_returns_none_and_caches_negative(self, mock_apis_cls):
        group = V1APIGroup(name="other.x-k8s.io", versions=[])
        mock_apis = MagicMock()
        mock_apis.get_api_versions.return_value = V1APIGroupList(groups=[group])
        mock_apis_cls.return_value = mock_apis

        version = discover_kueue_version()
        assert version is None

        # Negative hit cached
        version2 = discover_kueue_version()
        assert version2 is None
        assert mock_apis.get_api_versions.call_count == 1

    @patch("kubernetes.client.ApisApi")
    def test_discovery_404_returns_none(self, mock_apis_cls):
        mock_apis = MagicMock()
        mock_apis.get_api_versions.side_effect = ApiException(status=404)
        mock_apis_cls.return_value = mock_apis

        assert discover_kueue_version() is None

    @patch("kubernetes.client.ApisApi")
    def test_discovery_transient_error_not_cached(self, mock_apis_cls):
        mock_apis = MagicMock()
        mock_apis.get_api_versions.side_effect = ApiException(status=500)
        mock_apis_cls.return_value = mock_apis

        assert discover_kueue_version() is None

        # Second call should retry because 500 was not cached
        mock_apis.get_api_versions.side_effect = None
        gv = V1GroupVersionForDiscovery(group_version="kueue.x-k8s.io/v1beta1", version="v1beta1")
        mock_apis.get_api_versions.return_value = V1APIGroupList(
            groups=[V1APIGroup(name="kueue.x-k8s.io", versions=[gv], preferred_version=gv)]
        )

        assert discover_kueue_version() == "v1beta1"
        assert mock_apis.get_api_versions.call_count == 2


# ─── get_trainjob_queue_status tests ───────────────────────────────────────


class TestGetTrainjobQueueStatus:
    def test_terminal_status_skips_kueue_lookup(self):
        with patch("kubeflow_mcp.trainer.api.kueue.discover_kueue_version") as mock_disc:
            res_complete = get_trainjob_queue_status("job1", "default", job_status="Complete")
            res_failed = get_trainjob_queue_status("job1", "default", job_status="Failed")
            assert res_complete is None
            assert res_failed is None
            mock_disc.assert_not_called()

    @patch("kubeflow_mcp.trainer.api.kueue.discover_kueue_version", return_value=None)
    def test_kueue_not_available_returns_none(self, _disc):
        result = get_trainjob_queue_status("job1", "default", job_status="Created")
        assert result is None

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_trainjob_not_found_returns_none(self, mock_custom_api_fn, _disc):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = ApiException(status=404)
        mock_custom_api_fn.return_value = mock_api

        assert get_trainjob_queue_status("missing", "default", job_status="Created") is None

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_workload_queued_fixture(self, mock_custom_api_fn, _disc):
        wl = _load_fixture("workload_queued.yaml")
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "7b4d1a58-3cf2-4c6e-8a21-9d1a3b5c7e90",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert res["queue_name"] == "user-queue"
        assert res["state"] == "queued"
        assert res["reason"] == "Pending"
        assert "waiting for 4 nvidia.com/gpu" in res["message"]

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_workload_inadmissible_fixture(self, mock_custom_api_fn, _disc):
        wl = _load_fixture("workload_inadmissible.yaml")
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "7b4d1a58-3cf2-4c6e-8a21-9d1a3b5c7e90",
                "labels": {"kueue.x-k8s.io/queue-name": "non-existent-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert res["queue_name"] == "non-existent-queue"
        assert res["state"] == "inadmissible"
        assert res["reason"] == "Inadmissible"
        assert "doesn't exist" in res["message"]

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_workload_admitted_fixture(self, mock_custom_api_fn, _disc):
        wl = _load_fixture("workload_admitted.yaml")
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "7b4d1a58-3cf2-4c6e-8a21-9d1a3b5c7e90",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Running")
        assert res is not None
        assert res["queue_name"] == "user-queue"
        assert res["state"] == "admitted"
        assert res["reason"] == "Admitted"
        assert res["cluster_queue"] == "cluster-gpu"

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_workload_evicted_fixture(self, mock_custom_api_fn, _disc):
        wl = _load_fixture("workload_evicted.yaml")
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "7b4d1a58-3cf2-4c6e-8a21-9d1a3b5c7e90",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert res["state"] == "evicted"
        assert res["reason"] == "Preempted"
        assert res["requeue_count"] == 1

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_deactivated_spec_active_false(self, mock_custom_api_fn, _disc):
        wl = {
            "spec": {"queueName": "user-queue", "active": False},
            "status": {},
        }
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "123",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert res["state"] == "evicted"
        assert res["reason"] == "Deactivated"

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_reconciliation_lag_returns_reconciling_state(self, mock_custom_api_fn, _disc):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "123",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            },
            "spec": {"suspend": True},
        }
        # No workloads created yet
        mock_api.list_namespaced_custom_object.return_value = {"items": []}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert res["state"] == "queued"
        assert res["reason"] == "Reconciling"

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_running_job_with_queue_label_no_workload_returns_none(self, mock_custom_api_fn, _disc):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "123",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            },
            "spec": {"suspend": False},
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": []}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Running")
        assert res is None

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_message_truncation(self, mock_custom_api_fn, _disc):
        long_msg = "X" * 1000
        wl = {
            "spec": {"queueName": "user-queue"},
            "status": {
                "conditions": [
                    {
                        "type": "QuotaReserved",
                        "status": "False",
                        "reason": "Pending",
                        "message": long_msg,
                    }
                ]
            },
        }
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {
                "name": "llama-lora",
                "uid": "123",
                "labels": {"kueue.x-k8s.io/queue-name": "user-queue"},
            }
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
        assert res is not None
        assert len(res["message"]) == MAX_QUEUE_MESSAGE_LENGTH

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_rbac_403_fails_open(self, mock_custom_api_fn, _disc, caplog):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "llama-lora", "uid": "123"}
        }
        mock_api.list_namespaced_custom_object.side_effect = ApiException(status=403)
        mock_custom_api_fn.return_value = mock_api

        import logging

        with caplog.at_level(logging.WARNING, logger="kubeflow_mcp.trainer.api.kueue"):
            res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
            assert res is None
            assert "Permission denied listing Kueue Workloads" in caplog.text
            assert "workloads.kueue.x-k8s.io" in caplog.text

            # Verify it only warns once across subsequent calls
            caplog.clear()
            res2 = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
            assert res2 is None
            assert "Permission denied listing Kueue Workloads" not in caplog.text

            # After reset_kueue_cache, it can warn again
            reset_kueue_cache()
            res3 = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
            assert res3 is None
            assert "Permission denied listing Kueue Workloads" in caplog.text

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_non_403_error_does_not_warn(self, mock_custom_api_fn, _disc, caplog):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "llama-lora", "uid": "123"}
        }
        mock_api.list_namespaced_custom_object.side_effect = ApiException(status=500)
        mock_custom_api_fn.return_value = mock_api

        import logging

        with caplog.at_level(logging.WARNING, logger="kubeflow_mcp.trainer.api.kueue"):
            res = get_trainjob_queue_status("llama-lora", "default", job_status="Created")
            assert res is None
            assert caplog.text == ""

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_job_missing_uid_returns_none(self, mock_custom_api_fn, _disc):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {"metadata": {"name": "llama-lora"}}
        mock_custom_api_fn.return_value = mock_api

        assert get_trainjob_queue_status("llama-lora", "default") is None

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_no_workload_and_no_queue_label_returns_none(self, mock_custom_api_fn, _disc):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "llama-lora", "uid": "123", "labels": {}}
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": []}
        mock_custom_api_fn.return_value = mock_api

        assert get_trainjob_queue_status("llama-lora", "default") is None

    @patch("kubernetes.client.ApisApi")
    def test_discovery_fallback_default_when_versions_empty(self, mock_apis_cls):
        group = V1APIGroup(name="kueue.x-k8s.io", versions=[])
        group.preferred_version = None
        mock_apis = MagicMock()
        mock_apis.get_api_versions.return_value = V1APIGroupList(groups=[group])
        mock_apis_cls.return_value = mock_apis

        assert discover_kueue_version() == "v1beta1"

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_admitted_false_maps_to_queued(self, mock_custom_api_fn, _disc):
        wl = {
            "spec": {"queueName": "user-queue"},
            "status": {
                "conditions": [{"type": "Admitted", "status": "False", "reason": "Waiting"}]
            },
        }
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "llama-lora", "uid": "123"}
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default")
        assert res is not None
        assert res["state"] == "queued"
        assert res["reason"] == "Waiting"

    @patch(
        "kubeflow_mcp.trainer.api.kueue.discover_kueue_version",
        return_value="v1beta1",
    )
    @patch("kubeflow_mcp.trainer.api.kueue.get_custom_objects_api")
    def test_empty_conditions_maps_to_reconciling(self, mock_custom_api_fn, _disc):
        wl = {
            "spec": {"queueName": "user-queue"},
            "status": {"conditions": []},
        }
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"name": "llama-lora", "uid": "123"}
        }
        mock_api.list_namespaced_custom_object.return_value = {"items": [wl]}
        mock_custom_api_fn.return_value = mock_api

        res = get_trainjob_queue_status("llama-lora", "default")
        assert res is not None
        assert res["state"] == "queued"
        assert res["reason"] == "Reconciling"
