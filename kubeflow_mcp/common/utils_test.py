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

"""Tests for common/utils.py — K8s client singletons, namespace resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubeflow_mcp.common.utils import (
    MCP_MANAGED_LABEL,
    MCP_MANAGED_VALUE,
    _get_api_client,
    get_core_v1_api,
    get_custom_objects_api,
    get_trainer_client_for_namespace,
    get_trainer_effective_namespace,
    is_mcp_managed,
    reset_clients,
)

PATCH_CUSTOM_API = "kubeflow_mcp.common.utils.get_custom_objects_api"
PATCH_GET_API_CLIENT = "kubeflow_mcp.common.utils._get_api_client"
PATCH_TRAINER_CLIENT = "kubeflow_mcp.common.utils.get_trainer_client"


class TestIsMcpManaged:
    @patch(PATCH_CUSTOM_API)
    def test_returns_true_when_label_present(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"labels": {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}}
        }
        mock_api_fn.return_value = mock_api
        assert is_mcp_managed("my-job", "default") is True

    @patch(PATCH_CUSTOM_API)
    def test_returns_false_when_label_missing(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"labels": {"other-label": "value"}}
        }
        mock_api_fn.return_value = mock_api
        assert is_mcp_managed("my-job", "default") is False

    @patch(PATCH_CUSTOM_API)
    def test_returns_false_when_no_labels(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {"metadata": {}}
        mock_api_fn.return_value = mock_api
        assert is_mcp_managed("my-job", "default") is False

    @patch(PATCH_CUSTOM_API)
    def test_returns_false_on_404(self, mock_api_fn):
        from kubernetes.client.exceptions import ApiException

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        mock_api_fn.return_value = mock_api
        assert is_mcp_managed("missing-job", "default") is False

    @patch(PATCH_CUSTOM_API)
    def test_returns_none_on_other_api_error(self, mock_api_fn):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = Exception("connection refused")
        mock_api_fn.return_value = mock_api
        assert is_mcp_managed("my-job", "default") is None


class TestResetClients:
    def test_reset_clears_cached_clients(self):
        reset_clients()
        clients = [MagicMock(name="client-1"), MagicMock(name="client-2")]
        with (
            patch("kubernetes.config.load_config"),
            patch("kubernetes.client") as mock_client,
        ):
            mock_client.Configuration.get_default_copy.return_value = MagicMock()
            mock_client.ApiClient.side_effect = clients

            first = _get_api_client()
            assert mock_client.ApiClient.call_count == 1

            reset_clients()

            second = _get_api_client()
            assert mock_client.ApiClient.call_count == 2
            assert first is clients[0]
            assert second is clients[1]

        reset_clients()


class TestGetCoreV1Api:
    @patch(PATCH_GET_API_CLIENT)
    def test_returns_core_v1_api_instance(self, mock_api_client):
        mock_api_client.return_value = MagicMock()
        result = get_core_v1_api()
        from kubernetes.client import CoreV1Api

        assert isinstance(result, CoreV1Api)

    @patch(PATCH_GET_API_CLIENT)
    def test_uses_shared_api_client(self, mock_api_client):
        shared_client = MagicMock()
        mock_api_client.return_value = shared_client
        get_core_v1_api()
        mock_api_client.assert_called_once()


class TestGetCustomObjectsApi:
    @patch(PATCH_GET_API_CLIENT)
    def test_returns_custom_objects_api_instance(self, mock_api_client):
        mock_api_client.return_value = MagicMock()
        result = get_custom_objects_api()
        from kubernetes.client import CustomObjectsApi

        assert isinstance(result, CustomObjectsApi)

    @patch(PATCH_GET_API_CLIENT)
    def test_uses_shared_api_client(self, mock_api_client):
        shared_client = MagicMock()
        mock_api_client.return_value = shared_client
        get_custom_objects_api()
        mock_api_client.assert_called_once()


class TestGetTrainerEffectiveNamespace:
    def test_explicit_namespace_returned(self):
        assert get_trainer_effective_namespace("my-ns") == "my-ns"

    @patch(PATCH_TRAINER_CLIENT)
    def test_reads_backend_namespace(self, mock_trainer):
        mock_trainer.return_value = SimpleNamespace(
            backend=SimpleNamespace(namespace="from-kubeconfig")
        )
        assert get_trainer_effective_namespace(None) == "from-kubeconfig"

    @patch(PATCH_TRAINER_CLIENT)
    def test_falls_back_to_default(self, mock_trainer):
        mock_trainer.return_value = SimpleNamespace(backend=SimpleNamespace())
        assert get_trainer_effective_namespace(None) == "default"


class TestGetTrainerClientForNamespace:
    @patch(PATCH_TRAINER_CLIENT)
    def test_none_returns_singleton(self, mock_trainer):
        sentinel = MagicMock(name="singleton")
        mock_trainer.return_value = sentinel
        result = get_trainer_client_for_namespace(None)
        assert result is sentinel

    @patch("kubeflow_mcp.common.utils.TrainerClient")
    def test_explicit_namespace_creates_new_client(self, mock_cls):
        from kubeflow_mcp.common.utils import _ns_client_cache, _ns_client_lock

        with _ns_client_lock:
            _ns_client_cache.pop("test-ns", None)
        mock_cls.return_value = MagicMock(name="scoped-client")
        result = get_trainer_client_for_namespace("test-ns")
        assert result is mock_cls.return_value
        mock_cls.assert_called_once()
        with _ns_client_lock:
            _ns_client_cache.pop("test-ns", None)

    @patch("kubeflow_mcp.common.utils.TrainerClient")
    def test_cached_namespace_returns_same_client(self, mock_cls):
        from kubeflow_mcp.common.utils import _ns_client_cache, _ns_client_lock

        with _ns_client_lock:
            _ns_client_cache.pop("cache-ns", None)
        mock_cls.return_value = MagicMock(name="scoped-client")
        first = get_trainer_client_for_namespace("cache-ns")
        second = get_trainer_client_for_namespace("cache-ns")
        assert first is second
        assert mock_cls.call_count == 1
        with _ns_client_lock:
            _ns_client_cache.pop("cache-ns", None)
