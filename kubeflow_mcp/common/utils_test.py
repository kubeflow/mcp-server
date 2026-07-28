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

from unittest.mock import MagicMock, patch

from kubeflow_mcp.common.utils import (
    MCP_MANAGED_LABEL,
    MCP_MANAGED_VALUE,
    _get_api_client,
    is_mcp_managed,
    reset_clients,
)

PATCH_CUSTOM_API = "kubeflow_mcp.common.utils.get_custom_objects_api"


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

    # TODO(test): test get_core_v1_api returns singleton
    # TODO(test): test get_core_v1_api returns same instance on second call
    # TODO(test): test get_trainer_effective_namespace reads kubeconfig default
    # TODO(test): test get_trainer_client_for_namespace creates client with given ns
    # TODO(test): test get_custom_objects_api returns singleton
