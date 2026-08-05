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

from kubeflow_mcp.common.utils import _get_api_client, reset_clients


class TestIsMcpManaged:
    """is_mcp_managed(name, namespace) makes a K8s API call.
    Tests require mocking the CustomObjects API.
    """

    # TODO(test): test returns True when label present (mock get_custom_objects_api)
    # TODO(test): test returns False when label absent
    # TODO(test): test returns None on API error
    pass


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
