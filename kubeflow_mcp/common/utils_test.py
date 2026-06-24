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
"""Unit tests for common/utils.py helpers."""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.common.utils import (
    MCP_MANAGED_LABEL,
    MCP_MANAGED_VALUE,
    get_trainer_effective_namespace,
    is_mcp_managed,
    reset_clients,
)


class TestResetClients:
    def test_reset_clients_clears_cache(self):
        """reset_clients() should not raise and clears the LRU caches."""
        reset_clients()  # baseline — must not raise
        reset_clients()  # second call also safe

    def test_trainer_client_recreated_after_reset(self):
        """After reset, get_trainer_client() creates a fresh instance."""
        from unittest.mock import patch

        from kubeflow_mcp.common.utils import get_trainer_client

        with patch("kubeflow_mcp.common.utils.TrainerClient") as mock_tc:
            mock_tc.return_value = MagicMock()
            reset_clients()
            get_trainer_client()
            get_trainer_client()  # second call — should be cached
            assert mock_tc.call_count == 1  # only one new instance created

        reset_clients()


class TestGetTrainerEffectiveNamespace:
    def test_explicit_namespace_returned_as_is(self):
        assert get_trainer_effective_namespace("ml-team") == "ml-team"

    def test_none_resolves_from_backend(self):
        mock_client = MagicMock()
        mock_client.backend.namespace = "from-backend"

        with patch("kubeflow_mcp.common.utils.get_trainer_client", return_value=mock_client):
            ns = get_trainer_effective_namespace(None)

        assert ns == "from-backend"

    def test_none_falls_back_to_default_when_backend_has_no_namespace(self):
        mock_client = MagicMock()
        mock_client.backend.namespace = None

        with patch("kubeflow_mcp.common.utils.get_trainer_client", return_value=mock_client):
            ns = get_trainer_effective_namespace(None)

        assert ns == "default"


class TestIsMcpManaged:
    def test_returns_true_when_label_present(self):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {
            "metadata": {"labels": {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}}
        }

        with patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=mock_api):
            result = is_mcp_managed("my-job", "default")

        assert result is True

    def test_returns_false_when_label_absent(self):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = {"metadata": {"labels": {}}}

        with patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=mock_api):
            result = is_mcp_managed("my-job", "default")

        assert result is False

    def test_returns_false_when_job_not_found(self):
        mock_api = MagicMock()
        not_found_exc = Exception('{"code": 404, "reason": "NotFound"}')
        mock_api.get_namespaced_custom_object.side_effect = not_found_exc

        with (
            patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=mock_api),
            patch("kubeflow_mcp.common.types.is_k8s_not_found", return_value=True),
        ):
            result = is_mcp_managed("missing", "default")

        assert result is False

    def test_returns_none_on_api_error(self):
        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = Exception("connection refused")

        with (
            patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=mock_api),
            patch("kubeflow_mcp.common.types.is_k8s_not_found", return_value=False),
        ):
            result = is_mcp_managed("my-job", "default")

        assert result is None
