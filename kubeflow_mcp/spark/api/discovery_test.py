# Copyright The Kubeflow Authors.
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

"""Tests for spark discovery tools (list / get SparkConnect sessions).

The spark module imports the SDK lazily, so these run without the
``kubeflow[spark]`` extra installed; the SparkClient is mocked.
"""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.conftest import make_spark_session_info
from kubeflow_mcp.spark.api import discovery


class TestListSessions:
    def test_list_and_serialize(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info("a"),
            make_spark_session_info("b", state="Failed"),
        ]
        out = discovery.list_spark_sessions()
        assert out["success"] is True
        assert out["data"]["total"] == 2

    def test_state_filter(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info("a"),
            make_spark_session_info("b", state="Failed"),
        ]
        out = discovery.list_spark_sessions(state="failed")  # case-insensitive
        assert out["data"]["total"] == 1
        assert out["data"]["sessions"][0]["name"] == "b"

    def test_invalid_limit(self):
        out = discovery.list_spark_sessions(limit=0)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_missing_extra_friendly_error(self):
        with patch.object(
            discovery,
            "get_spark_client_for_namespace",
            side_effect=ImportError("SparkClient requires the Spark extra."),
        ):
            out = discovery.list_spark_sessions()
        assert out["success"] is False
        assert "Spark extra" in out["error"]


class TestGetSession:
    def test_get_ok(self, mock_spark_client):
        mock_spark_client.get_session.return_value = make_spark_session_info("x")
        out = discovery.get_spark_session("x")
        assert out["success"] is True
        assert out["data"]["name"] == "x"

    def test_not_found_maps_to_resource_not_found(self):
        from kubernetes.client.exceptions import ApiException

        exc = RuntimeError("SparkConnect not found: default/missing")
        exc.__cause__ = ApiException(status=404)
        client = MagicMock()
        client.get_session.side_effect = exc
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("missing")
        assert out["success"] is False
        assert out["error_code"] == "RESOURCE_NOT_FOUND"

    def test_failed_state_has_next_steps(self, mock_spark_client):
        mock_spark_client.get_session.return_value = make_spark_session_info("x", state="Failed")
        out = discovery.get_spark_session("x")
        assert "next_steps" in out["data"]
