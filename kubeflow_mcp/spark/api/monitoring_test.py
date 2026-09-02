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

"""Tests for spark monitoring tools (SparkConnect driver logs)."""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.spark.api import monitoring


class TestLogs:
    def test_logs_bounded_and_truncation_flagged(self, mock_spark_client):
        mock_spark_client.get_session_logs.return_value = iter([f"line{i}" for i in range(10)])
        out = monitoring.get_spark_session_logs("x", tail_lines=3)
        assert out["success"] is True
        assert out["data"]["lines"] == 3
        assert out["data"]["truncated"] is True
        assert out["data"]["logs"].splitlines() == ["line7", "line8", "line9"]

    def test_no_driver_pod_is_validation_error(self):
        client = MagicMock()
        client.get_session_logs.side_effect = RuntimeError("No driver pod for SparkConnect: d/x")
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x")
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"
