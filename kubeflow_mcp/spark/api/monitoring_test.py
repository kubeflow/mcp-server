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

import pytest

from kubeflow_mcp.spark.api import monitoring
from kubeflow_mcp.spark.api.monitoring import _TRUNCATION_MARKER


class TestLogs:
    def test_logs_enforce_total_character_budget_for_long_line(self):
        client = MagicMock()
        client.get_session_logs.return_value = iter(["x" * (monitoring.MAX_LOG_CHARS * 3)])
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("valid-name")
        assert out["success"] is True
        assert len(out["data"]["logs"]) <= monitoring.MAX_LOG_CHARS
        assert out["data"]["truncated"] is True
        assert "truncated" in out["data"]["logs"]

    def test_invalid_session_name_is_rejected_before_sdk_call(self):
        client = MagicMock()
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("Not a valid/name")
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"
        client.get_session_logs.assert_not_called()

    def test_logs_bounded_and_truncation_flagged(self, mock_spark_client):
        mock_spark_client.get_session_logs.return_value = iter([f"line{i}" for i in range(10)])
        out = monitoring.get_spark_session_logs("x", tail_lines=3)
        assert out["success"] is True
        assert out["data"]["lines"] == 3
        assert out["data"]["truncated"] is True
        # Truncated output carries the marker, then the tail of the window.
        assert out["data"]["logs"].splitlines() == [
            _TRUNCATION_MARKER.strip(),
            "line7",
            "line8",
            "line9",
        ]

    def test_reported_line_count_matches_returned_logs(self, mock_spark_client):
        """``lines`` must describe the payload actually returned, including when
        the character budget trims more than the line window did."""
        long_line = "x" * (monitoring.MAX_LOG_CHARS // 2)
        mock_spark_client.get_session_logs.return_value = iter([long_line] * 5)
        out = monitoring.get_spark_session_logs("x", tail_lines=5)
        assert out["data"]["truncated"] is True
        assert len(out["data"]["logs"]) <= monitoring.MAX_LOG_CHARS
        returned = out["data"]["logs"].splitlines()[1:]  # drop the marker
        assert out["data"]["lines"] == len(returned)

    @pytest.mark.parametrize(
        "message",
        [
            # Wording raised by released kubeflow[spark] 0.4.x.
            "No server pod for SparkConnect: default/x",
            # Wording matching the `driver_pod_name` field on SDK `main`.
            "No driver pod for SparkConnect: default/x",
        ],
    )
    def test_missing_server_pod_is_validation_error(self, message):
        client = MagicMock()
        client.get_session_logs.side_effect = RuntimeError(message)
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x")
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_missing_server_pod_detected_through_cause_chain(self):
        client = MagicMock()
        wrapped = RuntimeError("Failed to get logs for SparkConnect: default/x")
        wrapped.__cause__ = RuntimeError("No server pod for SparkConnect: default/x")
        client.get_session_logs.side_effect = wrapped
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x")
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_unrelated_sdk_failure_stays_sdk_error(self):
        client = MagicMock()
        client.get_session_logs.side_effect = RuntimeError("connection refused")
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x")
        assert out["error_code"] == "SDK_ERROR"
