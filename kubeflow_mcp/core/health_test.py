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

"""Tests for core health tools."""

from unittest.mock import patch

import pytest

from kubeflow_mcp.core.health import get_server_logs


def test_get_server_logs_rejects_invalid_level():
    with patch("kubeflow_mcp.core.health.get_log_buffer") as mock_get_logs:
        result = get_server_logs(level="INVALID")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    mock_get_logs.assert_not_called()


def test_get_server_logs_accepts_case_insensitive_level():
    with patch(
        "kubeflow_mcp.core.health.get_log_buffer",
        return_value=[{"level": "ERROR", "message": "failed"}],
    ):
        result = get_server_logs(level="error")

    assert result["success"] is True
    assert result["data"]["logs"] == [{"level": "ERROR", "message": "failed"}]


@pytest.mark.parametrize(
    ("alias", "record_level"),
    [("WARN", "WARNING"), ("FATAL", "CRITICAL")],
)
def test_get_server_logs_accepts_standard_level_aliases(alias, record_level):
    with patch(
        "kubeflow_mcp.core.health.get_log_buffer",
        return_value=[{"level": record_level, "message": "failed"}],
    ):
        result = get_server_logs(level=alias)

    assert result["success"] is True
    assert result["data"]["logs"] == [{"level": record_level, "message": "failed"}]
