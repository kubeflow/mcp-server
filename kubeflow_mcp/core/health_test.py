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

"""Unit tests for core/health.py - health check and server logs tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tests.common import SDK_ERROR, VALIDATION_ERROR

from kubeflow_mcp.common.utils import K8S_TIMEOUT
from kubeflow_mcp.core.health import (
    HEALTH_TOOL_ANNOTATIONS,
    HEALTH_TOOL_DESCRIPTIONS,
    HEALTH_TOOLS,
    get_server_logs,
    health_check,
)


class TestHealthCheck:
    """Tests for the health_check tool."""

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_health_check_healthy_when_k8s_succeeds(self, mock_get_core_v1_api: MagicMock) -> None:
        mock_v1 = MagicMock()
        mock_get_core_v1_api.return_value = mock_v1

        result = health_check()

        assert result["success"] is True
        data = result["data"]
        assert data["status"] == "healthy"
        assert data["kubernetes"] is True
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0
        assert "timestamp" in data
        mock_v1.list_namespace.assert_called_once_with(limit=1, _request_timeout=K8S_TIMEOUT)

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_health_check_degraded_when_k8s_fails(self, mock_get_core_v1_api: MagicMock) -> None:
        mock_v1 = MagicMock()
        mock_v1.list_namespace.side_effect = ConnectionError("K8s API unreachable")

        mock_get_core_v1_api.return_value = mock_v1
        result = health_check()

        # The tool should not crash even if k8s fails, it should return degraded status
        assert result["success"] is True
        data = result["data"]
        assert data["status"] == "degraded"
        assert data["kubernetes"] is False


class TestGetServerLogs:
    """Tests for the get_server_logs tool."""

    @pytest.fixture
    def sample_logs(self) -> list[dict[str, Any]]:
        return [
            {"timestamp": "2026-09-05T00:00:001Z", "level": "DEBUG", "message": "debug message"},
            {"timestamp": "2026-09-05T00:00:001Z", "level": "INFO", "message": "info message 1"},
            {
                "timestamp": "2026-09-05T00:00:001Z",
                "level": "WARNING",
                "message": "warning message",
            },
            {"timestamp": "2026-09-05T00:00:001Z", "level": "ERROR", "message": "error message"},
            {
                "timestamp": "2026-09-05T00:00:001Z",
                "level": "CRITICAL",
                "message": "critical message",
            },
            {"timestamp": "2026-09-05T00:00:001Z", "level": "INFO", "message": "info message 2"},
        ]

    def test_reject_limit_less_than_one(self) -> None:
        for invalid_limit in [0, -1, -50]:
            result = get_server_logs(limit=invalid_limit)
            assert result["success"] is False
            assert result["error_code"] == VALIDATION_ERROR
            assert f"limit must be >= 1, got {invalid_limit}" in result["error"]

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_filter_by_level_warning(
        self, mock_get_buffer: MagicMock, sample_logs: list[dict[str, Any]]
    ) -> None:
        mock_get_buffer.return_value = sample_logs

        result = get_server_logs(level="WARNING")

        assert result["success"] is True
        data = result["data"]
        assert data["total"] == 3
        levels = [log["level"] for log in data["logs"]]
        assert levels == ["WARNING", "ERROR", "CRITICAL"]
        assert data["logs"][0]["message"] == "warning message"
        assert data["logs"][1]["message"] == "error message"
        assert data["logs"][2]["message"] == "critical message"

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_handles_unexpected_exception(self, mock_get_buffer: MagicMock) -> None:
        mock_get_buffer.side_effect = RuntimeError("Buffer Locked")

        result = get_server_logs()

        assert result["success"] is False
        assert result["error_code"] == SDK_ERROR


class TestHealthMetadata:
    """Tests for metadata, registrations, and annotations."""

    def test_health_tools_exported(self) -> None:
        assert health_check in HEALTH_TOOLS
        assert get_server_logs in HEALTH_TOOLS
        assert len(HEALTH_TOOLS) == 2

    def test_descriptions_defined_for_all_tools(self) -> None:
        for tool in HEALTH_TOOLS:
            name = tool.__name__
            assert name in HEALTH_TOOL_DESCRIPTIONS
            assert len(HEALTH_TOOL_DESCRIPTIONS[name]) > 10

    def test_annotations_defined_for_all_tools(self) -> None:
        for tool in HEALTH_TOOLS:
            name = tool.__name__
            assert name in HEALTH_TOOL_ANNOTATIONS
            ann = HEALTH_TOOL_ANNOTATIONS[name]
            assert ann.get("readOnlyHint") is True
            assert ann.get("destructiveHint") is False
            assert ann.get("idempotentHint") is True
            assert ann.get("openWorldHint") is False
            assert "health" in ann.get("tags", [])
