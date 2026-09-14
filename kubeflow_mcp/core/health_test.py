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
from fastmcp import Client
from tests.common import SDK_ERROR, VALIDATION_ERROR

from kubeflow_mcp.common.utils import K8S_TIMEOUT
from kubeflow_mcp.core.health import (
    HEALTH_TOOL_ANNOTATIONS,
    HEALTH_TOOL_DESCRIPTIONS,
    HEALTH_TOOLS,
    get_server_logs,
    health_check,
)
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from kubeflow_mcp.core.server import create_server


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

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_reject_limit_less_than_one(self, mock_get_buffer: MagicMock) -> None:
        for invalid_limit in [0, -1, -50]:
            result = get_server_logs(limit=invalid_limit)
            assert result["success"] is False
            assert result["error_code"] == VALIDATION_ERROR
            assert f"limit must be >= 1, got {invalid_limit}" in result["error"]
        mock_get_buffer.assert_not_called()

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_filter_by_default_level(
        self, mock_get_buffer: MagicMock, sample_logs: list[dict[str, Any]]
    ) -> None:
        mock_get_buffer.return_value = sample_logs

        result = get_server_logs()

        assert result["success"] is True
        data = result["data"]

        assert data["total"] == 5
        assert data["buffer_size"] == 6

        levels = [log["level"] for log in data["logs"]]
        assert "DEBUG" not in levels
        assert levels == ["INFO", "WARNING", "ERROR", "CRITICAL", "INFO"]

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

    @pytest.mark.asyncio
    async def test_health_tools_registered_on_server(self) -> None:
        """Integration test: verify health tools are exposed via create_server() with metadata."""
        previous_persona = get_effective_persona()
        try:
            mcp = create_server()
            async with Client(mcp) as client:
                tools = await client.list_tools()
            tool_map = {tool.name: tool for tool in tools}

            assert "health_check" in tool_map
            assert "get_server_logs" in tool_map

            assert tool_map["health_check"].description == HEALTH_TOOL_DESCRIPTIONS["health_check"]
            assert (
                tool_map["get_server_logs"].description
                == HEALTH_TOOL_DESCRIPTIONS["get_server_logs"]
            )

            hc_ann = tool_map["health_check"].annotations
            assert hc_ann is not None
            assert hc_ann.readOnlyHint is True
            assert hc_ann.destructiveHint is False
            assert hc_ann.idempotentHint is True
            assert "health" in (hc_ann.tags or [])

            logs_ann = tool_map["get_server_logs"].annotations
            assert logs_ann is not None
            assert logs_ann.readOnlyHint is True
            assert logs_ann.destructiveHint is False
            assert "debug" in (logs_ann.tags or [])

        finally:
            set_effective_persona(previous_persona)


"""Tests for core health tools."""


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
