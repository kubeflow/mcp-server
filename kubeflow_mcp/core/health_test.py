"""Tests for health check tools."""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.core.health import get_server_logs, health_check


def test_health_check_k8s_ok():
    mock_v1 = MagicMock()
    with patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=mock_v1):
        result = health_check()

    assert result["data"]["status"] == "healthy"
    assert result["data"]["kubernetes"] is True
    assert "uptime_seconds" in result["data"]
    assert "timestamp" in result["data"]


def test_health_check_k8s_down():
    with patch(
        "kubeflow_mcp.common.utils.get_core_v1_api",
        side_effect=Exception("connection refused"),
    ):
        result = health_check()

    assert result["data"]["status"] == "degraded"
    assert result["data"]["kubernetes"] is False


def test_get_logs_returns_entries():
    mock_logs = [
        {"level": "INFO", "message": "server started", "timestamp": "2026-01-01T00:00:00"},
        {"level": "ERROR", "message": "something failed", "timestamp": "2026-01-01T00:00:01"},
    ]
    with patch("kubeflow_mcp.core.health.get_log_buffer", return_value=mock_logs):
        result = get_server_logs("INFO", 100)

    assert result["data"]["total"] == 2
    assert len(result["data"]["logs"]) == 2


def test_get_logs_filters_by_level():
    mock_logs = [
        {"level": "DEBUG", "message": "debug msg"},
        {"level": "INFO", "message": "info msg"},
        {"level": "ERROR", "message": "error msg"},
    ]
    with patch("kubeflow_mcp.core.health.get_log_buffer", return_value=mock_logs):
        result = get_server_logs("ERROR", 100)

    assert result["data"]["total"] == 1
    assert result["data"]["logs"][0]["level"] == "ERROR"


def test_get_logs_respects_limit():
    mock_logs = [{"level": "INFO", "message": f"msg {i}"} for i in range(20)]
    with patch("kubeflow_mcp.core.health.get_log_buffer", return_value=mock_logs):
        result = get_server_logs("INFO", 5)

    assert len(result["data"]["logs"]) == 5


def test_get_logs_handles_error():
    with patch(
        "kubeflow_mcp.core.health.get_log_buffer",
        side_effect=RuntimeError("buffer error"),
    ):
        result = get_server_logs("INFO", 100)

    assert "error" in result
