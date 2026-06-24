"""Tests for common types."""

from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse


def test_tool_response():
    resp = ToolResponse(data={"key": "value"})
    assert resp.success is True
    assert resp.data == {"key": "value"}

    dumped = resp.model_dump()
    assert dumped["success"] is True


def test_tool_error():
    err = ToolError(error="Something failed", error_code="TEST_ERROR")
    assert err.success is False
    assert err.error == "Something failed"
    assert err.error_code == "TEST_ERROR"


def test_preview_response():
    preview = PreviewResponse(config={"setting": 1})
    assert preview.success is True
    assert preview.status == "preview"
    assert preview.message == "Set confirmed=True to execute"
    assert preview.config == {"setting": 1}
