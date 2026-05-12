"""Tests for MCP server creation and tool registration."""

import asyncio
from unittest.mock import patch

from kubeflow_mcp.core.prompts import register_prompts
from kubeflow_mcp.core.server import CLIENT_MODULES, TOOL_ANNOTATIONS, create_server

WRITE_TOOLS = {name for name, ann in TOOL_ANNOTATIONS.items() if not ann.get("readOnlyHint", True)}
READ_TOOLS = {name for name, ann in TOOL_ANNOTATIONS.items() if ann.get("readOnlyHint", True)}


def _registered_tool_names(server) -> set[str]:
    """Extract registered tool names from a FastMCP server (async API)."""
    loop = asyncio.new_event_loop()
    try:
        tools = loop.run_until_complete(server._list_tools())
        return {t.name for t in tools}
    finally:
        loop.close()


def test_create_server_default():
    """Server creates with default trainer client."""
    server = create_server()
    assert server is not None
    assert server.name == "kubeflow-mcp"


def test_create_server_default_registers_all_trainer_tools():
    """Default persona (platform-admin) registers every trainer tool."""
    server = create_server(persona="platform-admin")
    names = _registered_tool_names(server)
    for tool_name in TOOL_ANNOTATIONS:
        assert tool_name in names, f"{tool_name} missing for platform-admin"


def test_create_server_readonly_persona():
    """Readonly persona only registers read-only tools."""
    server = create_server(persona="readonly")
    names = _registered_tool_names(server)

    for wt in WRITE_TOOLS:
        assert wt not in names, f"write tool {wt} should not be in readonly persona"

    assert "list_training_jobs" in names
    assert "get_training_job" in names


def test_create_server_read_only_policy():
    """read_only policy flag strips all write tools regardless of persona."""
    with patch(
        "kubeflow_mcp.core.server.is_read_only",
        return_value=True,
    ):
        server = create_server(persona="platform-admin")

    names = _registered_tool_names(server)
    for wt in WRITE_TOOLS:
        assert wt not in names, f"write tool {wt} registered despite read_only policy"

    for rt in READ_TOOLS:
        assert rt in names, f"read tool {rt} missing under read_only policy"


def test_create_server_unknown_client():
    """Server skips unknown clients gracefully."""
    server = create_server(clients=["trainer", "unknown"])
    assert server is not None


def test_client_modules_registered():
    """All expected client modules are registered."""
    assert "trainer" in CLIENT_MODULES
    assert "optimizer" in CLIENT_MODULES
    assert "hub" in CLIENT_MODULES


def test_trainer_module_has_tools():
    """Trainer module exports tools."""
    from kubeflow_mcp import trainer

    assert hasattr(trainer, "TOOLS")
    assert hasattr(trainer, "MODULE_INFO")
    assert len(trainer.TOOLS) > 0
    assert trainer.MODULE_INFO["status"] == "implemented"


def test_optimizer_module_is_stub():
    """Optimizer module is a stub."""
    from kubeflow_mcp import optimizer

    assert hasattr(optimizer, "TOOLS")
    assert len(optimizer.TOOLS) == 0
    assert optimizer.MODULE_INFO["status"] == "stub"


def test_hub_module_is_stub():
    """Hub module is a stub."""
    from kubeflow_mcp import hub

    assert hasattr(hub, "TOOLS")
    assert len(hub.TOOLS) == 0
    assert hub.MODULE_INFO["status"] == "stub"


def test_prompts_registered():
    """Server registers MCP prompts for workflows."""
    from fastmcp import FastMCP

    from kubeflow_mcp.common.constants import PROMPT_METADATA

    mcp = FastMCP("test")
    register_prompts(mcp)
    loop = asyncio.new_event_loop()
    try:
        prompts = loop.run_until_complete(mcp._list_prompts())
        prompt_names = {p.name for p in prompts}
        assert len(prompt_names) > 0, "No prompts registered"
        for name in PROMPT_METADATA:
            assert name in prompt_names, f"Prompt '{name}' not registered"
    finally:
        loop.close()


def test_audit_wrapper_emits_log(caplog):
    """Audit wrapper logs tool invocations with structured fields."""
    import logging

    from kubeflow_mcp.core.server import _audit_wrap

    def _fake_tool(**kwargs):
        return {"data": "ok"}

    wrapped = _audit_wrap(_fake_tool)
    with caplog.at_level(logging.INFO, logger="kubeflow_mcp.core.server"):
        result = wrapped(name="test-job")

    assert result == {"data": "ok"}
    assert any("tool_call" in r.message for r in caplog.records)
    audit_record = next(r for r in caplog.records if "tool_call" in r.message)
    assert getattr(audit_record, "tool", None) == "_fake_tool"
    assert getattr(audit_record, "success", None) is True
