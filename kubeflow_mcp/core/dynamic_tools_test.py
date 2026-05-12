"""Tests for dynamic tool modes (progressive and semantic)."""

import asyncio

from kubeflow_mcp.core.dynamic_tools import (
    TOOL_MODES,
    TOOL_REGISTRY,
    describe_tools,
    execute_tool,
    find_tools,
    get_mode_tools,
    init_dynamic_tools,
    list_tools,
)
from kubeflow_mcp.core.server import TOOL_DESCRIPTIONS, create_server


def _registered_tool_names(server) -> set[str]:
    loop = asyncio.new_event_loop()
    try:
        tools = loop.run_until_complete(server._list_tools())
        return {t.name for t in tools}
    finally:
        loop.close()


def _init_registry():
    """Helper: initialize dynamic registry with real trainer tools."""
    from kubeflow_mcp.trainer import TOOLS

    init_dynamic_tools(TOOLS, TOOL_DESCRIPTIONS)


class TestInitDynamicTools:
    def test_populates_registry(self):
        _init_registry()
        assert len(TOOL_REGISTRY) > 0
        assert "fine_tune" in TOOL_REGISTRY
        assert "get_cluster_resources" in TOOL_REGISTRY

    def test_categories_populated(self):
        from kubeflow_mcp.core.dynamic_tools import TOOL_HIERARCHY

        _init_registry()
        assert "planning" in TOOL_HIERARCHY
        assert "training" in TOOL_HIERARCHY
        assert len(TOOL_HIERARCHY["planning"]) > 0

    def test_registry_entry_structure(self):
        _init_registry()
        entry = TOOL_REGISTRY["fine_tune"]
        assert entry["name"] == "fine_tune"
        assert entry["category"] == "training"
        assert "description" in entry
        assert "full_doc" in entry
        assert callable(entry["func"])


class TestProgressiveMode:
    def setup_method(self):
        _init_registry()

    def test_list_tools_no_prefix(self):
        result = list_tools()
        assert "categories" in result
        assert "planning" in result["categories"]
        assert "training" in result["categories"]
        assert "category_tools" in result

    def test_list_tools_with_category(self):
        result = list_tools("planning")
        assert result["category"] == "planning"
        assert len(result["tools"]) > 0
        assert any(t["name"] == "get_cluster_resources" for t in result["tools"])

    def test_list_tools_with_prefix_match(self):
        result = list_tools("fine")
        assert len(result["matching_tools"]) > 0
        assert any(t["name"] == "fine_tune" for t in result["matching_tools"])

    def test_describe_tools_returns_schema(self):
        result = describe_tools(["fine_tune"])
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["name"] == "fine_tune"
        assert "parameters" in tool
        assert "model" in tool["parameters"]

    def test_describe_tools_max_limit(self):
        result = describe_tools(["a", "b", "c", "d", "e", "f"])
        assert "error" in result

    def test_describe_tools_unknown(self):
        result = describe_tools(["nonexistent_tool"])
        assert result["tools"][0]["error"] == "Tool not found"

    def test_execute_tool_unknown(self):
        result = execute_tool("nonexistent_tool")
        assert "error" in result
        assert "available" in result

    def test_progressive_meta_tools_count(self):
        tools = get_mode_tools("progressive")
        assert len(tools) == 3
        names = {f.__name__ for f in tools}
        assert names == {"list_tools", "describe_tools", "execute_tool"}


class TestSemanticMode:
    def setup_method(self):
        _init_registry()

    def test_find_tools_all(self):
        result = find_tools("all")
        assert result["total"] == len(TOOL_REGISTRY)
        assert len(result["tools"]) == len(TOOL_REGISTRY)

    def test_find_tools_keyword_fallback(self):
        result = find_tools("training logs")
        assert len(result["tools"]) > 0
        names = {t["name"] for t in result["tools"]}
        assert "get_training_logs" in names

    def test_find_tools_gpu(self):
        result = find_tools("check GPU availability")
        assert len(result["tools"]) > 0

    def test_find_tools_list_aliases(self):
        for alias in ["all tools", "show all", "available", "everything"]:
            result = find_tools(alias)
            assert result["total"] == len(TOOL_REGISTRY), f"Failed for alias: {alias}"

    def test_semantic_meta_tools_count(self):
        tools = get_mode_tools("semantic")
        assert len(tools) == 2
        names = {f.__name__ for f in tools}
        assert names == {"find_tools", "execute_tool"}


class TestServerModeIntegration:
    def setup_method(self):
        _init_registry()

    def test_full_mode_registers_all_tools(self):
        server = create_server(mode="full", persona="platform-admin")
        names = _registered_tool_names(server)
        assert "fine_tune" in names
        assert "get_cluster_resources" in names
        assert len(names) >= 18

    def test_progressive_mode_registers_meta_tools(self):
        server = create_server(mode="progressive", persona="platform-admin")
        names = _registered_tool_names(server)
        assert "list_tools" in names
        assert "describe_tools" in names
        assert "execute_tool" in names
        assert "fine_tune" not in names
        assert "health_check" in names

    def test_semantic_mode_registers_meta_tools(self):
        server = create_server(mode="semantic", persona="platform-admin")
        names = _registered_tool_names(server)
        assert "find_tools" in names
        assert "execute_tool" in names
        assert "fine_tune" not in names
        assert "health_check" in names

    def test_progressive_mode_fewer_tools_than_full(self):
        full = _registered_tool_names(create_server(mode="full", persona="platform-admin"))
        prog = _registered_tool_names(create_server(mode="progressive", persona="platform-admin"))
        assert len(prog) < len(full)

    def test_progressive_respects_persona(self):
        """Readonly persona limits tools accessible via execute_tool."""
        server = create_server(mode="progressive", persona="readonly")
        names = _registered_tool_names(server)
        assert "execute_tool" in names
        result = execute_tool("fine_tune", {"model": "test", "dataset": "test"})
        assert "error" in result

    def test_mode_options_documented(self):
        assert "full" in TOOL_MODES
        assert "progressive" in TOOL_MODES
        assert "semantic" in TOOL_MODES

    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid mode"):
            create_server(mode="turbo", persona="platform-admin")


class TestEdgeCases:
    def setup_method(self):
        _init_registry()

    def test_execute_tool_success(self):
        """execute_tool calls the underlying function and returns its result."""
        from unittest.mock import MagicMock

        TOOL_REGISTRY["_test_tool"] = {
            "name": "_test_tool",
            "func": MagicMock(return_value={"data": "ok"}),
            "description": "test",
            "full_doc": "test",
            "category": "testing",
        }
        result = execute_tool("_test_tool", {"arg1": "val1"})
        assert result == {"data": "ok"}
        TOOL_REGISTRY["_test_tool"]["func"].assert_called_once_with(arg1="val1")
        del TOOL_REGISTRY["_test_tool"]

    def test_describe_tools_empty_list(self):
        result = describe_tools([])
        assert result["tools"] == []

    def test_describe_tools_mixed_valid_and_invalid(self):
        result = describe_tools(["fine_tune", "nonexistent"])
        names = [t["name"] for t in result["tools"]]
        assert "fine_tune" in names
        assert "nonexistent" in names
        ft = next(t for t in result["tools"] if t["name"] == "fine_tune")
        ne = next(t for t in result["tools"] if t["name"] == "nonexistent")
        assert "parameters" in ft
        assert ne["error"] == "Tool not found"

    def test_init_empty_tools(self):
        init_dynamic_tools([], {})
        assert len(TOOL_REGISTRY) == 0
        result = list_tools()
        assert all(count == 0 for count in result["category_tools"].values())
