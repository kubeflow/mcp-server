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
from kubeflow_mcp.core.server import create_server
from kubeflow_mcp.trainer import CLIENT_TOOL_DESCRIPTIONS as TOOL_DESCRIPTIONS


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

    def test_semantic_mode_registers_meta_tools(self):
        server = create_server(mode="semantic", persona="platform-admin")
        names = _registered_tool_names(server)
        assert "find_tools" in names
        assert "execute_tool" in names
        assert "fine_tune" not in names

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

    def test_execute_tool_unknown_name(self):
        result = execute_tool("this_tool_does_not_exist")
        assert "error" in result
        assert "this_tool_does_not_exist" in result["error"]
        assert "available" in result

    def test_execute_tool_func_raises_records_failure_and_returns_error(self):
        from unittest.mock import MagicMock

        from kubeflow_mcp.core.resilience import reset_breakers

        reset_breakers()
        mock_fn = MagicMock(side_effect=RuntimeError("boom"))
        TOOL_REGISTRY["_raise_tool"] = {
            "name": "_raise_tool",
            "func": mock_fn,
            "description": "test",
            "full_doc": "test",
            "category": "testing",
        }
        result = execute_tool("_raise_tool", {})
        assert "error" in result
        assert "boom" in result["error"]
        assert result.get("error_code") == "SDK_ERROR"
        del TOOL_REGISTRY["_raise_tool"]
        reset_breakers()

    def test_execute_tool_circuit_open_returns_error(self):
        from unittest.mock import MagicMock

        from kubeflow_mcp.core.resilience import get_breaker, reset_breakers

        reset_breakers()
        TOOL_REGISTRY["_cb_tool"] = {
            "name": "_cb_tool",
            "func": MagicMock(return_value={"data": "ok"}),
            "description": "test",
            "full_doc": "test",
            "category": "testing",
        }
        breaker = get_breaker("_cb_tool")
        # Force circuit open by tripping it beyond threshold
        for _ in range(breaker.failure_threshold + 1):
            breaker.record_failure()

        result = execute_tool("_cb_tool", {})
        assert "error" in result
        assert result.get("error_code") == "CIRCUIT_OPEN"
        del TOOL_REGISTRY["_cb_tool"]
        reset_breakers()


class TestFindTools:
    def setup_method(self):
        _init_registry()

    def test_all_alias_returns_full_registry(self):
        for alias in ("all", "*", "list all", "available tools"):
            result = find_tools(alias)
            assert result["total"] == len(TOOL_REGISTRY), f"Failed for alias: {alias!r}"

    def test_keyword_fallback_matches_relevant_tools(self):
        """Without embeddings loaded, _keyword_search is the fallback."""
        result = find_tools("fine-tune language model training")
        assert "tools" in result
        names = [t["name"] for t in result["tools"]]
        # At least one training-related tool should surface
        assert any("train" in n or "fine_tune" in n or "run" in n for n in names)

    def test_keyword_fallback_mode_flag_present(self):
        """_keyword_search always tags the response with mode=keyword_fallback."""
        from kubeflow_mcp.core.dynamic_tools import _keyword_search

        result = _keyword_search("logs monitoring")
        assert result.get("mode") == "keyword_fallback"

    def test_keyword_search_ranks_exact_name_match_highly(self):
        from kubeflow_mcp.core.dynamic_tools import _keyword_search

        result = _keyword_search("fine_tune")
        names = [t["name"] for t in result["tools"]]
        assert names[0] == "fine_tune"

    def test_query_too_long_returns_error(self):
        from kubeflow_mcp.core.dynamic_tools import MAX_QUERY_LENGTH

        result = find_tools("x" * (MAX_QUERY_LENGTH + 1))
        assert "error" in result
        assert "too long" in result["error"].lower()

    def test_top_k_capped_at_maximum(self):
        from kubeflow_mcp.core.dynamic_tools import MAX_TOP_K, _keyword_search

        result = _keyword_search("training", top_k=99999)
        assert len(result["tools"]) <= MAX_TOP_K

    def test_no_match_returns_empty_tools_list(self):
        from kubeflow_mcp.core.dynamic_tools import _keyword_search

        result = _keyword_search("xyzzy_nonexistent_gibberish")
        assert result["tools"] == []

    def test_hint_present_in_result(self):
        result = find_tools("cluster resources")
        assert "hint" in result
