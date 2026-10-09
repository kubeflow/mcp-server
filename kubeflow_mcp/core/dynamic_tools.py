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

"""Dynamic toolsets for token-efficient tool discovery.

Implements two approaches from https://www.speakeasy.com/blog/100x-token-reduction-dynamic-toolsets:

1. **Progressive** — 3 meta-tools with hierarchical phase-based lookup.
   Agent calls list_tools() → describe_tools() → execute_tool().
   Initial token cost: ~85 tokens (vs ~200 for full mode).

2. **Semantic** — 2 meta-tools with embedding or keyword similarity search.
   Agent calls find_tools("natural language") → execute_tool().
   Initial token cost: ~69 tokens (vs ~200 for full mode).

Both modes register onto the MCP server like normal tools, so any MCP
client (Claude, Cursor, VS Code, MCP Inspector) benefits from reduced
tool schema overhead.
"""

import inspect
import logging
import threading
import warnings
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.common.constants import (
    TOOL_PHASES,
    TOOL_TO_PHASE,
    ErrorCode,
    is_infrastructure_error,
)
from kubeflow_mcp.core.resilience import get_breaker

logger = logging.getLogger(__name__)

# =============================================================================
# Progressive mode: list_tools → describe_tools → execute_tool
# =============================================================================


def _list_tools(
    tool_registry: dict[str, dict[str, Any]],
    tool_hierarchy: dict[str, list[str]],
    prefix: str = "",
) -> dict[str, Any]:
    """List available tools by category or prefix.

    Start with no prefix to see categories, then drill down.

    Args:
        prefix: Filter. Examples:
            - "" → list all categories with tool counts
            - "planning" → list planning tools
            - "training" → list training tools

    Returns:
        Categories and matching tools.
    """
    if not prefix:
        return {
            "categories": list(tool_hierarchy.keys()),
            "category_tools": {cat: len(tools) for cat, tools in tool_hierarchy.items()},
            "hint": "Use list_tools('category_name') to see tools in a category",
        }

    if prefix in tool_hierarchy:
        tools = tool_hierarchy[prefix]
        return {
            "category": prefix,
            "tools": [{"name": t, "description": tool_registry[t]["description"]} for t in tools],
            "hint": "Use describe_tools(['tool_name']) to get full schema",
        }

    matching = [
        {"name": name, "description": info["description"]}
        for name, info in tool_registry.items()
        if name.startswith(prefix) or prefix in name
    ]
    return {
        "prefix": prefix,
        "matching_tools": matching,
        "hint": "Use describe_tools(['tool_name']) to get full schema",
    }


def _describe_tools(
    tool_registry: dict[str, dict[str, Any]], tool_names: list[str]
) -> dict[str, Any]:
    """Get detailed schema for specific tools.

    Call after list_tools() to get parameter information before executing.

    Args:
        tool_names: List of tool names to describe (max 5 at a time).

    Returns:
        Tool schemas with parameter types and defaults.
    """
    if len(tool_names) > 5:
        return {"error": "Max 5 tools at a time to conserve tokens"}

    results: list[dict[str, Any]] = []
    for name in tool_names:
        if name not in tool_registry:
            results.append({"name": name, "error": "Tool not found"})
            continue

        tool = tool_registry[name]
        sig = inspect.signature(tool["func"])
        params: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            param_info: dict[str, Any] = {"type": "any"}
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            params[param_name] = param_info

        results.append(
            {
                "name": name,
                "category": tool["category"],
                "description": tool["full_doc"],
                "parameters": params,
            }
        )

    return {"tools": results}


def _execute_tool(
    tool_registry: dict[str, dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a discovered tool by name.

    Call after list_tools() and describe_tools() to run the actual tool.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as key-value pairs.

    Returns:
        Tool execution result.
    """
    if tool_name not in tool_registry:
        return {"error": f"Tool '{tool_name}' not found", "available": list(tool_registry.keys())}

    func = tool_registry[tool_name]["func"]
    args = arguments or {}

    # Reject bad arguments before touching the breaker: they are a caller mistake,
    # and returning after can_execute() would also leak a half-open probe slot.
    try:
        inspect.signature(func).bind(**args)
    except TypeError as e:
        return {
            "error": f"Invalid arguments for '{tool_name}': {e}",
            "error_code": ErrorCode.VALIDATION_ERROR,
            "tool": tool_name,
        }

    breaker = get_breaker(tool_name)
    if not breaker.can_execute():
        return {
            "error": f"Circuit breaker open for '{tool_name}' — K8s API may be degraded. Retries automatically after recovery timeout.",
            "error_code": ErrorCode.CIRCUIT_OPEN,
        }

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning, module="urllib3")
            result = func(**args)
        if isinstance(result, dict) and is_infrastructure_error(result):
            breaker.record_failure()
        else:
            breaker.record_success()
        if isinstance(result, dict):
            return result
        return {"result": result}
    except Exception as e:
        breaker.record_failure()
        return {"error": str(e), "error_code": ErrorCode.SDK_ERROR, "tool": tool_name}


# =============================================================================
# Semantic mode: find_tools → execute_tool
# =============================================================================


class _EmbeddingCache:
    """Lazy-loaded embedding cache for semantic search."""

    _shared_model = None
    _shared_model_lock = threading.Lock()

    def __init__(self, tool_registry: dict[str, dict[str, Any]]):
        self._tool_registry = tool_registry
        self._embeddings: dict[str, list[float]] | None = None
        self._model = None
        self._unavailable = False
        self._lock = threading.Lock()

    def get(self) -> tuple[dict[str, list[float]] | None, Any]:
        cached = self._cached()
        if cached is not None:
            return cached

        # One loader at a time. Without this, concurrent find_tools() calls all read
        # _unavailable as False and each start their own model download.
        with self._lock:
            cached = self._cached()
            if cached is not None:
                return cached
            return self._load()

    def _cached(self) -> tuple[dict[str, list[float]] | None, Any] | None:
        """Return the hit or the recorded failure, or None when a load is needed."""
        if self._embeddings is not None:
            return self._embeddings, self._model
        if self._unavailable:
            return None, None
        return None

    def _load(self) -> tuple[dict[str, list[float]] | None, Any]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.debug("sentence-transformers not installed, falling back to keyword search")
            return self._mark_unavailable()
        except Exception as e:
            # A broken native dependency raises OSError or RuntimeError from the import
            # itself, which would otherwise escape find_tools().
            logger.warning("Semantic search is unavailable, falling back to keyword search: %s", e)
            return self._mark_unavailable()

        try:
            with _EmbeddingCache._shared_model_lock:
                if _EmbeddingCache._shared_model is None:
                    _EmbeddingCache._shared_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._model = _EmbeddingCache._shared_model
            descriptions = [
                f"{info['description']}. Category: {info['category']}. {info['full_doc'][:200]}"
                for info in self._tool_registry.values()
            ]
            embeddings = self._model.encode(descriptions)
            self._embeddings = {
                name: emb.tolist()
                for name, emb in zip(self._tool_registry.keys(), embeddings, strict=True)
            }
            return self._embeddings, self._model
        except Exception as e:
            # Covers the model download and the encode pass: either way there are no
            # embeddings to search.
            logger.warning("Semantic search is unavailable, falling back to keyword search: %s", e)
            return self._mark_unavailable()

    def _mark_unavailable(self) -> tuple[None, None]:
        """Remember the failure so every call doesn't repeat work that just failed.

        Cleared by reset(), so this registry can retry once the underlying problem
        is fixed.
        """
        self._unavailable = True
        self._model = None
        return None, None

    def reset(self) -> None:
        with self._lock:
            self._embeddings = None
            self._model = None
            self._unavailable = False


MAX_QUERY_LENGTH = 500
MAX_TOP_K = 20


def _find_tools(
    tool_registry: dict[str, dict[str, Any]],
    embedding_cache: _EmbeddingCache,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Find relevant tools using semantic or keyword search.

    Describe what you want to accomplish in natural language.

    Args:
        query: Natural language description. Examples:
            - "all" → list every available tool
            - "check GPU availability in the cluster"
            - "fine-tune a language model"
            - "view logs from a training job"
            - "delete a failed job"
        top_k: Number of results (default 5, ignored when query="all").

    Returns:
        Matching tools ranked by relevance.
    """
    if len(query) > MAX_QUERY_LENGTH:
        return {"error": f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})"}
    top_k = max(1, min(top_k, MAX_TOP_K))
    query_lower = query.strip().lower()
    _list_all = {
        "*",
        "all",
        "list",
        "list all",
        "all tools",
        "available tools",
        "what tools",
        "show tools",
        "show all",
        "every tool",
        "everything",
        "available",
        "what's available",
        "whats available",
    }
    if query_lower in _list_all or "all tool" in query_lower or "available tool" in query_lower:
        return {
            "query": query,
            "total": len(tool_registry),
            "tools": [
                {"name": name, "description": info["description"], "category": info["category"]}
                for name, info in tool_registry.items()
            ],
            "hint": "Use execute_tool(tool_name, {args}) to run a tool",
        }

    embeddings, model = embedding_cache.get()

    if embeddings is None:
        return _keyword_search(tool_registry, query, top_k)

    try:
        import numpy as np

        query_embedding = model.encode([query])[0]
        scores = {}
        for name, tool_emb in embeddings.items():
            q_norm = np.linalg.norm(query_embedding)
            t_norm = np.linalg.norm(tool_emb)
            if q_norm == 0 or t_norm == 0:
                scores[name] = 0.0
            else:
                scores[name] = float(np.dot(query_embedding, tool_emb) / (q_norm * t_norm))
        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    except Exception:
        logger.debug("Embedding search failed, falling back to keyword search")
        return _keyword_search(tool_registry, query, top_k)

    return {
        "query": query,
        "tools": [
            {
                "name": name,
                "description": tool_registry[name]["description"],
                "category": tool_registry[name]["category"],
                "relevance": f"{score:.2f}",
            }
            for name, score in sorted_tools
        ],
        "hint": "Use execute_tool(tool_name, {args}) to run a tool",
    }


def _keyword_search(
    tool_registry: dict[str, dict[str, Any]], query: str, top_k: int = 5
) -> dict[str, Any]:
    """Fallback keyword search when embeddings unavailable."""
    query_lower = query.lower()
    keywords = query_lower.split()

    scores = {}
    for name, info in tool_registry.items():
        text = f"{name} {info['description']} {info['category']}".lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[name] = score

    sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {
        "query": query,
        "mode": "keyword_fallback",
        "tools": [
            {
                "name": name,
                "description": tool_registry[name]["description"],
                "category": tool_registry[name]["category"],
            }
            for name, _ in sorted_tools
        ],
        "hint": "Use execute_tool(tool_name, {args}) to run a tool",
    }


class DynamicToolRegistry:
    """Own the tools and discovery state exposed by one dynamic toolset."""

    def __init__(self, tool_funcs: list[Callable], descriptions: dict[str, str]):
        self.tool_registry: dict[str, dict[str, Any]] = {}
        self.tool_hierarchy: dict[str, list[str]] = {}
        self._embedding_cache = _EmbeddingCache(self.tool_registry)
        self.initialize(tool_funcs, descriptions)

    def initialize(self, tool_funcs: list[Callable], descriptions: dict[str, str]) -> None:
        """Replace this registry's contents and reset its semantic cache."""
        self._embedding_cache.reset()
        self.tool_registry.clear()
        self.tool_hierarchy.clear()
        for phase in TOOL_PHASES:
            self.tool_hierarchy[phase] = []

        for func in tool_funcs:
            name = func.__name__
            doc = func.__doc__ or ""
            category = TOOL_TO_PHASE.get(name, "other")
            short_desc = descriptions.get(name, doc.split("\n")[0] if doc else name)

            self.tool_registry[name] = {
                "name": name,
                "category": category,
                "description": short_desc,
                "full_doc": doc,
                "func": func,
            }
            self.tool_hierarchy.setdefault(category, []).append(name)

        logger.info(
            "Dynamic tool registry initialized: %s tools, %s categories",
            len(self.tool_registry),
            len(self.tool_hierarchy),
        )

    def list_tools(self, prefix: str = "") -> dict[str, Any]:
        """List this registry's available tools by category or prefix.

        Start with no prefix to see categories, then drill down.

        Args:
            prefix: Filter. Examples:
                - "" → list all categories with tool counts
                - "planning" → list planning tools
                - "training" → list training tools

        Returns:
            Categories and matching tools in this registry.
        """
        return _list_tools(self.tool_registry, self.tool_hierarchy, prefix)

    def describe_tools(self, tool_names: list[str]) -> dict[str, Any]:
        """Get detailed schema for specific tools in this registry.

        Call after list_tools() to get parameter information before executing.

        Args:
            tool_names: List of tool names to describe (max 5 at a time).

        Returns:
            Tool schemas with parameter types and defaults.
        """
        return _describe_tools(self.tool_registry, tool_names)

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a discovered tool from this registry by name.

        Call after list_tools() and describe_tools() to run the actual tool.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as key-value pairs.

        Returns:
            Tool execution result.
        """
        return _execute_tool(self.tool_registry, tool_name, arguments)

    def find_tools(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Find relevant tools in this registry using semantic or keyword search.

        Describe what you want to accomplish in natural language.

        Args:
            query: Natural language description. Examples:
                - "all" → list every available tool in this registry
                - "check GPU availability in the cluster"
                - "fine-tune a language model"
                - "view logs from a training job"
                - "delete a failed job"
            top_k: Number of results (default 5, ignored when query="all").

        Returns:
            Matching tools ranked by relevance within this registry.
        """
        return _find_tools(self.tool_registry, self._embedding_cache, query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search this registry using the semantic fallback."""
        return _keyword_search(self.tool_registry, query, top_k)


# Compatibility entry points retain a default registry for existing module-level APIs.
# New server instances use DynamicToolRegistry directly.
_default_registry = DynamicToolRegistry([], {})
TOOL_REGISTRY = _default_registry.tool_registry
TOOL_HIERARCHY = _default_registry.tool_hierarchy
_embedding_cache = _default_registry._embedding_cache


def init_dynamic_tools(tool_funcs: list[Callable], descriptions: dict[str, str]) -> None:
    """Initialize the default registry for existing module-level callers.

    New callers that need isolation should construct a DynamicToolRegistry.
    """
    _default_registry.initialize(tool_funcs, descriptions)


def list_tools(prefix: str = "") -> dict[str, Any]:
    """Compatibility wrapper for the default registry's list operation."""
    return _default_registry.list_tools(prefix)


def describe_tools(tool_names: list[str]) -> dict[str, Any]:
    """Compatibility wrapper for the default registry's describe operation."""
    return _default_registry.describe_tools(tool_names)


def execute_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility wrapper for the default registry's execute operation."""
    return _default_registry.execute_tool(tool_name, arguments)


def find_tools(query: str, top_k: int = 5) -> dict[str, Any]:
    """Compatibility wrapper for the default registry's semantic search."""
    return _default_registry.find_tools(query, top_k)


PROGRESSIVE_TOOLS: list[Callable] = [list_tools, describe_tools, execute_tool]
SEMANTIC_TOOLS: list[Callable] = [find_tools, execute_tool]


# =============================================================================
# Factory
# =============================================================================

TOOL_MODES = {
    "full": "All tools registered directly on MCP server",
    "progressive": "3 meta-tools: list_tools → describe_tools → execute_tool",
    "semantic": "2 meta-tools: find_tools → execute_tool",
}


def get_mode_tools(mode: str) -> list[Callable]:
    """Get meta-tool functions for the given mode."""
    if mode == "semantic":
        return SEMANTIC_TOOLS
    if mode == "progressive":
        return PROGRESSIVE_TOOLS
    raise ValueError(f"Unknown dynamic mode: {mode}. Use 'progressive' or 'semantic'.")
