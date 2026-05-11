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

# Populated by init_dynamic_tools() after server knows which tools are loaded.
TOOL_REGISTRY: dict[str, dict[str, Any]] = {}
TOOL_HIERARCHY: dict[str, list[str]] = {}


def init_dynamic_tools(
    tool_funcs: list[Callable],
    descriptions: dict[str, str],
) -> None:
    """Initialize the dynamic tool registry from loaded tool functions.

    Must be called before any meta-tool is invoked. Typically called by
    create_server() after collecting tools from client modules.
    """
    TOOL_REGISTRY.clear()
    TOOL_HIERARCHY.clear()

    for phase in TOOL_PHASES:
        TOOL_HIERARCHY[phase] = []

    for func in tool_funcs:
        name = func.__name__
        doc = func.__doc__ or ""
        category = TOOL_TO_PHASE.get(name, "other")
        short_desc = descriptions.get(name, doc.split("\n")[0] if doc else name)

        TOOL_REGISTRY[name] = {
            "name": name,
            "category": category,
            "description": short_desc,
            "full_doc": doc,
            "func": func,
        }
        TOOL_HIERARCHY.setdefault(category, []).append(name)

    logger.info(
        f"Dynamic tool registry initialized: {len(TOOL_REGISTRY)} tools, "
        f"{len(TOOL_HIERARCHY)} categories"
    )


# =============================================================================
# Progressive mode: list_tools → describe_tools → execute_tool
# =============================================================================


def list_tools(prefix: str = "") -> dict[str, Any]:
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
            "categories": list(TOOL_HIERARCHY.keys()),
            "category_tools": {cat: len(tools) for cat, tools in TOOL_HIERARCHY.items()},
            "hint": "Use list_tools('category_name') to see tools in a category",
        }

    if prefix in TOOL_HIERARCHY:
        tools = TOOL_HIERARCHY[prefix]
        return {
            "category": prefix,
            "tools": [{"name": t, "description": TOOL_REGISTRY[t]["description"]} for t in tools],
            "hint": "Use describe_tools(['tool_name']) to get full schema",
        }

    matching = [
        {"name": name, "description": info["description"]}
        for name, info in TOOL_REGISTRY.items()
        if name.startswith(prefix) or prefix in name
    ]
    return {
        "prefix": prefix,
        "matching_tools": matching,
        "hint": "Use describe_tools(['tool_name']) to get full schema",
    }


def describe_tools(tool_names: list[str]) -> dict[str, Any]:
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
        if name not in TOOL_REGISTRY:
            results.append({"name": name, "error": "Tool not found"})
            continue

        tool = TOOL_REGISTRY[name]
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


def execute_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a discovered tool by name.

    Call after list_tools() and describe_tools() to run the actual tool.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as key-value pairs.

    Returns:
        Tool execution result.
    """
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool '{tool_name}' not found", "available": list(TOOL_REGISTRY.keys())}

    breaker = get_breaker(tool_name)
    if not breaker.can_execute():
        return {
            "error": f"Circuit breaker open for '{tool_name}' — K8s API may be degraded. Retries automatically after recovery timeout.",
            "error_code": ErrorCode.CIRCUIT_OPEN,
        }

    func = TOOL_REGISTRY[tool_name]["func"]
    args = arguments or {}

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


PROGRESSIVE_TOOLS: list[Callable] = [list_tools, describe_tools, execute_tool]


# =============================================================================
# Semantic mode: find_tools → execute_tool
# =============================================================================


class _EmbeddingCache:
    """Lazy-loaded embedding cache for semantic search."""

    def __init__(self):
        self._embeddings: dict[str, list[float]] | None = None
        self._model = None

    def get(self) -> tuple[dict[str, list[float]] | None, Any]:
        if self._embeddings is not None:
            return self._embeddings, self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            descriptions = [
                f"{info['description']}. Category: {info['category']}. {info['full_doc'][:200]}"
                for info in TOOL_REGISTRY.values()
            ]
            embeddings = self._model.encode(descriptions)
            self._embeddings = {
                name: emb.tolist()
                for name, emb in zip(TOOL_REGISTRY.keys(), embeddings, strict=True)
            }
            return self._embeddings, self._model
        except ImportError:
            logger.debug("sentence-transformers not installed, falling back to keyword search")
            return None, None

    def reset(self) -> None:
        self._embeddings = None
        self._model = None


_embedding_cache = _EmbeddingCache()


MAX_QUERY_LENGTH = 500
MAX_TOP_K = 20


def find_tools(query: str, top_k: int = 5) -> dict[str, Any]:
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
            "total": len(TOOL_REGISTRY),
            "tools": [
                {"name": name, "description": info["description"], "category": info["category"]}
                for name, info in TOOL_REGISTRY.items()
            ],
            "hint": "Use execute_tool(tool_name, {args}) to run a tool",
        }

    embeddings, model = _embedding_cache.get()

    if embeddings is None:
        return _keyword_search(query, top_k)

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
        return _keyword_search(query, top_k)

    return {
        "query": query,
        "tools": [
            {
                "name": name,
                "description": TOOL_REGISTRY[name]["description"],
                "category": TOOL_REGISTRY[name]["category"],
                "relevance": f"{score:.2f}",
            }
            for name, score in sorted_tools
        ],
        "hint": "Use execute_tool(tool_name, {args}) to run a tool",
    }


def _keyword_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """Fallback keyword search when embeddings unavailable."""
    query_lower = query.lower()
    keywords = query_lower.split()

    scores = {}
    for name, info in TOOL_REGISTRY.items():
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
                "description": TOOL_REGISTRY[name]["description"],
                "category": TOOL_REGISTRY[name]["category"],
            }
            for name, _ in sorted_tools
        ],
        "hint": "Use execute_tool(tool_name, {args}) to run a tool",
    }


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
