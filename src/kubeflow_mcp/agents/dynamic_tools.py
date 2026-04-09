# Copyright 2026 The Kubeflow Authors
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
1. Progressive Search - Hierarchical discovery with prefix-based lookup
2. Semantic Search - Embeddings-based natural language discovery

Usage:
    # Progressive mode (~3K initial tokens)
    agent = OllamaAgent(model="qwen2.5:7b", tool_mode="progressive")

    # Semantic mode (~2K initial tokens, requires sentence-transformers)
    agent = OllamaAgent(model="qwen2.5:7b", tool_mode="semantic")

    # Full mode (all tools, ~200 tokens with compact descriptions) - default
    agent = OllamaAgent(model="qwen2.5:7b", tool_mode="full")
"""

import inspect
import warnings
from collections.abc import Callable
from typing import Any

try:
    from kubeflow_mcp.trainer import TOOLS
except ImportError:
    TOOLS = []  # type: ignore[assignment]  # trainer API not available (skeleton branch)

try:
    from kubeflow_mcp.common.constants import TOOL_TO_PHASE
except ImportError:
    TOOL_TO_PHASE: dict[str, str] = {}  # type: ignore[assignment]

try:
    from kubeflow_mcp.core.server import TOOL_DESCRIPTIONS
except ImportError:
    TOOL_DESCRIPTIONS: dict[str, str] = {}  # type: ignore[assignment]

# Build tool registry driven by TOOL_TO_PHASE and TOOL_DESCRIPTIONS from constants/server
# so adding a new tool only requires updating those two central maps.
TOOL_REGISTRY: dict[str, dict[str, Any]] = {}
TOOL_HIERARCHY: dict[str, list[str]] = {
    "planning": [],
    "training": [],
    "discovery": [],
    "monitoring": [],
    "lifecycle": [],
}

for _tool_func in TOOLS:
    _name = _tool_func.__name__
    _doc = _tool_func.__doc__ or ""
    _category = TOOL_TO_PHASE.get(_name, "other")
    _short_desc = TOOL_DESCRIPTIONS.get(_name, _doc.split("\n")[0] if _doc else _name)

    TOOL_REGISTRY[_name] = {
        "name": _name,
        "category": _category,
        "description": _short_desc,
        "full_doc": _doc,
        "func": _tool_func,
    }
    TOOL_HIERARCHY.setdefault(_category, []).append(_name)


# =============================================================================
# Progressive Search Implementation
# =============================================================================


def list_tools(prefix: str = "") -> dict[str, Any]:
    """List available tools by category or prefix.

    Use this to discover what tools are available. Start with no prefix to see
    categories, then drill down with specific prefixes.

    Args:
        prefix: Filter prefix. Examples:
            - "" → List all categories
            - "planning" → List planning tools
            - "training" → List training tools
            - "discovery" → List discovery tools

    Returns:
        {categories: [...], tools: [...]} based on prefix

    Example workflow:
        1. list_tools() → See categories: planning, training, discovery, monitoring, lifecycle
        2. list_tools("training") → See: fine_tune, run_custom_training, run_container_training
        3. describe_tools(["fine_tune"]) → Get full schema for fine_tune
        4. execute_tool("fine_tune", {model: "...", dataset: "..."})
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

    Call this after list_tools() to get full parameter information before executing.

    Args:
        tool_names: List of tool names to describe (max 5 at a time)

    Returns:
        {tools: [{name, description, parameters, returns}]}
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


def _format_friendly_error(result: dict[str, Any]) -> dict[str, Any]:
    """Convert technical errors to user-friendly messages.

    Checks both the top-level error string and the details dict (which
    contains the exception cause chain from exception_details()).
    """
    if result.get("success") is not False:
        return result

    error = result.get("error", "")
    error_code = result.get("error_code", "")
    # SDK wraps K8s HTTP errors; the cause chain is in details
    details = result.get("details") or {}
    detail_str = " ".join(str(v) for v in details.values())
    combined = f"{error} {detail_str}"

    if "401" in combined or "Unauthorized" in combined:
        result["friendly_error"] = "Not authorized to access the cluster. Check your kubeconfig."
        result["hint"] = "Run: kubectl config current-context && kubectl auth can-i list trainjobs"
    elif "403" in combined or "Forbidden" in combined:
        result["friendly_error"] = "Permission denied. Your account lacks RBAC access."
        result["hint"] = "Check RBAC: kubectl auth can-i list trainjobs -n <namespace>"
    elif "404" in combined or "not found" in combined.lower():
        result["friendly_error"] = "Resource not found."
    elif "Connection refused" in combined or "connection refused" in combined.lower():
        result["friendly_error"] = "Cannot connect to Kubernetes cluster."
        result["hint"] = "Is the cluster running? Check: kubectl cluster-info"
    elif "timeout" in combined.lower():
        result["friendly_error"] = "Request timed out. The cluster may be slow or unreachable."
    elif error_code == "SDK_ERROR" and "HuggingFace" in combined:
        result["friendly_error"] = "Could not fetch model info from HuggingFace."
        result["hint"] = "Check the model ID format (e.g., 'meta-llama/Llama-3.2-1B')"

    return result


def execute_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a discovered tool.

    Call this after using list_tools() and describe_tools() to run the actual tool.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments as key-value pairs

    Returns:
        Tool execution result
    """
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool '{tool_name}' not found", "available": list(TOOL_REGISTRY.keys())}

    func = TOOL_REGISTRY[tool_name]["func"]
    args = arguments or {}

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning, module="urllib3")
            result = func(**args)
        if isinstance(result, dict):
            return _format_friendly_error(result)
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "tool": tool_name, "arguments": args}


# Progressive search meta-tools (3 tools instead of 16)
PROGRESSIVE_TOOLS = [list_tools, describe_tools, execute_tool]


# =============================================================================
# Semantic Search Implementation
# =============================================================================

# Pre-computed tool descriptions for embedding
TOOL_DESCRIPTIONS_FOR_EMBEDDING = {
    name: f"{info['description']}. Category: {info['category']}. {info['full_doc'][:200]}"
    for name, info in TOOL_REGISTRY.items()
}


class _EmbeddingCache:
    """Lazy-loaded embedding cache. Holds model + per-tool vectors.

    Centralising state here (vs module globals) makes cache.reset() safe
    to call from tests without touching module-level names.
    """

    def __init__(self):
        self._embeddings: dict[str, list[float]] | None = None
        self._model = None

    def get(self) -> tuple[dict[str, list[float]] | None, Any]:
        if self._embeddings is not None:
            return self._embeddings, self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            descriptions = list(TOOL_DESCRIPTIONS_FOR_EMBEDDING.values())
            embeddings = self._model.encode(descriptions)
            self._embeddings = {
                name: emb.tolist()
                for name, emb in zip(
                    TOOL_DESCRIPTIONS_FOR_EMBEDDING.keys(), embeddings, strict=True
                )
            }
            return self._embeddings, self._model
        except ImportError:
            return None, None

    def reset(self) -> None:
        self._embeddings = None
        self._model = None


_embedding_cache = _EmbeddingCache()


def find_tools(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find relevant tools using semantic search.

    Describe what you want to accomplish in natural language, and this will
    return the most relevant tools.

    Args:
        query: Natural language description. Examples:
            - "all" - LIST ALL 16 AVAILABLE KUBEFLOW TOOLS (use when user asks what tools exist)
            - "check GPU availability in the cluster"
            - "fine-tune a language model"
            - "view logs from a training job"
            - "delete a failed job"
        top_k: Number of results to return (default 5, ignored when query="all")

    Returns:
        {tools: [{name, description, category}], hint: "Use execute_tool(name, args)"}
    """
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

    import numpy as np

    query_embedding = model.encode([query])[0]
    scores = {
        name: float(
            np.dot(query_embedding, tool_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(tool_emb))
        )
        for name, tool_emb in embeddings.items()
    }
    sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

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


# Semantic search meta-tools (2 tools instead of 16)
SEMANTIC_TOOLS = [find_tools, execute_tool]


# =============================================================================
# Factory Functions
# =============================================================================

# Pre-built schema snippet for the 5 most-common tools, injected into the
# progressive system prompt so the agent skips list_tools→describe_tools for
# the happy path (saves 2 LLM round-trips on ~80% of real queries).
_COMMON_TOOL_HINTS = "\n".join(
    f"  {name}: {TOOL_REGISTRY[name]['description']}"
    for name in [
        "get_cluster_resources",
        "estimate_resources",
        "list_runtimes",
        "list_training_jobs",
        "fine_tune",
        "get_training_logs",
    ]
    if name in TOOL_REGISTRY
)

_PROGRESSIVE_SYSTEM_PROMPT = f"""You are a Kubeflow training assistant. Help users manage ML training jobs on Kubernetes.

When greeted, introduce yourself briefly and offer these options:
- Check cluster resources (GPUs, nodes)
- Fine-tune a model (e.g., Llama, Gemma)
- List training jobs or runtimes
- Monitor a running job

Common tools you can call directly via execute_tool:
{_COMMON_TOOL_HINTS}

Categories for discovery:
  planning   → get_cluster_resources, estimate_resources
  training   → fine_tune, run_custom_training, run_container_training
  discovery  → list_runtimes, get_runtime, list_training_jobs, get_training_job
  monitoring → get_training_logs, get_training_events, wait_for_training
  lifecycle  → delete_training_job, suspend_training_job, resume_training_job

For less-common tasks: list_tools("category") → describe_tools(["tool_name"]) → execute_tool("tool_name", {{args}})

When the user asks to train or fine-tune:
1. execute_tool("get_cluster_resources") → check GPUs
2. execute_tool("estimate_resources", {{"model": "google/gemma-2b"}}) → check memory
3. execute_tool("list_runtimes") → check available runtimes
4. execute_tool("fine_tune", {{..., "confirmed": false}}) → show preview
5. Wait for user confirmation, then resubmit with confirmed=true

Use hf:// prefix for model/dataset URIs. If errors occur, explain them clearly.
"""

_SEMANTIC_SYSTEM_PROMPT = f"""You are a Kubeflow training assistant. Help users manage ML training jobs on Kubernetes.

When greeted, introduce yourself briefly and offer these options:
- Check cluster resources (GPUs, nodes)
- Fine-tune a model (e.g., Llama, Gemma)
- List training jobs or runtimes
- Monitor a running job

Common tools you can call directly via execute_tool:
{_COMMON_TOOL_HINTS}

IMPORTANT: When user asks "what tools are available", call find_tools("all").

Categories: planning (resources), training (fine_tune, custom, container), discovery (list_runtimes, list_training_jobs, get_runtime), monitoring (logs, events), lifecycle (delete, suspend, resume).

For other tasks, use find_tools("natural language query") to discover tools, then execute_tool().

When the user asks to train or fine-tune:
1. execute_tool("get_cluster_resources") → check GPUs
2. execute_tool("estimate_resources", {{"model": "google/gemma-2b"}}) → check memory
3. execute_tool("fine_tune", {{..., "confirmed": false}}) → show preview
4. Wait for user confirmation, then resubmit with confirmed=true

Use hf:// prefix for model/dataset URIs. If errors occur, explain them clearly.
"""


def get_dynamic_tools(mode: str = "progressive") -> list[Callable[..., Any]]:
    """Get meta-tools for dynamic discovery.

    Args:
        mode: "progressive" or "semantic"

    Returns:
        List of meta-tool functions
    """
    if mode == "semantic":
        return SEMANTIC_TOOLS  # type: ignore[return-value]
    return PROGRESSIVE_TOOLS  # type: ignore[return-value]


def get_dynamic_system_prompt(mode: str = "progressive") -> str:
    """Get system prompt for dynamic tool mode."""
    if mode == "semantic":
        return _SEMANTIC_SYSTEM_PROMPT
    return _PROGRESSIVE_SYSTEM_PROMPT
