# Copyright 2026 The Kubeflow Authors.
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

"""Agent-side wiring for progressive/semantic meta-tools.

Registers the same :mod:`kubeflow_mcp.core.dynamic_tools` implementation used by
the MCP server, backed by trainer + health tool callables (no MCP hop).
LlamaIndex-specific system prompts live here only.
"""

from collections.abc import Callable
from typing import Any

from kubeflow_mcp.core import dynamic_tools as _core_dynamic

init_dynamic_tools = _core_dynamic.init_dynamic_tools
TOOL_REGISTRY = _core_dynamic.TOOL_REGISTRY
PROGRESSIVE_TOOLS = _core_dynamic.PROGRESSIVE_TOOLS
SEMANTIC_TOOLS = _core_dynamic.SEMANTIC_TOOLS
get_dynamic_tools = _core_dynamic.get_dynamic_tools


def _union_tools_and_descriptions() -> tuple[list[Callable[..., Any]], dict[str, str]]:
    """Trainer tool callables plus merged short descriptions (matches prior agent scope)."""
    from kubeflow_mcp.core.health import HEALTH_TOOL_DESCRIPTIONS

    try:
        from kubeflow_mcp.trainer import CLIENT_TOOL_DESCRIPTIONS, TOOLS
    except ImportError:
        TOOLS = []
        CLIENT_TOOL_DESCRIPTIONS = {}

    descriptions = dict(CLIENT_TOOL_DESCRIPTIONS)
    descriptions.update(HEALTH_TOOL_DESCRIPTIONS)
    return list(TOOLS), descriptions


_tool_funcs, _descriptions = _union_tools_and_descriptions()
init_dynamic_tools(_tool_funcs, _descriptions)

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
3. execute_tool("list_runtimes") → check available runtimes
4. execute_tool("fine_tune", {{..., "confirmed": false}}) → show preview
5. Wait for user confirmation, then resubmit with confirmed=true

Use hf:// prefix for model/dataset URIs. If errors occur, explain them clearly.
"""


def get_dynamic_system_prompt(mode: str = "progressive") -> str:
    """System prompt for dynamic tool mode (LlamaIndex agent)."""
    if mode == "semantic":
        return _SEMANTIC_SYSTEM_PROMPT
    return _PROGRESSIVE_SYSTEM_PROMPT
