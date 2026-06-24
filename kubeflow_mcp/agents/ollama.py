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

"""Ollama agent using LlamaIndex FunctionAgent with native tool calling.

Requires optional dependencies:
    uv sync --extra agents-ollama
    pip install 'kubeflow-mcp[agents-ollama]'

For all agent backends (Ollama + LiteLLM): ``uv sync --extra agents``.

Usage:
    ollama serve
    uv run python -m kubeflow_mcp.agents.ollama
    uv run python -m kubeflow_mcp.agents.ollama --model qwen2.5:7b
"""

import io
import json
import logging
import re
import sys
from contextlib import redirect_stderr
from typing import Any

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("llama_index").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Check if being imported by Sphinx for documentation
_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.ollama import Ollama
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit("Error: required packages not installed\nRun: uv sync --extra agents-ollama")
    # Allow import to continue for autodoc even without dependencies
    FunctionAgent = None  # type: ignore[misc, assignment]
    ChatMemoryBuffer = None  # type: ignore[misc, assignment]
    FunctionTool = None  # type: ignore[misc, assignment]
    Ollama = None  # type: ignore[misc, assignment]

from kubeflow_mcp.agents.dynamic_tools import (  # noqa: E402
    PROGRESSIVE_TOOLS,
    SEMANTIC_TOOLS,
    get_dynamic_system_prompt,
    get_dynamic_tools,
)

try:
    from kubeflow_mcp.core.server import (  # noqa: E402
        build_agent_instruction_text,
        get_merged_client_tool_descriptions,
    )

    TOOL_DESCRIPTIONS = get_merged_client_tool_descriptions()
    SERVER_INSTRUCTIONS = build_agent_instruction_text()
except ImportError:
    SERVER_INSTRUCTIONS = "You are a Kubeflow training assistant."
    TOOL_DESCRIPTIONS: dict[str, str] = {}  # type: ignore[assignment]
try:
    from kubeflow_mcp.trainer import TOOLS  # noqa: E402
except ImportError:
    TOOLS = []  # type: ignore[assignment]  # trainer API not available (skeleton branch)

# Agent configuration defaults
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_REQUEST_TIMEOUT = 180.0  # LLM request timeout in seconds
DEFAULT_MEMORY_TOKEN_LIMIT = 16000  # Chat memory token limit (qwen3:8b has 32K context)

# Tool modes - counts computed dynamically from actual registries
_NUM_TOOLS = len(TOOLS)
_NUM_PROGRESSIVE = len(PROGRESSIVE_TOOLS)
_NUM_SEMANTIC = len(SEMANTIC_TOOLS)

# User-facing tool modes
# "full" uses in-process tools (efficient for local agent)
# "progressive" and "semantic" reduce token usage via meta-tools
TOOL_MODES = {
    "full": f"All {_NUM_TOOLS} tools loaded",
    "progressive": f"{_NUM_PROGRESSIVE} meta-tools with hierarchical discovery",
    "semantic": f"{_NUM_SEMANTIC} meta-tools with embedding search",
}

# Legacy aliases for backward compatibility
_MODE_ALIASES = {"static": "full", "mcp": "full"}

# Agent-specific additions to server instructions
AGENT_HINTS = """
AGENT-SPECIFIC:
- When greeted, introduce yourself briefly and offer to help with training tasks
- Model ID formats: estimate_resources() uses "google/gemma-2b", fine_tune() uses "hf://google/gemma-2b"
- Execute planning steps (1-4) together, only pause after showing the preview
- If no GPUs (gpu_total=0), suggest CPU training or inform user
"""

# System prompt combining server instructions with agent-specific hints
SYSTEM_PROMPT = SERVER_INSTRUCTIONS + AGENT_HINTS


def _create_tools(mode: str = "full") -> list[FunctionTool]:
    """Create LlamaIndex FunctionTools for the given mode.

    Uses compact descriptions from ``get_merged_client_tool_descriptions()`` for full mode (~200 tokens)
    instead of raw docstrings (~5K tokens).

    Args:
        mode: "full" | "progressive" | "semantic"

    Returns:
        List of FunctionTool objects
    """
    if mode in ("progressive", "semantic"):
        tool_funcs = get_dynamic_tools(mode)
    else:
        tool_funcs = TOOLS  # type: ignore[assignment]

    tools = []
    for tool_func in tool_funcs:
        doc = tool_func.__doc__ or ""
        desc = TOOL_DESCRIPTIONS.get(
            tool_func.__name__, doc.split("\n")[0] if doc else tool_func.__name__
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=tool_func,
                name=tool_func.__name__,
                description=desc,
            )
        )
    return tools


class OllamaAgent:
    """Ollama agent using LlamaIndex FunctionAgent with thinking support.

    Supports multiple tool modes for different context budgets:
        - "full": All tools loaded (~200 tokens) - default, best accuracy
        - "progressive": 3 meta-tools (~85 tokens) - hierarchical discovery
        - "semantic": 2 meta-tools (~69 tokens) - embedding-based discovery
    """

    _agent: "FunctionAgent | None"
    _tools: "list[FunctionTool] | None"
    _thinking_supported: bool | None
    _thinking_notified: bool
    memory: "ChatMemoryBuffer | None"
    llm: "Ollama | None"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        tool_mode: str = "full",
    ):
        self.model = model
        self.base_url = base_url
        # Resolve legacy aliases (static, mcp -> full)
        self.tool_mode = _MODE_ALIASES.get(tool_mode, tool_mode)
        self._agent = None
        self._tools = None
        self._thinking_supported = None  # None = unknown, True/False = tested
        self._thinking_notified = False
        self._use_thinking = True
        self._awaiting_confirmation = False  # True after agent shows a preview
        self.memory = None
        self.llm = None

        # Set system prompt based on mode
        if self.tool_mode in ("progressive", "semantic"):
            self._system_prompt = get_dynamic_system_prompt(tool_mode)
        else:
            # For static and mcp modes, use full prompt (mcp may override)
            self._system_prompt = SYSTEM_PROMPT

        # Dedicated event loop in background thread (prevents "Event loop is closed" errors)
        import asyncio
        import threading

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _create_llm(self, with_thinking: bool) -> "Ollama":
        """Create Ollama LLM with or without thinking mode."""
        return Ollama(
            model=self.model,
            base_url=self.base_url,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
            is_function_calling_model=True,
            thinking=with_thinking,
        )

    def _ensure_agent(self, with_thinking: bool | None = None):
        """Lazy initialization of agent."""
        if self._agent is not None:
            return

        if with_thinking is None:
            with_thinking = self._use_thinking

        with redirect_stderr(io.StringIO()):
            self._tools = _create_tools(mode=self.tool_mode)
            self.llm = self._create_llm(with_thinking)
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=DEFAULT_MEMORY_TOKEN_LIMIT)
            self._agent = FunctionAgent(
                tools=self._tools,
                llm=self.llm,
                memory=self.memory,
                system_prompt=self._system_prompt,
            )

    def set_thinking_mode(self, enabled: bool):
        """Toggle thinking mode - recreates LLM but preserves memory."""
        if self._use_thinking == enabled:
            return

        self._use_thinking = enabled

        if self._agent is not None:
            with redirect_stderr(io.StringIO()):
                use_thinking = enabled and (self._thinking_supported is not False)
                self.llm = self._create_llm(use_thinking)
                self._agent = FunctionAgent(
                    tools=self._tools,  # type: ignore[arg-type]
                    llm=self.llm,
                    memory=self.memory,
                    system_prompt=self._system_prompt,
                )

    def set_mode(self, mode: str) -> int:
        """Switch tool mode at runtime. Returns number of tools loaded."""
        # Handle legacy aliases (static, mcp -> full)
        resolved_mode = _MODE_ALIASES.get(mode, mode)

        if resolved_mode not in TOOL_MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(TOOL_MODES.keys())}")

        self.tool_mode = resolved_mode

        # Update system prompt based on mode
        if resolved_mode in ("progressive", "semantic"):
            self._system_prompt = get_dynamic_system_prompt(resolved_mode)
        else:
            self._system_prompt = SYSTEM_PROMPT

        # Force agent recreation with new tools
        self._agent = None
        self._tools = None
        self._ensure_agent(
            with_thinking=self._use_thinking and self._thinking_supported is not False
        )

        return len(self._tools) if self._tools else 0

    async def _chat_async(
        self,
        message: str,
        on_thinking=None,
        on_tool_call=None,
        on_tool_result=None,
    ) -> tuple[str, list[dict]]:
        """Async chat implementation with thinking support."""
        from llama_index.core.agent.workflow.workflow_events import (
            AgentOutput,
            AgentStream,
            ToolCallResult,
        )

        # Initialize with thinking if not yet tested
        if self._thinking_supported is None:
            self._ensure_agent(with_thinking=True)
        else:
            self._ensure_agent(with_thinking=self._thinking_supported and self._use_thinking)

        tool_calls = []
        seen_tools = set()

        try:
            assert self._agent is not None
            handler = self._agent.run(user_msg=message, memory=self.memory)

            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    # Stream thinking output (attribute may not exist in all SDK versions)
                    thinking_delta = getattr(event, "thinking_delta", None)
                    if thinking_delta and on_thinking:
                        on_thinking(thinking_delta)

                    # Collect tool calls
                    if event.tool_calls:
                        for tc in event.tool_calls:
                            key = f"{tc.tool_name}:{json.dumps(tc.tool_kwargs, sort_keys=True)}"
                            if key not in seen_tools:
                                seen_tools.add(key)
                                tool_info = {"name": tc.tool_name, "args": tc.tool_kwargs}
                                tool_calls.append(tool_info)
                                if on_tool_call:
                                    on_tool_call(tool_info)

                elif isinstance(event, ToolCallResult):
                    if on_tool_result:
                        result_info = {
                            "name": event.tool_name,
                            "result": event.tool_output.content if event.tool_output else None,
                        }
                        on_tool_result(result_info)

            result = await handler
            if isinstance(result, AgentOutput):
                response = result.response.content or ""
            else:
                response = str(result)

            # qwen3/deepseek sometimes puts the entire intent inside <think> and
            # emits nothing after it, leaving response empty even though the model
            # reasoned correctly. Strip residual tags so the retry logic fires.
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

            # Mark thinking as supported if we got here
            if self._thinking_supported is None:
                self._thinking_supported = True

        except Exception as e:
            error_msg = str(e)
            # Handle thinking mode not supported
            if "does not support thinking" in error_msg and self._thinking_supported is None:
                self._thinking_supported = False
                # Recreate agent without thinking
                self._agent = None
                self._ensure_agent(with_thinking=False)
                return await self._chat_async(message, on_thinking, on_tool_call, on_tool_result)
            raise

        return response, tool_calls

    def chat(
        self,
        message: str,
        on_thinking=None,
        on_tool_call=None,
        on_tool_result=None,
    ) -> tuple[str, list[dict]]:
        """Synchronous chat wrapper using dedicated event loop.

        Uses short polling intervals to allow Ctrl+C to interrupt.
        Includes retry logic for empty responses.
        """
        import asyncio

        def _poll_chat_async_result(msg: str) -> tuple[str, list[dict]]:
            future = asyncio.run_coroutine_threadsafe(
                self._chat_async(msg, on_thinking, on_tool_call, on_tool_result),
                self._loop,
            )
            while True:
                try:
                    return future.result(timeout=0.5)
                except TimeoutError:
                    continue
                except KeyboardInterrupt:
                    future.cancel()
                    raise

        response, tool_calls = _poll_chat_async_result(message)

        # Track whether agent just showed a preview (confirmed=False tool call)
        if tool_calls:
            last_tool = tool_calls[-1].get("name", "")
            last_args = tool_calls[-1].get("args", {})
            if last_tool in ("fine_tune", "run_custom_training", "run_container_training"):
                self._awaiting_confirmation = not last_args.get("confirmed", False)

        if not response.strip() and not tool_calls:
            logger.warning("Empty response from model, retrying")

            # Thinking mode causes qwen3/deepseek to reason but not emit a tool call.
            # Disable it first so the model outputs an action on the retry.
            if self._use_thinking:
                self.set_thinking_mode(False)

            if self._awaiting_confirmation:
                retry_msg = "User confirmed. Call the appropriate tool to complete the task."
            else:
                retry_msg = (
                    f"You know what to do. Call execute_tool() now. Original request: {message}"
                )

            response, tool_calls = _poll_chat_async_result(retry_msg)

            if not response.strip() and not tool_calls:
                response, tool_calls = _poll_chat_async_result(f"Execute the action now: {message}")

        if not response.strip() and not tool_calls:
            response = (
                "I couldn't generate a response. Try:\n"
                "- `/think` to toggle thinking mode\n"
                "- Be more specific about what you want\n"
                "- Use `/mode full` for more reliable responses"
            )

        return response, tool_calls

    def close(self):
        """Clean up agent resources."""
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2)
        except Exception:
            pass  # Ignore cleanup errors


def _check_ollama_model(model: str, url: str) -> tuple[bool, str]:
    """Check if model exists on Ollama server."""
    import httpx

    try:
        response = httpx.get(f"{url}/api/tags", timeout=10.0)
        response.raise_for_status()
        available = [m["name"] for m in response.json().get("models", [])]

        if model in available:
            return True, "Model ready"

        # Check for similar models
        base = model.split(":")[0]
        similar = [m for m in available if m.startswith(base)]
        if similar:
            return False, f"Not found. Try: {', '.join(similar[:3])}"
        return False, f"Not found. Pull with: ollama pull {model}"

    except httpx.ConnectError:
        return False, f"Cannot connect to {url}"
    except Exception as e:
        return False, str(e)


def run_chat(
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
    tool_mode: str = "full",
    thinking: bool = False,
) -> None:
    """Run interactive Ollama agent (Rich). Delegates to :mod:`kubeflow_mcp.agents.ollama_repl`."""
    from kubeflow_mcp.agents.ollama_repl import run_ollama_chat

    run_ollama_chat(model=model, url=url, tool_mode=tool_mode, thinking=thinking)


class OllamaProvider:
    """Entry-point provider for `kubeflow-mcp agent --provider ollama`."""

    name = "ollama"
    default_model = DEFAULT_MODEL
    requires = ["llama-index-core", "llama-index-llms-ollama", "rich"]

    def run(self, model: str, mode: str, **kwargs: Any) -> None:
        url = str(kwargs.get("url", DEFAULT_URL))
        thinking = bool(kwargs.get("thinking", False))
        run_chat(model=model, url=url, tool_mode=mode, thinking=thinking)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Kubeflow MCP Ollama Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tool modes:
  full        All tools loaded (~200 tokens) - default, best accuracy
  progressive 3 meta-tools (~85 tokens) - hierarchical discovery
  semantic    2 meta-tools (~69 tokens) - embedding search

Examples:
  # Default - all tools
  python -m kubeflow_mcp.agents.ollama

  # Progressive mode (minimal tokens, hierarchical discovery)
  python -m kubeflow_mcp.agents.ollama --mode progressive

  # Semantic mode (requires: pip install sentence-transformers)
  python -m kubeflow_mcp.agents.ollama --mode semantic
        """,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--url", default=DEFAULT_URL, help="Ollama server URL")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "progressive",
            "semantic",
            "static",
            "mcp",
        ],  # static/mcp are legacy aliases
        default="full",
        help="Tool loading mode: full (all tools), progressive (hierarchical), semantic (embedding search)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking output for supported models from startup",
    )
    args = parser.parse_args()

    run_chat(model=args.model, url=args.url, tool_mode=args.mode, thinking=args.thinking)


if __name__ == "__main__":
    main()
