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

"""Interactive terminal REPL for the Ollama-backed agent (Rich UI)."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from kubeflow_mcp.agents.terminal_ui import (
    format_tool_result_display,
    get_console,
    print_assistant_panel,
    print_error_panel,
    print_goodbye,
    print_plain,
    print_status,
    print_tip,
    print_tool_result_panel,
    print_user_panel,
    print_welcome_panel,
    setup_readline_history,
)


def _ollama_welcome_rows(
    model: str, url: str, tool_mode: str, tool_modes: dict[str, str]
) -> list[tuple[str, str]]:
    mode_desc = tool_modes.get(tool_mode, tool_mode)
    return [
        ("bold bright_cyan", "Kubeflow AI Agent"),
        ("bright_green", f"Model: {model}"),
        ("bright_white", f"Ollama: {url}"),
        ("bright_yellow", f"Tools: {mode_desc}"),
        ("", ""),
        ("bright_yellow", "Commands:"),
        ("white", "  /tools       - List available tools"),
        ("white", "  /mode        - Switch tool mode (static/progressive/semantic)"),
        ("white", "  /think       - Toggle thinking output"),
        ("white", "  /file <path> - Read file and analyze it"),
        ("white", "  /clear       - Clear conversation memory"),
        ("white", "  exit         - Quit the agent"),
    ]


def _print_chat_error_hints(c: Any, error_msg: str) -> None:
    if "does not support tools" in error_msg:
        print_plain(c, "💡 This model doesn't support function calling.", style="yellow")
        print_plain(
            c,
            "   Try: qwen2.5:7b, llama3.2, or mistral (with tools, no thinking)",
            style="yellow",
        )
        print_plain(c, "   Or: qwq:32b (has both thinking AND tools)", style="yellow")
    elif "connection" in error_msg.lower():
        print_plain(c, "💡 Check if Ollama is running: ollama serve", style="yellow")
    elif "timeout" in error_msg.lower():
        print_plain(c, "💡 Request timed out. Try a simpler query.", style="yellow")


def _repl_slash_tools(agent: Any, c: Any) -> None:
    tools = agent._tools or []
    c.print(f"\n[bold]Available tools ({len(tools)}):[/bold]")
    for t in tools:
        c.print(f"  [bright_cyan]{t.metadata.name}[/bright_cyan]")


def _repl_slash_think(agent: Any, show_thinking: list[bool], c: Any) -> None:
    show_thinking[0] = not show_thinking[0]
    agent.set_thinking_mode(show_thinking[0])
    status = "ON" if show_thinking[0] else "OFF"
    print_plain(c, f"Thinking mode: {status}", style="bright_yellow")
    if show_thinking[0]:
        print_plain(c, "Model reasoning will be shown during responses.", style="dim")


def _repl_slash_clear(agent: Any, show_thinking: list[bool], c: Any) -> None:
    if agent.memory:
        agent.memory.reset()
        agent._awaiting_confirmation = False
        print_plain(c, "✓ Conversation memory cleared", style="bright_green")
        print_plain(c, "Context reset - start fresh!", style="dim")
    else:
        print_plain(c, "No memory to clear", style="dim")
    if show_thinking[0]:
        print_plain(
            c,
            "Note: Only reasoning models (deepseek-r1, qwq, etc.) show thinking output",
            style="dim",
        )


def _repl_slash_mode(agent: Any, user_input: str, tool_modes: dict[str, str], c: Any) -> None:
    parts = user_input.split()
    if len(parts) == 1:
        c.print(f"\n[bold]Current mode:[/bold] {agent.tool_mode}")
        c.print("\n[bold]Available modes:[/bold]")
        for mode_name, mode_desc in tool_modes.items():
            marker = "→" if mode_name == agent.tool_mode else " "
            c.print(f"  {marker} [bright_cyan]{mode_name}[/bright_cyan]: {mode_desc}")
        c.print("\n[dim]Usage: /mode <name>[/dim]")
        return
    new_mode = parts[1].lower()
    try:
        print_plain(c, f"Switching to {new_mode} mode...", style="bright_cyan")
        num_tools = agent.set_mode(new_mode)
        print_plain(c, f"✓ Switched to {new_mode} ({num_tools} tools)", style="bright_green")
    except ValueError as e:
        print_plain(c, f"✗ {e}", style="bright_red")


def _repl_slash_file(user_input: str, c: Any) -> str | None:
    if user_input.lower() == "/file" or user_input[5:].strip() == "":
        print_plain(c, "Usage: /file <path>", style="bright_yellow")
        print_plain(c, "Example: /file examples/mnist_train.py", style="dim")
        print_plain(c, "         /file ~/scripts/train.py", style="dim")
        return None

    file_path = user_input[5:].strip()
    if file_path.startswith(" "):
        file_path = file_path[1:]

    try:
        from pathlib import Path

        path = Path(file_path).expanduser()
        if not path.exists():
            print_plain(c, f"✗ File not found: {file_path}", style="bright_red")
            print_plain(c, "Check the path and try again", style="dim")
            return None

        if not path.is_file():
            print_plain(c, f"✗ Not a file: {file_path}", style="bright_red")
            return None

        content = path.read_text()
        lines = len(content.splitlines())
        print_plain(c, f"✓ Read {path.name} ({lines} lines)", style="bright_green")

        ext = path.suffix.lower()
        lang = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "yaml": "yaml",
            "yml": "yaml",
            "json": "json",
        }.get(ext.lstrip("."), "")

        return (
            f"Here is the contents of `{path.name}`:\n\n```{lang}\n{content}\n```\n\n"
            "Please analyze this file and tell me what it does."
        )
    except Exception as e:
        print_plain(c, f"Error reading file: {e}", style="bright_red")
        return None


def _run_single_chat_turn(
    agent: Any,
    user_input: str,
    show_thinking: list[bool],
    thinking_buffer: list[str],
    c: Any,
) -> None:
    print_user_panel(c, user_input)

    print_status(c, "[bright_cyan]⏳ Thinking...[/bright_cyan]", end="\r")
    thinking_buffer.clear()
    first_output = [True]

    def on_thinking(delta: str) -> None:
        if show_thinking[0] and delta:
            if first_output[0]:
                c.print(" " * 20, end="\r")
                first_output[0] = False
            thinking_buffer.append(delta)
            c.print(
                f"[bright_magenta italic]{delta}[/bright_magenta italic]",
                end="",
                highlight=False,
            )

    def on_tool_call(tool_info: dict[str, Any]) -> None:
        if first_output[0]:
            c.print(" " * 20, end="\r")
            first_output[0] = False
        if thinking_buffer:
            c.print()
            thinking_buffer.clear()
        c.print()

        tool_name = tool_info.get("name", "unknown")
        tool_args = tool_info.get("args") or {}

        c.print(f"  [bright_yellow]🔧 {tool_name}[/bright_yellow]")

        if tool_args:
            args_str = json.dumps(tool_args, indent=2, default=str)
            for line in args_str.split("\n"):
                c.print(f"     [bright_white]{line}[/bright_white]")
        else:
            c.print("     [dim](no arguments)[/dim]")

        c.print("[bright_cyan]  ⏳ Executing...[/bright_cyan]", end="\r")

    def on_tool_result(result_info: dict[str, Any]) -> None:
        c.print(" " * 30, end="\r")
        if result_info.get("result"):
            result_str = format_tool_result_display(result_info["result"])
            print_tool_result_panel(c, result_str)

    try:
        response, _ = agent.chat(
            user_input,
            on_thinking=on_thinking if show_thinking[0] else None,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
    except Exception as e:
        c.print()
        print_error_panel(c, e)
        _print_chat_error_hints(c, str(e))
        return

    c.print(" " * 40, end="\r")

    if agent._thinking_supported is True and not agent._thinking_notified:
        agent._thinking_notified = True
        print_plain(
            c,
            "💭 Thinking supported. Use /think to see model reasoning.",
            style="dim",
        )

    if thinking_buffer:
        c.print()

    if response and response.strip():
        print_assistant_panel(c, response)


def _dispatch_repl_line(
    agent: Any,
    user_input: str,
    show_thinking: list[bool],
    tool_modes: dict[str, str],
    c: Any,
) -> tuple[Literal["empty", "exit", "continue", "chat"], str | None]:
    if not user_input:
        return "empty", None
    low = user_input.lower()
    if low in ("exit", "quit", "q"):
        agent.close()
        print_goodbye(c)
        return "exit", None
    if low == "/tools":
        _repl_slash_tools(agent, c)
        return "continue", None
    if low == "/think":
        _repl_slash_think(agent, show_thinking, c)
        return "continue", None
    if low == "/clear":
        _repl_slash_clear(agent, show_thinking, c)
        return "continue", None
    if low.startswith("/mode"):
        _repl_slash_mode(agent, user_input, tool_modes, c)
        return "continue", None
    if low.startswith("/file"):
        msg = _repl_slash_file(user_input, c)
        if msg is None:
            return "continue", None
        return "chat", msg
    return "chat", user_input


def run_ollama_chat(*, model: str, url: str, tool_mode: str, thinking: bool) -> None:
    """Interactive Ollama agent session (Rich terminal)."""
    from kubeflow_mcp.agents import ollama as core

    c = get_console()
    rows = _ollama_welcome_rows(model, url, tool_mode, core.TOOL_MODES)
    print_welcome_panel(
        panel_title="[bold bright_white]🚀 Ollama Agent[/bold bright_white]",
        border_style="bright_blue",
        rows=rows,
    )

    print_status(c, "[bright_cyan]Checking model...[/bright_cyan]", end="\r")
    model_ok, model_msg = core._check_ollama_model(model, url)
    if model_ok:
        print_status(c, f"[bright_green]✓ {model_msg}[/bright_green]          ")
    else:
        print_status(c, f"[bright_red]✗ {model_msg}[/bright_red]")
        return

    agent = core.OllamaAgent(model=model, base_url=url, tool_mode=tool_mode)

    print_status(c, "[bright_cyan]Loading tools...[/bright_cyan]", end="\r")
    try:
        agent._ensure_agent()
        tools_count = len(agent._tools) if agent._tools else 0
        print_status(c, f"[bright_green]✓ Loaded {tools_count} tools[/bright_green]")
    except Exception as e:
        print_status(c, f"[bright_red]✗ Failed to initialize: {e}[/bright_red]")
        return

    print_tip(c, "💡 Try: 'list training jobs' or 'check cluster resources'")

    setup_readline_history()

    show_thinking = [bool(thinking)]
    if thinking:
        agent.set_thinking_mode(True)
    thinking_buffer: list[str] = []

    while True:
        try:
            c.print()
            c.print("[bold bright_blue]You →[/bold bright_blue] ", end="")
            user_input = input().strip()

            action, chat_msg = _dispatch_repl_line(
                agent, user_input, show_thinking, core.TOOL_MODES, c
            )
            if action == "empty":
                continue
            if action == "exit":
                break
            if action == "continue":
                continue
            assert chat_msg is not None
            _run_single_chat_turn(agent, chat_msg, show_thinking, thinking_buffer, c)

        except KeyboardInterrupt:
            c.print(
                "\n[yellow]Interrupted. Press Ctrl+C again to quit, or continue typing.[/yellow]"
            )
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                agent.close()
                print_goodbye(c)
                break
            continue
        except EOFError:
            agent.close()
            c.print()
            print_goodbye(c)
            break
