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

"""Shared Rich terminal helpers for interactive CLI agents (Ollama, LiteLLM, …)."""

from __future__ import annotations

import json
import sys
from typing import Any

_SPHINX_BUILD = "sphinx" in sys.modules

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    if not _SPHINX_BUILD:
        sys.exit(
            "Error: required packages not installed\n"
            "Run: uv sync --extra agents-ollama   # or agents-litellm, or agents for all backends"
        )
    Console = None  # type: ignore[misc, assignment]
    Markdown = None  # type: ignore[misc, assignment]
    Panel = None  # type: ignore[misc, assignment]
    Table = None  # type: ignore[misc, assignment]
    Text = None  # type: ignore[misc, assignment]

_console: Console | None = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console


def setup_readline_history(history_file: str | None = None) -> None:
    try:
        import atexit
        import os
        import readline  # noqa: F401

        path = history_file or os.path.expanduser("~/.kubeflow_mcp_history")
        try:
            readline.read_history_file(path)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, path)
    except ImportError:
        pass


def print_welcome_panel(
    *,
    panel_title: str,
    border_style: str,
    rows: list[tuple[str, str]],
) -> None:
    """Print a titled panel; each row is (rich_style, text)."""
    c = get_console()
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="left")
    for style, text in rows:
        grid.add_row(Text(text, style=style))
    c.print()
    c.print(Panel(grid, title=panel_title, border_style=border_style, padding=(1, 2)))


def print_tip(c: Console, message: str, *, style: str = "bright_yellow") -> None:
    c.print()
    c.print(Text(message, style=style))


def print_user_panel(c: Console, user_text: str) -> None:
    c.print()
    c.print(
        Panel(
            Text(user_text, style="white"),
            title="[bold bright_blue]You[/bold bright_blue]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )


def print_assistant_panel(c: Console, markdown_text: str) -> None:
    c.print()
    c.print(
        Panel(
            Markdown(markdown_text),
            title="[bold bright_green]Assistant[/bold bright_green]",
            border_style="bright_green",
            padding=(0, 2),
        )
    )


def print_error_panel(c: Console, exc: Exception) -> None:
    body = f"{type(exc).__name__}: {exc}"
    c.print()
    c.print(
        Panel(
            Text(body, style="bright_red"),
            title="[bright_red bold]❌ Error[/bright_red bold]",
            border_style="red",
            padding=(0, 1),
        )
    )


def print_tool_result_panel(c: Console, result_text: str) -> None:
    c.print(
        Panel(
            Text(result_text, style="white"),
            title="[bright_green]Result[/bright_green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def format_tool_result_display(result: Any, max_lines: int = 15) -> str:
    if isinstance(result, dict):
        formatted = json.dumps(result, indent=2, default=str)
    else:
        formatted = str(result)
    lines = formatted.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return formatted


def print_goodbye(c: Console) -> None:
    c.print("[dim italic]Goodbye![/dim italic]")


def print_status(c: Console, message: str, *, end: str = "\n") -> None:
    c.print(message, end=end)


def print_plain(c: Console, message: str, *, style: str | None = None) -> None:
    if style:
        c.print(Text(message, style=style))
    else:
        c.print(message)
