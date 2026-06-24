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

"""LiteLLM-backed chat loop (gateway to many model APIs).

Full Kubeflow tool integration matches the Ollama provider; this module offers a
minimal interactive REPL for routing through LiteLLM. Install ``kubeflow-mcp[agents-litellm]``
(or ``[agents]`` for all backends).
"""

from typing import Any


class LiteLLMProvider:
    name = "litellm"
    default_model = "gpt-4o-mini"
    requires = ["litellm", "rich"]

    def run(self, model: str, mode: str, **kwargs: Any) -> None:
        del mode  # reserved for future tool-mode parity with Ollama
        try:
            import litellm
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel
        except ImportError as e:
            msg = (
                "Install optional deps: pip install 'kubeflow-mcp[agents-litellm]' (or '[agents]')"
            )
            raise RuntimeError(msg) from e

        console = Console()
        console.print(
            Panel.fit(
                f"[bold]LiteLLM provider[/bold]\nmodel={model}\n"
                "Type messages (exit / quit to leave). Kubeflow tools are not wired here yet; "
                "use [cyan]--provider ollama[/cyan] for full tool support.",
                title="kubeflow-mcp",
            )
        )
        messages: list[dict[str, str]] = []
        while True:
            console.print("[bold cyan]You[/bold cyan] ", end="")
            line = input().strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                break
            messages.append({"role": "user", "content": line})
            resp = litellm.completion(model=model, messages=messages)
            choice = resp.choices[0]
            content = choice.message.content or ""
            messages.append({"role": "assistant", "content": content})
            console.print(Panel(Markdown(content), title="Assistant", border_style="green"))
