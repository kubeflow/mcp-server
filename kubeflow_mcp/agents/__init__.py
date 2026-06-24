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

"""Pluggable CLI agents for Kubeflow MCP.

Heavy providers (Ollama, LiteLLM) are loaded lazily via :func:`__getattr__`
so ``import kubeflow_mcp.agents`` does not require optional dependencies.
"""

from kubeflow_mcp.agents.base import AgentProvider

__all__ = [
    "AgentProvider",
    "LiteLLMProvider",
    "OllamaAgent",
    "OllamaProvider",
]


def __getattr__(name: str):
    if name == "OllamaProvider":
        from kubeflow_mcp.agents.ollama import OllamaProvider as _OllamaProvider

        return _OllamaProvider
    if name == "OllamaAgent":
        from kubeflow_mcp.agents.ollama import OllamaAgent as _OllamaAgent

        return _OllamaAgent
    if name == "LiteLLMProvider":
        from kubeflow_mcp.agents.litellm_provider import LiteLLMProvider as _LiteLLMProvider

        return _LiteLLMProvider
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
