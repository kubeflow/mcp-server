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

"""Measure the size of the first Trainer tools/list result across loading modes."""

from collections.abc import Callable
from importlib import import_module

import pytest
from fastmcp import Client

from kubeflow_mcp.core import dynamic_tools, policy
from kubeflow_mcp.core.server import CLIENT_MODULES, create_server
from tests.benchmarks import TOOL_LIST_MODES


@pytest.mark.parametrize("mode", TOOL_LIST_MODES)
async def test_trainer_tool_list_size(
    mode: str,
    record_tool_list_size: Callable[[dict[str, object]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_name = "trainer"
    persona = "readonly"
    # Fail before create_server can fall back to health tools if Trainer cannot load.
    import_module(CLIENT_MODULES[client_name])

    # Use the built-in persona even if the operator has a local policy override.
    monkeypatch.setattr(policy, "_get_cached_policy", dict)
    # Server creation in progressive and semantic mode reinitializes this shared state.
    monkeypatch.setattr(dynamic_tools, "TOOL_REGISTRY", {})
    monkeypatch.setattr(dynamic_tools, "TOOL_HIERARCHY", {})
    monkeypatch.setattr(dynamic_tools, "_embedding_cache", dynamic_tools._EmbeddingCache())

    previous_persona = policy.get_effective_persona()
    try:
        server = create_server(clients=[client_name], persona=persona, mode=mode)
        async with Client(server) as client:
            result = await client.list_tools_mcp()
    finally:
        policy.set_effective_persona(previous_persona)

    assert result.tools, f"Trainer {mode} mode returned no tools"
    payload = result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    record_tool_list_size(
        {
            "client": client_name,
            "persona": persona,
            "mode": mode,
            "tool_count": len(result.tools),
            "payload_bytes": len(payload),
        }
    )
