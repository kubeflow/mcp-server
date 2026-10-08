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

"""Tier 1 safety judges: confirm gate and personas (see ARCHITECTURE.md, "Eval Pipeline").

These judges drive the server the way an agent does, through an in-memory MCP
client, and check invariants that must hold for every tool rather than for one
tool at a time:

- the tool annotations and the confirm gate agree with each other
- a mutating tool called with ``confirmed=False`` returns a preview and sends
  no write request to the Kubernetes API, in every tool mode
- each persona exposes exactly its allowlist over MCP, and hidden tools cannot
  be called, in every tool mode
"""

from typing import Any

import pytest
from fastmcp import Client

from kubeflow_mcp.core.policy import get_allowed_tools
from kubeflow_mcp.core.server import CLIENT_MODULES, create_server
from tests.eval.conftest import GATE_ARGS, payload

# Every registered client is loaded, not only the default one, so a new client's
# tools are judged as soon as it is added to CLIENT_MODULES.
CLIENTS = list(CLIENT_MODULES)

PERSONAS = ["readonly", "data-scientist", "ml-engineer", "platform-admin"]
PROXY_MODES = ["progressive", "semantic"]

# CONVENTIONS.md: "trainer runtime previews use ToolResponse not
# PreviewResponse, do not extend". Any other tool must use PreviewResponse.
LEGACY_PREVIEW_TOOLS = {"create_runtime", "patch_runtime", "delete_runtime"}


def is_preview(name: str, body: dict[str, Any]) -> bool:
    if body.get("status") == "preview":
        return True
    return name in LEGACY_PREVIEW_TOOLS and body.get("data", {}).get("action") == "preview"


async def _list_tools(persona: str) -> dict[str, Any]:
    async with Client(create_server(clients=CLIENTS, persona=persona)) as client:
        return {tool.name: tool for tool in await client.list_tools()}


async def test_annotations_agree_with_confirm_gate():
    """A tool is read-only exactly when it has no confirmed parameter."""
    for name, tool in (await _list_tools("platform-admin")).items():
        read_only = tool.annotations.read_only_hint is True
        confirmed = tool.input_schema.get("properties", {}).get("confirmed")
        if read_only:
            assert confirmed is None, f"{name} is readOnlyHint=True but accepts confirmed"
        else:
            assert confirmed is not None, f"{name} can write but has no confirmed parameter"
            assert confirmed.get("default") is False, f"{name}: confirmed must default to False"


async def test_every_mutating_tool_has_gate_arguments():
    tools = await _list_tools("platform-admin")
    mutating = {name for name, tool in tools.items() if tool.annotations.read_only_hint is not True}
    assert mutating == set(GATE_ARGS), "update GATE_ARGS so every mutating tool is judged"


@pytest.mark.parametrize("tool_name", sorted(GATE_ARGS))
async def test_unconfirmed_call_previews_without_writing(tool_name, k8s):
    async with Client(create_server(clients=CLIENTS, persona="platform-admin")) as client:
        result = await client.call_tool(
            tool_name, {**GATE_ARGS[tool_name], "confirmed": False}, raise_on_error=False
        )
    body = payload(result)

    assert k8s.writes == [], f"{tool_name} wrote to the cluster before confirmation"
    assert is_preview(tool_name, body), f"{tool_name} did not return a preview: {body}"


async def test_confirmed_call_does_write(k8s):
    """Control: the fake API does see writes, so an empty write list means something."""
    async with Client(create_server(clients=CLIENTS, persona="platform-admin")) as client:
        await client.call_tool(
            "update_training_job",
            {**GATE_ARGS["update_training_job"], "confirmed": True},
            raise_on_error=False,
        )
    assert any(method == "PATCH" for method, _ in k8s.writes), k8s.calls


@pytest.mark.parametrize("mode", PROXY_MODES)
@pytest.mark.parametrize("tool_name", sorted(GATE_ARGS))
async def test_proxied_unconfirmed_call_previews_without_writing(mode, tool_name, k8s):
    """The gate must hold when a tool is reached through the execute_tool meta-tool."""
    async with Client(
        create_server(clients=CLIENTS, persona="platform-admin", mode=mode)
    ) as client:
        result = await client.call_tool(
            "execute_tool",
            {"tool_name": tool_name, "arguments": {**GATE_ARGS[tool_name], "confirmed": False}},
            raise_on_error=False,
        )
    body = payload(result)

    assert k8s.writes == [], f"{mode}/{tool_name} wrote before confirmation"
    assert is_preview(tool_name, body), f"{mode}/{tool_name} did not return a preview: {body}"


@pytest.mark.parametrize("persona", PERSONAS)
async def test_persona_exposes_exactly_its_allowlist(persona):
    exposed = set(await _list_tools(persona))
    allowed = get_allowed_tools(persona)
    if allowed is None:
        return
    assert exposed == allowed


async def test_readonly_persona_exposes_no_mutating_tool():
    tools = await _list_tools("readonly")
    mutating = sorted(n for n, t in tools.items() if t.annotations.read_only_hint is not True)
    assert mutating == []


@pytest.mark.parametrize("persona", ["readonly", "data-scientist", "ml-engineer"])
async def test_hidden_tools_cannot_be_called(persona, k8s):
    hidden = sorted(set(GATE_ARGS) - get_allowed_tools(persona))
    async with Client(create_server(clients=CLIENTS, persona=persona)) as client:
        for name in hidden:
            result = await client.call_tool(
                name, {**GATE_ARGS[name], "confirmed": True}, raise_on_error=False
            )
            assert result.is_error, f"{persona} could call hidden tool {name}"
    assert k8s.writes == []


@pytest.mark.parametrize("mode", PROXY_MODES)
@pytest.mark.parametrize("persona", ["readonly", "data-scientist", "ml-engineer"])
async def test_proxied_hidden_tools_cannot_be_called(mode, persona, k8s):
    hidden = sorted(set(GATE_ARGS) - get_allowed_tools(persona))
    async with Client(create_server(clients=CLIENTS, persona=persona, mode=mode)) as client:
        for name in hidden:
            result = await client.call_tool(
                "execute_tool",
                {"tool_name": name, "arguments": {**GATE_ARGS[name], "confirmed": True}},
                raise_on_error=False,
            )
            body = payload(result)
            assert result.is_error or "error" in body, f"{mode}/{persona} ran hidden {name}"
    assert k8s.writes == []
