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

"""MCP tool schema snapshot test.

Snapshots the ``tools/list`` surface (tool names, parameters, types,
required/optional, output schemas, behavior annotations) into a committed
baseline and fails if the surface changes without an explicit update.

``AGENTS.md`` forbids tool schema changes without KEP / maintainer
discussion — this test turns that rule into a CI gate, analogous to
OpenAPI spec diffing for REST APIs.

The snapshot is purely structural: human-readable text (tool descriptions,
titles, per-parameter ``description`` fields) is stripped so documentation
edits do not churn the baseline.

To update the baseline after an intentional, approved schema change:

    make update-schema-snapshot
"""

import difflib
import json
import os
import pathlib
from typing import Any

import pytest
from fastmcp import Client

from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from kubeflow_mcp.core.server import create_server

# platform-admin exposes every tool, so its snapshot covers the full surface.
SNAPSHOT_PERSONA = "platform-admin"
SNAPSHOT_PATH = pathlib.Path(__file__).parent / "snapshots" / "tool_schema_platform_admin.json"

_UPDATE_ENV = "UPDATE_SCHEMA_SNAPSHOT"

# Text fields stripped from the snapshot so wording edits don't churn it.
_DOC_KEYS = {"description", "title"}


@pytest.fixture
def _restore_persona():
    """create_server() mutates the process-global persona; restore it."""
    previous = get_effective_persona()
    yield
    set_effective_persona(previous)


def _strip_doc_text(node: Any) -> Any:
    """Recursively drop human-readable text keys from a JSON schema."""
    if isinstance(node, dict):
        return {k: _strip_doc_text(v) for k, v in node.items() if k not in _DOC_KEYS}
    if isinstance(node, list):
        return [_strip_doc_text(item) for item in node]
    return node


async def _build_snapshot() -> dict[str, Any]:
    """Capture the structural tools/list surface via an in-memory MCP client."""
    mcp = create_server(persona=SNAPSHOT_PERSONA)
    async with Client(mcp) as client:
        tools = await client.list_tools()

    snapshot: dict[str, Any] = {}
    for tool in sorted(tools, key=lambda t: t.name):
        entry: dict[str, Any] = {
            "inputSchema": _strip_doc_text(tool.inputSchema),
        }
        if tool.outputSchema is not None:
            entry["outputSchema"] = _strip_doc_text(tool.outputSchema)
        if tool.annotations is not None:
            entry["annotations"] = _strip_doc_text(tool.annotations.model_dump(exclude_none=True))
        snapshot[tool.name] = entry
    return snapshot


def _render(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


async def test_tool_schema_matches_snapshot(_restore_persona):
    """Fail with a diff if the tools/list surface drifts from the baseline."""
    current = _render(await _build_snapshot())

    if os.environ.get(_UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(current, encoding="utf-8")
        return

    assert SNAPSHOT_PATH.exists(), (
        f"Schema snapshot baseline not found: {SNAPSHOT_PATH}\n"
        "Generate it with: make update-schema-snapshot"
    )

    baseline = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if current == baseline:
        return

    diff = "\n".join(
        difflib.unified_diff(
            baseline.splitlines(),
            current.splitlines(),
            fromfile="baseline (committed)",
            tofile="current (tools/list)",
            lineterm="",
        )
    )
    pytest.fail(
        "MCP tool schema surface changed.\n\n"
        f"{diff}\n\n"
        "Tool schemas are a stability contract (see AGENTS.md): changes to tool\n"
        "names, parameters, types, or required fields need KEP / maintainer\n"
        "approval. If this change is intentional and approved, refresh the\n"
        "baseline with:\n\n"
        "    make update-schema-snapshot\n\n"
        "and commit the updated snapshot alongside your change."
    )
