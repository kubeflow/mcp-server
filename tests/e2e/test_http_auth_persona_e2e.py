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

"""E2E tests for HTTP bearer auth and persona enforcement on the runtime image.

The persona is fixed when the server starts, so each persona needs its own server. CI
redeploys the image once per persona and runs this module with
``KUBEFLOW_MCP_E2E_PERSONA`` set to the persona the server was started with. Auth
failures (HTTP 401, no MCP session) are kept apart from persona denials (valid token,
tool not offered) so a regression in one cannot be mistaken for the other.

Requires ``KUBEFLOW_MCP_E2E=true``, ``KUBEFLOW_MCP_E2E_URL`` and
``KUBEFLOW_MCP_E2E_TOKEN``.
"""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.exceptions import MCPError

from tests.e2e.test_kubernetes_e2e import (
    NOOP_TRAIN_SCRIPT,
    _call_tool,
    _get_custom_runtime_name,
)

URL = os.getenv("KUBEFLOW_MCP_E2E_URL")
TOKEN = os.getenv("KUBEFLOW_MCP_E2E_TOKEN")
PERSONA = os.getenv("KUBEFLOW_MCP_E2E_PERSONA")

pytestmark = [
    pytest.mark.skipif(
        os.getenv("KUBEFLOW_MCP_E2E") != "true",
        reason="KUBEFLOW_MCP_E2E=true environment variable not set",
    ),
    pytest.mark.skipif(
        not (URL and TOKEN),
        reason="HTTP auth tests need KUBEFLOW_MCP_E2E_URL and KUBEFLOW_MCP_E2E_TOKEN",
    ),
]

NAMESPACE = "default"


@pytest.fixture
async def mcp_session() -> AsyncGenerator[ClientSession, None]:
    """Authenticated Streamable HTTP session.

    Not the shared fixture from test_kubernetes_e2e: the root conftest strips
    KUBEFLOW_MCP_* variables before every test, so it would fall back to a local stdio
    server. URL and token are read once at import instead. The session lives in its own
    task so anyio cancel scopes are entered and exited in the same task.
    """
    assert URL is not None
    session_future: asyncio.Future[ClientSession] = asyncio.Future()
    stop_event = asyncio.Event()

    async def run() -> None:
        try:
            async with (
                create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"}) as client,
                streamable_http_client(URL, http_client=client) as (read, write, *_),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                session_future.set_result(session)
                await stop_event.wait()
        except Exception as exc:
            if not session_future.done():
                session_future.set_exception(exc)
            else:
                raise

    task = asyncio.create_task(run())
    try:
        yield await session_future
    finally:
        stop_event.set()
        await asyncio.gather(task, return_exceptions=True)


_READONLY_ALLOWED = {"list_training_jobs", "get_cluster_resources", "list_runtimes"}
_DS_ALLOWED = _READONLY_ALLOWED | {"run_custom_training", "delete_training_job", "fine_tune"}
_MLE_ALLOWED = _DS_ALLOWED | {"run_container_training", "update_training_job", "inspect_crd"}

# Representative tools each persona may and may not be offered. Derived from
# kubeflow_mcp/core/policy.py, but spelled out so a policy change fails here loudly.
ALLOWED: dict[str, set[str]] = {
    "readonly": _READONLY_ALLOWED,
    "data-scientist": _DS_ALLOWED,
    "ml-engineer": _MLE_ALLOWED,
    "platform-admin": _MLE_ALLOWED | {"delete_runtime"},
}
DENIED: dict[str, set[str]] = {
    "readonly": {"fine_tune", "run_custom_training", "delete_training_job", "delete_runtime"},
    "data-scientist": {"run_container_training", "update_training_job", "delete_runtime"},
    "ml-engineer": {"delete_runtime"},
    "platform-admin": set(),
}


def _initialize_request() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "e2e-auth", "version": "0"},
        },
    }


def _post_initialize(headers: dict[str, str]) -> httpx.Response:
    assert URL is not None
    return httpx.post(
        URL,
        json=_initialize_request(),
        headers={"Accept": "application/json, text/event-stream", **headers},
        timeout=30,
    )


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-token"),
        pytest.param({"Authorization": "Bearer not-the-real-token"}, id="invalid-token"),
        pytest.param({"Authorization": "Basic Zm9vOmJhcg=="}, id="wrong-scheme"),
    ],
)
def test_http_rejects_missing_or_invalid_credentials(headers: dict[str, str]) -> None:
    resp = _post_initialize(headers)
    assert resp.status_code == 401, resp.text
    assert "mcp-session-id" not in resp.headers


def test_http_accepts_valid_token() -> None:
    resp = _post_initialize({"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200, resp.text


@pytest.fixture
def persona() -> str:
    if PERSONA not in ALLOWED:
        pytest.skip(f"KUBEFLOW_MCP_E2E_PERSONA must be one of {sorted(ALLOWED)}, got {PERSONA!r}")
    assert PERSONA is not None
    return PERSONA


async def _denied(session: ClientSession, tool: str, args: dict[str, Any]) -> None:
    """Assert the server refuses ``tool``, whether as a protocol error or an error result."""
    try:
        resp = await session.call_tool(tool, arguments=args)
    except MCPError:
        return
    assert resp.is_error, f"{tool} was not denied: {resp.content}"


async def _job_names(session: ClientSession) -> list[str]:
    data = await _call_tool(session, "list_training_jobs", {"namespace": NAMESPACE})
    assert data["success"] is True, data
    return [j["name"] for j in data["data"]["jobs"]]


@pytest.mark.asyncio
async def test_persona_tool_list(mcp_session: ClientSession, persona: str) -> None:
    offered = {t.name for t in (await mcp_session.list_tools()).tools}
    assert ALLOWED[persona] <= offered, f"{persona} missing {ALLOWED[persona] - offered}"
    assert not (DENIED[persona] & offered), f"{persona} offered {DENIED[persona] & offered}"


@pytest.mark.asyncio
async def test_persona_denied_mutation_changes_nothing(
    mcp_session: ClientSession,
    persona: str,
) -> None:
    """A denied mutating call, even with confirmed=True, must not create a TrainJob."""
    denied = sorted(DENIED[persona] & {"run_custom_training", "run_container_training"})
    if not denied:
        pytest.skip(f"{persona} is allowed every mutating training tool")
    runtime = await _get_custom_runtime_name(mcp_session)
    before = await _job_names(mcp_session)

    for tool in denied:
        name = f"e2e-denied-{uuid.uuid4().hex[:6]}"
        args: dict[str, Any] = {"name": name, "namespace": NAMESPACE, "confirmed": True}
        if tool == "run_custom_training":
            args |= {"script": NOOP_TRAIN_SCRIPT, "runtime": runtime, "gpu_per_node": 0}
        else:
            args |= {"image": "busybox:1.36", "runtime": runtime, "gpu_per_node": 0}
        await _denied(mcp_session, tool, args)
        assert name not in await _job_names(mcp_session)

    assert sorted(await _job_names(mcp_session)) == sorted(before)


@pytest.mark.asyncio
async def test_persona_allowed_mutation_executes(
    mcp_session: ClientSession,
    persona: str,
) -> None:
    """Counterpart to the denial test: a permitted persona really creates the job."""
    if "run_custom_training" not in ALLOWED[persona] or persona == "readonly":
        pytest.skip(f"{persona} may not run training")
    runtime = await _get_custom_runtime_name(mcp_session)
    name = f"e2e-allowed-{uuid.uuid4().hex[:6]}"
    try:
        data = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": runtime,
                "name": name,
                "namespace": NAMESPACE,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert data["success"] is True, json.dumps(data)
        assert name in await _job_names(mcp_session)
    finally:
        await mcp_session.call_tool(
            "delete_training_job",
            arguments={"name": name, "namespace": NAMESPACE, "confirmed": True},
        )
