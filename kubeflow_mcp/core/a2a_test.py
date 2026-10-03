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

from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from kubeflow_mcp.core.a2a import (
    A2A_PATH,
    AGENT_CARD_PATH,
    build_agent_card,
    extract_delegation,
    register_a2a_routes,
)


def fine_tune(model: str, dataset: str, confirmed: bool = False) -> dict[str, Any]:
    """Stand-in for the real tool, preserving its preview-until-confirmed contract."""
    if not confirmed:
        return {"preview": True, "message": "Review config and set confirmed=True to submit job"}
    return {"job_name": "ft-1", "status": "Created"}


def list_training_jobs() -> dict[str, Any]:
    return {"jobs": []}


def exploding_tool() -> dict[str, Any]:
    raise RuntimeError("kaboom")


def type_error_tool() -> dict[str, Any]:
    """A tool whose own body raises TypeError, as an SDK call might."""
    raise TypeError("unsupported operand type(s) inside the tool")


def identity_probe() -> dict[str, Any]:
    """Report the audit identity visible to a tool while it runs."""
    from kubeflow_mcp.core.middleware import get_user_id

    return {"audit_user": get_user_id()}


def _a2a_client(
    *,
    tools: dict[str, Any] | None = None,
    authenticated: bool = False,
    clients: list[str] | None = None,
) -> TestClient:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    register_a2a_routes(
        mcp,
        tools=tools if tools is not None else {"fine_tune": fine_tune},
        loaded_clients=clients if clients is not None else ["trainer"],
        auth_provider=auth,
    )
    return TestClient(mcp.http_app(transport="streamable-http"))


def _send(tool: str, arguments: dict[str, Any] | None = None, method: str = "message/send") -> dict:
    """Build a standard A2A message/send request carrying a structured DataPart."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": "m-1",
                "parts": [{"kind": "data", "data": {"tool": tool, "arguments": arguments or {}}}],
            }
        },
    }


# ─── Agent Card ───────────────────────────────────────────


def test_agent_card_carries_required_fields() -> None:
    with _a2a_client() as client:
        card = client.get(AGENT_CARD_PATH).json()

    for field in ("name", "url", "version", "protocolVersion", "capabilities", "skills"):
        assert field in card, f"required Agent Card field '{field}' missing"


def test_agent_card_url_is_absolute_and_points_at_endpoint() -> None:
    """An orchestrator calls this URL directly, so a path alone is useless."""
    with _a2a_client() as client:
        card = client.get(AGENT_CARD_PATH).json()

    assert card["url"].startswith("http")
    assert card["url"].endswith(A2A_PATH)


def test_agent_card_skills_are_structured_objects() -> None:
    with _a2a_client() as client:
        skills = client.get(AGENT_CARD_PATH).json()["skills"]

    assert skills
    for skill in skills:
        assert isinstance(skill, dict)
        for field in ("id", "name", "description", "tags"):
            assert field in skill


def test_agent_card_describes_capabilities_not_tool_surface() -> None:
    """The card is unauthenticated, so it must not enumerate tools or schemas."""
    with _a2a_client() as client:
        card = client.get(AGENT_CARD_PATH).json()

    skill_ids = {skill["id"] for skill in card["skills"]}
    assert "fine_tune" not in skill_ids
    assert "list_training_jobs" not in skill_ids
    assert "fine-tuning" in skill_ids
    assert "tools" not in card
    assert "inputSchema" not in repr(card)


def test_agent_card_skills_do_not_vary_with_persona() -> None:
    """Persona shapes what /a2a will run, never what the public card advertises."""
    restrictive = build_agent_card(url="http://x/a2a", loaded_clients=["trainer"])
    permissive = build_agent_card(url="http://x/a2a", loaded_clients=["trainer"])

    assert restrictive["skills"] == permissive["skills"]


def test_agent_card_reflects_loaded_clients() -> None:
    card = build_agent_card(url="http://x/a2a", loaded_clients=["trainer", "optimizer"])
    skill_ids = {skill["id"] for skill in card["skills"]}

    assert "fine-tuning" in skill_ids
    assert "hyperparameter-optimization" in skill_ids
    assert "model-registry" not in skill_ids


def test_agent_card_advertises_security_scheme_only_when_auth_enabled() -> None:
    secured = build_agent_card(url="http://x/a2a", loaded_clients=[], auth_required=True)
    open_card = build_agent_card(url="http://x/a2a", loaded_clients=[], auth_required=False)

    assert secured["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
    assert "securitySchemes" not in open_card


def test_agent_card_declares_streaming_unsupported() -> None:
    """The MVP is stateless request/response; the card must not promise otherwise."""
    with _a2a_client() as client:
        capabilities = client.get(AGENT_CARD_PATH).json()["capabilities"]

    assert capabilities["streaming"] is False


def test_agent_card_stays_public_when_auth_is_enabled() -> None:
    with _a2a_client(authenticated=True) as client:
        assert client.get(AGENT_CARD_PATH).status_code == 200


# ─── Authentication ────────────────────────────────────────


def test_delegation_rejects_missing_credentials() -> None:
    with _a2a_client(authenticated=True) as client:
        response = client.post(A2A_PATH, json=_send("fine_tune"))

    assert response.status_code == 401


def test_delegation_rejects_wrong_token() -> None:
    with _a2a_client(authenticated=True) as client:
        response = client.post(
            A2A_PATH, json=_send("fine_tune"), headers={"Authorization": "Bearer wrong"}
        )

    assert response.status_code == 401


def test_delegation_accepts_valid_token() -> None:
    with _a2a_client(authenticated=True) as client:
        response = client.post(
            A2A_PATH,
            json=_send("fine_tune", {"model": "m", "dataset": "d"}),
            headers={"Authorization": "Bearer secret"},
        )

    assert response.status_code == 200
    assert "result" in response.json()


# ─── Structured delegation ─────────────────────────────────


def test_delegation_runs_tool_and_returns_result_inline() -> None:
    """Stateless: the result rides back on the response, with no task to poll."""
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json=_send("fine_tune", {"model": "m", "dataset": "d", "confirmed": True})
        ).json()

    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert body["result"]["parts"][0]["data"] == {"job_name": "ft-1", "status": "Created"}
    assert "taskId" not in body["result"]


def test_delegation_accepts_legacy_tasks_send_alias() -> None:
    with _a2a_client() as client:
        response = client.post(
            A2A_PATH,
            json=_send("fine_tune", {"model": "m", "dataset": "d"}, method="tasks/send"),
        )

    assert response.status_code == 200
    assert "result" in response.json()


def test_delegation_preserves_confirm_gate_by_default() -> None:
    """The gate exists to stop one-hop job submission; delegation must not skip it."""
    with _a2a_client() as client:
        body = client.post(A2A_PATH, json=_send("fine_tune", {"model": "m", "dataset": "d"})).json()

    assert body["result"]["parts"][0]["data"]["preview"] is True


def test_delegation_honours_explicit_confirmation() -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json=_send("fine_tune", {"model": "m", "dataset": "d", "confirmed": True})
        ).json()

    assert body["result"]["parts"][0]["data"]["job_name"] == "ft-1"


def test_delegation_refuses_tool_outside_persona_surface() -> None:
    """The tool map is the authorization boundary: absent means unreachable."""
    with _a2a_client(tools={"list_training_jobs": list_training_jobs}) as client:
        body = client.post(A2A_PATH, json=_send("fine_tune", {"model": "m"})).json()

    assert body["error"]["code"] == -32004
    assert "not available" in body["error"]["message"]


def test_delegation_rejects_natural_language_only_request() -> None:
    """NL planning belongs to the orchestrator; this agent executes structured calls."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "parts": [{"kind": "text", "text": "fine-tune model X on dataset Y"}],
            }
        },
    }
    with _a2a_client() as client:
        body = client.post(A2A_PATH, json=request).json()

    assert body["error"]["code"] == -32602
    assert "DataPart" in body["error"]["message"]


# ─── Malformed input ───────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"[]", id="array"),
        pytest.param(b"null", id="null"),
        pytest.param(b'"string"', id="scalar-string"),
        pytest.param(b"42", id="scalar-int"),
    ],
)
def test_delegation_rejects_non_object_request_without_crashing(raw: bytes) -> None:
    """Arrays, null, and scalars are valid JSON but invalid requests, never a 500."""
    with _a2a_client() as client:
        response = client.post(A2A_PATH, content=raw, headers={"Content-Type": "application/json"})

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32600


@pytest.mark.parametrize(
    "params",
    [
        pytest.param([], id="params-array"),
        pytest.param("text", id="params-scalar"),
    ],
)
def test_delegation_rejects_non_object_params(params: Any) -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": params}
        ).json()

    assert body["error"]["code"] == -32602


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"parts": []}, id="empty-parts"),
        pytest.param({"parts": "not-a-list"}, id="parts-not-list"),
        pytest.param({}, id="no-parts"),
        pytest.param([], id="message-not-object"),
    ],
)
def test_delegation_rejects_malformed_message(message: Any) -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {"message": message},
            },
        ).json()

    assert body["error"]["code"] == -32602


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"tool": "", "arguments": {}}, id="empty-tool-name"),
        pytest.param({"tool": 123, "arguments": {}}, id="tool-not-string"),
        pytest.param({"arguments": {}}, id="tool-missing"),
        pytest.param({"tool": "fine_tune", "arguments": []}, id="arguments-not-object"),
    ],
)
def test_delegation_rejects_malformed_datapart(data: Any) -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": {"parts": [{"kind": "data", "data": data}]}},
    }
    with _a2a_client() as client:
        body = client.post(A2A_PATH, json=request).json()

    assert body["error"]["code"] == -32602


def test_delegation_rejects_unsupported_method() -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json={"jsonrpc": "2.0", "id": 1, "method": "tasks/cancel", "params": {}}
        ).json()

    assert body["error"]["code"] == -32601


def test_delegation_rejects_wrong_jsonrpc_version() -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json={"jsonrpc": "1.0", "id": 1, "method": "message/send", "params": {}}
        ).json()

    assert body["error"]["code"] == -32600


def test_delegation_rejects_unparseable_body() -> None:
    with _a2a_client() as client:
        response = client.post(
            A2A_PATH, content=b"{not json", headers={"Content-Type": "application/json"}
        )

    assert response.json()["error"]["code"] == -32700


def test_delegation_reports_bad_arguments_as_caller_error() -> None:
    """A wrong argument set is invalid-params, not an internal fault."""
    with _a2a_client() as client:
        body = client.post(A2A_PATH, json=_send("fine_tune", {"nonexistent": "x"})).json()

    assert body["error"]["code"] == -32602


def test_delegation_reports_internal_type_error_as_internal_error() -> None:
    """A TypeError from inside the tool is a server fault, not bad arguments."""
    with _a2a_client(tools={"type_error_tool": type_error_tool}) as client:
        body = client.post(A2A_PATH, json=_send("type_error_tool")).json()

    assert body["error"]["code"] == -32603
    assert "unsupported operand" not in body["error"]["message"]


def test_delegation_separates_bad_arguments_from_tool_faults() -> None:
    """The same tool yields invalid-params for a bad call and internal for its own fault."""
    with _a2a_client(tools={"type_error_tool": type_error_tool}) as client:
        bad_args = client.post(A2A_PATH, json=_send("type_error_tool", {"nope": 1})).json()
        internal = client.post(A2A_PATH, json=_send("type_error_tool")).json()

    assert bad_args["error"]["code"] == -32602
    assert internal["error"]["code"] == -32603


# ─── Audit identity ────────────────────────────────────────


def test_delegated_call_records_authenticated_subject() -> None:
    """A delegated action must be attributable to whoever authenticated."""
    with _a2a_client(tools={"identity_probe": identity_probe}, authenticated=True) as client:
        body = client.post(
            A2A_PATH,
            json=_send("identity_probe"),
            headers={"Authorization": "Bearer secret"},
        ).json()

    assert body["result"]["parts"][0]["data"]["audit_user"] == "api-key"


def test_audit_identity_does_not_leak_after_the_call() -> None:
    from kubeflow_mcp.core.middleware import get_user_id

    with _a2a_client(tools={"identity_probe": identity_probe}, authenticated=True) as client:
        client.post(
            A2A_PATH,
            json=_send("identity_probe"),
            headers={"Authorization": "Bearer secret"},
        )

    assert get_user_id() is None


def test_unauthenticated_server_records_no_subject() -> None:
    """With auth disabled there is no verified caller to attribute the call to."""
    with _a2a_client(tools={"identity_probe": identity_probe}) as client:
        body = client.post(A2A_PATH, json=_send("identity_probe")).json()

    assert body["result"]["parts"][0]["data"]["audit_user"] is None


def test_audit_wrapper_sees_subject_on_a2a_tool_call(monkeypatch) -> None:
    """End-to-end: the audit wrapper that stamps user.id sees the verified caller.

    Spies on the identity lookup inside ``_audit_wrap`` rather than the probe
    tool, so the assertion covers the real audit path a delegated call takes.
    """
    from kubeflow_mcp.core import server as server_module
    from kubeflow_mcp.core.auth import APIKeyVerifier
    from kubeflow_mcp.core.middleware import get_user_id as real_get_user_id

    seen: list[str | None] = []

    def _recording_get_user_id() -> str | None:
        value = real_get_user_id()
        seen.append(value)
        return value

    monkeypatch.setattr(server_module, "get_user_id", _recording_get_user_id)

    auth = APIKeyVerifier(expected_token="secret")
    mcp = server_module.create_server(persona="readonly", auth_provider=auth)
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.post(
            A2A_PATH,
            json=_send("health_check"),
            headers={"Authorization": "Bearer secret"},
        )

    assert response.status_code == 200
    assert "api-key" in seen, f"audit wrapper never saw the authenticated subject: {seen}"


def test_token_subject_prefers_sub_claim_over_client_id() -> None:
    """Under JWT the caller's own identity is the sub claim, not the client."""
    from kubeflow_mcp.core.a2a import _token_subject

    class _Token:
        claims = {"sub": "kunal@example.com"}
        client_id = "some-client"

    assert _token_subject(_Token()) == "kunal@example.com"


def test_token_subject_falls_back_to_client_id() -> None:
    from kubeflow_mcp.core.a2a import _token_subject

    class _Token:
        claims: dict[str, Any] = {}
        client_id = "api-key"

    assert _token_subject(_Token()) == "api-key"


def test_delegation_reports_tool_failure_as_internal_error() -> None:
    with _a2a_client(tools={"exploding_tool": exploding_tool}) as client:
        body = client.post(A2A_PATH, json=_send("exploding_tool")).json()

    assert body["error"]["code"] == -32603
    # The exception text stays in the logs rather than going back to the caller.
    assert "kaboom" not in body["error"]["message"]


def test_error_responses_echo_request_id() -> None:
    with _a2a_client() as client:
        body = client.post(
            A2A_PATH, json={"jsonrpc": "2.0", "id": "abc", "method": "nope", "params": {}}
        ).json()

    assert body["id"] == "abc"


# ─── Wiring through the real server factory ────────────────


def _server_client(persona: str) -> TestClient:
    from kubeflow_mcp.core.server import create_server

    mcp = create_server(persona=persona)
    return TestClient(mcp.http_app(transport="streamable-http"))


def test_server_factory_mounts_agent_card() -> None:
    with _server_client("readonly") as client:
        response = client.get(AGENT_CARD_PATH)

    assert response.status_code == 200
    assert response.json()["name"] == "kubeflow-mcp-server"


def test_readonly_persona_cannot_delegate_a_write_tool() -> None:
    """Persona filtering happens in create_server; /a2a inherits it rather than re-checking."""
    with _server_client("readonly") as client:
        body = client.post(A2A_PATH, json=_send("fine_tune", {"model": "m", "dataset": "d"})).json()

    assert body["error"]["code"] == -32004


def test_data_scientist_persona_can_delegate_a_write_tool() -> None:
    """The same tool the readonly persona was refused resolves for a permitted persona."""
    with _server_client("data-scientist") as client:
        body = client.post(A2A_PATH, json=_send("fine_tune", {"model": "m", "dataset": "d"})).json()

    assert "error" not in body or body["error"]["code"] != -32004


# ─── extract_delegation unit coverage ──────────────────────


def test_extract_delegation_reads_datapart_past_leading_textpart() -> None:
    """A client may prepend a human-readable part; the DataPart still governs."""
    tool, arguments = extract_delegation(
        {
            "message": {
                "parts": [
                    {"kind": "text", "text": "please fine-tune this"},
                    {"kind": "data", "data": {"tool": "fine_tune", "arguments": {"model": "m"}}},
                ]
            }
        }
    )

    assert tool == "fine_tune"
    assert arguments == {"model": "m"}


def test_extract_delegation_defaults_absent_arguments_to_empty() -> None:
    tool, arguments = extract_delegation(
        {"message": {"parts": [{"kind": "data", "data": {"tool": "health_check"}}]}}
    )

    assert tool == "health_check"
    assert arguments == {}
