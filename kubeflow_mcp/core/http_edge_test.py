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

from fastmcp import FastMCP
from starlette.testclient import TestClient

from kubeflow_mcp import __version__
from kubeflow_mcp.core.http_edge import register_probe_routes
from kubeflow_mcp.core.server import create_server


def _probe_client(*, authenticated: bool = False, ready: bool = True) -> TestClient:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    register_probe_routes(mcp, is_ready=ready)
    return TestClient(mcp.http_app(transport="streamable-http"))


def test_health_probe_reports_process_liveness() -> None:
    with _probe_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_probe_reports_server_readiness() -> None:
    with _probe_client() as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_probe_rejects_incomplete_startup() -> None:
    with _probe_client(ready=False) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_probe_rejects_unknown_configured_client() -> None:
    mcp = create_server(clients=["missing-client"])
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_ready_probe_rejects_missing_resource_file(monkeypatch) -> None:
    import kubeflow_mcp.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "CLIENT_RESOURCES",
        {"trainer://guides/missing": ("resources/missing.md", "Missing resource")},
    )

    mcp = create_server()
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_probes_remain_available_when_mcp_auth_is_enabled() -> None:
    with _probe_client(authenticated=True) as client:
        health_response = client.get("/health")
        ready_response = client.get("/ready")
        mcp_response = client.post("/mcp")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert mcp_response.status_code == 401


def test_server_card_reports_metadata() -> None:
    mcp = create_server(persona="platform-admin", mode="progressive")
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    card = response.json()
    assert (
        card["$schema"]
        == "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
    )
    assert card["name"] == "io.github.kubeflow/mcp-server"
    assert card["title"] == "Kubeflow MCP Server"
    assert card["version"] == __version__
    assert card["transport"]["type"] == "streamable-http"
    assert card["transport"]["endpoint"] == "/mcp"
    assert len(card["remotes"]) == 1
    assert card["remotes"][0]["type"] == "streamable-http"
    assert card["remotes"][0]["url"].startswith("http://")
    assert card["remotes"][0]["url"].endswith("/mcp")
    assert card["metadata"]["persona"] == "platform-admin"
    assert card["metadata"]["mode"] == "progressive"
    assert "trainer" in card["metadata"]["clients"]
    assert card["capabilities"]["tools"] is True
    assert card["capabilities"]["resources"] is True


def test_server_card_supports_sse_transport() -> None:
    mcp = create_server(transport="sse")
    with TestClient(mcp.http_app(transport="sse")) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    card = response.json()
    assert card["transport"]["type"] == "sse"
    assert card["transport"]["endpoint"] == "/sse"
    assert card["remotes"][0]["type"] == "sse"
    assert card["remotes"][0]["url"].endswith("/sse")


def test_server_card_handles_empty_or_failed_clients() -> None:
    mcp = create_server(clients=["nonexistent_client"])
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    card = response.json()
    assert card["metadata"]["clients"] == []


def test_server_card_remains_available_when_auth_is_enabled() -> None:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret")
    mcp = create_server(persona="readonly", auth_provider=auth)
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        card_response = client.get("/.well-known/mcp.json")
        mcp_response = client.post("/mcp")

    assert card_response.status_code == 200
    assert mcp_response.status_code == 401


def test_server_card_respects_configured_public_url(monkeypatch) -> None:
    monkeypatch.setenv("KUBEFLOW_MCP_PUBLIC_URL", "https://mcp.kubeflow.org/base")
    mcp = create_server(persona="readonly")
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    card = response.json()
    assert card["remotes"][0]["url"] == "https://mcp.kubeflow.org/base/mcp"


def test_server_card_respects_forwarded_headers() -> None:
    mcp = create_server(persona="readonly")
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get(
            "/.well-known/mcp.json",
            headers={
                "x-forwarded-proto": "https, http",
                "x-forwarded-host": "mcp.example.com:8443, proxy.internal",
            },
        )

    assert response.status_code == 200
    card = response.json()
    assert card["remotes"][0]["url"] == "https://mcp.example.com:8443/mcp"
