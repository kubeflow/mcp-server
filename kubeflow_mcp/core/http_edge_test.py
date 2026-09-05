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

from kubeflow_mcp.core.http_edge import register_probe_routes
from kubeflow_mcp.core.server import create_server


def _probe_client(*, authenticated: bool = False, ready: bool = True) -> TestClient:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    auth_type = "bearer" if authenticated else None
    register_probe_routes(mcp, is_ready=ready, clients=["trainer"], auth_type=auth_type)
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
        mcp_card_response = client.get("/.well-known/mcp.json")
        mcp_response = client.post("/mcp")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert mcp_card_response.status_code == 200
    assert mcp_response.status_code == 401


def test_discovery_endpoint_serves_mcp_card() -> None:
    from kubeflow_mcp import __version__

    with _probe_client(authenticated=True) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "kubeflow-mcp"
    assert data["version"] == __version__
    assert data["transports"] == [{"type": "streamable-http", "url": "/mcp"}]
    assert data["authentication"] == {"type": "bearer"}
    assert data["capabilities"]["tools"] is True
    assert data["clients"] == ["trainer"]


def test_discovery_endpoint_no_auth() -> None:
    with _probe_client(authenticated=False) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    data = response.json()
    assert "authentication" not in data


def test_discovery_endpoint_with_create_server() -> None:
    mcp = create_server(clients=["trainer"])
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/.well-known/mcp.json")

    assert response.status_code == 200
    data = response.json()
    assert data["clients"] == ["trainer"]
