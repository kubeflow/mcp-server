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

from kubeflow_mcp.core import http_edge
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


def test_metrics_route_reports_unavailable_when_metrics_are_disabled(monkeypatch) -> None:
    monkeypatch.setattr(http_edge, "prometheus_enabled", lambda: False)

    with _probe_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == {"status": "metrics_disabled"}


def test_metrics_route_serves_prometheus_text_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(http_edge, "prometheus_enabled", lambda: True)
    monkeypatch.setattr(
        http_edge,
        "_collect_prometheus_metrics",
        lambda: (b"# HELP up 1\nup 1.0\n", "text/plain; version=0.0.4; charset=utf-8"),
    )

    with _probe_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "up 1.0" in response.text


def test_metrics_route_reports_unavailable_when_exporter_is_missing(monkeypatch) -> None:
    def _raise_import_error() -> tuple[bytes, str]:
        raise ImportError("No module named 'prometheus_client'")

    monkeypatch.setattr(http_edge, "prometheus_enabled", lambda: True)
    monkeypatch.setattr(http_edge, "_collect_prometheus_metrics", _raise_import_error)

    with _probe_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == {"status": "metrics_unavailable"}


def test_metrics_route_reports_unavailable_when_collection_fails(monkeypatch) -> None:
    def _raise_runtime_error() -> tuple[bytes, str]:
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(http_edge, "prometheus_enabled", lambda: True)
    monkeypatch.setattr(http_edge, "_collect_prometheus_metrics", _raise_runtime_error)

    with _probe_client() as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == {"status": "metrics_unavailable"}


def test_metrics_route_remains_unauthenticated(monkeypatch) -> None:
    monkeypatch.setattr(http_edge, "prometheus_enabled", lambda: True)
    monkeypatch.setattr(
        http_edge,
        "_collect_prometheus_metrics",
        lambda: (b"up 1.0\n", "text/plain; version=0.0.4; charset=utf-8"),
    )

    with _probe_client(authenticated=True) as client:
        metrics_response = client.get("/metrics")
        mcp_response = client.post("/mcp")

    assert metrics_response.status_code == 200
    assert mcp_response.status_code == 401
