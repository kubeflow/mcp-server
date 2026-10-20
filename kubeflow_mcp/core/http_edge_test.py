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

from unittest.mock import patch

from fastmcp import FastMCP
from starlette.testclient import TestClient

from kubeflow_mcp.core.health import ReadinessDiagnostic
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


def test_ready_probe_includes_diagnostics_on_failure() -> None:
    diagnostics = [
        ReadinessDiagnostic(
            check="trainer_crd",
            status="fail",
            message="CRD trainjobs.trainer.kubeflow.org not found",
            category="crd_missing",
            resource="trainjobs.trainer.kubeflow.org",
        ),
        ReadinessDiagnostic(
            check="rbac:list:trainjobs:default",
            status="fail",
            message="Permission denied: list trainjobs",
            category="permission_denied",
            resource="trainjobs",
            verb="list",
            namespace="default",
        ),
    ]
    mcp = FastMCP("test-server")
    register_probe_routes(mcp, is_ready=False, diagnostics=diagnostics)
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert len(data["diagnostics"]) == 2
    assert data["diagnostics"][0]["category"] == "crd_missing"
    assert data["diagnostics"][0]["resource"] == "trainjobs.trainer.kubeflow.org"
    assert data["diagnostics"][1]["category"] == "permission_denied"
    assert data["diagnostics"][1]["verb"] == "list"


def test_ready_probe_backward_compatible_without_diagnostics() -> None:
    mcp = FastMCP("test-server")
    register_probe_routes(mcp, is_ready=False)
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_create_server_readiness_healthy_when_k8s_ok() -> None:
    with patch(
        "kubeflow_mcp.core.server.run_readiness_diagnostics",
        return_value=(
            True,
            [
                ReadinessDiagnostic(
                    check="k8s_api",
                    status="pass",
                    message="Kubernetes API reachable",
                )
            ],
        ),
    ):
        mcp = create_server()
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_create_server_readiness_fails_with_diagnostics_when_k8s_fails() -> None:
    failed_diag = ReadinessDiagnostic(
        check="trainer_crd",
        status="fail",
        message="CRD trainjobs.trainer.kubeflow.org not found",
        category="crd_missing",
        resource="trainjobs.trainer.kubeflow.org",
    )
    with patch(
        "kubeflow_mcp.core.server.run_readiness_diagnostics",
        return_value=(False, [failed_diag]),
    ):
        mcp = create_server()
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert len(data["diagnostics"]) == 1
    assert data["diagnostics"][0]["category"] == "crd_missing"
    assert data["diagnostics"][0]["check"] == "trainer_crd"


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
