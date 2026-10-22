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

from unittest.mock import MagicMock, patch

import pytest
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


def test_ready_probe_omits_diagnostics_from_payload_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
    with caplog.at_level("WARNING"):
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data == {"status": "not_ready"}
    assert "diagnostics" not in data
    assert any("CRD trainjobs.trainer.kubeflow.org not found" in r.message for r in caplog.records)
    assert any("Permission denied: list trainjobs" in r.message for r in caplog.records)


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


def test_create_server_readiness_fails_without_leaking_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level("WARNING"):
            with TestClient(mcp.http_app(transport="streamable-http")) as client:
                response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data == {"status": "not_ready"}
    assert "diagnostics" not in data
    assert any("CRD trainjobs.trainer.kubeflow.org not found" in r.message for r in caplog.records)


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


def test_ready_probe_recovers_when_diagnostics_recover() -> None:
    """Verify /ready refreshes diagnostics dynamically and recovers without server restart."""
    failed_diag = ReadinessDiagnostic(
        check="k8s_api",
        status="fail",
        message="Kubernetes API unreachable",
        category="api_unreachable",
    )
    passed_diag = ReadinessDiagnostic(
        check="k8s_api",
        status="pass",
        message="Kubernetes API reachable",
    )
    with patch(
        "kubeflow_mcp.core.server.run_readiness_diagnostics",
        side_effect=[
            (False, [failed_diag]),
            (True, [passed_diag]),
        ],
    ):
        mcp = create_server()
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            # 1. Initial probe fails (e.g. cluster temporarily unavailable)
            resp1 = client.get("/ready")
            assert resp1.status_code == 503
            assert resp1.json() == {"status": "not_ready"}
            assert "diagnostics" not in resp1.json()

            # 2. Cluster recovers; next probe succeeds without server restart
            resp2 = client.get("/ready")
            assert resp2.status_code == 200
            assert resp2.json() == {"status": "ready"}


def test_ready_probe_handles_unexpected_exception_in_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = FastMCP("test-server")
    register_probe_routes(
        mcp,
        readiness_check=MagicMock(side_effect=RuntimeError("Unexpected probe failure")),
    )
    with caplog.at_level("WARNING"):
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data == {"status": "not_ready"}
    assert "error" not in data
    assert any("Unexpected probe failure" in r.message for r in caplog.records)
