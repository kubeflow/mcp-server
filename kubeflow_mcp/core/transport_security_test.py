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

"""Tests for DNS rebinding protection on the HTTP transport."""

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from kubeflow_mcp.core.config import SecurityConfig
from kubeflow_mcp.core.transport_security import (
    DNSRebindingProtectionMiddleware,
    build_transport_security_settings,
)


def _make_client(config: SecurityConfig) -> TestClient:
    async def ok(_request):
        return JSONResponse({"ok": True})

    settings = build_transport_security_settings(config)
    app = Starlette(
        routes=[
            Route("/mcp", ok, methods=["GET", "POST"]),
            Route("/health", ok, methods=["GET"]),
            Route("/ready", ok, methods=["GET"]),
        ],
        middleware=[Middleware(DNSRebindingProtectionMiddleware, settings=settings)],
    )
    # raise_server_exceptions keeps behaviour close to a real ASGI server.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_probe_paths_bypass_host_validation(path: str):
    # kubelet sends liveness/readiness probes with Host: <pod-ip>:<port>,
    # which is never in the loopback allowlist. These paths are
    # unauthenticated and return no sensitive data, so they must respond
    # regardless of Host (see issue #91).
    client = _make_client(SecurityConfig())
    resp = client.get(path, headers={"host": "10.244.0.5:8000"})
    assert resp.status_code == 200


def test_mcp_path_still_protected_when_probes_are_exempt():
    # Exempting /health and /ready must not weaken validation elsewhere.
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "attacker.example.com"},
    )
    assert resp.status_code == 421


def test_defaults_allow_loopback_host():
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "localhost:8000"},
    )
    assert resp.status_code == 200


def test_defaults_allow_ipv6_loopback_host():
    # FastMCP may bind ::1; clients then send a bracketed IPv6 Host header.
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "[::1]:8000", "origin": "http://[::1]:8000"},
    )
    assert resp.status_code == 200


def test_rejects_unknown_host_with_421():
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "attacker.example.com"},
    )
    assert resp.status_code == 421


def test_rejects_unknown_origin_with_403():
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "localhost:8000", "origin": "http://evil.example.com"},
    )
    assert resp.status_code == 403


def test_rejects_non_json_post_with_400():
    client = _make_client(SecurityConfig())
    resp = client.post(
        "/mcp",
        content=b"not json",
        headers={"host": "localhost:8000", "content-type": "text/plain"},
    )
    assert resp.status_code == 400


def test_disabled_protection_allows_any_host():
    client = _make_client(SecurityConfig(enable_dns_rebinding_protection=False))
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "attacker.example.com"},
    )
    assert resp.status_code == 200


def test_custom_allowlist_is_honored():
    client = _make_client(SecurityConfig(allowed_hosts=["mcp.internal:8000"]))
    allowed = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "mcp.internal:8000"},
    )
    assert allowed.status_code == 200

    # Loopback is no longer implicitly allowed once an explicit list is set.
    rejected = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"host": "localhost:8000"},
    )
    assert rejected.status_code == 421


def test_get_stream_without_origin_allowed():
    # SSE GET streams carry no Origin header from non-browser MCP clients.
    client = _make_client(SecurityConfig())
    resp = client.get("/mcp", headers={"host": "127.0.0.1:8000"})
    assert resp.status_code == 200


class TestLoadConfigSecurityEnvVars:
    """load_config() env-var handling for the security section."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, monkeypatch: pytest.MonkeyPatch):
        # Ignore any ~/.kubeflow-mcp.yaml on the host running the tests.
        from kubeflow_mcp.core import config

        monkeypatch.setattr(config, "_find_config_file", lambda: None)
        for var in (
            "KUBEFLOW_MCP_DNS_REBINDING_PROTECTION",
            "KUBEFLOW_MCP_ALLOWED_HOSTS",
            "KUBEFLOW_MCP_ALLOWED_ORIGINS",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_defaults_when_unset(self):
        from kubeflow_mcp.core.config import load_config

        cfg = load_config()
        assert cfg.security.enable_dns_rebinding_protection is True
        assert cfg.security.allowed_hosts == []
        assert cfg.security.allowed_origins == []

    def test_allowed_hosts_csv_parsing(self, monkeypatch: pytest.MonkeyPatch):
        from kubeflow_mcp.core.config import load_config

        monkeypatch.setenv("KUBEFLOW_MCP_ALLOWED_HOSTS", "mcp.example.com, mcp.example.com:* ,")
        cfg = load_config()
        assert cfg.security.allowed_hosts == ["mcp.example.com", "mcp.example.com:*"]

    def test_allowed_origins_csv_parsing(self, monkeypatch: pytest.MonkeyPatch):
        from kubeflow_mcp.core.config import load_config

        monkeypatch.setenv("KUBEFLOW_MCP_ALLOWED_ORIGINS", "https://mcp.example.com,http://[::1]:*")
        cfg = load_config()
        assert cfg.security.allowed_origins == [
            "https://mcp.example.com",
            "http://[::1]:*",
        ]

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "False", " OFF "])
    def test_protection_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch, value: str):
        from kubeflow_mcp.core.config import load_config

        monkeypatch.setenv("KUBEFLOW_MCP_DNS_REBINDING_PROTECTION", value)
        cfg = load_config()
        assert cfg.security.enable_dns_rebinding_protection is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
    def test_protection_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch, value: str):
        from kubeflow_mcp.core.config import load_config

        monkeypatch.setenv("KUBEFLOW_MCP_DNS_REBINDING_PROTECTION", value)
        cfg = load_config()
        assert cfg.security.enable_dns_rebinding_protection is True
