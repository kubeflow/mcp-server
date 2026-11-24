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

"""Tests for core/auth.py — API key verifier, auth provider builder, JWT checks."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.server.auth.providers.jwt import JWTVerifier
from joserfc import jwk, jwt
from starlette.testclient import TestClient

from kubeflow_mcp.core.auth import (
    APIKeyVerifier,
    AuthContext,
    StrictJWTVerifier,
    build_auth_provider,
)
from kubeflow_mcp.core.config import AuthConfig
from kubeflow_mcp.core.server import create_server

ISSUER = "https://idp.example.com"
AUDIENCE = "kubeflow-mcp"
SIGNING_KEY = jwk.RSAKey.generate_key(2048, parameters={"kid": "test-key"})
OTHER_KEY = jwk.RSAKey.generate_key(2048, parameters={"kid": "test-key"})

# Tokens the server must refuse even though the JWKS key signed them.
REJECTED_CLAIMS = [
    pytest.param({"aud": "grafana"}, id="issued-for-another-app"),
    pytest.param({"aud": None}, id="no-audience"),
    pytest.param({"iss": "https://other-tenant.example.com"}, id="other-issuer"),
    pytest.param({"iss": None}, id="no-issuer"),
    pytest.param({"exp": None}, id="no-expiry"),
    pytest.param({"exp": int(time.time()) - 600}, id="expired"),
    pytest.param({"nbf": int(time.time()) + 3600}, id="not-yet-valid"),
    pytest.param({"iat": int(time.time()) + 3600}, id="issued-in-the-future"),
]


def _token(key: jwk.RSAKey = SIGNING_KEY, **claims: object) -> str:
    """RS256 token for ISSUER/AUDIENCE; a claim set to None is left out."""
    now = int(time.time())
    body = {"sub": "alice", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300}
    body.update(claims)
    body = {k: v for k, v in body.items() if v is not None}
    return jwt.encode({"alg": "RS256", "kid": "test-key"}, body, key, algorithms=["RS256"])


@pytest.fixture
def jwt_verifier():
    """The verifier build_auth_provider returns, with the JWKS fetch stubbed out."""
    provider = build_auth_provider(
        AuthConfig(jwks_uri="https://idp.example.com/jwks", issuer=ISSUER, audience=AUDIENCE)
    )
    public_pem = SIGNING_KEY.as_pem(private=False)
    with patch.object(provider, "_get_verification_key", AsyncMock(return_value=public_pem)):
        yield provider


class TestAPIKeyVerifier:
    async def test_accepts_valid_token(self):
        verifier = APIKeyVerifier(expected_token="test-secret")
        result = await verifier.verify_token("test-secret")
        assert result is not None
        assert result.client_id == "api-key"

    async def test_rejects_invalid_token(self):
        verifier = APIKeyVerifier(expected_token="test-secret")
        result = await verifier.verify_token("wrong-token")
        assert result is None

    async def test_rejects_non_ascii_token(self):
        verifier = APIKeyVerifier(expected_token="test-secret")
        result = await verifier.verify_token("café")
        assert result is None

    async def test_accepts_valid_non_ascii_token(self):
        verifier = APIKeyVerifier(expected_token="café")
        result = await verifier.verify_token("café")
        assert result is not None
        assert result.client_id == "api-key"

    async def test_rejects_mismatched_non_ascii_token(self):
        verifier = APIKeyVerifier(expected_token="café")
        result = await verifier.verify_token("fancé")
        assert result is None

    async def test_constant_time_comparison(self):
        verifier = APIKeyVerifier(expected_token="correct-token")
        result = await verifier.verify_token("wrong-length")
        assert result is None

    async def test_empty_token_rejection(self):
        verifier = APIKeyVerifier(expected_token="test-secret")
        result = await verifier.verify_token("")
        assert result is None

    @patch("hmac.compare_digest", return_value=True)
    async def test_timing_safe_comparison(self, mock_compare):
        verifier = APIKeyVerifier(expected_token="test-secret")
        await verifier.verify_token("test-secret")
        mock_compare.assert_called_once_with(b"test-secret", b"test-secret")


class TestBuildAuthProvider:
    def test_returns_none_when_no_auth(self):
        config = AuthConfig()
        assert build_auth_provider(config) is None

    def test_returns_api_key_verifier(self):
        config = AuthConfig(auth_token="my-token")
        provider = build_auth_provider(config)
        assert isinstance(provider, APIKeyVerifier)

    def test_returns_jwt_verifier(self):
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json", issuer=ISSUER, audience=AUDIENCE
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, StrictJWTVerifier)
        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://example.com/jwks.json"
        assert provider.issuer == ISSUER
        assert provider.audience == AUDIENCE

    def test_jwks_uri_takes_precedence(self):
        config = AuthConfig(
            auth_token="my-token",
            jwks_uri="https://example.com/jwks.json",
            issuer=ISSUER,
            audience=AUDIENCE,
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, JWTVerifier)

    @pytest.mark.parametrize(
        ("issuer", "audience"),
        [
            pytest.param(None, None, id="neither"),
            pytest.param(ISSUER, None, id="no-audience"),
            pytest.param(None, AUDIENCE, id="no-issuer"),
            pytest.param("", "", id="empty-strings"),
        ],
    )
    def test_jwks_without_issuer_and_audience_is_refused(self, issuer, audience):
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json", issuer=issuer, audience=audience
        )

        with pytest.raises(ValueError, match="KUBEFLOW_MCP_JWT_AUDIENCE") as exc:
            build_auth_provider(config)
        assert "KUBEFLOW_MCP_JWT_ISSUER" in str(exc.value)


class TestJWTVerification:
    async def test_accepts_token_issued_for_this_server(self, jwt_verifier):
        access = await jwt_verifier.verify_token(_token())

        assert access is not None
        assert access.subject == "alice"

    @pytest.mark.parametrize("claims", REJECTED_CLAIMS)
    async def test_rejects(self, jwt_verifier, claims):
        assert await jwt_verifier.verify_token(_token(**claims)) is None

    async def test_rejects_token_signed_by_another_key(self, jwt_verifier):
        assert await jwt_verifier.verify_token(_token(key=OTHER_KEY)) is None

    async def test_allows_small_clock_skew_on_nbf(self, jwt_verifier):
        token = _token(nbf=int(time.time()) + 30)

        assert await jwt_verifier.verify_token(token) is not None


class TestJWTOverHTTP:
    """The same checks as an MCP client sees them, through the server's HTTP app."""

    @pytest.fixture
    def client(self, jwt_verifier):
        mcp = create_server(auth_provider=jwt_verifier)
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            yield client

    @staticmethod
    def _initialize(client: TestClient, token: str | None):
        headers = {"Accept": "application/json, text/event-stream"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "auth-test", "version": "0"},
                },
            },
        )

    def test_token_issued_for_this_server_is_served(self, client):
        response = self._initialize(client, _token())

        assert response.status_code == 200
        assert '"serverInfo"' in response.text

    @pytest.mark.parametrize("claims", REJECTED_CLAIMS)
    def test_rejected_token_gets_401(self, client, claims):
        response = self._initialize(client, _token(**claims))

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "serverInfo" not in response.text

    def test_token_signed_by_another_key_gets_401(self, client):
        assert self._initialize(client, _token(key=OTHER_KEY)).status_code == 401

    def test_request_without_a_token_gets_401(self, client):
        assert self._initialize(client, None).status_code == 401

    def test_probes_stay_open(self, client):
        assert client.get("/health").status_code == 200


class TestAuthContext:
    def test_defaults(self):
        ctx = AuthContext()
        assert ctx.user is None
        assert ctx.groups is None
        assert ctx.impersonate is None
