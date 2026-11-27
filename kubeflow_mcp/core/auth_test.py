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
from collections.abc import Iterator
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    import httpx

ISSUER = "https://idp.example.com"
AUDIENCE = "kubeflow-mcp"
SIGNING_KEY = jwk.RSAKey.generate_key(2048, parameters={"kid": "test-key"})
OTHER_KEY = jwk.RSAKey.generate_key(2048, parameters={"kid": "test-key"})

# Tokens the server must refuse even though the JWKS key signed them. Times are
# offsets in seconds from when _token() builds the token.
REJECTED_CLAIMS = [
    pytest.param({"aud": "grafana"}, id="issued-for-another-app"),
    pytest.param(
        {"aud": ["https://idp.example.com/userinfo", "grafana"]}, id="audience-list-for-other-apps"
    ),
    pytest.param({"aud": None}, id="no-audience"),
    pytest.param({"iss": "https://other-tenant.example.com"}, id="other-issuer"),
    pytest.param({"iss": None}, id="no-issuer"),
    pytest.param({"exp": None}, id="no-expiry"),
    pytest.param({"exp": -600}, id="expired"),
    pytest.param({"nbf": 3600}, id="not-yet-valid"),
    pytest.param({"iat": 3600}, id="issued-in-the-future"),
]


def _token(key: jwk.RSAKey = SIGNING_KEY, **claims: object) -> str:
    """RS256 token for ISSUER/AUDIENCE that expires in five minutes unless ``exp`` is given.

    ``exp``, ``nbf`` and ``iat`` are offsets in seconds from now, so they stay
    relative to the request. A claim set to None is left out.
    """
    now = int(time.time())
    body: dict[str, object] = {"sub": "alice", "iss": ISSUER, "aud": AUDIENCE, "iat": 0, "exp": 300}
    body.update(claims)
    for name in ("exp", "nbf", "iat"):
        offset = body.get(name)
        if isinstance(offset, int):
            body[name] = now + offset
    body = {k: v for k, v in body.items() if v is not None}
    return jwt.encode({"alg": "RS256", "kid": "test-key"}, body, key, algorithms=["RS256"])


@pytest.fixture
def jwt_verifier() -> Iterator[StrictJWTVerifier]:
    """The verifier build_auth_provider returns, with the JWKS fetch stubbed out."""
    provider = build_auth_provider(
        AuthConfig(jwks_uri="https://idp.example.com/jwks", issuer=ISSUER, audience=AUDIENCE)
    )
    assert isinstance(provider, StrictJWTVerifier)
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
    def test_returns_none_when_no_auth(self) -> None:
        config = AuthConfig()
        assert build_auth_provider(config) is None

    def test_returns_api_key_verifier(self) -> None:
        config = AuthConfig(auth_token="my-token")
        provider = build_auth_provider(config)
        assert isinstance(provider, APIKeyVerifier)

    def test_returns_jwt_verifier(self) -> None:
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json", issuer=ISSUER, audience=AUDIENCE
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, StrictJWTVerifier)
        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://example.com/jwks.json"
        assert provider.issuer == ISSUER
        assert provider.audience == AUDIENCE

    def test_jwks_uri_takes_precedence(self) -> None:
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
            pytest.param("  ", "  ", id="whitespace-only"),
            pytest.param(ISSUER, " ", id="whitespace-only-audience"),
        ],
    )
    def test_jwks_without_issuer_and_audience_is_refused(
        self, issuer: str | None, audience: str | None
    ) -> None:
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json", issuer=issuer, audience=audience
        )

        with pytest.raises(ValueError, match="KUBEFLOW_MCP_JWT_AUDIENCE") as exc:
            build_auth_provider(config)
        assert "KUBEFLOW_MCP_JWT_ISSUER" in str(exc.value)

    def test_trims_whitespace_around_issuer_and_audience(self) -> None:
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json", issuer=f" {ISSUER} ", audience=f"{AUDIENCE}\n"
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, StrictJWTVerifier)
        assert provider.issuer == ISSUER
        assert provider.audience == AUDIENCE

    @pytest.mark.parametrize("jwks_uri", ["   ", "\n", "\t \n"])
    def test_blank_jwks_uri_is_refused(self, jwks_uri: str) -> None:
        config = AuthConfig(jwks_uri=jwks_uri, issuer=ISSUER, audience=AUDIENCE)

        with pytest.raises(ValueError, match="KUBEFLOW_MCP_JWKS_URI"):
            build_auth_provider(config)

    def test_trims_whitespace_around_jwks_uri(self) -> None:
        # A Kubernetes Secret created with --from-file keeps the file's trailing newline.
        config = AuthConfig(
            jwks_uri=" https://example.com/jwks.json\n", issuer=ISSUER, audience=AUDIENCE
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, StrictJWTVerifier)
        assert provider.jwks_uri == "https://example.com/jwks.json"


class TestJWTVerification:
    async def test_accepts_token_issued_for_this_server(
        self, jwt_verifier: StrictJWTVerifier
    ) -> None:
        access = await jwt_verifier.verify_token(_token())

        assert access is not None
        assert access.subject == "alice"

    async def test_accepts_audience_list_that_includes_this_server(
        self, jwt_verifier: StrictJWTVerifier
    ) -> None:
        # Some identity providers (Auth0, for one) put several audiences in access tokens.
        token = _token(aud=["https://idp.example.com/userinfo", AUDIENCE])

        assert await jwt_verifier.verify_token(token) is not None

    @pytest.mark.parametrize("claims", REJECTED_CLAIMS)
    async def test_rejects(
        self, jwt_verifier: StrictJWTVerifier, claims: dict[str, object]
    ) -> None:
        assert await jwt_verifier.verify_token(_token(**claims)) is None

    async def test_rejects_token_signed_by_another_key(
        self, jwt_verifier: StrictJWTVerifier
    ) -> None:
        assert await jwt_verifier.verify_token(_token(key=OTHER_KEY)) is None

    async def test_allows_small_clock_skew_on_nbf(self, jwt_verifier: StrictJWTVerifier) -> None:
        assert await jwt_verifier.verify_token(_token(nbf=30)) is not None

    async def test_rejects_token_expired_within_leeway(
        self, jwt_verifier: StrictJWTVerifier
    ) -> None:
        # The minute of leeway is only for nbf and iat: FastMCP rejects a passed exp first.
        assert await jwt_verifier.verify_token(_token(exp=-30)) is None


class TestJWTOverHTTP:
    """The same checks as an MCP client sees them, through the server's HTTP app."""

    @pytest.fixture
    def client(self, jwt_verifier: StrictJWTVerifier) -> Iterator[TestClient]:
        mcp = create_server(auth_provider=jwt_verifier)
        with TestClient(mcp.http_app(transport="streamable-http")) as client:
            yield client

    @staticmethod
    def _initialize(client: TestClient, token: str | None) -> httpx.Response:
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

    def test_token_issued_for_this_server_is_served(self, client: TestClient) -> None:
        response = self._initialize(client, _token())

        assert response.status_code == 200
        assert '"serverInfo"' in response.text

    @pytest.mark.parametrize("claims", REJECTED_CLAIMS)
    def test_rejected_token_gets_401(self, client: TestClient, claims: dict[str, object]) -> None:
        response = self._initialize(client, _token(**claims))

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "serverInfo" not in response.text

    def test_token_signed_by_another_key_gets_401(self, client: TestClient) -> None:
        assert self._initialize(client, _token(key=OTHER_KEY)).status_code == 401

    def test_request_without_a_token_gets_401(self, client: TestClient) -> None:
        assert self._initialize(client, None).status_code == 401

    def test_probes_stay_open(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200


class TestAuthContext:
    def test_defaults(self):
        ctx = AuthContext()
        assert ctx.user is None
        assert ctx.groups is None
        assert ctx.impersonate is None
