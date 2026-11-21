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

"""Tests for core/auth.py — API key verifier, auth provider builder."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from fastmcp.server.auth.providers.jwt import JWTVerifier

from kubeflow_mcp.core.auth import (
    APIKeyVerifier,
    AuthContext,
    build_auth_provider,
    discover_oidc_configuration,
)
from kubeflow_mcp.core.config import AuthConfig


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
        config = AuthConfig(jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)

        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://example.com/jwks.json"

    def test_jwks_uri_takes_precedence(self):
        config = AuthConfig(auth_token="my-token", jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)

        assert isinstance(provider, JWTVerifier)

    def test_oidc_issuer_auto_discovery(self):
        metadata = {
            "issuer": "https://accounts.google.com",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
        }
        with patch(
            "kubeflow_mcp.core.auth.discover_oidc_configuration", return_value=metadata
        ) as mock_disc:
            config = AuthConfig(oidc_issuer="https://accounts.google.com", audience="my-app")
            provider = build_auth_provider(config)

        mock_disc.assert_called_once_with("https://accounts.google.com", client=None)
        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://www.googleapis.com/oauth2/v3/certs"
        assert provider.issuer == "https://accounts.google.com"
        assert provider.audience == "my-app"

    def test_issuer_fallback_auto_discovery(self):
        metadata = {
            "issuer": "https://keycloak.example.com/realms/kf",
            "jwks_uri": "https://keycloak.example.com/realms/kf/protocol/openid-connect/certs",
        }
        with patch(
            "kubeflow_mcp.core.auth.discover_oidc_configuration", return_value=metadata
        ) as mock_disc:
            config = AuthConfig(issuer="https://keycloak.example.com/realms/kf")
            provider = build_auth_provider(config)

        mock_disc.assert_called_once_with("https://keycloak.example.com/realms/kf", client=None)
        assert isinstance(provider, JWTVerifier)
        assert (
            provider.jwks_uri
            == "https://keycloak.example.com/realms/kf/protocol/openid-connect/certs"
        )

    def test_jwks_uri_skips_oidc_discovery(self):
        with patch("kubeflow_mcp.core.auth.discover_oidc_configuration") as mock_disc:
            config = AuthConfig(
                jwks_uri="https://static.example.com/jwks.json",
                oidc_issuer="https://accounts.google.com",
            )
            provider = build_auth_provider(config)

        mock_disc.assert_not_called()
        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://static.example.com/jwks.json"


class TestDiscoverOIDCConfiguration:
    def test_successful_openid_configuration_discovery(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/.well-known/openid-configuration":
                return httpx.Response(
                    200,
                    json={
                        "issuer": "https://idp.example.com",
                        "jwks_uri": "https://idp.example.com/keys",
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        metadata = discover_oidc_configuration("https://idp.example.com", client=client)
        assert metadata["jwks_uri"] == "https://idp.example.com/keys"

    def test_fallback_to_oauth_authorization_server(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/.well-known/openid-configuration":
                return httpx.Response(404)
            if request.url.path == "/.well-known/oauth-authorization-server":
                return httpx.Response(
                    200,
                    json={
                        "issuer": "https://oauth.example.com",
                        "jwks_uri": "https://oauth.example.com/jwks",
                    },
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        metadata = discover_oidc_configuration("https://oauth.example.com", client=client)
        assert metadata["jwks_uri"] == "https://oauth.example.com/jwks"

    def test_rejects_invalid_issuer_urls(self):
        with pytest.raises(ValueError, match="OIDC issuer must be a non-empty URL string"):
            discover_oidc_configuration("")

        with pytest.raises(ValueError, match="must start with http:// or https://"):
            discover_oidc_configuration("ftp://idp.example.com")

        with pytest.raises(ValueError, match="must start with http:// or https://"):
            discover_oidc_configuration("invalid-domain-without-scheme")

    def test_rejects_missing_jwks_uri_in_metadata(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"issuer": "https://idp.example.com"})

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        with pytest.raises(ValueError, match="returned metadata without a valid 'jwks_uri'"):
            discover_oidc_configuration("https://idp.example.com", client=client)

    def test_rejects_non_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        with pytest.raises(ValueError, match="not valid JSON"):
            discover_oidc_configuration("https://idp.example.com", client=client)

    def test_rejects_unreachable_issuer(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        with pytest.raises(ValueError, match="Failed to discover OIDC configuration"):
            discover_oidc_configuration("https://idp.example.com", client=client)


class TestAuthContext:
    def test_defaults(self):
        ctx = AuthContext()
        assert ctx.user is None
        assert ctx.groups is None
        assert ctx.impersonate is None
