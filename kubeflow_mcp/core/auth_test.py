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

from unittest.mock import AsyncMock, patch

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

from kubeflow_mcp.core.auth import APIKeyVerifier, AuthContext, build_auth_provider
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

    def test_returns_none_when_empty_strings(self):
        config = AuthConfig(auth_token="", jwks_uri="")
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

    def test_returns_jwt_verifier_with_issuer_and_audience(self):
        config = AuthConfig(
            jwks_uri="https://example.com/jwks.json",
            issuer="https://issuer.example.com",
            audience="kubeflow-mcp",
        )
        provider = build_auth_provider(config)

        assert isinstance(provider, JWTVerifier)
        assert provider.jwks_uri == "https://example.com/jwks.json"
        assert provider.issuer == "https://issuer.example.com"
        assert provider.audience == "kubeflow-mcp"

    def test_jwks_uri_takes_precedence(self):
        config = AuthConfig(auth_token="my-token", jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)

        assert isinstance(provider, JWTVerifier)


class TestAuthContext:
    def test_defaults(self):
        ctx = AuthContext()
        assert ctx.user is None
        assert ctx.groups is None
        assert ctx.impersonate is None

    def test_custom_values(self):
        ctx = AuthContext(
            user="test-user",
            groups=["kubeflow-admins", "data-engineers"],
            impersonate="target-user",
        )
        assert ctx.user == "test-user"
        assert ctx.groups == ["kubeflow-admins", "data-engineers"]
        assert ctx.impersonate == "target-user"


class TestJWTVerifierIntegration:
    async def test_verify_token_invalid_jwt_format(self):
        config = AuthConfig(jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)
        assert isinstance(provider, JWTVerifier)
        result = await provider.verify_token("not-a-valid-jwt-token")
        assert result is None

    @patch.object(JWTVerifier, "load_access_token", new_callable=AsyncMock)
    async def test_verify_token_delegates_to_load_access_token(self, mock_load):
        token_obj = AccessToken(
            token="valid.jwt.token",
            client_id="test-client",
            scopes=["read", "write"],
        )
        mock_load.return_value = token_obj

        config = AuthConfig(jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)
        assert isinstance(provider, JWTVerifier)

        result = await provider.verify_token("valid.jwt.token")
        assert result == token_obj
        mock_load.assert_awaited_once_with("valid.jwt.token")

    @patch.object(JWTVerifier, "load_access_token", new_callable=AsyncMock)
    async def test_verify_token_expired_or_invalid_claims(self, mock_load):
        mock_load.return_value = None

        config = AuthConfig(jwks_uri="https://example.com/jwks.json")
        provider = build_auth_provider(config)
        assert isinstance(provider, JWTVerifier)

        result = await provider.verify_token("expired.jwt.token")
        assert result is None
        mock_load.assert_awaited_once_with("expired.jwt.token")
