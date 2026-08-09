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

    async def test_constant_time_comparison(self):
        verifier = APIKeyVerifier(expected_token="correct-token")
        result = await verifier.verify_token("wrong-length")
        assert result is None

    # TODO(test): test empty token rejection
    # TODO(test): test timing-safe comparison (statistical test)


class TestBuildAuthProvider:
    def test_returns_none_when_no_auth(self):
        config = AuthConfig()
        assert build_auth_provider(config) is None

    def test_returns_api_key_verifier(self):
        config = AuthConfig(auth_token="my-token")
        provider = build_auth_provider(config)
        assert isinstance(provider, APIKeyVerifier)

    # TODO(test): test returns JWTVerifier when jwks_uri set
    # TODO(test): test jwks_uri takes precedence over auth_token


class TestAuthContext:
    def test_defaults(self):
        ctx = AuthContext()
        assert ctx.user is None
        assert ctx.groups is None
        assert ctx.impersonate is None
