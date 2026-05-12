"""Tests for authentication context and providers."""

import asyncio
from unittest.mock import MagicMock, patch

from kubeflow_mcp.core.auth import APIKeyVerifier, AuthContext, build_auth_provider


def test_auth_context_defaults():
    ctx = AuthContext()
    assert isinstance(ctx, AuthContext)
    assert ctx.user is None
    assert ctx.groups is None
    assert ctx.impersonate is None


def test_auth_context_dataclass():
    ctx = AuthContext(user="alice", groups=["ml-team"], impersonate=None)
    assert ctx.user == "alice"
    assert ctx.groups == ["ml-team"]


# ─── APIKeyVerifier ───────────────────────────────────────────────────────────


def test_api_key_verifier_accepts_correct_token():
    verifier = APIKeyVerifier(expected_token="secret-key")
    result = asyncio.run(verifier.verify_token("secret-key"))
    assert result is not None
    assert result.client_id == "api-key"


def test_api_key_verifier_rejects_wrong_token():
    verifier = APIKeyVerifier(expected_token="secret-key")
    result = asyncio.run(verifier.verify_token("wrong-key"))
    assert result is None


def test_api_key_verifier_rejects_empty_token():
    verifier = APIKeyVerifier(expected_token="secret-key")
    result = asyncio.run(verifier.verify_token(""))
    assert result is None


# ─── build_auth_provider ──────────────────────────────────────────────────────


def test_build_auth_provider_returns_none_when_no_config():
    cfg = MagicMock()
    cfg.jwks_uri = None
    cfg.auth_token = None

    assert build_auth_provider(cfg) is None


def test_build_auth_provider_returns_api_key_verifier():
    cfg = MagicMock()
    cfg.jwks_uri = None
    cfg.auth_token = "my-secret"

    provider = build_auth_provider(cfg)
    assert isinstance(provider, APIKeyVerifier)


def test_build_auth_provider_prefers_jwt_over_api_key():
    cfg = MagicMock()
    cfg.jwks_uri = "https://example.com/.well-known/jwks.json"
    cfg.auth_token = "also-set"
    cfg.issuer = None
    cfg.audience = None

    with patch("fastmcp.server.auth.providers.jwt.JWTVerifier", autospec=True) as mock_jwt:
        mock_jwt.return_value = MagicMock()
        result = build_auth_provider(cfg)
        assert result is not None
