"""Tests for authentication context."""

from kubeflow_mcp.core.auth import AuthContext, get_auth_context


def test_get_auth_context_returns_default():
    ctx = get_auth_context()
    assert isinstance(ctx, AuthContext)
    assert ctx.user is None
    assert ctx.groups is None
    assert ctx.impersonate is None


def test_auth_context_dataclass():
    ctx = AuthContext(user="alice", groups=["ml-team"], impersonate=None)
    assert ctx.user == "alice"
    assert ctx.groups == ["ml-team"]


def test_auth_context_defaults():
    ctx = AuthContext()
    assert ctx.user is None
    assert ctx.groups is None
    assert ctx.impersonate is None
