"""Tests for authentication context."""

from kubeflow_mcp.core.auth import AuthContext


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
