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

"""Tests for AuditIdentityMiddleware ContextVar bridge."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kubeflow_mcp.core.middleware import (
    AuditIdentityMiddleware,
    _request_id_var,
    _session_id_var,
    _user_id_var,
    get_mcp_request_id,
    get_mcp_session_id,
    get_user_id,
)


def _make_context(
    *,
    session_id: object = None,
    request_id: object = None,
    user_id: str | None = None,
    has_fastmcp_ctx: bool = True,
    meta_as_dict: bool = True,
) -> SimpleNamespace:
    """Build a minimal mock MiddlewareContext for testing."""
    ctx = SimpleNamespace()

    if has_fastmcp_ctx:
        fastmcp_ctx = SimpleNamespace()
        fastmcp_ctx.session_id = session_id
        fastmcp_ctx.request_id = request_id
        ctx.fastmcp_context = fastmcp_ctx
    else:
        ctx.fastmcp_context = None

    if user_id is not None:
        if meta_as_dict:
            meta = {"user_id": user_id}
        else:
            meta = SimpleNamespace(user_id=user_id)
        ctx.request_context = SimpleNamespace(meta=meta)
    else:
        ctx.request_context = None

    return ctx


class TestAuditIdentityMiddleware:
    """Tests for the AuditIdentityMiddleware __call__ method."""

    def _run(self, coro):
        """Helper to run async code in tests."""
        return asyncio.get_event_loop().run_until_complete(coro)

    @pytest.fixture(autouse=True)
    def _clean_contextvars(self):
        """Ensure ContextVars are clean before and after each test."""
        _session_id_var.set(None)
        _request_id_var.set(None)
        _user_id_var.set(None)
        yield
        _session_id_var.set(None)
        _request_id_var.set(None)
        _user_id_var.set(None)

    async def test_extracts_session_and_request_ids(self) -> None:
        """Middleware sets session and request ContextVars from fastmcp_context."""
        context = _make_context(session_id="sess-123", request_id=42)
        mw = AuditIdentityMiddleware()

        captured: dict[str, str | None] = {}

        async def call_next(_ctx):
            captured["session"] = get_mcp_session_id()
            captured["request"] = get_mcp_request_id()
            return "ok"

        result = await mw(context, call_next)

        assert result == "ok"
        assert captured["session"] == "sess-123"
        assert captured["request"] == "42"

    async def test_extracts_user_id_from_dict_meta(self) -> None:
        """Middleware extracts user_id when meta is a dict."""
        context = _make_context(user_id="alice@example.com", meta_as_dict=True)
        mw = AuditIdentityMiddleware()

        captured: dict[str, str | None] = {}

        async def call_next(_ctx):
            captured["user"] = get_user_id()
            return "ok"

        await mw(context, call_next)
        assert captured["user"] == "alice@example.com"

    async def test_extracts_user_id_from_object_meta(self) -> None:
        """Middleware extracts user_id when meta is an object with attributes."""
        context = _make_context(user_id="bob@example.com", meta_as_dict=False)
        mw = AuditIdentityMiddleware()

        captured: dict[str, str | None] = {}

        async def call_next(_ctx):
            captured["user"] = get_user_id()
            return "ok"

        await mw(context, call_next)
        assert captured["user"] == "bob@example.com"

    async def test_cleanup_after_request(self) -> None:
        """ContextVars are reset after the middleware completes."""
        context = _make_context(
            session_id="sess-cleanup",
            request_id=99,
            user_id="cleanup-user",
        )
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            # Values should be set during the call
            assert get_mcp_session_id() == "sess-cleanup"
            return "ok"

        await mw(context, call_next)

        # After completion, values should be restored to None
        assert get_mcp_session_id() is None
        assert get_mcp_request_id() is None
        assert get_user_id() is None

    async def test_cleanup_on_exception(self) -> None:
        """ContextVars are reset even when call_next raises."""
        context = _make_context(session_id="sess-err", request_id=1)
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await mw(context, call_next)

        assert get_mcp_session_id() is None
        assert get_mcp_request_id() is None
        assert get_user_id() is None

    async def test_missing_fastmcp_context(self) -> None:
        """Middleware handles missing fastmcp_context gracefully."""
        context = _make_context(has_fastmcp_ctx=False)
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            return "ok"

        result = await mw(context, call_next)

        assert result == "ok"
        assert get_mcp_session_id() is None
        assert get_mcp_request_id() is None

    async def test_none_session_and_request_ids(self) -> None:
        """When session_id and request_id are None, ContextVars stay None."""
        context = _make_context(session_id=None, request_id=None)
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            assert get_mcp_session_id() is None
            assert get_mcp_request_id() is None
            return "ok"

        await mw(context, call_next)

    async def test_no_request_context_for_user_id(self) -> None:
        """When request_context is None, user_id stays None."""
        context = _make_context(session_id="s1")
        # request_context is None by default when user_id is not passed
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            assert get_user_id() is None
            return "ok"

        await mw(context, call_next)

    async def test_token_based_reset_restores_previous_value(self) -> None:
        """Token reset restores previous ContextVar value, not just None."""
        # Simulate an outer layer that set a value
        _session_id_var.set("outer-session")

        context = _make_context(session_id="inner-session")
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            # During the call, the inner value should be active
            assert get_mcp_session_id() == "inner-session"
            return "ok"

        await mw(context, call_next)

        # After middleware completes, the *outer* value should be restored
        assert get_mcp_session_id() == "outer-session"


class TestGetterFunctions:
    """Tests for the public getter functions."""

    def test_defaults_are_none(self) -> None:
        """All getters return None when no ContextVar is set."""
        # Reset to defaults
        _session_id_var.set(None)
        _request_id_var.set(None)
        _user_id_var.set(None)

        assert get_mcp_session_id() is None
        assert get_mcp_request_id() is None
        assert get_user_id() is None

    def test_getters_return_set_values(self) -> None:
        """Getters return values when ContextVars are populated."""
        _session_id_var.set("test-session")
        _request_id_var.set("test-request")
        _user_id_var.set("test-user")

        try:
            assert get_mcp_session_id() == "test-session"
            assert get_mcp_request_id() == "test-request"
            assert get_user_id() == "test-user"
        finally:
            _session_id_var.set(None)
            _request_id_var.set(None)
            _user_id_var.set(None)
