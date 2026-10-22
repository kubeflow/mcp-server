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
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.auth import AccessToken
from kubernetes import client as k8s_client

from kubeflow_mcp.common.utils import get_core_v1_api
from kubeflow_mcp.core.middleware import (
    AuditIdentityMiddleware,
    RequestClientConfigurationMiddleware,
    ToolErrorMiddleware,
    _request_id_var,
    _user_id_var,
    get_mcp_request_id,
    get_user_id,
)
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from kubeflow_mcp.core.server import create_server


def _make_context(
    *,
    request_id: object = None,
    user_id: str | None = None,
    has_fastmcp_ctx: bool = True,
    meta_as_dict: bool = True,
) -> SimpleNamespace:
    """Build a minimal mock MiddlewareContext for testing."""
    ctx = SimpleNamespace()

    if has_fastmcp_ctx:
        fastmcp_ctx = SimpleNamespace()
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
        _request_id_var.set(None)
        _user_id_var.set(None)
        yield
        _request_id_var.set(None)
        _user_id_var.set(None)

    async def test_extracts_request_id(self) -> None:
        """Middleware sets the request ContextVar from fastmcp_context."""
        context = _make_context(request_id=42)
        mw = AuditIdentityMiddleware()

        captured: dict[str, str | None] = {}

        async def call_next(_ctx):
            captured["request"] = get_mcp_request_id()
            return "ok"

        result = await mw(context, call_next)

        assert result == "ok"
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
            request_id=99,
            user_id="cleanup-user",
        )
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            # Values should be set during the call
            assert get_mcp_request_id() == "99"
            return "ok"

        await mw(context, call_next)

        # After completion, values should be restored to None
        assert get_mcp_request_id() is None
        assert get_user_id() is None

    async def test_cleanup_on_exception(self) -> None:
        """ContextVars are reset even when call_next raises."""
        context = _make_context(request_id=1)
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await mw(context, call_next)

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
        assert get_mcp_request_id() is None

    async def test_none_request_id(self) -> None:
        """When request_id is None, the ContextVar stays None."""
        context = _make_context(request_id=None)
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            assert get_mcp_request_id() is None
            return "ok"

        await mw(context, call_next)

    async def test_no_request_context_for_user_id(self) -> None:
        """When request_context is None, user_id stays None."""
        context = _make_context(request_id=1)
        # request_context is None by default when user_id is not passed
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            assert get_user_id() is None
            return "ok"

        await mw(context, call_next)

    async def test_token_based_reset_restores_previous_value(self) -> None:
        """Token reset restores previous ContextVar value, not just None."""
        # Simulate an outer layer that set a value
        _request_id_var.set("outer-request")

        context = _make_context(request_id="inner-request")
        mw = AuditIdentityMiddleware()

        async def call_next(_ctx):
            # During the call, the inner value should be active
            assert get_mcp_request_id() == "inner-request"
            return "ok"

        await mw(context, call_next)

        # After middleware completes, the *outer* value should be restored
        assert get_mcp_request_id() == "outer-request"


class TestGetterFunctions:
    """Tests for the public getter functions."""

    def test_defaults_are_none(self) -> None:
        """All getters return None when no ContextVar is set."""
        # Reset to defaults
        _request_id_var.set(None)
        _user_id_var.set(None)

        assert get_mcp_request_id() is None
        assert get_user_id() is None

    def test_getters_return_set_values(self) -> None:
        """Getters return values when ContextVars are populated."""
        _request_id_var.set("test-request")
        _user_id_var.set("test-user")

        try:
            assert get_mcp_request_id() == "test-request"
            assert get_user_id() == "test-user"
        finally:
            _request_id_var.set(None)
            _user_id_var.set(None)


class TestToolErrorMiddleware:
    @staticmethod
    def _server() -> FastMCP:
        mcp = FastMCP("test")
        mcp.add_middleware(ToolErrorMiddleware())

        @mcp.tool
        def fails() -> dict:
            return {"success": False, "error": "bad name", "error_code": "VALIDATION_ERROR"}

        @mcp.tool
        def works() -> dict:
            return {"success": True, "items": []}

        return mcp

    async def test_error_result_sets_is_error(self) -> None:
        async with Client(self._server()) as client:
            result = await client.call_tool_mcp("fails", {})
        assert result.is_error is True
        assert result.structured_content == {
            "success": False,
            "error": "bad name",
            "error_code": "VALIDATION_ERROR",
        }

    async def test_success_result_keeps_is_error_false(self) -> None:
        async with Client(self._server()) as client:
            result = await client.call_tool_mcp("works", {})
        assert result.is_error is False
        assert result.structured_content == {"success": True, "items": []}

    async def test_server_marks_validation_error(self) -> None:
        previous_persona = get_effective_persona()
        try:
            async with Client(create_server()) as client:
                result = await client.call_tool_mcp("get_training_job", {"name": "Bad_Name"})
        finally:
            set_effective_persona(previous_persona)
        assert result.is_error is True
        assert result.structured_content["error_code"] == "VALIDATION_ERROR"


def _bearer_configuration(token: str) -> k8s_client.Configuration:
    cfg = k8s_client.Configuration()
    cfg.host = "https://kubernetes.example"
    cfg.api_key = {"authorization": f"Bearer {token}"}
    return cfg


class TestRequestClientConfigurationMiddleware:
    """The per-request client hook runs each call as its caller, or refuses it."""

    @staticmethod
    def _server(provider) -> tuple[FastMCP, list[str]]:
        mcp = FastMCP("test")
        mcp.add_middleware(RequestClientConfigurationMiddleware(provider))
        executed: list[str] = []

        @mcp.tool
        def whoami() -> dict:
            # A sync tool runs in a worker thread, so this also checks that the
            # request configuration crosses into it.
            executed.append("whoami")
            api = get_core_v1_api()
            return {"authorization": api.api_client.configuration.api_key["authorization"]}

        return mcp, executed

    async def test_tool_clients_use_provider_configuration(self) -> None:
        token = AccessToken(token="alice-token", client_id="alice", scopes=[])
        seen: list[AccessToken | None] = []

        def provider(access_token):
            seen.append(access_token)
            return _bearer_configuration(access_token.token)

        mcp, _ = self._server(provider)
        with patch("kubeflow_mcp.core.middleware.get_access_token", return_value=token):
            async with Client(mcp) as client:
                result = await client.call_tool_mcp("whoami", {})
        assert result.is_error is False
        assert result.structured_content == {"authorization": "Bearer alice-token"}
        assert seen == [token]

    async def test_async_provider_is_awaited(self) -> None:
        async def provider(_token):
            return _bearer_configuration("async-token")

        mcp, _ = self._server(provider)
        async with Client(mcp) as client:
            result = await client.call_tool_mcp("whoami", {})
        assert result.structured_content == {"authorization": "Bearer async-token"}

    async def test_none_from_provider_refuses_without_running_tool(self) -> None:
        """No credentials for the caller must never mean running as the server."""
        mcp, executed = self._server(lambda _token: None)
        async with Client(mcp) as client:
            result = await client.call_tool_mcp("whoami", {})
        assert result.is_error is True
        assert result.structured_content["error_code"] == "PERMISSION_DENIED"
        assert executed == []

    async def test_provider_error_refuses_and_does_not_log_message(self, caplog) -> None:
        """Provider errors can echo the token, so only the exception type is logged."""

        def provider(_token):
            raise ValueError("token secret-value rejected by exchange")

        mcp, executed = self._server(provider)
        with caplog.at_level(logging.DEBUG, logger="kubeflow_mcp.core.middleware"):
            async with Client(mcp) as client:
                result = await client.call_tool_mcp("whoami", {})
        assert result.is_error is True
        assert result.structured_content["error_code"] == "PERMISSION_DENIED"
        assert executed == []
        assert "secret-value" not in caplog.text
        assert "secret-value" not in str(result.structured_content)

    async def test_create_server_wires_provider(self) -> None:
        """With a provider set, upstream tools run under it (here: refused)."""
        previous_persona = get_effective_persona()
        try:
            mcp = create_server(client_configuration_provider=lambda _token: None)
            async with Client(mcp) as client:
                result = await client.call_tool_mcp("get_training_job", {"name": "my-job"})
        finally:
            set_effective_persona(previous_persona)
        assert result.is_error is True
        assert result.structured_content["error_code"] == "PERMISSION_DENIED"

    async def test_write_tool_runs_as_caller(self) -> None:
        """The ownership read and the delete itself both use the caller's configuration."""
        from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE

        ownership_auth: list[str] = []

        def get_namespaced_custom_object(api, **_kwargs):
            ownership_auth.append(api.api_client.configuration.api_key["authorization"])
            return {"metadata": {"labels": {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}}}

        trainer = MagicMock()
        previous_persona = get_effective_persona()
        try:
            mcp = create_server(
                persona="data-scientist",
                client_configuration_provider=lambda _token: _bearer_configuration("bob-token"),
            )
            with (
                patch("kubeflow_mcp.common.utils.TrainerClient", return_value=trainer) as cls,
                patch.object(
                    k8s_client.CustomObjectsApi,
                    "get_namespaced_custom_object",
                    get_namespaced_custom_object,
                ),
            ):
                async with Client(mcp) as client:
                    result = await client.call_tool_mcp(
                        "delete_training_job",
                        {"name": "my-job", "namespace": "team-b", "confirmed": True},
                    )
        finally:
            set_effective_persona(previous_persona)
        assert result.is_error is False, result.structured_content
        assert ownership_auth == ["Bearer bob-token"]
        backend_config = cls.call_args.kwargs["backend_config"]
        assert backend_config.namespace == "team-b"
        assert backend_config.client_configuration.api_key == {"authorization": "Bearer bob-token"}
        trainer.delete_job.assert_called_once_with(name="my-job")

    def test_create_server_default_has_no_hook(self) -> None:
        """Single-tenant deployments keep the server's own kubeconfig identity."""
        previous_persona = get_effective_persona()
        try:
            mcp = create_server()
        finally:
            set_effective_persona(previous_persona)
        assert not any(isinstance(m, RequestClientConfigurationMiddleware) for m in mcp.middleware)
