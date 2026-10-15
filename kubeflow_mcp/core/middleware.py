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

"""Middleware to bridge FastMCP async context into sync tool wrappers via ContextVars.

FastMCP's ``CurrentContext()`` dependency injection may not reliably propagate
into sync wrappers.  This module uses :mod:`contextvars` to capture request
and user identity from the async middleware layer so that the
synchronous ``_audit_wrap`` in :mod:`kubeflow_mcp.core.server` can read them
without depending on DI.

:class:`RequestClientConfigurationMiddleware` uses the same mechanism to give
each tool call its own Kubernetes client configuration, for deployments that
run every call as the caller's Kubernetes identity.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
from typing import TYPE_CHECKING, Any

from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.common.utils import request_client_configuration

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastmcp.server.auth import AccessToken
    from kubernetes.client import Configuration

    ClientConfigurationProvider = Callable[
        [AccessToken | None],
        Configuration | None | Awaitable[Configuration | None],
    ]

logger = logging.getLogger(__name__)

# ContextVars populated by AuditIdentityMiddleware, read by _audit_wrap
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_request_id", default=None
)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_user_id", default=None
)


def get_mcp_request_id() -> str | None:
    """Return the MCP request ID for the current request, or None."""
    return _request_id_var.get()


def get_user_id() -> str | None:
    """Return the user identity for the current request, or None."""
    return _user_id_var.get()


class AuditIdentityMiddleware:
    """FastMCP-compatible middleware that captures identity into ContextVars.

    Extracts ``request_id`` and optionally ``user_id``
    from the FastMCP ``MiddlewareContext`` and stores them in module-level
    :class:`contextvars.ContextVar` instances.  Downstream sync code
    (e.g. ``_audit_wrap``) can retrieve these values via the public
    ``get_mcp_*`` helpers without needing async or DI.

    Usage::

        from kubeflow_mcp.core.middleware import AuditIdentityMiddleware
        mcp = FastMCP("kubeflow-mcp-server")
        mcp.add_middleware(AuditIdentityMiddleware())
    """

    async def __call__(self, context: Any, call_next: Any) -> Any:
        """Capture identity from context, then delegate to the next handler."""
        # Use tokens so reset() restores the *previous* value rather than
        # unconditionally writing None (correct ContextVar cleanup pattern).
        request_token = _request_id_var.set(None)
        user_token = _user_id_var.set(None)

        # Extract the request ID from FastMCP context. There is no session ID to
        # capture: the 2026-07-28 MCP protocol is sessionless.
        fastmcp_ctx = None
        try:
            fastmcp_ctx = getattr(context, "fastmcp_context", None)
        except Exception:
            pass

        if fastmcp_ctx is not None:
            try:
                request_id = getattr(fastmcp_ctx, "request_id", None)
                if request_id is not None:
                    _request_id_var.set(str(request_id))
            except Exception:
                pass

        # Extract user identity (from auth or transport metadata)
        try:
            request_context = getattr(context, "request_context", None)
            if request_context is not None:
                meta = getattr(request_context, "meta", None)
                if meta is not None:
                    if isinstance(meta, dict):
                        user_id = meta.get("user_id")
                    else:
                        user_id = getattr(meta, "user_id", None)
                    if user_id is not None:
                        _user_id_var.set(str(user_id))
        except Exception:
            pass

        try:
            return await call_next(context)
        finally:
            # Restore ContextVars to their previous values
            _request_id_var.reset(request_token)
            _user_id_var.reset(user_token)


class ToolErrorMiddleware(Middleware):
    """Mark tool results that carry ``error`` or ``error_code`` with ``isError``."""

    async def on_call_tool(self, context: Any, call_next: Any) -> ToolResult:
        result = await call_next(context)
        data = result.structured_content
        if (
            not result.is_error
            and isinstance(data, dict)
            and ("error" in data or "error_code" in data)
        ):
            return ToolResult(
                content=result.content,
                structured_content=data,
                meta=result.meta,
                is_error=True,
            )
        return result


class RequestClientConfigurationMiddleware(Middleware):
    """Run each tool call with a Kubernetes client configuration chosen for its caller.

    Opt-in hook for multi-user deployments. For every tool call the *provider*
    is given the verified access token (``None`` when the transport carries
    none, e.g. stdio) and returns the ``kubernetes.client.Configuration`` to
    use, for example one carrying the caller's own bearer token so the API
    server applies the caller's RBAC. Every API helper and ``TrainerClient`` in
    :mod:`kubeflow_mcp.common.utils` is then built from it for that call only.

    Fails closed: if the provider returns ``None`` or raises, the call is
    refused with ``PERMISSION_DENIED`` and never falls back to the server's own
    kubeconfig or ServiceAccount. The provider may be sync or async.

    Usage::

        def provider(token):
            if token is None:
                return None
            cfg = kubernetes.client.Configuration()
            cfg.host = "https://kubernetes.default.svc"
            cfg.ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
            cfg.api_key = {"authorization": f"Bearer {token.token}"}
            return cfg

        mcp = create_server(auth_provider=..., client_configuration_provider=provider)
    """

    def __init__(self, provider: ClientConfigurationProvider) -> None:
        self._provider = provider

    async def on_call_tool(self, context: Any, call_next: Any) -> ToolResult:
        try:
            configuration = self._provider(get_access_token())
            if inspect.isawaitable(configuration):
                configuration = await configuration
        except Exception as exc:
            # The exception text may echo the token, so only its type is logged.
            logger.warning(
                "client_configuration_provider_failed",
                extra={"error_type": type(exc).__qualname__},
            )
            configuration = None
        if configuration is None:
            logger.warning(
                "tool_call_refused",
                extra={
                    "audit": True,
                    "tool": getattr(getattr(context, "message", None), "name", None),
                    "error_code": ErrorCode.PERMISSION_DENIED,
                },
            )
            return ToolResult(
                structured_content=ToolError(
                    error=(
                        "No Kubernetes credentials could be established for this caller; "
                        "the request was refused rather than run as the server."
                    ),
                    error_code=ErrorCode.PERMISSION_DENIED,
                ).model_dump(exclude_none=True),
                is_error=True,
            )
        with request_client_configuration(configuration):
            return await call_next(context)
