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
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Iterator
from typing import Any

from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult

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


@contextlib.contextmanager
def audit_identity(user_id: str | None) -> Iterator[None]:
    """Bind the caller identity for the duration of a block.

    ``AuditIdentityMiddleware`` only runs for MCP requests. HTTP endpoints
    mounted beside the MCP app — notably A2A delegation — authenticate their
    own callers, so they bind the verified identity here rather than leaving
    the audit log to record the call with no subject attached.

    Reuses the same ContextVar ``_audit_wrap`` already reads, so there is one
    identity path rather than two that can disagree.
    """
    token = _user_id_var.set(str(user_id) if user_id is not None else None)
    try:
        yield
    finally:
        _user_id_var.reset(token)


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
