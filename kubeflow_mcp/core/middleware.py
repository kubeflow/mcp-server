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
into sync wrappers.  This module uses :mod:`contextvars` to capture session,
request, and user identity from the async middleware layer so that the
synchronous ``_audit_wrap`` in :mod:`kubeflow_mcp.core.server` can read them
without depending on DI.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ContextVars populated by AuditIdentityMiddleware, read by _audit_wrap
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_session_id", default=None
)
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_request_id", default=None
)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_user_id", default=None
)


def get_mcp_session_id() -> str | None:
    """Return the MCP session ID for the current request, or None."""
    return _session_id_var.get()


def get_mcp_request_id() -> str | None:
    """Return the MCP request ID for the current request, or None."""
    return _request_id_var.get()


def get_user_id() -> str | None:
    """Return the user identity for the current request, or None."""
    return _user_id_var.get()


class AuditIdentityMiddleware:
    """FastMCP-compatible middleware that captures identity into ContextVars.

    Extracts ``session_id``, ``request_id``, and optionally ``user_id``
    from the FastMCP ``MiddlewareContext`` and stores them in module-level
    :class:`contextvars.ContextVar` instances.  Downstream sync code
    (e.g. ``_audit_wrap``) can retrieve these values via the public
    ``get_mcp_*`` helpers without needing async or DI.

    Usage::

        from kubeflow_mcp.core.middleware import AuditIdentityMiddleware
        mcp = FastMCP("kubeflow-mcp-server")
        mcp.add_middleware(AuditIdentityMiddleware)
    """

    async def __call__(self, context: Any, call_next: Any) -> Any:
        """Capture identity from context, then delegate to the next handler."""
        # Use tokens so reset() restores the *previous* value rather than
        # unconditionally writing None (correct ContextVar cleanup pattern).
        session_token = _session_id_var.set(None)
        request_token = _request_id_var.set(None)
        user_token = _user_id_var.set(None)

        # Extract session + request IDs from FastMCP context
        fastmcp_ctx = None
        try:
            fastmcp_ctx = getattr(context, "fastmcp_context", None)
        except Exception:
            pass

        if fastmcp_ctx is not None:
            try:
                session_id = getattr(fastmcp_ctx, "session_id", None)
                if session_id is not None:
                    _session_id_var.set(str(session_id))
            except Exception:
                pass
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
            _session_id_var.reset(session_token)
            _request_id_var.reset(request_token)
            _user_id_var.reset(user_token)
