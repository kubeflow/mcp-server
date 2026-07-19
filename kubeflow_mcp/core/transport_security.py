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

"""DNS rebinding protection for the HTTP/SSE transports.

FastMCP's streamable-HTTP transport does not validate the ``Host``/``Origin``
headers by default, so a page loaded in a victim's browser could use DNS
rebinding to reach a locally-bound server. This module rejects requests whose
``Host``/``Origin`` are not on an allowlist before they reach the MCP handler.

The header validation itself is delegated to the MCP SDK's reference
implementation (:class:`mcp.server.transport_security.TransportSecurityMiddleware`)
so behaviour matches the protocol conformance suite: an invalid ``Host`` yields
``421``, an invalid ``Origin`` yields ``403``, and a POST without a JSON
``Content-Type`` yields ``400``.

The middleware is implemented as a pure-ASGI wrapper (rather than
``BaseHTTPMiddleware``) so it does not buffer the streaming SSE responses the
transport relies on — it only inspects request headers and either short-circuits
with an error response or passes the untouched send/receive channels through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from kubeflow_mcp.core.config import SecurityConfig

# Loopback defaults used when no explicit allowlist is configured. These cover
# both the IPv4 (127.0.0.1) and IPv6 (::1) loopback binds FastMCP may use, plus
# the ``localhost`` name. IPv6 hosts arrive bracketed in the Host/Origin header
# (e.g. ``[::1]:8000``), so the allowlist must use the bracketed form. The ``:*``
# suffix matches any port. An absent Origin (typical for non-browser MCP clients)
# is always allowed by the SDK validator.
_DEFAULT_ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
    "[::1]",
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
]
_DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://[::1]",
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]


def build_transport_security_settings(
    config: SecurityConfig,
) -> TransportSecuritySettings:
    """Translate our ``SecurityConfig`` into the SDK's transport settings.

    Falls back to loopback defaults when the allowlists are left empty so
    protection is on out of the box for the default bind.
    """
    allowed_hosts = config.allowed_hosts or _DEFAULT_ALLOWED_HOSTS
    allowed_origins = config.allowed_origins or _DEFAULT_ALLOWED_ORIGINS
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=config.enable_dns_rebinding_protection,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


# Paths that are exempt from Host/Origin validation. These probes are
# unauthenticated by design, return no sensitive data, and must respond
# regardless of the Host header a caller uses to reach them — notably
# kubelet, which sends liveness/readiness probes with `Host: <pod-ip>:<port>`
# rather than a loopback name. Rejecting those probes causes CrashLoopBackOff
# in any Kubernetes/OpenShift deployment (see issue #91).
_EXEMPT_PATHS = frozenset({"/health", "/ready"})


class DNSRebindingProtectionMiddleware:
    """Pure-ASGI middleware that validates Host/Origin before the MCP handler."""

    def __init__(self, app: ASGIApp, settings: TransportSecuritySettings) -> None:
        self.app = app
        self._validator = TransportSecurityMiddleware(settings)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        # Reading headers off the scope does not consume the receive channel, so
        # the downstream app still gets the full request body.
        request = Request(scope, receive)
        is_post = scope.get("method") == "POST"
        error = await self._validator.validate_request(request, is_post=is_post)
        if error is not None:
            await error(scope, receive, send)
            return

        await self.app(scope, receive, send)
