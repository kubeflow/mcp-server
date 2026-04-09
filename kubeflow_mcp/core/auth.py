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

"""HTTP transport authentication for kubeflow-mcp.

Two modes, selected by config:

1. **API key** (``auth_token``): Simple bearer token comparison for dev/staging.
   Set via ``--auth-token``, ``KUBEFLOW_MCP_AUTH_TOKEN``, or config file.

2. **JWT** (``jwks_uri``): Production-grade JWT verification via FastMCP's
   ``JWTVerifier``. Validates signatures, expiry, issuer, and audience.

Ignored when ``transport=stdio`` (stdio inherits OS-level security).
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp.server.auth import AccessToken, TokenVerifier

if TYPE_CHECKING:
    from kubeflow_mcp.core.config import AuthConfig

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """Authentication context for requests."""

    user: str | None = None
    groups: list[str] | None = None
    impersonate: str | None = None


class APIKeyVerifier(TokenVerifier):
    """Simple bearer token verifier for dev/staging deployments.

    Compares the incoming ``Authorization: Bearer <token>`` header against
    a pre-shared secret using constant-time comparison.
    """

    def __init__(self, expected_token: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._expected = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token, self._expected):
            return AccessToken(
                token=token,
                client_id="api-key",
                scopes=[],
                claims={"auth_method": "api_key"},
            )
        return None


def build_auth_provider(auth_config: AuthConfig) -> TokenVerifier | None:
    """Build the appropriate auth provider from config, or None if auth is disabled."""
    if auth_config.jwks_uri:
        from fastmcp.server.auth.providers.jwt import JWTVerifier

        logger.info("HTTP auth: JWT verification via %s", auth_config.jwks_uri)
        return JWTVerifier(
            jwks_uri=auth_config.jwks_uri,
            issuer=auth_config.issuer,
            audience=auth_config.audience,
        )

    if auth_config.auth_token:
        logger.info("HTTP auth: API key verification enabled")
        return APIKeyVerifier(expected_token=auth_config.auth_token)

    return None
