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
   ``JWTVerifier``. Checks the signature against the JWKS, the issuer, the
   audience, and that the token carries an ``exp`` that has not passed.
   ``issuer`` and ``audience`` are required with ``jwks_uri``: without them
   every token the identity provider signs, for any application, would be
   accepted.

Ignored when ``transport=stdio`` (stdio inherits OS-level security).
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.providers.jwt import JWTVerifier
from joserfc.errors import JoseError
from joserfc.jwt import JWTClaimsRegistry

if TYPE_CHECKING:
    from kubeflow_mcp.core.config import AuthConfig

logger = logging.getLogger(__name__)

# FastMCP checks ``exp`` only when the token has one and never looks at ``nbf``,
# so a token without ``exp`` would never expire. The minute of clock skew only
# applies to ``nbf`` and ``iat``: FastMCP rejects a token whose ``exp`` has passed,
# with no leeway, before this check runs.
_TIME_CLAIMS = JWTClaimsRegistry(leeway=60, exp={"essential": True})


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
        if hmac.compare_digest(token.encode(), self._expected.encode()):
            return AccessToken(
                token=token,
                client_id="api-key",
                scopes=[],
                claims={"auth_method": "api_key"},
            )
        return None


class StrictJWTVerifier(JWTVerifier):
    """``JWTVerifier`` that also rejects tokens with no ``exp``, or before ``nbf``/``iat``."""

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = await super().load_access_token(token)
        if access is None:
            return None
        try:
            _TIME_CLAIMS.validate(access.claims)
        except JoseError as e:
            logger.warning(
                "Bearer token rejected for client %s: %s",
                access.client_id,
                e.description or e.error,
            )
            return None
        return access


def build_auth_provider(auth_config: AuthConfig) -> TokenVerifier | None:
    """Build the appropriate auth provider from config, or None if auth is disabled.

    Raises:
        ValueError: ``jwks_uri`` is only whitespace, or is set without both
            ``issuer`` and ``audience`` (a value that is only whitespace counts
            as missing).
    """
    if auth_config.jwks_uri:
        # A secret created with --from-file ends in a newline, which would make
        # every JWKS fetch fail.
        jwks_uri = auth_config.jwks_uri.strip()
        if not jwks_uri:
            raise ValueError(
                "KUBEFLOW_MCP_JWKS_URI (auth.jwks_uri in the config file) is blank. "
                "Set it to your identity provider's JWKS URL, or unset it."
            )
        issuer = (auth_config.issuer or "").strip()
        audience = (auth_config.audience or "").strip()
        if not (issuer and audience):
            raise ValueError(
                "JWT auth needs an issuer and an audience: set KUBEFLOW_MCP_JWT_ISSUER and "
                "KUBEFLOW_MCP_JWT_AUDIENCE (or auth.issuer and auth.audience in the config "
                "file). Without them the server accepts any token the JWKS signs, including "
                "tokens issued to other applications."
            )

        logger.info(
            "HTTP auth: JWT verification via %s (issuer=%s, audience=%s)",
            jwks_uri,
            issuer,
            audience,
        )
        return StrictJWTVerifier(
            jwks_uri=jwks_uri,
            issuer=issuer,
            audience=audience,
        )

    if auth_config.auth_token:
        logger.info("HTTP auth: API key verification enabled")
        return APIKeyVerifier(expected_token=auth_config.auth_token)

    return None
