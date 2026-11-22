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

Three modes, selected by config:

1. **API key** (``auth_token``): Simple bearer token comparison for dev/staging.
   Set via ``--auth-token``, ``KUBEFLOW_MCP_AUTH_TOKEN``, or config file.

2. **JWT** (``jwks_uri``): Production-grade JWT verification via FastMCP's
   ``JWTVerifier``. Validates signatures, expiry, issuer, and audience.

3. **Native OIDC / OAuth 2.1 Discovery** (``oidc_issuer`` or ``issuer``):
   RFC 8414 and OpenID Connect Discovery auto-resolves ``jwks_uri`` from
   the identity provider's well-known configuration endpoint (ROADMAP Phase 4).

Ignored when ``transport=stdio`` (stdio inherits OS-level security).
"""

from __future__ import annotations

import hmac
import logging
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier

if TYPE_CHECKING:
    from kubeflow_mcp.core.config import AuthConfig

logger = logging.getLogger(__name__)


def _validate_metadata(
    data: Any,
    endpoint_url: str,
    expected_issuer: str,
) -> dict[str, Any]:
    """Validate that discovered IdP metadata contains a valid HTTPS jwks_uri and matching issuer."""
    if not isinstance(data, dict):
        raise ValueError(f"OIDC discovery response at {endpoint_url} must be a JSON object")

    jwks_uri = data.get("jwks_uri")
    if not (isinstance(jwks_uri, str) and jwks_uri.startswith("https://")):
        raise ValueError(
            f"OIDC discovery at {endpoint_url} returned metadata without a valid HTTPS 'jwks_uri'"
        )

    meta_issuer = data.get("issuer")
    if not isinstance(meta_issuer, str) or meta_issuer.rstrip("/") != expected_issuer:
        raise ValueError(
            f"OIDC discovery at {endpoint_url} returned issuer {meta_issuer!r}, "
            f"which does not match expected issuer {expected_issuer!r}"
        )

    return data


def discover_oidc_configuration(
    issuer: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Perform RFC 8414 / OpenID Connect Discovery to discover IdP metadata.

    Attempts ``/.well-known/openid-configuration`` first, then falls back to
    RFC 8414 OAuth 2.0 Authorization Server metadata.

    Returns the parsed JSON dictionary containing at least ``jwks_uri`` and matching ``issuer``.

    Raises:
        ValueError: If issuer URL is invalid, non-HTTPS, endpoints return errors,
            or metadata lacks a valid HTTPS ``jwks_uri`` or has a mismatched issuer.
    """
    if not issuer or not isinstance(issuer, str):
        raise ValueError("OIDC issuer must be a non-empty URL string")

    clean_issuer = issuer.strip().rstrip("/")
    if not clean_issuer.startswith("https://"):
        raise ValueError(f"OIDC issuer URL must start with https://, got: {issuer!r}")

    parsed = urllib.parse.urlsplit(clean_issuer)
    if not parsed.netloc:
        raise ValueError(f"OIDC issuer URL is missing host: {issuer!r}")

    origin = f"{parsed.scheme}://{parsed.netloc}"
    issuer_path = parsed.path.rstrip("/")

    # RFC 8414 Section 3 defines authorization server metadata path by inserting
    # /.well-known/oauth-authorization-server between the origin and path component,
    # whereas OpenID Connect Discovery concatenates /.well-known/openid-configuration.
    urls_to_try = [
        f"{clean_issuer}/.well-known/openid-configuration",
        f"{origin}/.well-known/oauth-authorization-server{issuer_path}",
    ]

    last_error: Exception | None = None
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout)
        close_client = True

    try:
        for url in urls_to_try:
            try:
                resp = client.get(url, headers={"Accept": "application/json"})
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as e:
                        raise ValueError(
                            f"OIDC discovery response at {url} is not valid JSON: {e}"
                        ) from e
                    return _validate_metadata(data, url, clean_issuer)
                elif resp.status_code != 404:
                    resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
    finally:
        if close_client:
            client.close()

    error_msg = f"Failed to discover OIDC configuration for issuer '{clean_issuer}'"
    if last_error:
        error_msg += f": {last_error}"
    raise ValueError(error_msg)


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


def build_auth_provider(
    auth_config: AuthConfig,
    *,
    http_client: httpx.Client | None = None,
) -> TokenVerifier | None:
    """Build the appropriate auth provider from config, or None if auth is disabled."""
    if auth_config.jwks_uri:
        from fastmcp.server.auth.providers.jwt import JWTVerifier

        logger.info("HTTP auth: JWT verification via %s", auth_config.jwks_uri)
        return JWTVerifier(
            jwks_uri=auth_config.jwks_uri,
            issuer=auth_config.issuer,
            audience=auth_config.audience,
        )

    effective_issuer = auth_config.oidc_issuer or auth_config.issuer
    if effective_issuer:
        from fastmcp.server.auth.providers.jwt import JWTVerifier

        metadata = discover_oidc_configuration(effective_issuer, client=http_client)
        discovered_jwks_uri = metadata["jwks_uri"]
        discovered_issuer = metadata.get("issuer")
        if not isinstance(discovered_issuer, str) or not discovered_issuer:
            raise ValueError(
                f"OIDC discovery metadata for '{effective_issuer}' lacks a valid 'issuer'"
            )

        if auth_config.issuer and auth_config.issuer.rstrip("/") != discovered_issuer.rstrip("/"):
            raise ValueError(
                f"Configured issuer '{auth_config.issuer}' does not match "
                f"discovered metadata issuer '{discovered_issuer}'"
            )

        expected_issuer = auth_config.issuer or discovered_issuer

        logger.info(
            "HTTP auth: Native OIDC discovery succeeded for issuer '%s' (jwks_uri=%s)",
            effective_issuer,
            discovered_jwks_uri,
        )
        return JWTVerifier(
            jwks_uri=discovered_jwks_uri,
            issuer=expected_issuer,
            audience=auth_config.audience,
        )

    if auth_config.auth_token:
        logger.info("HTTP auth: API key verification enabled")
        return APIKeyVerifier(expected_token=auth_config.auth_token)

    return None
