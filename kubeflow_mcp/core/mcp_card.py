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

"""MCP Server Card for pre-connect HTTP discovery (SEP-2127).

A Server Card is a small, static JSON document that lets a client learn who
this server is and where to connect *before* paying for a full initialization
handshake.

Two deliberate choices, both taken from the SEP:

**No primitives.** SEP-2127 excludes tools, resources, and prompts from the
card by design. The surface this server exposes varies by persona, policy
file, and which client modules loaded, so a static document cannot represent
it honestly. Worse, a client that trusted a static tool list for an
access-control decision would be trusting a document that the runtime is free
to contradict. Primitives stay discoverable through ``tools/list`` with the
caller's own identity attached.

**Derived from server.json.** The SEP defines the card as ``server.json``
minus ``packages`` and with its own ``$schema``. Reading the repository's
existing ``server.json`` keeps the registry entry and the served card from
drifting apart, instead of maintaining the same identity in two places.

The card is served at two paths because the spec has not settled on one:

- ``/mcp/server-card`` — the MCP endpoint URL plus ``/server-card``, the
  default reserved by the current SEP-2127 draft.
- ``/.well-known/mcp-server-card`` — the ``.well-known`` URI written into the
  SEP body, and the suffix queued for IANA registration.

Serving both costs one extra route and keeps the server readable by clients
written against either revision. Both are unauthenticated: the card carries
only public metadata, and discovery is the one thing that has to work before
a caller holds a credential.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kubeflow_mcp import __version__

logger = logging.getLogger(__name__)

SERVER_CARD_SCHEMA = "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json"

# The SEP-2127 draft reserves the MCP endpoint URL plus "/server-card"; the SEP
# body documents a .well-known URI. Both are served until the spec converges.
SERVER_CARD_PATH = "/mcp/server-card"
SERVER_CARD_WELL_KNOWN_PATH = "/.well-known/mcp-server-card"

# Server cards are public, immutable-ish metadata: cache them and let browser
# clients read them cross-origin.
_CARD_HEADERS = {
    "Cache-Control": "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET",
    "Access-Control-Allow-Headers": "Content-Type",
}

# Fallback identity, used only when server.json cannot be read (e.g. an
# installed wheel that did not ship it). Keep in sync with server.json.
_FALLBACK_NAME = "io.github.kubeflow/mcp-server"
_FALLBACK_TITLE = "Kubeflow MCP Server"
_FALLBACK_DESCRIPTION = (
    "AI-assisted Kubeflow Training via Model Context Protocol — plan, submit, "
    "monitor, and manage TrainJobs."
)
_FALLBACK_REPOSITORY = {
    "url": "https://github.com/kubeflow/mcp-server",
    "source": "github",
}

# Fields the Server Card shares with server.json. "packages" is deliberately
# absent: it describes how to run a local server, which SEP-2127 keeps out of
# .well-known for both RFC 8615 and security reasons.
_INHERITED_FIELDS = ("name", "version", "description", "title", "websiteUrl", "repository", "icons")


def _server_json_path() -> Path:
    """Locate server.json at the repository root, if it is present."""
    return Path(__file__).resolve().parents[2] / "server.json"


@lru_cache(maxsize=1)
def _load_server_json() -> dict[str, Any]:
    """Read server.json once, tolerating its absence in installed packages."""
    path = _server_json_path()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            logger.warning("server.json is not a JSON object, ignoring it")
            return {}
        return data
    except FileNotFoundError:
        logger.debug("server.json not found at %s, using fallback identity", path)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read server.json (%s), using fallback identity", exc)
        return {}


def build_server_card(mcp_url: str | None = None) -> dict[str, Any]:
    """Build the SEP-2127 Server Card.

    Args:
        mcp_url: Absolute URL of the MCP endpoint, used to populate ``remotes``.
            Omitted when the public URL is unknown, since a relative URL there
            would be useless to the aggregators the card exists to serve.

    Returns:
        A Server Card dict. Never contains tools, resources, or prompts.
    """
    source = _load_server_json()

    card: dict[str, Any] = {"$schema": SERVER_CARD_SCHEMA}
    for field in _INHERITED_FIELDS:
        value = source.get(field)
        if value is not None:
            card[field] = value

    # Required fields must be present even when server.json is unavailable.
    card.setdefault("name", _FALLBACK_NAME)
    card.setdefault("version", __version__)
    card.setdefault("description", _FALLBACK_DESCRIPTION)
    card.setdefault("title", _FALLBACK_TITLE)
    card.setdefault("repository", dict(_FALLBACK_REPOSITORY))

    if mcp_url:
        card["remotes"] = [{"type": "streamable-http", "url": mcp_url}]

    return card


def _resolve_mcp_url(request: Request) -> str | None:
    """Derive the absolute MCP endpoint URL from the incoming request.

    Returns None when the host is unknown, so the card omits ``remotes``
    rather than advertising an endpoint a remote client cannot reach.
    """
    base = str(request.base_url).rstrip("/")
    if not base:
        return None
    return f"{base}/mcp"


def register_server_card_routes(mcp: FastMCP) -> None:
    """Register the unauthenticated Server Card routes on the HTTP app."""

    async def _serve_card(request: Request) -> Response:
        card = build_server_card(mcp_url=_resolve_mcp_url(request))
        return JSONResponse(card, headers=dict(_CARD_HEADERS))

    @mcp.custom_route(SERVER_CARD_PATH, methods=["GET"], include_in_schema=False)
    async def server_card(request: Request) -> Response:
        return await _serve_card(request)

    @mcp.custom_route(SERVER_CARD_WELL_KNOWN_PATH, methods=["GET"], include_in_schema=False)
    async def server_card_well_known(request: Request) -> Response:
        return await _serve_card(request)
