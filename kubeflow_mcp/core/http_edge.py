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

"""HTTP liveness and readiness probes, and discovery routes for network transports."""

import os
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kubeflow_mcp import __version__


def register_probe_routes(mcp: FastMCP, *, is_ready: bool = True) -> None:
    """Register unauthenticated probes on the FastMCP HTTP application."""

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "healthy"})

    @mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
    async def ready(_request: Request) -> Response:
        if not is_ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})


def register_server_card_route(
    mcp: FastMCP,
    *,
    persona: str = "readonly",
    clients: list[str] | None = None,
    mode: str = "full",
    transport: str = "http",
) -> None:
    """Register the RFC 8615 well-known MCP Server Card auto-discovery endpoint (ROADMAP Phase 4)."""
    if transport == "sse":
        transport_type = "sse"
        endpoint = "/sse"
    else:
        transport_type = "streamable-http"
        endpoint = "/mcp"

    effective_clients = ["trainer"] if clients is None else list(clients)

    server_card: dict[str, Any] = {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        "name": "io.github.kubeflow/mcp-server",
        "title": "Kubeflow MCP Server",
        "version": __version__,
        "description": "AI-assisted Kubeflow Training and ML workflows via Model Context Protocol.",
        "transport": {
            "type": transport_type,
            "endpoint": endpoint,
        },
        "capabilities": {
            "tools": True,
            "resources": True,
            "prompts": False,
            "logging": True,
        },
        "metadata": {
            "persona": persona,
            "mode": mode,
            "clients": effective_clients,
            "personas_supported": [
                "readonly",
                "data-scientist",
                "ml-engineer",
                "platform-admin",
            ],
            "modes_supported": ["full", "progressive", "semantic"],
        },
    }

    @mcp.custom_route("/.well-known/mcp.json", methods=["GET"], include_in_schema=False)
    async def server_card_route(request: Request) -> Response:
        # Public origin derivation:
        # 1. Reuse explicitly configured public base URL if available
        public_url = os.environ.get("KUBEFLOW_MCP_PUBLIC_URL") or os.environ.get(
            "KUBEFLOW_MCP_BASE_URL"
        )
        if public_url:
            base_url = public_url.rstrip("/")
        else:
            # 2. Check X-Forwarded headers if present from reverse proxy/ingress
            forwarded_proto = request.headers.get("x-forwarded-proto")
            forwarded_host = request.headers.get("x-forwarded-host")
            if forwarded_proto and forwarded_host:
                proto = forwarded_proto.split(",")[0].strip()
                host = forwarded_host.split(",")[0].strip()
                base_url = f"{proto}://{host}".rstrip("/")
            else:
                base_url = str(request.base_url).rstrip("/")

        card = dict(server_card)
        card["remotes"] = [
            {
                "type": transport_type,
                "url": f"{base_url}{endpoint}",
            }
        ]
        return JSONResponse(card)
