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
) -> None:
    """Register the RFC 8615 well-known MCP Server Card auto-discovery endpoint (ROADMAP Phase 4)."""
    server_card: dict[str, Any] = {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        "name": "io.github.kubeflow/mcp-server",
        "title": "Kubeflow MCP Server",
        "version": __version__,
        "description": "AI-assisted Kubeflow Training and ML workflows via Model Context Protocol.",
        "transport": {
            "type": "streamable-http",
            "endpoint": "/mcp",
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
            "clients": list(clients or ["trainer"]),
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
    async def server_card_route(_request: Request) -> Response:
        return JSONResponse(server_card)
