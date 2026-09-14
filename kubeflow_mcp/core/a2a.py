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

"""A2A (Agent-to-Agent) protocol endpoints for horizontal delegation."""

import logging
import re

from fastmcp import FastMCP
from fastmcp.server.auth import TokenVerifier
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kubeflow_mcp.trainer.api.training import fine_tune

logger = logging.getLogger(__name__)


def register_a2a_routes(
    mcp: FastMCP,
    clients: list[str],
    persona: str,
    auth_provider: TokenVerifier | None = None,
) -> None:
    """Register A2A protocol routes on the FastMCP application."""

    @mcp.custom_route("/.well-known/agent-card.json", methods=["GET"], include_in_schema=False)
    async def agent_card(_request: Request) -> Response:
        skills = []
        if "trainer" in clients:
            skills.append("fine-tune")
            skills.append("distributed-training")
        if "optimizer" in clients:
            skills.append("hyperparameter-optimization")

        card = {
            "name": "kubeflow-mcp-server",
            "description": f"Kubeflow AI Agent (Persona: {persona})",
            "skills": skills,
            "endpoints": {"a2a": "/a2a"},
        }
        return JSONResponse(card)

    @mcp.custom_route("/a2a", methods=["POST"], include_in_schema=False)
    async def a2a_endpoint(request: Request) -> Response:
        # 1. Auth check
        if auth_provider:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse({"error": "Missing or invalid token"}, status_code=401)
            token = auth_header.split(" ")[1]
            access_token = await auth_provider.verify_token(token)
            if not access_token:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
                status_code=400,
            )

        method = body.get("method")
        params = body.get("params", {})
        msg_id = body.get("id")

        if method != "tasks/send":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": msg_id,
                },
                status_code=404,
            )

        task_str = params.get("task", "")

        # Minimum viable parser: "fine-tune model X on dataset Y"
        match = re.search(r"fine-tune model (\S+) on dataset (\S+)", task_str, re.IGNORECASE)
        if match:
            model = match.group(1)
            dataset = match.group(2)

            try:
                # Direct tool call
                # In MVP we assume confirmed=True since A2A is direct task delegation
                result = fine_tune(model=model, dataset=dataset, confirmed=True)
                return JSONResponse({"jsonrpc": "2.0", "result": result, "id": msg_id})
            except Exception as e:
                return JSONResponse(
                    {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": msg_id},
                    status_code=500,
                )

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Could not parse task description"},
                "id": msg_id,
            },
            status_code=400,
        )
