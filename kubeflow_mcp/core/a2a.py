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

"""A2A Agent Card and task delegation endpoint.

MCP is the vertical interface — an agent reaching down to tools. A2A is the
horizontal one — an orchestrator (LangGraph, ADK, Strands) delegating a unit
of work to this server as a peer. This module adds the horizontal half:

- ``/.well-known/agent-card.json`` — unauthenticated discovery.
- ``/a2a`` — authenticated JSON-RPC 2.0 delegation.

Four design decisions worth stating, because each one is a place this could
have gone wrong:

**Structured payloads, not natural language.** Delegation carries an explicit
``{"tool": ..., "arguments": {...}}``, the same shape as an MCP tool call. The
orchestrator already has a planner; asking this server to re-derive intent
from prose would duplicate that badly and make the mapping from request to
side effect impossible to audit. The payload rides in an A2A ``DataPart``,
which is exactly the part type the spec provides for structured JSON, so the
envelope stays standard while the contents stay unambiguous.

**The confirm gate is the caller's to satisfy.** Mutating tools take
``confirmed`` and return a preview until it is set. Arguments are passed
through untouched, so a delegated call defaults to a preview exactly like a
direct one. Forcing ``confirmed=True`` here would let any authenticated
caller submit a training job in one hop, which is precisely the gate's job to
prevent.

**Persona gating is inherited, not re-implemented.** The endpoint is handed
the same filtered, audit-wrapped callables that were registered as MCP tools,
after persona, policy, and read-only filtering have all run. A tool the
persona cannot reach is not in the map, so it cannot be delegated. There is
no second copy of the access rules to drift.

**The Agent Card describes capabilities, not the tool surface.** It is served
without auth, so it advertises what this server does ("fine-tuning", "job
monitoring") rather than enumerating tool names and schemas. The concrete
surface stays behind auth on ``/a2a`` and behind ``tools/list`` on MCP, where
the caller's identity is known.

Delegation is stateless: ``message/send`` runs the tool and returns its result
inline. No task store, no SSE. That keeps the endpoint horizontally scalable,
and streaming can be added later for a consumer that actually needs it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kubeflow_mcp import __version__

logger = logging.getLogger(__name__)

AGENT_CARD_PATH = "/.well-known/agent-card.json"
A2A_PATH = "/a2a"

# Revision of the A2A specification this implementation was written against.
A2A_PROTOCOL_VERSION = "0.3.0"

# "message/send" is the standard A2A method. "tasks/send" is the older name,
# accepted so clients written against the earlier revision keep working.
_SEND_METHODS = frozenset({"message/send", "tasks/send"})

# JSON-RPC 2.0 reserved codes.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
# A2A-defined: the request was understood but this agent will not serve it.
_UNSUPPORTED_OPERATION = -32004

_CARD_HEADERS = {
    "Cache-Control": "public, max-age=3600",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET",
    "Access-Control-Allow-Headers": "Content-Type",
}

# High-level capabilities per client module. Deliberately coarse: this is
# public, so it names what the server can do, never which tools implement it
# or what arguments they take. A client module may override this by exporting
# its own ``A2A_SKILLS``, the same way it exports INSTRUCTION_SECTIONS.
_CLIENT_SKILLS: dict[str, list[dict[str, Any]]] = {
    "trainer": [
        {
            "id": "fine-tuning",
            "name": "Model fine-tuning",
            "description": (
                "Fine-tune a foundation model on a dataset using Kubeflow Trainer, "
                "including LoRA configuration and resource planning."
            ),
            "tags": ["training", "fine-tuning", "llm"],
        },
        {
            "id": "distributed-training",
            "name": "Distributed training",
            "description": (
                "Submit and manage multi-node distributed training jobs on Kubernetes."
            ),
            "tags": ["training", "distributed", "kubernetes"],
        },
        {
            "id": "job-monitoring",
            "name": "Training job monitoring",
            "description": (
                "Inspect status, logs, and events for training jobs, and diagnose failures."
            ),
            "tags": ["monitoring", "observability", "logs"],
        },
    ],
    "optimizer": [
        {
            "id": "hyperparameter-optimization",
            "name": "Hyperparameter optimization",
            "description": "Run and monitor hyperparameter search experiments.",
            "tags": ["optimization", "hpo", "tuning"],
        },
    ],
    "hub": [
        {
            "id": "model-registry",
            "name": "Model registry",
            "description": "Discover and inspect registered models and their versions.",
            "tags": ["registry", "models", "metadata"],
        },
    ],
}


def _error(
    code: int,
    message: str,
    msg_id: Any = None,
    *,
    status_code: int = 200,
) -> JSONResponse:
    """Build a JSON-RPC error response.

    JSON-RPC carries application-level failures in the body with HTTP 200;
    reserving non-200 for transport and auth failures keeps the two layers
    from being confused by intermediaries.
    """
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": msg_id},
        status_code=status_code,
    )


def _safe_id(body: Any) -> Any:
    """Extract the JSON-RPC id defensively.

    The id is echoed in error responses, including errors about the request
    being malformed, so reading it must not itself raise on a non-object body.
    """
    if isinstance(body, Mapping):
        msg_id = body.get("id")
        if isinstance(msg_id, (str, int)) or msg_id is None:
            return msg_id
    return None


def build_skills(
    loaded_clients: list[str], loaded_modules: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Build the card's skill list from the loaded client modules.

    Skills reflect what this deployment loaded, not what the caller's persona
    may reach — the card is public, so it stays persona-agnostic.
    """
    skills: list[dict[str, Any]] = []
    for client in loaded_clients:
        module = (loaded_modules or {}).get(client)
        declared = getattr(module, "A2A_SKILLS", None) if module is not None else None
        for skill in declared or _CLIENT_SKILLS.get(client, []):
            skills.append(dict(skill))
    return skills


def build_agent_card(
    *,
    url: str,
    loaded_clients: list[str],
    loaded_modules: Mapping[str, Any] | None = None,
    auth_required: bool = False,
) -> dict[str, Any]:
    """Build the A2A Agent Card.

    Args:
        url: Absolute URL of the ``/a2a`` endpoint. Orchestrators use this to
            call the agent, so it must be absolute rather than a path.
        loaded_clients: Client modules loaded by the server.
        loaded_modules: Imported client modules, consulted for ``A2A_SKILLS``.
        auth_required: Whether ``/a2a`` enforces a bearer token, which decides
            if the card advertises a security scheme.

    Returns:
        A schema-complete Agent Card.
    """
    card: dict[str, Any] = {
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "name": "kubeflow-mcp-server",
        "description": (
            "Kubeflow training agent. Delegates model fine-tuning, distributed "
            "training, and job monitoring to a Kubeflow cluster."
        ),
        "url": url,
        "version": __version__,
        "provider": {
            "organization": "Kubeflow",
            "url": "https://github.com/kubeflow/mcp-server",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": build_skills(loaded_clients, loaded_modules),
    }

    if auth_required:
        card["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearerAuth": []}]

    return card


def extract_delegation(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull the tool name and arguments out of A2A ``message/send`` params.

    The payload is expected in a ``DataPart``::

        {"message": {"role": "user", "parts": [
            {"kind": "data", "data": {"tool": "fine_tune", "arguments": {...}}}
        ]}}

    Raises:
        ValueError: With a caller-facing message when the shape is wrong. The
            caller sees invalid-params rather than a 500.
    """
    message = params.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("params.message must be an object")

    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("params.message.parts must be a non-empty array")

    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if part.get("kind") != "data":
            continue
        data = part.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("DataPart.data must be an object")

        tool = data.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("DataPart.data.tool must be a non-empty string")

        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise ValueError("DataPart.data.arguments must be an object")

        return tool, dict(arguments)

    raise ValueError(
        "No DataPart found in params.message.parts. Delegation requires a part with "
        'kind="data" carrying {"tool": ..., "arguments": {...}}. '
        "Natural-language TextParts are not interpreted by this agent."
    )


async def _authenticate(request: Request, auth_provider: Any) -> bool:
    """Verify the bearer token on a delegation request.

    ``custom_route`` handlers sit outside FastMCP's MCP-level auth, so the
    check is made here explicitly rather than assumed.
    """
    if auth_provider is None:
        return True

    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False

    try:
        return await auth_provider.verify_token(token) is not None
    except Exception:
        logger.warning("a2a_token_verification_failed", exc_info=True)
        return False


def register_a2a_routes(
    mcp: FastMCP,
    *,
    tools: Mapping[str, Callable[..., Any]] | None = None,
    loaded_clients: list[str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
    auth_provider: Any = None,
) -> None:
    """Register the Agent Card and delegation endpoint on the HTTP app.

    Args:
        mcp: The FastMCP server to mount routes on.
        tools: Name to callable map of tools this server exposes, already
            filtered by persona and policy and already audit-wrapped. The map
            *is* the authorization boundary for delegation.
        loaded_clients: Client module names, used to describe capabilities.
        loaded_modules: Imported client modules, consulted for ``A2A_SKILLS``.
        auth_provider: FastMCP token verifier, or None when auth is disabled.
    """
    tool_map: Mapping[str, Callable[..., Any]] = tools or {}
    clients = loaded_clients or []
    auth_required = auth_provider is not None

    @mcp.custom_route(AGENT_CARD_PATH, methods=["GET"], include_in_schema=False)
    async def agent_card(request: Request) -> Response:
        base = str(request.base_url).rstrip("/")
        card = build_agent_card(
            url=f"{base}{A2A_PATH}",
            loaded_clients=clients,
            loaded_modules=loaded_modules,
            auth_required=auth_required,
        )
        return JSONResponse(card, headers=dict(_CARD_HEADERS))

    @mcp.custom_route(A2A_PATH, methods=["POST"], include_in_schema=False)
    async def a2a_endpoint(request: Request) -> Response:  # noqa: C901
        if not await _authenticate(request, auth_provider):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            body = await request.json()
        except Exception:
            return _error(_PARSE_ERROR, "Invalid JSON payload")

        # A JSON-RPC request is an object. Arrays, null, and scalars are
        # rejected here so that reading fields below cannot raise.
        if not isinstance(body, Mapping):
            return _error(_INVALID_REQUEST, "Request must be a JSON object")

        msg_id = _safe_id(body)

        if body.get("jsonrpc") != "2.0":
            return _error(_INVALID_REQUEST, 'Missing or invalid "jsonrpc": must be "2.0"', msg_id)

        method = body.get("method")
        if not isinstance(method, str):
            return _error(_INVALID_REQUEST, "Method must be a string", msg_id)
        if method not in _SEND_METHODS:
            return _error(
                _METHOD_NOT_FOUND,
                f"Method not found: {method}. Supported: {', '.join(sorted(_SEND_METHODS))}",
                msg_id,
            )

        params = body.get("params", {})
        if not isinstance(params, Mapping):
            return _error(_INVALID_PARAMS, "params must be an object", msg_id)

        try:
            tool_name, arguments = extract_delegation(params)
        except ValueError as exc:
            return _error(_INVALID_PARAMS, str(exc), msg_id)

        tool_func = tool_map.get(tool_name)
        if tool_func is None:
            # The tool is absent because it does not exist or because the
            # active persona cannot reach it; both are refusals to serve.
            return _error(
                _UNSUPPORTED_OPERATION,
                f"Tool '{tool_name}' is not available to this agent.",
                msg_id,
            )

        try:
            # Tools are synchronous; run off the event loop so one delegated
            # call cannot stall the server for everyone else.
            result = await run_in_threadpool(lambda: tool_func(**arguments))
        except TypeError as exc:
            # A bad argument set is the caller's error, not a server fault.
            return _error(_INVALID_PARAMS, f"Invalid arguments for '{tool_name}': {exc}", msg_id)
        except Exception:
            logger.error("a2a_delegation_failed", extra={"tool": tool_name}, exc_info=True)
            return _error(_INTERNAL_ERROR, f"Tool '{tool_name}' failed", msg_id)

        # Stateless: the result comes back inline on the response. There is no
        # task store to poll and no stream to subscribe to.
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": uuid.uuid4().hex,
                    "parts": [{"kind": "data", "data": result}],
                },
                "id": msg_id,
            }
        )
