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

"""Shared types for kubeflow-mcp tools."""

import logging
import traceback
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def is_k8s_not_found(exc: Exception) -> bool:
    """Return True only for Kubernetes API 404 responses.

    Avoids the pitfall of ``"not found" in str(e).lower()`` which also matches
    ``ModuleNotFoundError``, ``AttributeError``, and any exception whose message
    happens to contain the substring "not found".
    """
    try:
        from kubernetes.client.exceptions import ApiException
    except ImportError:
        return False
    if isinstance(exc, ApiException) and exc.status == 404:
        return True
    cause = exc.__cause__ or exc.__context__
    if isinstance(cause, ApiException) and cause.status == 404:
        return True
    return False


def has_k8s_status(exc: BaseException, *statuses: int) -> bool:
    """Return True if *exc* or anything in its cause/context chain has one of *statuses*.

    The SDK can wrap an ``ApiException`` more than once, so the whole chain is
    walked rather than a single level. Matching on the ``status`` attribute, not
    the message text, avoids false hits on unrelated errors.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "status", None) in statuses:
            return True
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return False


def is_k8s_forbidden(exc: BaseException) -> bool:
    """Return True for Kubernetes 401/403 responses.

    These mean the caller lacks access, not that the cluster is unhealthy, so
    tools report them as ``PERMISSION_DENIED`` — a code the circuit breaker
    treats as neutral. Reporting them as ``SDK_ERROR`` would let one user's
    denied requests open the shared per-tool breaker for everyone.
    """
    return has_k8s_status(exc, 401, 403)


def exception_details(exc: Exception) -> dict[str, Any]:
    """Extract structured details from an exception for ToolError.details.

    Includes the cause chain so downstream error handlers (e.g. the Ollama
    agent's _format_friendly_error) can surface HTTP status codes that the SDK
    wraps inside a generic message.

    Tracebacks are only included when the ``kubeflow_mcp`` logger is at DEBUG
    level, to avoid leaking internal paths and library versions to clients.
    """
    details: dict[str, Any] = {"exception": type(exc).__name__, "message": str(exc)}
    cause = exc.__cause__ or exc.__context__
    if cause:
        details["cause"] = f"{type(cause).__name__}: {cause}"

    root_logger = logging.getLogger("kubeflow_mcp")
    if root_logger.isEnabledFor(logging.DEBUG):
        tb = traceback.format_exc()
        if tb and tb.strip() != "NoneType: None":
            details["traceback"] = tb

    return details


class ToolPermission(NamedTuple):
    """A Kubernetes API permission that a tool's own requests need.

    Declared per tool in each client module's ``CLIENT_TOOL_PERMISSIONS`` so an
    authorization check can ask whether a caller holds them (#565). Only requests
    the server sends to the Kubernetes API count, including the ones the Kubeflow
    SDK makes on the tool's behalf; requests authorized elsewhere (the Pipelines
    API server) and best-effort reads whose failure the tool tolerates do not.
    """

    verb: str
    group: str  # "" is the core API group
    resource: str  # plural, optionally with a subresource, e.g. "pods/log"
    cluster_wide: bool = False  # cluster-scoped, or across all namespaces
    # Parameter that triggers the request; None means every call makes it.
    when: str | None = None


class ToolResponse(BaseModel):
    """Standard success response."""

    success: Literal[True] = True
    data: dict[str, Any]


class ToolError(BaseModel):
    """Standard error response."""

    success: Literal[False] = False
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None
    hint: str | None = None  # Suggest relevant MCP prompt for recovery


class PreviewResponse(BaseModel):
    """Response for two-phase confirmation pattern."""

    success: Literal[True] = True
    status: Literal["preview"] = "preview"
    message: str = "Set confirmed=True to execute"
    config: dict[str, Any]


ToolResult = ToolResponse | ToolError | PreviewResponse
