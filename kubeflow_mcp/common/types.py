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
from typing import Any, Literal

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
