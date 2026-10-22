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

"""HTTP liveness and readiness probes for network transports."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from kubeflow_mcp.core.health import ReadinessDiagnostic

logger = logging.getLogger(__name__)


def register_probe_routes(
    mcp: FastMCP,
    *,
    is_ready: bool = True,
    diagnostics: list[ReadinessDiagnostic] | None = None,
    readiness_check: Callable[[], tuple[bool, list[ReadinessDiagnostic] | None]] | None = None,
) -> None:
    """Register unauthenticated probes on the FastMCP HTTP application."""

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "healthy"})

    @mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
    async def ready(_request: Request) -> Response:
        current_ready = is_ready
        current_diagnostics = diagnostics

        if readiness_check is not None and current_ready:
            try:
                current_ready, raw_diags = readiness_check()
                current_diagnostics = (
                    [d for d in raw_diags if d.status != "pass"] if raw_diags else None
                )
            except Exception as e:
                logger.warning("Readiness probe check failed unexpectedly: %s", e)
                return JSONResponse({"status": "not_ready"}, status_code=503)

        if not current_ready:
            if current_diagnostics:
                for diag in current_diagnostics:
                    check = getattr(diag, "check", "")
                    msg = getattr(diag, "message", str(diag))
                    cat = getattr(diag, "category", "")
                    logger.warning(
                        "Readiness probe check '%s' failed: %s (category: %s)",
                        check,
                        msg,
                        cat or "none",
                    )
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})
