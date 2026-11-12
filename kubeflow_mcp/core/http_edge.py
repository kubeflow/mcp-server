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

"""HTTP liveness, readiness, and metrics routes for network transports."""

import logging

from fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kubeflow_mcp.core.telemetry import prometheus_enabled

logger = logging.getLogger(__name__)


def _collect_prometheus_metrics() -> tuple[bytes, str]:
    """Render the default Prometheus registry.

    Imported lazily: prometheus_client arrives with the optional ``otel``
    dependency group, so the module must stay importable without it.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return generate_latest(), CONTENT_TYPE_LATEST


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

    @mcp.custom_route("/metrics", methods=["GET"], include_in_schema=False)
    async def metrics(_request: Request) -> Response:
        # Scrapers need a stable endpoint, so an unconfigured or failing collector
        # answers 503 rather than an empty body that would read as "zero traffic".
        if not prometheus_enabled():
            return JSONResponse(
                {"status": "metrics_disabled"},
                status_code=503,
                media_type="application/json",
            )

        try:
            payload, content_type = await run_in_threadpool(_collect_prometheus_metrics)
        except ImportError:
            logger.warning(
                "Prometheus metrics are enabled but prometheus_client is not importable. "
                "Install OpenTelemetry dependencies from source with: uv sync --group otel"
            )
            return JSONResponse({"status": "metrics_unavailable"}, status_code=503)
        except Exception:
            logger.exception("Failed to collect Prometheus metrics")
            return JSONResponse({"status": "metrics_unavailable"}, status_code=503)

        return Response(payload, media_type=content_type)
