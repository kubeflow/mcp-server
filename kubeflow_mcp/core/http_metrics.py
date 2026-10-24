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

"""HTTP RED metrics and a readiness gauge for the network transports.

``HTTPMetricsMiddleware`` is a pure-ASGI middleware that records one sample per
HTTP request into ``http.server.request.duration`` (OpenTelemetry HTTP semantic
conventions). Request rate, error rate and latency all derive from that one
histogram. Its attributes are bounded by construction: the request method
(anything outside the well-known set collapses to ``_OTHER``), the response
status code, and the matched route *template* -- never the raw path.

``kubeflow_mcp.server.ready`` is an observable gauge fed from the same flag the
``/ready`` probe answers with, so the gauge and the probe cannot disagree.

Instruments are created through the OpenTelemetry API, which fastmcp always
installs. Until a ``MeterProvider`` is configured they are no-ops; once one is
installed (the shared metrics setup from #302) they start recording through the
same instrument objects. Neither fastmcp nor the MCP SDK emit HTTP metrics, so
nothing here duplicates a framework signal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.metrics import Observation
except ImportError:  # pragma: no cover - fastmcp always installs opentelemetry-api
    _otel_metrics = None  # type: ignore[assignment]
    Observation = None  # type: ignore[assignment,misc]

# OpenTelemetry HTTP semantic conventions.
METHOD_ATTR = "http.request.method"
STATUS_ATTR = "http.response.status_code"
ROUTE_ATTR = "http.route"
OTHER_METHOD = "_OTHER"

REQUEST_DURATION_METRIC = "http.server.request.duration"
SERVER_READY_METRIC = "kubeflow_mcp.server.ready"

# Bucket advice from the semantic conventions for http.server.request.duration, in seconds.
REQUEST_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)

_KNOWN_METHODS = frozenset(
    {"CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"}
)

_server_ready = False


def set_server_ready(ready: bool) -> None:
    """Record the readiness the ``/ready`` probe reports; the gauge follows it."""
    global _server_ready
    _server_ready = bool(ready)


def is_server_ready() -> bool:
    return _server_ready


def _observe_server_ready(_options: Any = None) -> Iterable[Any]:
    """Observable-gauge callback: 1 while the server reports ready, else 0."""
    if Observation is None:  # pragma: no cover - API missing
        return []
    return [Observation(1 if _server_ready else 0)]


class _NoopInstrument:
    def record(self, amount: float, attributes: dict[str, Any] | None = None) -> None:
        return None


def _histogram(name: str, description: str, unit: str, boundaries: tuple[float, ...]) -> Any:
    """Create a histogram, or a no-op stand-in when the OpenTelemetry API is missing."""
    if _otel_metrics is None:
        return _NoopInstrument()
    try:
        return _otel_metrics.get_meter("kubeflow_mcp").create_histogram(
            name,
            unit=unit,
            description=description,
            explicit_bucket_boundaries_advisory=list(boundaries),
        )
    except Exception as e:  # defensive: metrics must never break startup
        logger.debug("Could not create metric %s: %s", name, e)
        return _NoopInstrument()


def _observable_gauge(name: str, description: str, unit: str, callback: Any) -> Any:
    """Create an observable gauge, or a no-op stand-in when the API is missing."""
    if _otel_metrics is None:
        return _NoopInstrument()
    try:
        return _otel_metrics.get_meter("kubeflow_mcp").create_observable_gauge(
            name, callbacks=[callback], unit=unit, description=description
        )
    except Exception as e:  # defensive: metrics must never break startup
        logger.debug("Could not create metric %s: %s", name, e)
        return _NoopInstrument()


_request_duration = _histogram(
    REQUEST_DURATION_METRIC,
    "Duration of HTTP server requests.",
    unit="s",
    boundaries=REQUEST_DURATION_BUCKETS,
)
_server_ready_gauge = _observable_gauge(
    SERVER_READY_METRIC,
    "1 while the server reports ready on /ready, 0 otherwise.",
    unit="1",
    callback=_observe_server_ready,
)


def route_template(scope: Scope) -> str | None:
    """The matched route's template (``/mcp``, ``/health``), or None if nothing matched.

    Starlette stores the matched ``Route``/``Mount`` on the scope, so this is the
    declared pattern rather than the raw request path, which keeps cardinality bounded.
    """
    path = getattr(scope.get("route"), "path", None)
    return path if isinstance(path, str) and path else None


def request_attributes(scope: Scope, status_code: int) -> dict[str, Any]:
    """Bounded attribute set for one request: method, status code and route template."""
    method = scope.get("method", "")
    attributes: dict[str, Any] = {
        METHOD_ATTR: method if method in _KNOWN_METHODS else OTHER_METHOD,
        STATUS_ATTR: int(status_code),
    }
    route = route_template(scope)
    if route is not None:
        attributes[ROUTE_ATTR] = route
    return attributes


def record_request(scope: Scope, status_code: int, duration_s: float) -> None:
    """Record one HTTP request sample."""
    _request_duration.record(duration_s, request_attributes(scope, status_code))


class HTTPMetricsMiddleware:
    """Pure-ASGI middleware that records one duration sample per HTTP request.

    Place it outermost so that requests rejected by other middleware (auth, DNS
    rebinding protection) are counted too. Non-HTTP scopes (lifespan, websocket)
    pass straight through. A request that ends without a response having started
    (client went away) has no status to classify and is not recorded.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Starlette's ServerErrorMiddleware, above us, answers 500 for this.
            if status_code is None:
                status_code = 500
            raise
        finally:
            if status_code is not None:
                try:
                    record_request(scope, status_code, time.perf_counter() - start)
                except Exception as e:  # metrics must never break a request
                    logger.debug("Could not record HTTP metrics: %s", e)
