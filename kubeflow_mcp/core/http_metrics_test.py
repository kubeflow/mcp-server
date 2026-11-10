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

"""Tests for HTTP RED metrics and the readiness gauge."""

import asyncio
from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.testclient import TestClient

from kubeflow_mcp.core import http_metrics
from kubeflow_mcp.core.http_edge import register_probe_routes
from kubeflow_mcp.core.http_metrics import (
    METHOD_ATTR,
    OTHER_METHOD,
    ROUTE_ATTR,
    STATUS_ATTR,
    HTTPMetricsMiddleware,
)


class _Recorder:
    """Stands in for the duration histogram and keeps every record() call."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, dict[str, Any]]] = []

    def record(self, amount: float, attributes: dict[str, Any] | None = None) -> None:
        self.samples.append((amount, dict(attributes or {})))

    @property
    def attributes(self) -> list[dict[str, Any]]:
        return [attrs for _, attrs in self.samples]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(http_metrics, "_request_duration", rec)
    return rec


@pytest.fixture(autouse=True)
def _restore_readiness():
    before = http_metrics.is_server_ready()
    yield
    http_metrics.set_server_ready(before)


def _client(*, ready: bool = True, authenticated: bool = False) -> TestClient:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    register_probe_routes(mcp, is_ready=ready)

    @mcp.custom_route("/boom", methods=["GET"], include_in_schema=False)
    async def boom(_request: Request) -> Response:
        raise RuntimeError("handler failed")

    @mcp.custom_route("/items/{item_id}", methods=["GET"], include_in_schema=False)
    async def item(request: Request) -> Response:
        return JSONResponse({"item": request.path_params["item_id"]})

    app = mcp.http_app(transport="streamable-http", middleware=[Middleware(HTTPMetricsMiddleware)])
    return TestClient(app)


def _ready_observation() -> int:
    observations = list(http_metrics._observe_server_ready())
    assert len(observations) == 1
    return observations[0].value


# ─── Readiness gauge ────────────────────────────────────────────────────────


def test_ready_gauge_is_one_when_the_ready_probe_answers_200(recorder):
    with _client(ready=True) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert _ready_observation() == 1


def test_ready_gauge_is_zero_when_the_ready_probe_answers_503(recorder):
    with _client(ready=False) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert _ready_observation() == 0


def test_ready_gauge_follows_readiness_changes():
    http_metrics.set_server_ready(True)
    assert _ready_observation() == 1
    http_metrics.set_server_ready(False)
    assert _ready_observation() == 0


# ─── Request samples: representative responses ──────────────────────────────


def test_successful_request_is_recorded_with_method_status_and_route(recorder):
    with _client() as client:
        assert client.get("/health").status_code == 200

    assert recorder.attributes == [{METHOD_ATTR: "GET", STATUS_ATTR: 200, ROUTE_ATTR: "/health"}]
    duration, _ = recorder.samples[0]
    assert 0 <= duration < 5


def test_not_ready_probe_is_recorded_as_503(recorder):
    with _client(ready=False) as client:
        client.get("/ready")

    assert recorder.attributes == [{METHOD_ATTR: "GET", STATUS_ATTR: 503, ROUTE_ATTR: "/ready"}]


def test_route_attribute_is_the_template_not_the_raw_path(recorder):
    with _client() as client:
        for item_id in ("alpha", "beta", "gamma"):
            assert client.get(f"/items/{item_id}").status_code == 200

    assert {attrs[ROUTE_ATTR] for attrs in recorder.attributes} == {"/items/{item_id}"}


def test_unmatched_path_has_no_route_attribute(recorder):
    with _client() as client:
        assert client.get("/no/such/path").status_code == 404

    assert recorder.attributes == [{METHOD_ATTR: "GET", STATUS_ATTR: 404}]


def test_method_not_allowed_keeps_the_route_template(recorder):
    with _client() as client:
        assert client.post("/health").status_code == 405

    assert recorder.attributes == [{METHOD_ATTR: "POST", STATUS_ATTR: 405, ROUTE_ATTR: "/health"}]


def test_unknown_methods_collapse_to_other(recorder):
    with _client() as client:
        client.request("BREW", "/health")

    assert recorder.attributes[0][METHOD_ATTR] == OTHER_METHOD


def test_auth_rejections_on_the_mcp_route_are_counted(recorder):
    with _client(authenticated=True) as client:
        assert client.post("/mcp").status_code == 401

    assert recorder.attributes == [{METHOD_ATTR: "POST", STATUS_ATTR: 401, ROUTE_ATTR: "/mcp"}]


def test_handler_exception_is_recorded_as_500_and_still_raised(recorder):
    with _client() as client:
        with pytest.raises(RuntimeError, match="handler failed"):
            client.get("/boom")

    assert recorder.attributes == [{METHOD_ATTR: "GET", STATUS_ATTR: 500, ROUTE_ATTR: "/boom"}]


def test_lifespan_and_other_non_http_scopes_are_ignored(recorder):
    with _client():
        pass  # TestClient runs startup/shutdown lifespan events only

    assert recorder.samples == []


def test_sse_message_mount_is_recorded_by_its_prefix(recorder):
    app = FastMCP("sse-server").http_app(
        transport="sse", middleware=[Middleware(HTTPMetricsMiddleware)]
    )
    with TestClient(app) as client:
        response = client.post("/messages/?session_id=0123456789abcdef0123456789abcdef", json={})

    assert response.status_code == 404  # unknown session
    assert recorder.attributes == [{METHOD_ATTR: "POST", STATUS_ATTR: 404, ROUTE_ATTR: "/messages"}]


def test_request_without_a_response_is_not_recorded(recorder):
    async def silent_app(scope, receive, send):
        return  # client went away before anything was sent

    scope = {"type": "http", "method": "GET", "path": "/health"}
    asyncio.run(HTTPMetricsMiddleware(silent_app)(scope, None, None))

    assert recorder.samples == []


def test_metrics_failure_never_breaks_the_request(monkeypatch: pytest.MonkeyPatch):
    class _Broken:
        def record(self, amount, attributes=None):
            raise RuntimeError("exporter exploded")

    monkeypatch.setattr(http_metrics, "_request_duration", _Broken())

    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ─── Attribute helpers ──────────────────────────────────────────────────────


def test_request_attributes_are_bounded_to_method_status_and_route():
    class _Route:
        path = "/mcp"

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp?session=abc",
        "route": _Route(),
        "headers": [(b"user-agent", b"x")],
    }

    assert http_metrics.request_attributes(scope, 200) == {
        METHOD_ATTR: "POST",
        STATUS_ATTR: 200,
        ROUTE_ATTR: "/mcp",
    }


def test_attribute_keys_follow_http_semantic_conventions():
    assert METHOD_ATTR == "http.request.method"
    assert STATUS_ATTR == "http.response.status_code"
    assert ROUTE_ATTR == "http.route"
    assert http_metrics.REQUEST_DURATION_METRIC == "http.server.request.duration"


# ─── Instrument creation ────────────────────────────────────────────────────


def test_instruments_exist_without_a_meter_provider():
    """fastmcp ships opentelemetry-api, so instruments exist even with no SDK."""
    otel_metrics = pytest.importorskip("opentelemetry.metrics")

    assert isinstance(http_metrics._request_duration, otel_metrics.Histogram)
    assert isinstance(http_metrics._server_ready_gauge, otel_metrics.ObservableGauge)
    # No provider installed: recording must be a harmless no-op.
    assert http_metrics._request_duration.record(0.1, {METHOD_ATTR: "GET"}) is None


def test_instruments_fall_back_to_noop_without_the_otel_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(http_metrics, "_otel_metrics", None)

    histogram = http_metrics._histogram("x", "d", unit="s", boundaries=(0.1,))
    gauge = http_metrics._observable_gauge("y", "d", unit="1", callback=lambda *_: [])

    assert isinstance(histogram, http_metrics._NoopInstrument)
    assert isinstance(gauge, http_metrics._NoopInstrument)
    assert histogram.record(0.1, {}) is None


def test_instruments_fall_back_to_noop_when_the_meter_raises(monkeypatch: pytest.MonkeyPatch):
    class _BrokenMetricsApi:
        @staticmethod
        def get_meter(_name: str):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(http_metrics, "_otel_metrics", _BrokenMetricsApi)

    assert isinstance(
        http_metrics._histogram("x", "d", unit="s", boundaries=(0.1,)),
        http_metrics._NoopInstrument,
    )
    assert isinstance(
        http_metrics._observable_gauge("y", "d", unit="1", callback=lambda *_: []),
        http_metrics._NoopInstrument,
    )


def test_metrics_record_through_the_sdk_when_a_provider_exists(monkeypatch: pytest.MonkeyPatch):
    """End to end with the real SDK, when it is installed (`uv sync --group otel`)."""
    sdk_metrics = pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = sdk_metrics.MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("kubeflow_mcp")
    histogram = meter.create_histogram(
        http_metrics.REQUEST_DURATION_METRIC,
        unit="s",
        explicit_bucket_boundaries_advisory=list(http_metrics.REQUEST_DURATION_BUCKETS),
    )
    meter.create_observable_gauge(
        http_metrics.SERVER_READY_METRIC, callbacks=[http_metrics._observe_server_ready]
    )
    monkeypatch.setattr(http_metrics, "_request_duration", histogram)

    with _client(ready=False) as client:
        client.get("/health")
        client.get("/health")
        client.get("/ready")

    def _points(name: str):
        return [
            point
            for resource_metrics in reader.get_metrics_data().resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
            if metric.name == name
            for point in metric.data.data_points
        ]

    by_attrs = {
        (dict(p.attributes)[ROUTE_ATTR], dict(p.attributes)[STATUS_ATTR]): p
        for p in _points(http_metrics.REQUEST_DURATION_METRIC)
    }
    assert by_attrs[("/health", 200)].count == 2
    assert by_attrs[("/ready", 503)].count == 1
    assert list(by_attrs[("/health", 200)].explicit_bounds) == list(
        http_metrics.REQUEST_DURATION_BUCKETS
    )

    (ready_point,) = _points(http_metrics.SERVER_READY_METRIC)
    assert ready_point.value == 0
    provider.shutdown()
