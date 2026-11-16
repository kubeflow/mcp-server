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

"""Tests for rate-limit and circuit-breaker metrics."""

from contextlib import contextmanager
from typing import Any

import pytest

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.core import dynamic_tools, metrics
from kubeflow_mcp.core.metrics import FROM_STATE_ATTR, TO_STATE_ATTR, TOOL_NAME_ATTR
from kubeflow_mcp.core.resilience import (
    CircuitBreaker,
    CircuitState,
    get_breaker,
    reset_breakers,
)
from kubeflow_mcp.core.server import _audit_wrap


class _Recorder:
    """Stands in for an OTel counter and keeps every add() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []

    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        self.calls.append((amount, dict(attributes or {})))


class _FakeSpan:
    def set_attribute(self, key: str, value: object) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_status(self, code: object, description: str | None = None) -> None:
        pass


class _FakeTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        yield _FakeSpan()


class _Limiter:
    def __init__(self, allow: bool) -> None:
        self._allow = allow

    def acquire(self, tokens: float = 1.0) -> bool:
        return self._allow


class _Breaker:
    def __init__(self, allow: bool) -> None:
        self._allow = allow

    def acquire(self) -> int | None:
        return 0 if self._allow else None

    def record_success(self, generation: int | None = None) -> None:
        pass

    def record_failure(self, generation: int | None = None) -> None:
        pass

    def release(self, generation: int | None = None) -> None:
        pass


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Recorder]:
    """Swap the module-level instruments for recorders."""
    rec = {
        "rate_limited": _Recorder(),
        "circuit_open": _Recorder(),
        "transitions": _Recorder(),
    }
    monkeypatch.setattr(metrics, "_rate_limited", rec["rate_limited"])
    monkeypatch.setattr(metrics, "_circuit_open", rec["circuit_open"])
    monkeypatch.setattr(metrics, "_circuit_transitions", rec["transitions"])
    return rec


def _wrapped_tool(monkeypatch: pytest.MonkeyPatch, *, limiter: object | None, breaker: object):
    import kubeflow_mcp.core.server as server_mod

    monkeypatch.setattr(server_mod, "_rate_limiter", limiter)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-306")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "readonly")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: _FakeTracer())
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    def list_jobs(**_kwargs):
        return {"success": True, "data": {}}

    return _audit_wrap(list_jobs)


# ─── Record functions: metric and attribute contract ────────────────────────


def test_record_rate_limited_counts_one_call_labelled_with_the_tool(recorders):
    metrics.record_rate_limited("fine_tune")

    assert recorders["rate_limited"].calls == [(1, {TOOL_NAME_ATTR: "fine_tune"})]
    assert recorders["circuit_open"].calls == []
    assert recorders["transitions"].calls == []


def test_record_circuit_open_counts_one_call_labelled_with_the_tool(recorders):
    metrics.record_circuit_open("fine_tune")

    assert recorders["circuit_open"].calls == [(1, {TOOL_NAME_ATTR: "fine_tune"})]
    assert recorders["rate_limited"].calls == []


def test_record_circuit_transition_labels_tool_and_both_states(recorders):
    metrics.record_circuit_transition("fine_tune", "closed", "open")

    assert recorders["transitions"].calls == [
        (
            1,
            {
                TOOL_NAME_ATTR: "fine_tune",
                FROM_STATE_ATTR: "closed",
                TO_STATE_ATTR: "open",
            },
        )
    ]


def test_tool_attribute_matches_the_span_attribute():
    """Metrics and the tool spans in _audit_wrap must be joinable on the same key."""
    assert TOOL_NAME_ATTR == "gen_ai.tool.name"


# ─── Instrument creation ────────────────────────────────────────────────────


def test_instruments_are_api_counters_without_a_meter_provider():
    """fastmcp ships opentelemetry-api, so instruments exist even with no SDK."""
    otel_metrics = pytest.importorskip("opentelemetry.metrics")

    for instrument in (metrics._rate_limited, metrics._circuit_open, metrics._circuit_transitions):
        assert isinstance(instrument, otel_metrics.Counter)
        # No provider installed: recording must be a harmless no-op.
        assert instrument.add(1, {TOOL_NAME_ATTR: "noop"}) is None


def test_counter_falls_back_to_noop_without_the_otel_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(metrics, "_otel_metrics", None)

    counter = metrics._counter("kubeflow_mcp.test", "test", unit="{call}")

    assert isinstance(counter, metrics._NoopCounter)
    assert counter.add(1, {TOOL_NAME_ATTR: "x"}) is None


def test_counter_falls_back_to_noop_when_the_meter_cannot_create_it(
    monkeypatch: pytest.MonkeyPatch,
):
    class _BrokenMetricsApi:
        @staticmethod
        def get_meter(_name: str):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(metrics, "_otel_metrics", _BrokenMetricsApi)

    assert isinstance(
        metrics._counter("kubeflow_mcp.test", "test", unit="{call}"), metrics._NoopCounter
    )


def test_counters_record_through_the_sdk_when_a_provider_exists(monkeypatch: pytest.MonkeyPatch):
    """End to end with the real SDK, when it is installed (`uv sync --group otel`)."""
    sdk_metrics = pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = sdk_metrics.MeterProvider(metric_readers=[reader])
    counter = provider.get_meter("kubeflow_mcp").create_counter(
        metrics.RATE_LIMITED_METRIC, unit="{call}"
    )
    monkeypatch.setattr(metrics, "_rate_limited", counter)

    metrics.record_rate_limited("fine_tune")
    metrics.record_rate_limited("fine_tune")
    metrics.record_rate_limited("list_jobs")

    points = {
        dict(point.attributes)[TOOL_NAME_ATTR]: point.value
        for resource_metrics in reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == metrics.RATE_LIMITED_METRIC
        for point in metric.data.data_points
    }
    assert points == {"fine_tune": 2, "list_jobs": 1}
    provider.shutdown()


# ─── _audit_wrap hooks ──────────────────────────────────────────────────────


def test_audit_wrap_counts_rate_limited_calls(monkeypatch: pytest.MonkeyPatch, recorders):
    wrapped = _wrapped_tool(
        monkeypatch, limiter=_Limiter(allow=False), breaker=_Breaker(allow=True)
    )

    result = wrapped()

    assert result["error_code"] == ErrorCode.RATE_LIMITED
    assert recorders["rate_limited"].calls == [(1, {TOOL_NAME_ATTR: "list_jobs"})]
    assert recorders["circuit_open"].calls == []


def test_audit_wrap_counts_circuit_open_calls(monkeypatch: pytest.MonkeyPatch, recorders):
    wrapped = _wrapped_tool(
        monkeypatch, limiter=_Limiter(allow=True), breaker=_Breaker(allow=False)
    )

    result = wrapped()

    assert result["error_code"] == ErrorCode.CIRCUIT_OPEN
    assert recorders["circuit_open"].calls == [(1, {TOOL_NAME_ATTR: "list_jobs"})]
    assert recorders["rate_limited"].calls == []


def test_audit_wrap_counts_nothing_for_a_normal_call(monkeypatch: pytest.MonkeyPatch, recorders):
    wrapped = _wrapped_tool(monkeypatch, limiter=_Limiter(allow=True), breaker=_Breaker(allow=True))

    assert wrapped() == {"success": True, "data": {}}
    assert recorders["rate_limited"].calls == []
    assert recorders["circuit_open"].calls == []


def test_audit_wrap_rate_limit_rejection_is_unchanged(monkeypatch: pytest.MonkeyPatch, recorders):
    """Counting must not alter the response callers already rely on."""
    wrapped = _wrapped_tool(
        monkeypatch, limiter=_Limiter(allow=False), breaker=_Breaker(allow=True)
    )

    assert wrapped() == {
        "error": "Rate limit exceeded. Retry after a brief pause.",
        "error_code": ErrorCode.RATE_LIMITED,
    }


# ─── Dynamic execute_tool hook ──────────────────────────────────────────────


def probe_tool(name: str) -> dict:
    """Read-only tool used to exercise execute_tool."""
    return {"success": True, "data": {"name": name}}


@pytest.fixture
def dynamic_registry():
    reset_breakers()
    dynamic_tools.init_dynamic_tools([probe_tool], {})
    yield
    dynamic_tools.TOOL_REGISTRY.clear()
    dynamic_tools.TOOL_HIERARCHY.clear()
    dynamic_tools._embedding_cache.reset()
    reset_breakers()


def test_execute_tool_counts_circuit_open_rejections(dynamic_registry, recorders):
    breaker = get_breaker("probe_tool")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    result = dynamic_tools.execute_tool("probe_tool", {"name": "x"})

    assert result["error_code"] == ErrorCode.CIRCUIT_OPEN
    assert recorders["circuit_open"].calls == [(1, {TOOL_NAME_ATTR: "probe_tool"})]


def test_execute_tool_counts_nothing_when_the_circuit_is_closed(dynamic_registry, recorders):
    result = dynamic_tools.execute_tool("probe_tool", {"name": "ok"})

    assert result == {"success": True, "data": {"name": "ok"}}
    assert recorders["circuit_open"].calls == []


# ─── Circuit breaker transitions ────────────────────────────────────────────


def _transitions(recorder: _Recorder) -> list[tuple[str, str, str]]:
    return [
        (attrs[TOOL_NAME_ATTR], attrs[FROM_STATE_ATTR], attrs[TO_STATE_ATTR])
        for amount, attrs in recorder.calls
        if amount == 1
    ]


def test_breaker_counts_every_state_transition(recorders):
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=1, name="fine_tune"
    )

    breaker.record_failure()  # CLOSED -> OPEN
    generation = breaker.acquire()  # OPEN -> HALF_OPEN (recovery timeout elapsed)
    breaker.record_success(generation)  # HALF_OPEN -> CLOSED
    breaker.record_failure()  # CLOSED -> OPEN
    generation = breaker.acquire()  # OPEN -> HALF_OPEN
    breaker.record_failure(generation)  # HALF_OPEN -> OPEN

    assert breaker.state == CircuitState.OPEN
    assert _transitions(recorders["transitions"]) == [
        ("fine_tune", "closed", "open"),
        ("fine_tune", "open", "half_open"),
        ("fine_tune", "half_open", "closed"),
        ("fine_tune", "closed", "open"),
        ("fine_tune", "open", "half_open"),
        ("fine_tune", "half_open", "open"),
    ]


def test_breaker_counts_nothing_while_it_stays_closed(recorders):
    breaker = CircuitBreaker(failure_threshold=3, name="list_jobs")

    for _ in range(5):
        generation = breaker.acquire()
        breaker.record_success(generation)
    breaker.record_failure()
    breaker.record_failure()  # still below the threshold

    assert breaker.state == CircuitState.CLOSED
    assert recorders["transitions"].calls == []


def test_breaker_does_not_count_a_late_failure_while_already_open(recorders):
    """A failure reported after the trip re-runs the CLOSED -> OPEN branch; no state change."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0, name="fine_tune")

    breaker.record_failure()  # CLOSED -> OPEN
    breaker.record_failure()  # in-flight call reporting late
    assert breaker.acquire() is None  # still open, recovery timeout not elapsed

    assert _transitions(recorders["transitions"]) == [("fine_tune", "closed", "open")]


def test_breaker_thresholds_are_unchanged_by_the_name_field():
    breaker = CircuitBreaker()

    assert breaker.name == ""
    assert (breaker.failure_threshold, breaker.recovery_timeout, breaker.half_open_max_calls) == (
        5,
        30.0,
        3,
    )


def test_get_breaker_labels_the_breaker_with_its_tool_name():
    reset_breakers()
    try:
        assert get_breaker("fine_tune").name == "fine_tune"
        assert get_breaker("fine_tune") is get_breaker("fine_tune")
    finally:
        reset_breakers()
