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

"""OpenTelemetry tracing tests."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kubeflow_mcp.core import telemetry
from kubeflow_mcp.core.server import _audit_wrap


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.exceptions: list[BaseException] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)


class _FakeTracer:
    def __init__(self, span: _FakeSpan) -> None:
        self._span = span

    @contextmanager
    def start_as_current_span(self, _name: str):
        yield self._span


class _FakeBreaker:
    def __init__(self, can_execute: bool = True) -> None:
        self._can_execute = can_execute
        self.successes = 0
        self.failures = 0

    def can_execute(self) -> bool:
        return self._can_execute

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self) -> None:
        self.failures += 1


def test_get_tracer_returns_noop_when_otel_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", False)
    tracer = telemetry.get_tracer("test")
    with tracer.start_as_current_span("span") as span:
        span.set_attribute("k", "v")
        span.record_exception(ValueError("boom"))


def test_setup_tracing_noop_when_otel_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(telemetry, "_tracing_initialized", False)
    assert telemetry.setup_tracing("http://collector:4318/v1/traces") is False


def test_setup_tracing_configures_provider_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeProvider:
        def __init__(self, resource: object) -> None:
            calls["resource"] = resource
            self._processors: list[object] = []

        def add_span_processor(self, processor: object) -> None:
            self._processors.append(processor)
            calls["processor"] = processor

    class _FakeResource:
        @staticmethod
        def create(data: dict[str, str]) -> dict[str, str]:
            return data

    class _FakeBatchProcessor:
        def __init__(self, exporter: object) -> None:
            self.exporter = exporter

    class _FakeExporter:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint
            calls["endpoint"] = endpoint

    fake_trace = SimpleNamespace()

    def _set_tracer_provider(provider: object) -> None:
        calls["provider"] = provider

    def _get_tracer(name: str) -> str:
        return f"tracer:{name}"

    def _get_tracer_provider() -> object:
        return object()

    fake_trace.set_tracer_provider = _set_tracer_provider
    fake_trace.get_tracer = _get_tracer
    fake_trace.get_tracer_provider = _get_tracer_provider

    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_tracing_initialized", False)
    monkeypatch.setattr(telemetry, "_configured_endpoint", None, raising=False)
    monkeypatch.setattr(telemetry, "Resource", _FakeResource, raising=False)
    monkeypatch.setattr(telemetry, "TracerProvider", _FakeProvider, raising=False)
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", _FakeBatchProcessor, raising=False)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", _FakeExporter, raising=False)
    monkeypatch.setattr(telemetry, "_otel_trace", fake_trace, raising=False)

    assert telemetry.setup_tracing("http://collector:4318/v1/traces", "kubeflow-mcp") is True
    assert calls["endpoint"] == "http://collector:4318/v1/traces"
    assert calls["resource"] == {"service.name": "kubeflow-mcp"}
    assert telemetry.get_tracer("unit") == "tracer:unit"


def test_setup_tracing_treats_whitespace_endpoint_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_tracing_initialized", False)
    monkeypatch.setattr(telemetry, "_configured_endpoint", None, raising=False)
    assert telemetry.setup_tracing("   ") is False


def test_setup_tracing_rejects_invalid_endpoint() -> None:
    with pytest.raises(ValueError, match="Invalid OpenTelemetry endpoint"):
        telemetry.setup_tracing("localhost:4318/v1/traces")


def test_setup_tracing_reuses_existing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _ExistingProvider:
        def __init__(self) -> None:
            self.processors: list[object] = []

        def add_span_processor(self, processor: object) -> None:
            self.processors.append(processor)
            calls["processor"] = processor

    class _FakeBatchProcessor:
        def __init__(self, exporter: object) -> None:
            self.exporter = exporter

    class _FakeExporter:
        def __init__(self, endpoint: str) -> None:
            calls["endpoint"] = endpoint

    provider = _ExistingProvider()
    fake_trace = SimpleNamespace()
    fake_trace.get_tracer_provider = lambda: provider
    fake_trace.get_tracer = lambda _name: "tracer"
    fake_trace.set_tracer_provider = lambda _provider: calls.update({"set_called": True})

    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_tracing_initialized", False)
    monkeypatch.setattr(telemetry, "_configured_endpoint", None, raising=False)
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", _FakeBatchProcessor, raising=False)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", _FakeExporter, raising=False)
    monkeypatch.setattr(telemetry, "_otel_trace", fake_trace, raising=False)

    assert telemetry.setup_tracing("http://collector:4318/v1/traces", "kubeflow-mcp") is True
    assert calls["endpoint"] == "http://collector:4318/v1/traces"
    assert "processor" in calls
    assert "set_called" not in calls


def test_audit_wrap_sets_span_attributes_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import kubeflow_mcp.core.server as server_mod

    span = _FakeSpan()
    breaker = _FakeBreaker()
    monkeypatch.setattr(server_mod, "_rate_limiter", None)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-123")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "ml-engineer")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: _FakeTracer(span))
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    def sample_tool(**_kwargs):
        return {"ok": True}

    wrapped = _audit_wrap(sample_tool)
    wrapped()

    assert span.attributes["tool.name"] == "sample_tool"
    assert span.attributes["correlation_id"] == "cid-123"
    assert span.attributes["kubeflow.persona"] == "ml-engineer"
    assert span.attributes["tool.success"] is True
    assert "tool.duration_ms" in span.attributes
    assert breaker.successes == 1


def test_audit_wrap_records_exception_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import kubeflow_mcp.core.server as server_mod

    span = _FakeSpan()
    breaker = _FakeBreaker()
    monkeypatch.setattr(server_mod, "_rate_limiter", None)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-123")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "readonly")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: _FakeTracer(span))
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    def failing_tool(**_kwargs):
        raise RuntimeError("boom")

    wrapped = _audit_wrap(failing_tool)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped()

    assert span.attributes["tool.success"] is False
    assert "tool.duration_ms" in span.attributes
    assert len(span.exceptions) == 1
    assert isinstance(span.exceptions[0], RuntimeError)
    assert breaker.failures == 1
