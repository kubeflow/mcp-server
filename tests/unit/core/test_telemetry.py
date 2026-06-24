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
        self.status_code: object | None = None
        self.status_description: str | None = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)

    def set_status(self, code: object, description: str | None = None) -> None:
        self.status_code = code
        self.status_description = description


class _FakeTracer:
    def __init__(self, span: _FakeSpan) -> None:
        self._span = span
        self.last_span_name: str | None = None
        self.last_span_kwargs: dict[str, object] = {}

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        self.last_span_name = name
        self.last_span_kwargs = kwargs
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

        def shutdown(self) -> None:
            pass

    class _FakeResource:
        @staticmethod
        def create(data: dict[str, str]) -> dict[str, str]:
            return data

    class _FakeBatchProcessor:
        def __init__(self, exporter: object, **kwargs) -> None:
            self.exporter = exporter

    class _FakeExporter:
        def __init__(self, endpoint: str, **kwargs) -> None:
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


def test_setup_tracing_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_tracing() reads OTEL_EXPORTER_OTLP_ENDPOINT when endpoint is None."""
    calls: dict[str, object] = {}

    class _FakeBatchProcessor:
        def __init__(self, exporter: object, **kwargs) -> None:
            self.exporter = exporter

    class _FakeExporter:
        def __init__(self, endpoint: str, **kwargs) -> None:
            calls["endpoint"] = endpoint

    class _FakeProvider:
        def __init__(self, resource: object) -> None:
            pass

        def add_span_processor(self, processor: object) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class _FakeResource:
        @staticmethod
        def create(data: dict[str, str]) -> dict[str, str]:
            return data

    fake_trace = SimpleNamespace()
    fake_trace.set_tracer_provider = lambda _p: None
    fake_trace.get_tracer = lambda _name: "tracer"
    fake_trace.get_tracer_provider = lambda: object()

    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_tracing_initialized", False)
    monkeypatch.setattr(telemetry, "_configured_endpoint", None, raising=False)
    monkeypatch.setattr(telemetry, "Resource", _FakeResource, raising=False)
    monkeypatch.setattr(telemetry, "TracerProvider", _FakeProvider, raising=False)
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", _FakeBatchProcessor, raising=False)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", _FakeExporter, raising=False)
    monkeypatch.setattr(telemetry, "_otel_trace", fake_trace, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-collector:4318")

    assert telemetry.setup_tracing(endpoint=None) is True
    assert calls["endpoint"] == "http://env-collector:4318/v1/traces"


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
        def __init__(self, exporter: object, **kwargs) -> None:
            self.exporter = exporter

    class _FakeExporter:
        def __init__(self, endpoint: str, **kwargs) -> None:
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
    tracer = _FakeTracer(span)
    breaker = _FakeBreaker()
    monkeypatch.setattr(server_mod, "_rate_limiter", None)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-123")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "ml-engineer")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: tracer)
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    def sample_tool(**_kwargs):
        return {"ok": True}

    wrapped = _audit_wrap(sample_tool)
    wrapped()

    # OTel MCP semantic conventions
    assert span.attributes["gen_ai.tool.name"] == "sample_tool"
    assert span.attributes["mcp.method.name"] == "tools/call"
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tracer.last_span_name == "tools/call sample_tool"
    # mcp.protocol.version from SDK
    from kubeflow_mcp.core.server import _MCP_PROTOCOL_VERSION

    if _MCP_PROTOCOL_VERSION:
        assert span.attributes["mcp.protocol.version"] == _MCP_PROTOCOL_VERSION
    # Custom Kubeflow enrichment
    assert span.attributes["correlation_id"] == "cid-123"
    assert span.attributes["kubeflow.persona"] == "ml-engineer"
    assert span.attributes["tool.success"] is True
    assert "tool.duration_ms" in span.attributes
    assert span.attributes["tool.args_preview"] == "{}"
    assert breaker.successes == 1
    # Verify SpanKind.SERVER is passed when OTel is available
    from kubeflow_mcp.core.server import SpanKind as _SpanKind

    if _SpanKind is not None:
        assert tracer.last_span_kwargs.get("kind") == _SpanKind.SERVER


def test_audit_wrap_sets_mcp_context_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ContextVars populated by middleware are reflected as span attributes."""
    import kubeflow_mcp.core.middleware as mw_mod
    import kubeflow_mcp.core.server as server_mod

    span = _FakeSpan()
    tracer = _FakeTracer(span)
    breaker = _FakeBreaker()
    monkeypatch.setattr(server_mod, "_rate_limiter", None)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-ctx")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "readonly")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: tracer)
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    # Simulate what AuditIdentityMiddleware does: set ContextVars
    mw_mod._session_id_var.set("sess-abc")
    mw_mod._request_id_var.set("42")
    mw_mod._user_id_var.set("alice@example.com")

    def sample_tool(**_kwargs):
        return {"ok": True}

    try:
        wrapped = _audit_wrap(sample_tool)
        wrapped()

        assert span.attributes["mcp.session.id"] == "sess-abc"
        assert span.attributes["mcp.request.id"] == "42"
        assert span.attributes["user.id"] == "alice@example.com"
    finally:
        # Clean up ContextVars
        mw_mod._session_id_var.set(None)
        mw_mod._request_id_var.set(None)
        mw_mod._user_id_var.set(None)


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
    assert span.attributes["error.type"] == "RuntimeError"
    assert breaker.failures == 1
    # record_exception is no longer called manually; start_as_current_span
    # auto-records it.  Instead, set_status(ERROR) must be called.
    assert len(span.exceptions) == 0
    from kubeflow_mcp.core.server import _StatusCode

    if _StatusCode is not None:
        status = span.status_code
        assert status.status_code == _StatusCode.ERROR
        assert status.description == "boom"


def test_audit_wrap_circuit_breaker_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Circuit-open path sets tool.success=False on the span before the early return."""
    import kubeflow_mcp.core.server as server_mod

    span = _FakeSpan()
    tracer = _FakeTracer(span)
    breaker = _FakeBreaker(can_execute=False)
    monkeypatch.setattr(server_mod, "_rate_limiter", None)
    monkeypatch.setattr(server_mod, "with_correlation_id", lambda: "cid-456")
    monkeypatch.setattr(server_mod, "get_effective_persona", lambda: "readonly")
    monkeypatch.setattr(server_mod, "get_tracer", lambda _name: tracer)
    monkeypatch.setattr(server_mod, "get_breaker", lambda _tool: breaker)

    def sample_tool(**_kwargs):
        return {"ok": True}

    wrapped = _audit_wrap(sample_tool)
    result = wrapped()

    assert span.attributes["tool.success"] is False
    assert "tool.duration_ms" in span.attributes
    assert "error" in result
    assert result["error_code"] == "CIRCUIT_OPEN"
