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

"""OpenTelemetry tracing and metrics helpers with safe no-op fallback."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

try:
    # ``Resource`` is imported by the tracing block above, so metrics are only
    # considered available when the base SDK import succeeded as well.
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    _METRICS_AVAILABLE = _OTEL_AVAILABLE
except ImportError:
    _METRICS_AVAILABLE = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exception: BaseException, **kwargs: Any) -> None:
        return None

    def set_status(self, status: Any, description: str | None = None) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Generator[_NoopSpan, None, None]:
        yield _NoopSpan()


class _NoopInstrument:
    """Stand-in for a metric instrument when OTel metrics are unavailable."""

    def add(self, amount: int | float, attributes: Any = None, **kwargs: Any) -> None:
        return None

    def record(self, amount: int | float, attributes: Any = None, **kwargs: Any) -> None:
        return None

    def set(self, amount: int | float, attributes: Any = None, **kwargs: Any) -> None:
        return None


class _NoopMeter:
    def create_counter(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_gauge(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_observable_counter(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()

    def create_observable_up_down_counter(self, name: str, **kwargs: Any) -> _NoopInstrument:
        return _NoopInstrument()


_NOOP_TRACER = _NoopTracer()
_NOOP_METER = _NoopMeter()
_tracing_initialized = False
_configured_endpoint: str | None = None
_setup_lock = threading.Lock()

_metrics_initialized = False
_configured_metrics_endpoint: str | None = None
_prometheus_enabled = False
_metrics_setup_lock = threading.Lock()


def _normalize_and_validate_endpoint(endpoint: str | None, signal_path: str = "/v1/traces") -> str:
    """Normalize and validate OTLP endpoint."""
    normalized_endpoint = endpoint.strip() if endpoint is not None else ""
    if not normalized_endpoint:
        return ""

    parsed = urlparse(normalized_endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Invalid OpenTelemetry endpoint. Use a full HTTP(S) URL, "
            "for example: http://localhost:4318"
        )
    # Append the standard OTLP signal path when only a base URL is provided,
    # matching the OTel SDK convention for OTEL_EXPORTER_OTLP_ENDPOINT.
    if not parsed.path or parsed.path == "/":
        normalized_endpoint = parsed._replace(path=signal_path).geturl()
    return normalized_endpoint


def setup_tracing(endpoint: str | None = None, service_name: str = "kubeflow-mcp-server") -> bool:
    """Configure OpenTelemetry tracing.

    Returns True when tracing is configured, False when disabled/unavailable.
    """
    global _configured_endpoint, _tracing_initialized

    # Fall back to standard OTel env var when no explicit endpoint is provided
    if not endpoint or not endpoint.strip():
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    normalized_endpoint = _normalize_and_validate_endpoint(endpoint)
    if not normalized_endpoint:
        return False

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry endpoint configured but OTel packages are not installed. "
            "Install OpenTelemetry dependencies from source with: uv sync --group otel"
        )
        return False

    with _setup_lock:
        if _tracing_initialized:
            if _configured_endpoint and _configured_endpoint != normalized_endpoint:
                logger.warning(
                    "Tracing already initialized for endpoint '%s'; ignoring new endpoint '%s'.",
                    _configured_endpoint,
                    normalized_endpoint,
                )
            return True

        exporter = OTLPSpanExporter(
            endpoint=normalized_endpoint,
            timeout=2,
        )
        processor = BatchSpanProcessor(exporter, export_timeout_millis=2000)
        current_provider = _otel_trace.get_tracer_provider()

        if hasattr(current_provider, "add_span_processor"):
            current_provider.add_span_processor(processor)
        else:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(processor)
            _otel_trace.set_tracer_provider(provider)
            atexit.register(provider.shutdown)

        _tracing_initialized = True
        _configured_endpoint = normalized_endpoint
        return True


def get_tracer(name: str = "kubeflow_mcp") -> Any:
    """Return OpenTelemetry tracer or no-op tracer when OTel is unavailable."""
    if not _OTEL_AVAILABLE:
        return _NOOP_TRACER
    return _otel_trace.get_tracer(name)


def setup_metrics(
    endpoint: str | None = None,
    service_name: str = "kubeflow-mcp-server",
    enable_prometheus: bool = True,
) -> bool:
    """Configure the OpenTelemetry MeterProvider.

    A Prometheus reader is attached whenever the exporter is installed, so a
    scrape endpoint can serve from the default ``prometheus_client`` registry.
    An OTLP reader is added on top when an endpoint is configured. Returns True
    when a meter provider is in place, False when metrics are disabled or the
    optional dependencies are missing.
    """
    global _configured_metrics_endpoint, _metrics_initialized, _prometheus_enabled

    # Fall back to standard OTel env var when no explicit endpoint is provided
    if not endpoint or not endpoint.strip():
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    normalized_endpoint = _normalize_and_validate_endpoint(endpoint, signal_path="/v1/metrics")

    if not _METRICS_AVAILABLE:
        if normalized_endpoint:
            logger.warning(
                "OpenTelemetry endpoint configured but OTel packages are not installed. "
                "Install OpenTelemetry dependencies from source with: uv sync --group otel"
            )
        return False

    with _metrics_setup_lock:
        if _metrics_initialized:
            if _configured_metrics_endpoint and _configured_metrics_endpoint != normalized_endpoint:
                logger.warning(
                    "Metrics already initialized for endpoint '%s'; ignoring new endpoint '%s'.",
                    _configured_metrics_endpoint,
                    normalized_endpoint,
                )
            return True

        readers: list[Any] = []
        prometheus_reader_added = False
        if enable_prometheus:
            if _PROMETHEUS_AVAILABLE:
                readers.append(PrometheusMetricReader())
                prometheus_reader_added = True
            else:
                logger.warning(
                    "Prometheus metrics requested but the exporter is not installed. "
                    "Install OpenTelemetry dependencies from source with: uv sync --group otel"
                )

        if normalized_endpoint:
            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=normalized_endpoint, timeout=2)
                )
            )

        if not readers:
            return False

        # Readers cannot be attached to an existing MeterProvider, and replacing
        # a global provider would silently drop whoever installed it first.
        current_provider = _otel_metrics.get_meter_provider()
        if isinstance(current_provider, MeterProvider):
            logger.warning(
                "A MeterProvider is already installed globally; "
                "skipping kubeflow-mcp meter provider setup."
            )
            _metrics_initialized = True
            return True

        resource = Resource.create({"service.name": service_name})
        provider = MeterProvider(resource=resource, metric_readers=readers)
        _otel_metrics.set_meter_provider(provider)
        atexit.register(provider.shutdown)

        _metrics_initialized = True
        _configured_metrics_endpoint = normalized_endpoint or None
        _prometheus_enabled = prometheus_reader_added
        return True


def get_meter(name: str = "kubeflow_mcp") -> Any:
    """Return OpenTelemetry meter or no-op meter when OTel metrics are unavailable."""
    if not _METRICS_AVAILABLE:
        return _NOOP_METER
    return _otel_metrics.get_meter(name)


def metrics_enabled() -> bool:
    """Return True when a meter provider has been configured."""
    return _metrics_initialized


def prometheus_enabled() -> bool:
    """Return True when metrics are collected by a Prometheus reader."""
    return _prometheus_enabled
