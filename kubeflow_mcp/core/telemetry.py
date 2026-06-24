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

"""OpenTelemetry tracing helpers with safe no-op fallback."""

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


_NOOP_TRACER = _NoopTracer()
_tracing_initialized = False
_configured_endpoint: str | None = None
_setup_lock = threading.Lock()


def _normalize_and_validate_endpoint(endpoint: str | None) -> str:
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
    # Append the standard OTLP trace path when only a base URL is provided,
    # matching the OTel SDK convention for OTEL_EXPORTER_OTLP_ENDPOINT.
    if not parsed.path or parsed.path == "/":
        normalized_endpoint = normalized_endpoint.rstrip("/") + "/v1/traces"
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
            "Install with: pip install '.[otel]'"
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
            timeout=2000,
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
