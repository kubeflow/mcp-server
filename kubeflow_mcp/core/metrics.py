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

"""Resilience metrics: rate-limit and circuit-breaker events as OpenTelemetry metrics.

Instruments are created through the OpenTelemetry *API*, which fastmcp always
installs. Until a ``MeterProvider`` is configured they are no-ops; once one is
installed (the shared metrics setup from #302) the API's proxy meter starts
recording through the same instrument objects, so this module never needs to
know whether the optional SDK is present.

Attribute values are bounded by construction: the tool name comes from the
registered tool set, and the breaker states from ``CircuitState``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as _otel_metrics
except ImportError:  # pragma: no cover - fastmcp always installs opentelemetry-api
    _otel_metrics = None  # type: ignore[assignment]

# Same key the tool spans use, so metrics and traces can be joined on it.
TOOL_NAME_ATTR = "gen_ai.tool.name"
FROM_STATE_ATTR = "circuit_breaker.from_state"
TO_STATE_ATTR = "circuit_breaker.to_state"

RATE_LIMITED_METRIC = "kubeflow_mcp.tool.rate_limited"
CIRCUIT_OPEN_METRIC = "kubeflow_mcp.tool.circuit_open"
CIRCUIT_TRANSITIONS_METRIC = "kubeflow_mcp.circuit_breaker.transitions"


class _NoopCounter:
    def add(self, amount: int, attributes: dict[str, Any] | None = None) -> None:
        return None


def _counter(name: str, description: str, unit: str) -> Any:
    """Create a counter, or a no-op stand-in when the OpenTelemetry API is missing."""
    if _otel_metrics is None:
        return _NoopCounter()
    try:
        return _otel_metrics.get_meter("kubeflow_mcp").create_counter(
            name, unit=unit, description=description
        )
    except Exception as e:  # defensive: metrics must never break startup
        logger.debug("Could not create metric %s: %s", name, e)
        return _NoopCounter()


_rate_limited = _counter(
    RATE_LIMITED_METRIC,
    "Tool calls rejected by the rate limiter.",
    unit="{call}",
)
_circuit_open = _counter(
    CIRCUIT_OPEN_METRIC,
    "Tool calls rejected because the tool's circuit breaker was open.",
    unit="{call}",
)
_circuit_transitions = _counter(
    CIRCUIT_TRANSITIONS_METRIC,
    "Circuit breaker state transitions.",
    unit="{transition}",
)


def record_rate_limited(tool: str) -> None:
    """Count a tool call refused by the rate limiter."""
    _rate_limited.add(1, {TOOL_NAME_ATTR: tool})


def record_circuit_open(tool: str) -> None:
    """Count a tool call refused because the circuit for *tool* was open."""
    _circuit_open.add(1, {TOOL_NAME_ATTR: tool})


def record_circuit_transition(tool: str, from_state: str, to_state: str) -> None:
    """Count a circuit breaker moving from *from_state* to *to_state*."""
    _circuit_transitions.add(
        1,
        {TOOL_NAME_ATTR: tool, FROM_STATE_ATTR: from_state, TO_STATE_ATTR: to_state},
    )
