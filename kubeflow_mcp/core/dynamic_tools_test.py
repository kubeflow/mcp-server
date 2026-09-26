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

"""Tests for dynamic tool discovery, execution, and cache invalidation."""

from unittest.mock import Mock

import pytest

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.core import dynamic_tools
from kubeflow_mcp.core.resilience import CircuitState, get_breaker

_calls: list[str] = []


def probe_tool(name: str) -> dict:
    """Read-only tool used to exercise execute_tool."""
    _calls.append(name)
    return {"success": True, "data": {"name": name}}


def failing_tool(name: str) -> dict:
    """Tool that raises, standing in for an SDK or API failure."""
    raise RuntimeError("cluster unreachable")


@pytest.fixture(autouse=True)
def _registry():
    _calls.clear()
    dynamic_tools.init_dynamic_tools([probe_tool, failing_tool], {})
    yield
    dynamic_tools.TOOL_REGISTRY.clear()
    dynamic_tools.TOOL_HIERARCHY.clear()


@pytest.mark.parametrize(
    "arguments",
    [
        {"nmae": "typo"},
        {"name": "ok", "extra": "unexpected"},
        {},
    ],
)
def test_bad_arguments_return_validation_error(arguments):
    result = dynamic_tools.execute_tool("probe_tool", arguments)

    assert result["error_code"] == ErrorCode.VALIDATION_ERROR
    assert result["tool"] == "probe_tool"
    assert _calls == []


def test_bad_arguments_do_not_open_breaker():
    for _ in range(get_breaker("probe_tool").failure_threshold + 2):
        dynamic_tools.execute_tool("probe_tool", {"nmae": "typo"})

    breaker = get_breaker("probe_tool")
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0

    result = dynamic_tools.execute_tool("probe_tool", {"name": "ok"})
    assert result == {"success": True, "data": {"name": "ok"}}


def test_bad_arguments_do_not_consume_half_open_slot():
    breaker = get_breaker("probe_tool")
    breaker.state = CircuitState.HALF_OPEN
    breaker.half_open_calls = 0

    dynamic_tools.execute_tool("probe_tool", {"nmae": "typo"})

    assert breaker.half_open_calls == 0


def test_tool_exception_still_counts_as_breaker_failure():
    breaker = get_breaker("failing_tool")
    for _ in range(breaker.failure_threshold):
        result = dynamic_tools.execute_tool("failing_tool", {"name": "x"})
        assert result["error_code"] == ErrorCode.SDK_ERROR

    assert breaker.state == CircuitState.OPEN


def test_init_dynamic_tools_resets_embedding_cache(monkeypatch):
    reset = Mock()
    monkeypatch.setattr(dynamic_tools._embedding_cache, "reset", reset)

    dynamic_tools.init_dynamic_tools([], {})

    reset.assert_called_once_with()
