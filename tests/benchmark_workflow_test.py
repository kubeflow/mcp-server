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

"""Regression coverage for benchmark setup and output publication."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from kubeflow_mcp.core import dynamic_tools, policy, server
from tests.benchmarks import TOOL_LIST_MODES
from tests.benchmarks import conftest as benchmark_hooks
from tests.benchmarks import test_token_usage as tool_list_benchmark


async def test_tool_list_benchmark_fails_if_trainer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(server.CLIENT_MODULES, "trainer", "missing_benchmark_trainer")
    record_result = Mock()

    with pytest.raises(ImportError, match="missing_benchmark_trainer"):
        await tool_list_benchmark.test_trainer_tool_list_size(
            "semantic", record_result, monkeypatch
        )

    record_result.assert_not_called()


async def test_tool_list_benchmark_preserves_dynamic_tool_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dynamic_tools, "TOOL_REGISTRY", {"existing_tool": {}})
    monkeypatch.setattr(dynamic_tools, "TOOL_HIERARCHY", {"phase": ["existing_tool"]})
    cache = dynamic_tools._EmbeddingCache()
    cache._mark_unavailable()
    monkeypatch.setattr(dynamic_tools, "_embedding_cache", cache)
    monkeypatch.setattr(policy, "_effective_persona", "platform-admin")

    with monkeypatch.context() as benchmark_patch:
        await tool_list_benchmark.test_trainer_tool_list_size(
            "progressive", Mock(), benchmark_patch
        )

    assert policy.get_effective_persona() == "platform-admin"
    assert dynamic_tools.TOOL_REGISTRY == {"existing_tool": {}}
    assert dynamic_tools.TOOL_HIERARCHY == {"phase": ["existing_tool"]}
    assert dynamic_tools._embedding_cache is cache
    assert cache._cached() == (None, None)


@pytest.mark.parametrize(
    ("exit_status", "modes", "should_publish"),
    [
        (pytest.ExitCode.OK, TOOL_LIST_MODES, True),
        (pytest.ExitCode.OK, ("semantic",), False),
        (pytest.ExitCode.TESTS_FAILED, TOOL_LIST_MODES, False),
    ],
)
def test_tool_list_report_requires_a_successful_comparison(
    exit_status: pytest.ExitCode,
    modes: tuple[str, ...],
    should_publish: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = tmp_path / "token_usage.json"
    html_path = tmp_path / "index.html"
    payload_path.write_text("previous result")
    html_path.write_text("previous report")
    monkeypatch.setattr(benchmark_hooks, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(benchmark_hooks, "LATENCY_RESULTS", [])
    monkeypatch.setattr(benchmark_hooks, "TOOL_LIST_RESULTS", [{"mode": mode} for mode in modes])

    benchmark_hooks.pytest_terminal_summary(Mock(), exit_status, None)

    if should_publish:
        assert [
            result["mode"] for result in json.loads(payload_path.read_text())["results"]
        ] == list(TOOL_LIST_MODES)
        assert "Initial tools/list payload" in html_path.read_text()
    else:
        assert payload_path.read_text() == "previous result"
        assert html_path.read_text() == "previous report"
