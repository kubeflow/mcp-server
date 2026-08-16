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

"""Tests for common/constants.py — error classification, phase maps."""

from kubeflow_mcp.common.constants import (
    TOOL_NEXT_HINTS,
    TOOL_PHASES,
    TOOL_TO_PHASE,
    ErrorCode,
    is_infrastructure_error,
)


class TestIsInfrastructureError:
    def test_kubernetes_error_is_infrastructure(self):
        result = {"error": "api failed", "error_code": ErrorCode.KUBERNETES_ERROR}
        assert is_infrastructure_error(result) is True

    def test_timeout_is_infrastructure(self):
        result = {"error": "timed out", "error_code": ErrorCode.TIMEOUT}
        assert is_infrastructure_error(result) is True

    def test_validation_error_is_not_infrastructure(self):
        result = {"error": "bad input", "error_code": ErrorCode.VALIDATION_ERROR}
        assert is_infrastructure_error(result) is False

    def test_none_result_is_not_infrastructure(self):
        assert is_infrastructure_error(None) is False

    def test_success_result_is_not_infrastructure(self):
        assert is_infrastructure_error({"data": {"ok": True}}) is False

    def test_sdk_error_is_infrastructure(self):
        result = {"error": "sdk failed", "error_code": ErrorCode.SDK_ERROR}
        assert is_infrastructure_error(result) is True

    def test_circuit_open_is_not_infrastructure(self):
        result = {"error": "circuit open", "error_code": ErrorCode.CIRCUIT_OPEN}
        assert is_infrastructure_error(result) is False

    def test_rate_limited_is_not_infrastructure(self):
        result = {"error": "rate limited", "error_code": ErrorCode.RATE_LIMITED}
        assert is_infrastructure_error(result) is False


class TestToolPhases:
    def test_contains_all_expected_phases(self):
        expected = {
            "planning",
            "discovery",
            "training",
            "monitoring",
            "lifecycle",
            "platform",
            "health",
        }
        assert set(TOOL_PHASES.keys()) == expected

    def test_tool_to_phase_covers_all_tools(self):
        all_tools = {tool for tools in TOOL_PHASES.values() for tool in tools}
        assert set(TOOL_TO_PHASE.keys()) == all_tools

    def test_tool_to_phase_reverse_mapping_is_correct(self):
        for phase, tools in TOOL_PHASES.items():
            for tool in tools:
                assert TOOL_TO_PHASE[tool] == phase


class TestToolNextHints:
    def test_training_tools_have_hints(self):
        for tool in TOOL_PHASES["training"]:
            assert tool in TOOL_NEXT_HINTS, f"Missing hint for training tool: {tool}"

    def test_monitoring_tools_have_hints(self):
        for tool in TOOL_PHASES["monitoring"]:
            assert tool in TOOL_NEXT_HINTS, f"Missing hint for monitoring tool: {tool}"

    def test_hints_are_non_empty_strings(self):
        for tool, hint in TOOL_NEXT_HINTS.items():
            assert isinstance(hint, str), f"Hint for {tool} is not a string"
            assert len(hint) > 0, f"Hint for {tool} is empty"
