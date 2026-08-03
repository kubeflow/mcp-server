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

"""Architecture tests for the optimizer module.

Validates that the module structure, exports, and metadata match the
conventions established by the trainer module (KEP-34 compliance).
"""

import inspect
from typing import Any

from kubeflow_mcp.common.constants import TOOL_NEXT_HINTS, TOOL_PHASES, TOOL_TO_PHASE
from kubeflow_mcp.core.policy import DESTRUCTIVE_TOOLS, PERSONAS


class TestModuleExports:
    """Verify optimizer module exports match trainer conventions."""

    def test_module_info_fields(self):
        from kubeflow_mcp.optimizer import MODULE_INFO

        assert MODULE_INFO["name"] == "optimizer"
        assert MODULE_INFO["status"] == "implemented"
        assert MODULE_INFO["sdk_client"] == "kubeflow.optimizer.OptimizerClient"
        assert MODULE_INFO["sdk_version"] == ">=0.4.0"

    def test_tools_list_not_empty(self):
        from kubeflow_mcp.optimizer import TOOLS

        assert len(TOOLS) == 17, f"Expected 17 tools, got {len(TOOLS)}"

    def test_all_tools_are_callables(self):
        from kubeflow_mcp.optimizer import TOOLS

        for tool in TOOLS:
            assert callable(tool), f"{tool} is not callable"
            assert hasattr(tool, "__name__"), f"{tool} has no __name__"

    def test_tool_names_are_unique(self):
        from kubeflow_mcp.optimizer import TOOLS

        names = [t.__name__ for t in TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_all_tools_have_descriptions(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_DESCRIPTIONS, TOOLS

        tool_names = {t.__name__ for t in TOOLS}
        desc_names = set(CLIENT_TOOL_DESCRIPTIONS.keys())
        missing = tool_names - desc_names
        assert not missing, f"Tools missing descriptions: {missing}"

    def test_all_tools_have_annotations(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS, TOOLS

        tool_names = {t.__name__ for t in TOOLS}
        ann_names = set(CLIENT_TOOL_ANNOTATIONS.keys())
        missing = tool_names - ann_names
        assert not missing, f"Tools missing annotations: {missing}"

    def test_annotations_have_required_fields(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS

        required = {"title", "readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            missing = required - set(ann.keys())
            assert not missing, f"{name} annotation missing: {missing}"

    def test_all_tools_return_dict(self):
        """All tool functions must have return type dict[str, Any]."""
        from kubeflow_mcp.optimizer import TOOLS

        for tool in TOOLS:
            hints = inspect.get_annotations(tool)
            assert "return" in hints, f"{tool.__name__} missing return annotation"
            assert hints["return"] == dict[str, Any], (
                f"{tool.__name__} return type should be dict[str, Any], got {hints['return']}"
            )

    def test_client_resources_exist(self):
        from kubeflow_mcp.optimizer import CLIENT_RESOURCES

        assert len(CLIENT_RESOURCES) >= 2
        for uri, (path, _desc) in CLIENT_RESOURCES.items():
            assert uri.startswith("optimizer://"), (
                f"Resource URI should start with optimizer://: {uri}"
            )
            assert path.endswith(".md"), f"Resource path should end with .md: {path}"

    def test_instruction_sections_exist(self):
        from kubeflow_mcp.optimizer import INSTRUCTION_SECTIONS

        assert "optimizer_planning" in INSTRUCTION_SECTIONS
        assert "optimizer_optimization" in INSTRUCTION_SECTIONS
        assert "optimizer_monitoring" in INSTRUCTION_SECTIONS

    def test_phase_to_section_mapping(self):
        from kubeflow_mcp.optimizer import PHASE_TO_SECTION

        assert PHASE_TO_SECTION["optimizer_planning"] == "optimizer_planning"
        assert PHASE_TO_SECTION["optimizer_discovery"] is None
        assert PHASE_TO_SECTION["optimizer_optimization"] == "optimizer_optimization"
        assert PHASE_TO_SECTION["optimizer_monitoring"] == "optimizer_monitoring"
        assert PHASE_TO_SECTION["optimizer_lifecycle"] == "optimizer_monitoring"


class TestToolPhaseRegistration:
    """Verify all optimizer tools are registered in TOOL_PHASES."""

    def test_all_tools_in_phases(self):
        from kubeflow_mcp.optimizer import TOOLS

        tool_names = {t.__name__ for t in TOOLS}
        phase_tools = set()
        for phase, tools in TOOL_PHASES.items():
            if phase.startswith("optimizer_"):
                phase_tools.update(tools)

        missing = tool_names - phase_tools
        assert not missing, f"Tools not in any optimizer phase: {missing}"

        extra = phase_tools - tool_names
        assert not extra, f"Phases reference non-existent tools: {extra}"

    def test_all_tools_in_next_hints(self):
        from kubeflow_mcp.optimizer import TOOLS

        tool_names = {t.__name__ for t in TOOLS}
        hint_names = set(TOOL_NEXT_HINTS.keys())
        missing = tool_names - hint_names
        assert not missing, f"Tools missing TOOL_NEXT_HINTS: {missing}"

    def test_tool_to_phase_reverse_mapping(self):
        from kubeflow_mcp.optimizer import TOOLS

        tool_names = {t.__name__ for t in TOOLS}
        for name in tool_names:
            assert name in TOOL_TO_PHASE, f"{name} not in TOOL_TO_PHASE reverse mapping"
            assert TOOL_TO_PHASE[name].startswith("optimizer_"), (
                f"{name} mapped to non-optimizer phase: {TOOL_TO_PHASE[name]}"
            )


class TestPersonaIntegration:
    """Verify optimizer tools are in the correct persona allowlists (KEP-34)."""

    def _resolve_persona_tools(self, persona: str) -> set[str]:
        """Resolve full tool set including inherited tools."""
        config = PERSONAS[persona]
        if config["tools"] == "*":
            return {"*"}
        tools = set(config["tools"])
        if "inherit" in config:
            tools |= self._resolve_persona_tools(config["inherit"])
        return tools

    def test_readonly_has_13_optimizer_tools(self):
        """KEP-34: readonly gets all 13 read-only optimizer tools."""
        readonly_tools = self._resolve_persona_tools("readonly")
        optimizer_readonly = {
            "katib_pre_flight",
            "list_experiments",
            "get_experiment",
            "get_experiment_status",
            "get_trial",
            "get_successful_trials",
            "list_suggestions",
            "get_experiment_trials",
            "get_best_trial",
            "get_suggestion",
            "wait_for_experiment",
            "get_experiment_trial_logs",
            "get_experiment_events",
        }
        assert optimizer_readonly <= readonly_tools, (
            f"Missing from readonly: {optimizer_readonly - readonly_tools}"
        )

    def test_data_scientist_has_create_and_delete(self):
        """KEP-34: data-scientist adds create_hpo_experiment + delete_experiment."""
        ds_tools = self._resolve_persona_tools("data-scientist")
        assert "create_hpo_experiment" in ds_tools
        assert "delete_experiment" in ds_tools

    def test_data_scientist_lacks_advanced_tools(self):
        """KEP-34: data-scientist should NOT have create_experiment_from_spec."""
        ds_tools = self._resolve_persona_tools("data-scientist")
        assert "create_experiment_from_spec" not in ds_tools
        assert "update_experiment" not in ds_tools

    def test_ml_engineer_has_advanced_tools(self):
        """KEP-34: ml-engineer adds create_experiment_from_spec + update_experiment."""
        mle_tools = self._resolve_persona_tools("ml-engineer")
        assert "create_experiment_from_spec" in mle_tools
        assert "update_experiment" in mle_tools

    def test_platform_admin_unrestricted(self):
        assert PERSONAS["platform-admin"]["tools"] == "*"

    def test_delete_experiment_is_destructive(self):
        assert "delete_experiment" in DESTRUCTIVE_TOOLS


class TestAnnotationSemantics:
    """Verify annotation hints match tool behavior (KEP-34 MCP Tools tables)."""

    def test_read_only_tools_marked_correctly(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS

        read_only_tools = {
            "katib_pre_flight",
            "list_experiments",
            "get_experiment",
            "get_experiment_status",
            "get_trial",
            "get_successful_trials",
            "list_suggestions",
            "get_experiment_trials",
            "get_best_trial",
            "get_suggestion",
            "wait_for_experiment",
            "get_experiment_trial_logs",
            "get_experiment_events",
        }
        for name in read_only_tools:
            ann = CLIENT_TOOL_ANNOTATIONS[name]
            assert ann["readOnlyHint"] is True, f"{name} should be readOnlyHint=True"
            assert ann["destructiveHint"] is False, f"{name} should be destructiveHint=False"

    def test_mutating_tools_marked_correctly(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS

        for name in ("create_hpo_experiment", "create_experiment_from_spec"):
            ann = CLIENT_TOOL_ANNOTATIONS[name]
            assert ann["readOnlyHint"] is False
            assert ann["idempotentHint"] is False  # create is not idempotent

    def test_delete_experiment_is_destructive(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS

        ann = CLIENT_TOOL_ANNOTATIONS["delete_experiment"]
        assert ann["destructiveHint"] is True
        assert ann["idempotentHint"] is True

    def test_update_experiment_not_destructive(self):
        from kubeflow_mcp.optimizer import CLIENT_TOOL_ANNOTATIONS

        ann = CLIENT_TOOL_ANNOTATIONS["update_experiment"]
        assert ann["readOnlyHint"] is False
        assert ann["destructiveHint"] is False
        assert ann["idempotentHint"] is True


class TestCrossClientHints:
    """Verify cross-client TOOL_NEXT_HINTS bridge trainer ↔ optimizer (KEP-34)."""

    def test_wait_for_training_hints_hpo(self):
        hint = TOOL_NEXT_HINTS["wait_for_training"]
        assert "create_hpo_experiment" in hint

    def test_get_best_trial_hints_fine_tune(self):
        hint = TOOL_NEXT_HINTS["get_best_trial"]
        assert "fine_tune" in hint

    def test_create_hpo_hints_monitoring(self):
        hint = TOOL_NEXT_HINTS["create_hpo_experiment"]
        assert "get_experiment_status" in hint or "wait_for_experiment" in hint

    def test_wait_for_experiment_hints_best_trial(self):
        hint = TOOL_NEXT_HINTS["wait_for_experiment"]
        assert "get_best_trial" in hint
