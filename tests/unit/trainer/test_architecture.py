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

"""Tests for the plugin architecture: instruction composition, resource loading,
persona gating, tool metadata consistency, and tier derivation."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common.constants import TOOL_PHASES, TOOL_TO_PHASE
from kubeflow_mcp.core.policy import DESTRUCTIVE_TOOLS, get_allowed_tools
from kubeflow_mcp.core.server import (
    _build_server_instructions,
    _derive_tier,
    _sections_for_persona,
)
from kubeflow_mcp.trainer import (
    CLIENT_RESOURCES,
    CLIENT_TOOL_ANNOTATIONS,
    CLIENT_TOOL_DESCRIPTIONS,
    INSTRUCTION_SECTIONS,
    PHASE_TO_SECTION,
    TOOLS,
)

# ─── Tool metadata consistency ──────────────────────────────────────────────


class TestToolMetadataConsistency:
    def test_all_tools_have_descriptions(self):
        tool_names = {t.__name__ for t in TOOLS}
        desc_names = set(CLIENT_TOOL_DESCRIPTIONS.keys())
        assert tool_names == desc_names

    def test_all_tools_have_annotations(self):
        tool_names = {t.__name__ for t in TOOLS}
        ann_names = set(CLIENT_TOOL_ANNOTATIONS.keys())
        assert tool_names == ann_names

    def test_all_tools_in_tool_phases(self):
        tool_names = {t.__name__ for t in TOOLS}
        phased_tools = set(TOOL_TO_PHASE.keys())
        assert tool_names == phased_tools

    def test_annotation_schema(self):
        required_keys = {"title", "readOnlyHint", "destructiveHint", "idempotentHint", "tags"}
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            missing = required_keys - set(ann.keys())
            assert not missing, f"Tool '{name}' missing annotation keys: {missing}"

    def test_destructive_tools_have_destructive_hint(self):
        for tool_name in DESTRUCTIVE_TOOLS:
            if tool_name in CLIENT_TOOL_ANNOTATIONS:
                assert CLIENT_TOOL_ANNOTATIONS[tool_name]["destructiveHint"] is True, (
                    f"Tool '{tool_name}' is in DESTRUCTIVE_TOOLS but destructiveHint is False"
                )

    def test_platform_tools_registered(self):
        tool_names = {t.__name__ for t in TOOLS}
        platform_tools = {
            "inspect_crd",
            "inspect_controller",
            "patch_runtime",
            "create_runtime",
            "delete_runtime",
        }
        assert platform_tools.issubset(tool_names)


# ─── Persona gating ────────────────────────────────────────────────────────


class TestPersonaGating:
    def test_readonly_has_preflight(self):
        tools = get_allowed_tools("readonly")
        assert "pre_flight" in tools
        assert "check_compatibility" in tools

    def test_readonly_cannot_train(self):
        tools = get_allowed_tools("readonly")
        assert "fine_tune" not in tools
        assert "run_custom_training" not in tools

    def test_data_scientist_inherits_readonly(self):
        readonly_tools = get_allowed_tools("readonly")
        ds_tools = get_allowed_tools("data-scientist")
        assert readonly_tools.issubset(ds_tools)

    def test_ml_engineer_inherits_data_scientist(self):
        ds_tools = get_allowed_tools("data-scientist")
        eng_tools = get_allowed_tools("ml-engineer")
        assert ds_tools.issubset(eng_tools)

    def test_ml_engineer_has_crd_tools(self):
        tools = get_allowed_tools("ml-engineer")
        assert "inspect_crd" in tools
        assert "inspect_controller" in tools

    def test_ml_engineer_cannot_delete_runtime(self):
        tools = get_allowed_tools("ml-engineer")
        assert "delete_runtime" not in tools
        assert "create_runtime" not in tools

    def test_platform_admin_unrestricted(self):
        tools = get_allowed_tools("platform-admin")
        assert tools is None

    def test_unknown_persona_raises(self):
        with pytest.raises(ValueError, match="Unknown persona"):
            get_allowed_tools("nonexistent")

    def test_delete_runtime_is_destructive(self):
        assert "delete_runtime" in DESTRUCTIVE_TOOLS


# ─── Instruction composition ───────────────────────────────────────────────


@dataclass
class TierTestCase:
    name: str
    tier: str
    full_text: str
    expected_contains: str | None = None
    expected_not_contains: str | None = None


class TestInstructionComposition:
    def test_sections_for_readonly(self):
        sections = _sections_for_persona("readonly")
        assert "planning" in sections
        assert "monitoring" in sections
        assert "training" not in sections
        assert "platform" not in sections

    def test_sections_for_data_scientist(self):
        sections = _sections_for_persona("data-scientist")
        assert "training" in sections
        assert "platform" not in sections

    def test_sections_for_platform_admin(self):
        sections = _sections_for_persona("platform-admin")
        assert "platform" in sections

    def test_section_order(self):
        sections = _sections_for_persona("platform-admin")
        expected_order = ["planning", "monitoring", "training", "platform"]
        assert sections == expected_order

    @pytest.mark.parametrize(
        "test_case",
        [
            TierTestCase(
                name="full tier preserves all content",
                tier="full",
                full_text="Use pre_flight() first.\nRead trainer://guides/platform-fixes.",
                expected_contains="trainer://",
            ),
            TierTestCase(
                name="compact strips resource refs",
                tier="compact",
                full_text="Use pre_flight() first.\nRead trainer://guides/platform-fixes.",
                expected_not_contains="trainer://",
            ),
            TierTestCase(
                name="compact preserves non-resource lines",
                tier="compact",
                full_text="Use pre_flight() first.\nRead trainer://guides/platform-fixes.",
                expected_contains="pre_flight",
            ),
            TierTestCase(
                name="minimal extracts tool names",
                tier="minimal",
                full_text="Call pre_flight() first, then fine_tune().",
                expected_contains="pre_flight, fine_tune",
            ),
        ],
    )
    def test_derive_tier(self, test_case: TierTestCase):
        result = _derive_tier(test_case.full_text, test_case.tier)
        if test_case.expected_contains:
            assert test_case.expected_contains in result, (
                f"[{test_case.name}] Expected '{test_case.expected_contains}' in result"
            )
        if test_case.expected_not_contains:
            assert test_case.expected_not_contains not in result, (
                f"[{test_case.name}] Did not expect '{test_case.expected_not_contains}' in result"
            )

    def test_instruction_sections_have_full_tier(self):
        for section_name, tiers in INSTRUCTION_SECTIONS.items():
            assert "full" in tiers, f"Section '{section_name}' missing 'full' tier"
            assert len(tiers["full"]) > 10, f"Section '{section_name}' full tier is too short"

    def test_build_instructions_includes_header(self):
        import importlib

        modules = {"trainer": importlib.import_module("kubeflow_mcp.trainer")}
        instructions = _build_server_instructions(modules, "readonly", "full")
        assert "Kubeflow MCP Server" in instructions

    def test_build_instructions_full_includes_resources(self):
        import importlib

        modules = {"trainer": importlib.import_module("kubeflow_mcp.trainer")}
        instructions = _build_server_instructions(modules, "platform-admin", "full")
        assert "RESOURCES" in instructions
        assert "trainer://guides/training-patterns" in instructions

    def test_build_instructions_compact_excludes_resources_section(self):
        import importlib

        modules = {"trainer": importlib.import_module("kubeflow_mcp.trainer")}
        instructions = _build_server_instructions(modules, "platform-admin", "compact")
        assert "RESOURCES (read on demand):" not in instructions

    def test_build_instructions_minimal_is_short(self):
        import importlib

        modules = {"trainer": importlib.import_module("kubeflow_mcp.trainer")}
        full = _build_server_instructions(modules, "platform-admin", "full")
        minimal = _build_server_instructions(modules, "platform-admin", "minimal")
        assert len(minimal) < len(full) * 0.3


# ─── Resource loading ──────────────────────────────────────────────────────


class TestResourceLoading:
    def test_client_resources_defined(self):
        assert len(CLIENT_RESOURCES) == 3

    def test_resource_files_exist(self):
        from pathlib import Path

        import kubeflow_mcp.trainer as trainer_module

        base = Path(trainer_module.__file__).parent
        for uri, (filename, _desc) in CLIENT_RESOURCES.items():
            path = base / filename
            assert path.exists(), f"Resource file missing: {path} (URI: {uri})"
            content = path.read_text()
            assert len(content) > 50, f"Resource file too small: {path}"

    def test_resource_uris_use_trainer_scheme(self):
        for uri in CLIENT_RESOURCES:
            assert uri.startswith("trainer://"), f"URI '{uri}' should use trainer:// scheme"

    def test_register_resources_with_mock_mcp(self):
        import kubeflow_mcp.trainer as trainer_module
        from kubeflow_mcp.core.resources import register_resources

        mock_mcp = MagicMock()
        mock_mcp.resource.return_value = lambda fn: fn

        register_resources(mock_mcp, {"trainer": trainer_module})
        assert mock_mcp.resource.call_count == 3


# ─── PHASE_TO_SECTION mapping ─────────────────────────────────────────────


class TestPhaseToSection:
    def test_all_phases_mapped(self):
        for phase in TOOL_PHASES:
            assert phase in PHASE_TO_SECTION, f"Phase '{phase}' not in PHASE_TO_SECTION"

    def test_discovery_maps_to_none(self):
        assert PHASE_TO_SECTION["discovery"] is None

    def test_platform_maps_to_platform(self):
        assert PHASE_TO_SECTION["platform"] == "platform"


# ─── Config and CLI ───────────────────────────────────────────────────────


class TestConfig:
    def test_instruction_tier_in_config(self):
        from kubeflow_mcp.core.config import ServerConfig

        config = ServerConfig()
        assert config.instruction_tier == "full"

    def test_instruction_tier_custom(self):
        from kubeflow_mcp.core.config import ServerConfig

        config = ServerConfig(instruction_tier="compact")
        assert config.instruction_tier == "compact"

    def test_load_config_env_override(self):
        from kubeflow_mcp.core.config import load_config

        with patch.dict("os.environ", {"KUBEFLOW_MCP_INSTRUCTION_TIER": "minimal"}):
            cfg = load_config()
            assert cfg.server.instruction_tier == "minimal"
