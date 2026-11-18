# Copyright The Kubeflow Authors.
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

"""Module contract, persona and instruction tests for the pipelines client."""

import inspect
from pathlib import Path

import pytest

from kubeflow_mcp.common.constants import SDK_COMPATIBILITY, TOOL_NEXT_HINTS, TOOL_TO_PHASE
from kubeflow_mcp.core.policy import DESTRUCTIVE_TOOLS, get_allowed_tools
from kubeflow_mcp.core.server import CLIENT_MODULES, _build_server_instructions
from kubeflow_mcp.pipelines import (
    CLIENT_RESOURCES,
    CLIENT_TOOL_ANNOTATIONS,
    CLIENT_TOOL_DESCRIPTIONS,
    INSTRUCTION_SECTIONS,
    MODULE_INFO,
    PHASE_TO_SECTION,
    SECTION_ORDER,
    TOOLS,
)

TOOL_NAMES = {t.__name__ for t in TOOLS}
READ_TOOLS = {
    "list_pipelines",
    "get_pipeline",
    "list_pipeline_runs",
    "get_pipeline_run",
    "get_pipeline_logs",
    "list_experiments",
    "get_experiment",
}
MUTATING_TOOLS = {
    "create_pipeline_run",
    "stop_pipeline_run",
    "retry_pipeline_run",
    "create_experiment",
    "delete_experiment",
    "create_pipeline",
    "delete_pipeline",
}
ADMIN_ONLY_TOOLS = {"delete_experiment", "delete_pipeline"}
ANNOTATION_KEYS = {
    "title",
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
    "tags",
}


class TestModuleContract:
    def test_registered_with_the_server(self):
        assert CLIENT_MODULES["pipelines"] == "kubeflow_mcp.pipelines"

    def test_tools_match_the_issue(self):
        assert TOOL_NAMES == READ_TOOLS | MUTATING_TOOLS

    def test_every_tool_has_description_and_annotations(self):
        assert set(CLIENT_TOOL_DESCRIPTIONS) == TOOL_NAMES
        assert set(CLIENT_TOOL_ANNOTATIONS) == TOOL_NAMES
        for name, annotation in CLIENT_TOOL_ANNOTATIONS.items():
            assert ANNOTATION_KEYS <= set(annotation), name

    def test_every_tool_has_a_phase_tag_and_hint(self):
        for name in TOOL_NAMES:
            phase = TOOL_TO_PHASE[name]
            assert phase.startswith("pipelines_"), name
            assert phase in CLIENT_TOOL_ANNOTATIONS[name]["tags"], name
            assert name in TOOL_NEXT_HINTS, name

    def test_read_only_hints(self):
        for name in READ_TOOLS:
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is True, name
        for name in MUTATING_TOOLS:
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is False, name

    def test_mutating_tools_have_the_confirm_gate(self):
        for tool in TOOLS:
            if tool.__name__ in MUTATING_TOOLS:
                param = inspect.signature(tool).parameters["confirmed"]
                assert param.default is False, tool.__name__

    def test_stop_is_marked_destructive(self):
        assert CLIENT_TOOL_ANNOTATIONS["stop_pipeline_run"]["destructiveHint"] is True
        assert CLIENT_TOOL_DESCRIPTIONS["stop_pipeline_run"].startswith("[DESTRUCTIVE]")
        assert "stop_pipeline_run" in DESTRUCTIVE_TOOLS

    @pytest.mark.parametrize("name", sorted({"delete_experiment", "delete_pipeline"}))
    def test_deletes_are_marked_destructive(self, name):
        assert CLIENT_TOOL_ANNOTATIONS[name]["destructiveHint"] is True
        assert CLIENT_TOOL_DESCRIPTIONS[name].startswith("[DESTRUCTIVE]")
        assert name in DESTRUCTIVE_TOOLS

    def test_destructive_set_matches_hints(self):
        for name in TOOL_NAMES:
            hint = CLIENT_TOOL_ANNOTATIONS[name]["destructiveHint"]
            assert hint is (name in DESTRUCTIVE_TOOLS), name

    def test_hints_never_skip_the_preview(self):
        for name in TOOL_NAMES:
            assert "confirmed=True" not in TOOL_NEXT_HINTS[name], name

    def test_sdk_compatibility_entry(self):
        entry = SDK_COMPATIBILITY["clients"]["pipelines"]
        assert entry["status"] == MODULE_INFO["status"] == "implemented"
        assert entry["sdk_client"] == MODULE_INFO["sdk_client"]

    def test_resource_files_exist(self):
        root = Path(__file__).resolve().parents[1]
        for uri, (path, _) in CLIENT_RESOURCES.items():
            assert uri.startswith("pipelines://guides/"), uri
            assert (root / path).is_file(), path


class TestPersonas:
    def test_readonly_gets_only_read_tools(self):
        allowed = get_allowed_tools("readonly")
        assert READ_TOOLS <= allowed
        assert not MUTATING_TOOLS & allowed

    def test_data_scientist_can_create_but_not_stop_or_retry(self):
        allowed = get_allowed_tools("data-scientist")
        assert "create_pipeline_run" in allowed
        assert "stop_pipeline_run" not in allowed
        assert "retry_pipeline_run" not in allowed

    def test_data_scientist_can_create_experiments_but_not_pipelines(self):
        allowed = get_allowed_tools("data-scientist")
        assert "create_experiment" in allowed
        assert "create_pipeline" not in allowed
        assert "delete_experiment" not in allowed

    def test_ml_engineer_gets_every_tool_but_the_admin_deletes(self):
        allowed = get_allowed_tools("ml-engineer")
        assert TOOL_NAMES - ADMIN_ONLY_TOOLS <= allowed
        assert not ADMIN_ONLY_TOOLS & allowed

    @pytest.mark.parametrize("persona", ["readonly", "data-scientist", "ml-engineer"])
    def test_admin_only_deletes(self, persona):
        assert not ADMIN_ONLY_TOOLS & get_allowed_tools(persona)

    def test_platform_admin_gets_every_tool(self):
        # None means unrestricted.
        assert get_allowed_tools("platform-admin") is None


class TestInstructions:
    def test_every_phase_maps_to_a_known_section(self):
        for name in TOOL_NAMES:
            section = PHASE_TO_SECTION[TOOL_TO_PHASE[name]]
            assert section in INSTRUCTION_SECTIONS
            assert section in SECTION_ORDER

    @pytest.mark.parametrize(
        ("persona", "has_runs"), [("readonly", False), ("data-scientist", True)]
    )
    def test_run_guidance_follows_persona(self, persona, has_runs):
        import kubeflow_mcp.pipelines as module

        text = _build_server_instructions({"pipelines": module}, persona, "full")
        assert "KUBEFLOW PIPELINES:" in text
        assert ("PIPELINE RUNS:" in text) is has_runs

    @pytest.mark.parametrize(
        ("persona", "has_authoring"),
        [("readonly", False), ("data-scientist", False), ("ml-engineer", True)],
    )
    def test_authoring_guidance_follows_persona(self, persona, has_authoring):
        import kubeflow_mcp.pipelines as module

        text = _build_server_instructions({"pipelines": module}, persona, "full")
        assert ("PIPELINE AUTHORING" in text) is has_authoring
