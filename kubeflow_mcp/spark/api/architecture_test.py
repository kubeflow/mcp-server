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

"""Architecture tests for the spark client module: tool metadata, registration
wiring, persona gating, and resource loading.

Mirrors ``trainer/api/architecture_test.py`` as required by
``docs/CONVENTIONS.md`` before a module may claim ``status: implemented``.
"""

from pathlib import Path

from kubeflow_mcp.common.constants import (
    SDK_COMPATIBILITY,
    TOOL_NEXT_HINTS,
    TOOL_PHASES,
    TOOL_TO_PHASE,
)
from kubeflow_mcp.core.policy import DESTRUCTIVE_TOOLS, get_allowed_tools
from kubeflow_mcp.core.server import CLIENT_MODULES
from kubeflow_mcp.spark import (
    CLIENT_RESOURCES,
    CLIENT_TOOL_ANNOTATIONS,
    CLIENT_TOOL_DESCRIPTIONS,
    INSTRUCTION_SECTIONS,
    MODULE_INFO,
    PHASE_TO_SECTION,
    TOOLS,
)

SPARK_TOOL_NAMES = {t.__name__ for t in TOOLS}
READ_TOOLS = {"list_spark_sessions", "get_spark_session", "get_spark_session_logs"}
WRITE_TOOLS = {"create_spark_session", "delete_spark_session"}


# ─── Tool metadata consistency ──────────────────────────────────────────────


class TestToolMetadataConsistency:
    def test_all_tools_have_descriptions(self):
        assert SPARK_TOOL_NAMES == set(CLIENT_TOOL_DESCRIPTIONS.keys())

    def test_all_tools_have_annotations(self):
        assert SPARK_TOOL_NAMES == set(CLIENT_TOOL_ANNOTATIONS.keys())

    def test_all_tools_in_tool_phases(self):
        orphan = SPARK_TOOL_NAMES - set(TOOL_TO_PHASE.keys())
        assert not orphan, f"Spark TOOLS entries missing phase map: {orphan}"

    def test_annotation_schema(self):
        # CONVENTIONS.md: every tool carries these annotation keys.
        required = {
            "title",
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
            "tags",
        }
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            missing = required - set(ann.keys())
            assert not missing, f"Tool '{name}' missing annotation keys: {missing}"

    def test_annotation_tags_include_phase(self):
        # Phases are namespaced per client (``spark_discovery``) while tags carry
        # the bare phase family (``discovery``), matching trainer's annotations.
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            family = TOOL_TO_PHASE[name].removeprefix("spark_")
            assert family in ann["tags"], (
                f"Tool '{name}' tags must include its phase family '{family}'"
            )

    def test_tool_names_follow_verb_noun(self):
        verbs = {"list", "get", "create", "delete", "update", "run", "wait", "inspect", "patch"}
        for name in SPARK_TOOL_NAMES:
            assert name.islower(), f"'{name}' must be snake_case"
            assert name.split("_")[0] in verbs, f"'{name}' must start with a standard verb"

    def test_module_status_implemented(self):
        assert MODULE_INFO["status"] == "implemented"
        assert MODULE_INFO["name"] and MODULE_INFO["description"]

    def test_read_tools_marked_readonly(self):
        for name in READ_TOOLS:
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is True

    def test_write_tools_not_readonly(self):
        for name in WRITE_TOOLS:
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is False

    def test_destructive_tools_have_destructive_hint(self):
        for name in DESTRUCTIVE_TOOLS & SPARK_TOOL_NAMES:
            assert CLIENT_TOOL_ANNOTATIONS[name]["destructiveHint"] is True, (
                f"Tool '{name}' is in DESTRUCTIVE_TOOLS but destructiveHint is False"
            )

    def test_destructive_description_carries_marker(self):
        for name in DESTRUCTIVE_TOOLS & SPARK_TOOL_NAMES:
            assert "[DESTRUCTIVE]" in CLIENT_TOOL_DESCRIPTIONS[name]


# ─── Registration wiring ────────────────────────────────────────────────────


class TestWiring:
    def test_module_registered(self):
        assert CLIENT_MODULES["spark"] == "kubeflow_mcp.spark"

    def test_delete_is_destructive(self):
        assert "delete_spark_session" in DESTRUCTIVE_TOOLS

    def test_spark_phases_present(self):
        for phase in ("spark_discovery", "spark_sessions", "spark_monitoring"):
            assert phase in TOOL_PHASES

    def test_every_tool_has_next_hint(self):
        missing = SPARK_TOOL_NAMES - set(TOOL_NEXT_HINTS.keys())
        assert not missing, f"Spark tools missing TOOL_NEXT_HINTS entries: {missing}"

    def test_sdk_compatibility_declares_spark(self):
        clients = SDK_COMPATIBILITY["clients"]
        assert clients["spark"]["status"] == "implemented"
        assert clients["spark"]["sdk_client"] == "kubeflow.spark.SparkClient"

    def test_phase_to_section_maps_to_valid_slots(self):
        # Sections must be one of the server's fixed slots (or None).
        valid = {"planning", "monitoring", "training", "platform", None}
        for phase, section in PHASE_TO_SECTION.items():
            assert section in valid, f"{phase} -> {section} is not a valid section slot"

    def test_instruction_sections_have_full_tier(self):
        for _section, tiers in INSTRUCTION_SECTIONS.items():
            assert "full" in tiers and len(tiers["full"]) > 10


# ─── Persona gating ─────────────────────────────────────────────────────────


class TestPersonaGating:
    def test_readonly_has_read_tools_only(self):
        tools = get_allowed_tools("readonly")
        assert READ_TOOLS <= tools
        assert not (WRITE_TOOLS & tools)

    def test_data_scientist_inherits_readonly(self):
        assert get_allowed_tools("readonly") <= get_allowed_tools("data-scientist")

    def test_data_scientist_has_lifecycle_tools(self):
        assert WRITE_TOOLS <= get_allowed_tools("data-scientist")

    def test_ml_engineer_inherits_data_scientist(self):
        assert get_allowed_tools("data-scientist") <= get_allowed_tools("ml-engineer")

    def test_platform_admin_is_unrestricted(self):
        # ``None`` is the policy layer's "all tools" sentinel — platform-admin is
        # never enumerated, so spark tools need no explicit entry for it.
        assert get_allowed_tools("platform-admin") is None


# ─── Resources ──────────────────────────────────────────────────────────────


class TestResources:
    def test_resource_files_exist(self):
        import kubeflow_mcp.spark as spark_module

        base = Path(spark_module.__file__).parent
        assert len(CLIENT_RESOURCES) >= 1
        for uri, (filename, _desc) in CLIENT_RESOURCES.items():
            assert uri.startswith("spark://"), f"URI '{uri}' should use spark:// scheme"
            path = base / filename
            assert path.exists(), f"Resource file missing: {path}"
            assert len(path.read_text(encoding="utf-8")) > 50
