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

"""Tool description integrity — supply chain defense stubs.

Verifies that registered tools have complete, consistent metadata.
Prevents silent drift in tool descriptions across releases.

See also: kubeflow_mcp/trainer/api/architecture_test.py for full metadata consistency tests.
"""

import hashlib

import pytest
from fastmcp import Client

from kubeflow_mcp import __version__
from kubeflow_mcp.common.constants import TOOL_NEXT_HINTS, TOOL_TO_PHASE
from kubeflow_mcp.common.types import PreviewResponse, ToolResponse
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from kubeflow_mcp.core.server import _inject_meta, create_server
from kubeflow_mcp.trainer import CLIENT_TOOL_ANNOTATIONS, CLIENT_TOOL_DESCRIPTIONS


class TestToolDescriptionIntegrity:
    def test_descriptions_are_non_empty(self):
        for name, desc in CLIENT_TOOL_DESCRIPTIONS.items():
            assert len(desc) > 10, f"Tool '{name}' has suspiciously short description"

    def test_annotations_have_read_only_hint(self):
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            assert "readOnlyHint" in ann, f"Tool '{name}' missing readOnlyHint"

    def test_description_checksums_generated(self):
        """Compute checksums for future baseline pinning (see TODO below)."""
        checksums = {
            name: hashlib.sha256(desc.encode()).hexdigest()[:16]
            for name, desc in sorted(CLIENT_TOOL_DESCRIPTIONS.items())
        }
        assert checksums
        assert all(len(digest) == 16 for digest in checksums.values())

    BASELINE_TOOL_DESCRIPTION_CHECKSUMS = {
        "check_compatibility": "573e13e9d156d057",
        "create_runtime": "4a91f8bf0943cff7",
        "delete_runtime": "9be2a848a27133ec",
        "delete_training_job": "7a230704e6e9cbca",
        "estimate_resources": "06c315693b2fff3d",
        "fine_tune": "263902adb969a394",
        "get_cluster_resources": "d436bd50321a4b46",
        "get_runtime": "f8c86da73be2a5d3",
        "get_training_events": "acc2f87c0a493f09",
        "get_training_job": "23014441e9139a88",
        "get_training_logs": "18a5a86ffb60e5b9",
        "inspect_controller": "70a7ca6464ed602d",
        "inspect_crd": "960ddaa7c98a9f79",
        "list_runtimes": "cd64d3b91647788b",
        "list_training_jobs": "8041a71321166d53",
        "patch_runtime": "681a0b91a85e48ea",
        "pre_flight": "0cf3cacccf2f7104",
        "run_container_training": "cc49f789d27e3250",
        "run_custom_training": "ae983eca798ece9a",
        "update_training_job": "64c03e724ebe68c0",
        "wait_for_training": "23cdfb81cfe136d2",
    }

    def test_description_checksums_match_baseline(self):
        """Pin checksum baseline and assert equality across releases."""
        checksums = {
            name: hashlib.sha256(desc.encode()).hexdigest()[:16]
            for name, desc in sorted(CLIENT_TOOL_DESCRIPTIONS.items())
        }
        assert checksums == self.BASELINE_TOOL_DESCRIPTION_CHECKSUMS


class TestCreateServer:
    """Tests for create_server tool registration, persona filtering, modes, and policy."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("persona", "expected_count"),
        [
            ("readonly", 12),
            ("data-scientist", 16),
            ("ml-engineer", 20),
            ("platform-admin", 23),
        ],
    )
    async def test_create_server_deterministic_tools_per_persona(self, persona, expected_count):
        """create_server produces deterministic tool set per persona."""
        previous = get_effective_persona()
        try:
            s1 = create_server(persona=persona)
            tools1 = [t.name for t in await s1.list_tools()]
            s2 = create_server(persona=persona)
            tools2 = [t.name for t in await s2.list_tools()]
            assert len(tools1) == expected_count
            assert tools1 == tools2
        finally:
            set_effective_persona(previous)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "persona",
        ["readonly", "data-scientist", "ml-engineer", "platform-admin"],
    )
    async def test_health_tools_always_included_regardless_of_persona(self, persona):
        """Health tools are always included regardless of persona."""
        previous = get_effective_persona()
        try:
            server = create_server(persona=persona)
            tool_names = {t.name for t in await server.list_tools()}
            assert {"health_check", "get_server_logs"}.issubset(tool_names)
        finally:
            set_effective_persona(previous)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "expected_meta_tools"),
        [
            ("progressive", {"list_tools", "describe_tools", "execute_tool"}),
            ("semantic", {"find_tools", "execute_tool"}),
        ],
    )
    async def test_dynamic_mode_tools_are_subset_of_full_mode(self, mode, expected_meta_tools):
        """Dynamic mode meta-tools proxy a subset of full mode tools."""
        previous = get_effective_persona()
        try:
            full_server = create_server(persona="platform-admin", mode="full")
            full_tools = {t.name for t in await full_server.list_tools()}

            dynamic_server = create_server(persona="readonly", mode=mode)
            meta_tools = {t.name for t in await dynamic_server.list_tools()}
            assert meta_tools == expected_meta_tools

            from kubeflow_mcp.core.dynamic_tools import TOOL_REGISTRY

            assert set(TOOL_REGISTRY.keys()).issubset(full_tools)
        finally:
            set_effective_persona(previous)

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode raises ValueError with allowed choices."""
        with pytest.raises(ValueError, match="Invalid mode 'invalid'"):
            create_server(mode="invalid")

    @pytest.mark.asyncio
    async def test_read_only_policy_excludes_write_tools(self, monkeypatch):
        """read_only policy excludes write tools even for platform-admin."""
        previous = get_effective_persona()
        try:
            monkeypatch.setattr("kubeflow_mcp.core.server.is_read_only", lambda: True)
            server = create_server(persona="platform-admin")
            tool_names = {t.name for t in await server.list_tools()}
            assert len(tool_names) == 15
            write_tools = {
                "fine_tune",
                "run_custom_training",
                "run_container_training",
                "delete_training_job",
                "update_training_job",
                "create_runtime",
                "patch_runtime",
                "delete_runtime",
            }
            assert tool_names.isdisjoint(write_tools)
        finally:
            set_effective_persona(previous)

    def test_unknown_client_logs_warning_and_skips(self, caplog):
        """create_server warns on unknown client module and skips it."""
        with caplog.at_level("WARNING"):
            create_server(clients=["unknown_client"])
        assert "Unknown client 'unknown_client'" in caplog.text

    def test_failed_client_import_logs_warning(self, monkeypatch, caplog):
        """create_server warns when client module fails to import."""

        def mock_import(name):
            raise ImportError("Simulated failure")

        monkeypatch.setattr("importlib.import_module", mock_import)
        with caplog.at_level("WARNING"):
            create_server(clients=["trainer"])
        assert "Failed to import client 'trainer'" in caplog.text

    def test_auth_provider_attached(self):
        """create_server attaches auth_provider to FastMCP when provided."""
        mock_auth = object()
        server = create_server(auth_provider=mock_auth)
        assert server.auth is mock_auth


class TestInjectMetaBlockedResponses:
    """``_meta.next`` must not advance the agent past reported blockers.

    ``pre_flight`` and ``check_compatibility`` return a successful envelope with
    the verdict inside ``data``, so the guidance has to read the payload rather
    than the envelope alone.
    """

    def test_next_withheld_when_blockers_present(self):
        result = _inject_meta(
            {"success": True, "data": {"compatible": False, "blockers": ["K8s too old"]}},
            "check_compatibility",
        )
        assert "next" not in result["_meta"]

    def test_next_withheld_for_nested_compatibility_report(self):
        """pre_flight nests the compatibility verdict one level down."""
        result = _inject_meta(
            {
                "success": True,
                "data": {"compatibility": {"compatible": False, "blockers": ["CRD missing"]}},
            },
            "pre_flight",
        )
        assert "next" not in result["_meta"]

    def test_phase_preserved_when_blocked(self):
        """Only the advance hint is wrong; the workflow phase is still accurate."""
        result = _inject_meta(
            {"success": True, "data": {"compatible": False, "blockers": ["K8s too old"]}},
            "pre_flight",
        )
        assert result["_meta"]["phase"] == TOOL_TO_PHASE["pre_flight"]

    def test_next_present_when_compatible(self):
        result = _inject_meta(
            {"success": True, "data": {"compatible": True, "blockers": []}},
            "check_compatibility",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["check_compatibility"]

    def test_next_present_for_unrelated_payload(self):
        """Tools whose data carries no compatibility verdict are unaffected."""
        result = _inject_meta({"success": True, "data": {"jobs": []}}, "list_training_jobs")
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["list_training_jobs"]

    def test_degraded_health_check_keeps_conditional_hint(self):
        """health_check reports degradation without blockers; its hint is conditional."""
        result = _inject_meta(
            {"success": True, "data": {"status": "degraded", "kubernetes": False}},
            "health_check",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["health_check"]

    def test_error_envelope_still_short_circuits(self):
        result = _inject_meta(
            {"error": "boom", "error_code": "SDK_ERROR"},
            "pre_flight",
        )
        assert "_meta" not in result

    def test_non_dict_result_passes_through(self):
        assert _inject_meta("not a dict", "pre_flight") == "not a dict"

    def test_empty_blockers_list_is_not_blocked(self):
        """An empty blockers list is a clean report, not a blocked one."""
        result = _inject_meta(
            {"success": True, "data": {"compatible": True, "blockers": []}},
            "pre_flight",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["pre_flight"]

    def test_compatible_absent_does_not_block(self):
        """Payloads that carry neither key keep their hint (guard must not overreach)."""
        for payload in ({"runtimes": []}, {"jobs": [], "total": 0}, {"logs": ""}):
            result = _inject_meta({"success": True, "data": payload}, "list_runtimes")
            assert result["_meta"]["next"] == TOOL_NEXT_HINTS["list_runtimes"]

    def test_nested_sibling_payloads_do_not_trigger(self):
        """pre_flight nests cluster/estimate/runtimes; only `compatibility` is consulted."""
        result = _inject_meta(
            {
                "success": True,
                "data": {
                    "compatibility": {"compatible": True, "blockers": []},
                    "cluster": {"gpu_total": 2},
                    "runtimes": {"runtimes": [{"name": "torchtune-llama3.2-1b"}]},
                },
            },
            "pre_flight",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["pre_flight"]


class TestInjectMetaPreviewResponses:
    """``_meta.next`` must not describe the step after an action that has not run.

    With ``confirmed=False`` a mutating tool only returns a preview, but the hints
    in ``TOOL_NEXT_HINTS`` assume the action ran ("Monitor with get_training_job",
    "Confirm removal with list_runtimes()").
    """

    MUTATING_TOOLS = [
        "fine_tune",
        "run_custom_training",
        "run_container_training",
        "delete_training_job",
        "update_training_job",
        "create_runtime",
        "patch_runtime",
        "delete_runtime",
    ]

    def test_every_mutating_tool_has_a_post_action_hint(self):
        """Guards the premise: each of these tools has a hint that could leak."""
        for tool_name in self.MUTATING_TOOLS:
            assert TOOL_NEXT_HINTS.get(tool_name), tool_name

    def test_next_withheld_for_preview_response(self):
        for tool_name in self.MUTATING_TOOLS:
            result = _inject_meta(PreviewResponse(config={"name": "job-a"}).model_dump(), tool_name)
            assert "next" not in result["_meta"], tool_name

    def test_next_withheld_for_legacy_runtime_preview(self):
        """Runtime tools preview through ToolResponse with data.action="preview"."""
        for tool_name in ("create_runtime", "patch_runtime", "delete_runtime"):
            result = _inject_meta(
                ToolResponse(data={"action": "preview", "runtime": "rt-a"}).model_dump(),
                tool_name,
            )
            assert "next" not in result["_meta"], tool_name

    def test_legacy_preview_shape_only_counts_for_runtime_tools(self):
        """Another tool returning data.action="preview" keeps its hint."""
        for tool_name in TOOL_NEXT_HINTS:
            if tool_name in ("create_runtime", "patch_runtime", "delete_runtime"):
                continue
            result = _inject_meta(ToolResponse(data={"action": "preview"}).model_dump(), tool_name)
            assert result["_meta"]["next"] == TOOL_NEXT_HINTS[tool_name], tool_name

    def test_phase_preserved_for_preview(self):
        result = _inject_meta(PreviewResponse(config={}).model_dump(), "fine_tune")
        assert result["_meta"]["phase"] == TOOL_TO_PHASE["fine_tune"]

    def test_next_present_after_confirmed_submission(self):
        result = _inject_meta(
            ToolResponse(data={"job_id": "job-a", "status": "Created"}).model_dump(),
            "fine_tune",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["fine_tune"]

    def test_next_present_after_confirmed_update(self):
        """Executed update_training_job carries data.action="suspend", not "preview"."""
        result = _inject_meta(
            ToolResponse(data={"job": "job-a", "action": "suspend"}).model_dump(),
            "update_training_job",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["update_training_job"]

    def test_job_status_in_data_is_not_a_preview(self):
        """A job's own status lives in data; only the envelope status marks a preview."""
        result = _inject_meta(
            ToolResponse(data={"name": "job-a", "status": "preview"}).model_dump(),
            "get_training_job",
        )
        assert result["_meta"]["next"] == TOOL_NEXT_HINTS["get_training_job"]

    async def test_preview_over_mcp_has_no_next_hint(self):
        """End to end: patch_runtime previews without touching the cluster."""
        previous = get_effective_persona()
        try:
            async with Client(create_server(persona="platform-admin")) as client:
                result = await client.call_tool(
                    "patch_runtime",
                    {"name": "rt-a", "patch": {"metadata": {"labels": {"a": "b"}}}},
                )
        finally:
            set_effective_persona(previous)
        body = result.structured_content
        assert body["data"]["action"] == "preview"
        assert body["_meta"] == {"phase": TOOL_TO_PHASE["patch_runtime"]}


async def test_server_info_reports_package_version():
    previous = get_effective_persona()
    try:
        async with Client(create_server()) as client:
            # The sessionless protocol has no initialize handshake; FastMCP exposes
            # the server info it discovered on both protocol eras.
            assert client.server_info.version == __version__
    finally:
        set_effective_persona(previous)
