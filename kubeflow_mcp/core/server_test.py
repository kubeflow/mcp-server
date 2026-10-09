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

from kubeflow_mcp import __version__, trainer
from kubeflow_mcp.common.constants import TOOL_NEXT_HINTS, TOOL_TO_PHASE
from kubeflow_mcp.common.types import PreviewResponse, ToolResponse
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

    # TODO(test): pin checksum baseline and assert equality across releases
    # TODO(test): test create_server produces deterministic tool set per persona
    # TODO(test): test health tools are always included regardless of persona
    # TODO(test): test dynamic mode tools are subset of full mode


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
        async with Client(create_server(persona="platform-admin")) as client:
            result = await client.call_tool(
                "patch_runtime",
                {"name": "rt-a", "patch": {"metadata": {"labels": {"a": "b"}}}},
            )
        body = result.structured_content
        assert body["data"]["action"] == "preview"
        assert body["_meta"] == {"phase": TOOL_TO_PHASE["patch_runtime"]}


async def test_server_info_reports_package_version():
    async with Client(create_server()) as client:
        # The sessionless protocol has no initialize handshake; FastMCP exposes
        # the server info it discovered on both protocol eras.
        assert client.server_info.version == __version__


@pytest.mark.parametrize("mode", ["progressive", "semantic"])
async def test_dynamic_server_registries_are_isolated(mode: str, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def harmless_delete_training_job(name: str, confirmed: bool = False) -> dict:
        """Harmless stand-in for the admin-only delete tool."""
        calls.append(name)
        return {"success": True, "data": {"name": name, "confirmed": confirmed}}

    harmless_delete_training_job.__name__ = "delete_training_job"
    monkeypatch.setattr(
        trainer,
        "TOOLS",
        [
            harmless_delete_training_job if tool.__name__ == "delete_training_job" else tool
            for tool in trainer.TOOLS
        ],
    )

    server_a = create_server(persona="readonly", mode=mode)
    server_b = create_server(persona="platform-admin", mode=mode)

    async def list_dynamic_tool_names(client: Client) -> set[str]:
        if mode == "progressive":
            categories = (await client.call_tool("list_tools", {})).structured_content["categories"]
            names = set()
            for category in categories:
                tools = (
                    await client.call_tool("list_tools", {"prefix": category})
                ).structured_content["tools"]
                names.update(tool["name"] for tool in tools)
            return names

        tools = (await client.call_tool("find_tools", {"query": "all"})).structured_content["tools"]
        return {tool["name"] for tool in tools}

    async with Client(server_a) as client_a, Client(server_b) as client_b:
        tools_a_before = await list_dynamic_tool_names(client_a)
        tools_b = await list_dynamic_tool_names(client_b)

        assert "delete_training_job" not in tools_a_before
        assert "delete_training_job" in tools_b

        result_b = await client_b.call_tool(
            "execute_tool",
            {
                "tool_name": "delete_training_job",
                "arguments": {"name": "stub-job", "confirmed": False},
            },
        )
        assert result_b.structured_content["data"]["name"] == "stub-job"
        assert calls == ["stub-job"]

        result_a = await client_a.call_tool_mcp(
            "execute_tool",
            {
                "tool_name": "delete_training_job",
                "arguments": {"name": "stub-job", "confirmed": False},
            },
        )
        assert result_a.is_error is True
        assert result_a.structured_content["error"] == "Tool 'delete_training_job' not found"
        assert calls == ["stub-job"]
        assert await list_dynamic_tool_names(client_a) == tools_a_before
