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


def test_audit_wrap_execute_tool_circuit_breaker_isolation(monkeypatch):
    from kubeflow_mcp.common.constants import ErrorCode
    from kubeflow_mcp.core import dynamic_tools, server
    from kubeflow_mcp.core.resilience import CircuitBreaker, CircuitState

    breakers = {}

    def test_get_breaker(name):
        return breakers.setdefault(name, CircuitBreaker())

    # Keep this test independent of breakers left behind by other tests.
    monkeypatch.setattr(server, "get_breaker", test_get_breaker)
    monkeypatch.setattr(dynamic_tools, "get_breaker", test_get_breaker)

    def make_fail(name):
        def fn():
            return {"error": "k8s connection reset", "error_code": ErrorCode.KUBERNETES_ERROR}

        fn.__name__ = name
        return fn

    failing_tools = [make_fail(f"k8s_err_{i}") for i in range(5)]

    def healthy_worker():
        return {"status": "ok"}

    dynamic_tools.init_dynamic_tools(failing_tools + [healthy_worker], {})
    audited_exec = server._audit_wrap(
        dynamic_tools.execute_tool,
        is_dynamic_execute_tool=True,
    )

    for i in range(5):
        res = audited_exec(tool_name=f"k8s_err_{i}")
        assert res["error_code"] == ErrorCode.KUBERNETES_ERROR
        assert breakers[f"k8s_err_{i}"].state == CircuitState.CLOSED
        assert breakers[f"k8s_err_{i}"].failure_count == 1

    # No shared breaker is even created for the execute_tool wrapper.
    assert "execute_tool" not in breakers

    # Healthy tool executes successfully through audited execute_tool
    healthy_res = audited_exec(tool_name="healthy_worker")
    assert healthy_res == {"status": "ok"}

    # The bypass is specific to dynamic-mode execution, not any function
    # that happens to have the same name.
    def execute_tool(tool_name):
        return {"status": "ok", "tool": tool_name}

    ordinary_tool = server._audit_wrap(execute_tool)
    assert ordinary_tool(tool_name="standalone") == {"status": "ok", "tool": "standalone"}
    assert "execute_tool" in breakers


def test_audit_wrap_preserves_dynamic_tool_meta(monkeypatch):
    from kubeflow_mcp.common.constants import TOOL_NEXT_HINTS, TOOL_TO_PHASE
    from kubeflow_mcp.core.dynamic_tools import execute_tool, init_dynamic_tools
    from kubeflow_mcp.core.server import _audit_wrap

    def meta_worker():
        return {"status": "ok"}

    init_dynamic_tools([meta_worker], {})
    monkeypatch.setitem(TOOL_TO_PHASE, "meta_worker", "evaluation")
    monkeypatch.setitem(TOOL_NEXT_HINTS, "meta_worker", "View metrics")

    audited_exec = _audit_wrap(execute_tool, is_dynamic_execute_tool=True)
    res = audited_exec(tool_name="meta_worker")
    assert res["status"] == "ok"
    assert res.get("_meta") == {"phase": "evaluation", "next": "View metrics"}


async def test_progressive_mode_e2e_meta_and_breaker():
    previous = get_effective_persona()
    try:
        async with Client(create_server(mode="progressive")) as client:
            # execute_tool returns _meta workflow guidance in progressive mode
            res = await client.call_tool("execute_tool", {"tool_name": "health_check"})
            assert res.structured_content["_meta"]["phase"] == "health"
            assert "next" in res.structured_content["_meta"]
    finally:
        set_effective_persona(previous)
