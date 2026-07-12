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

"""Tests for the spark client module: metadata, wiring, and tool behavior.

All SDK imports in the spark module are lazy, so these tests run without the
``kubeflow[spark]`` extra installed; the SparkClient is mocked.
"""

import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common.constants import TOOL_PHASES, TOOL_TO_PHASE
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
from kubeflow_mcp.spark.api import discovery, monitoring, sessions
from kubeflow_mcp.spark.types import session_info_to_dict

SPARK_TOOL_NAMES = {t.__name__ for t in TOOLS}


def _fake_session(
    name="spark-connect-ab12", state="Ready", namespace="default", driver_pod="drv-0"
):
    """Build a duck-typed SparkConnectInfo (str-enum state, datetime timestamp).

    Uses ``pod_name`` to match the released ``kubeflow[spark]`` 0.4.x SDK (the
    declared compatibility baseline); unreleased SDK ``main`` renamed it to
    ``driver_pod_name`` (covered separately by the fallback test below).
    """
    return types.SimpleNamespace(
        name=name,
        namespace=namespace,
        state=types.SimpleNamespace(value=state),
        pod_name=driver_pod,
        pod_ip="10.0.0.5",
        service_name=f"{name}-svc",
        creation_timestamp=datetime(2026, 7, 9, 12, 0, 0),
    )


# ─── Metadata consistency (mirrors trainer architecture test) ───────────────


class TestToolMetadataConsistency:
    def test_all_tools_have_descriptions(self):
        assert SPARK_TOOL_NAMES == set(CLIENT_TOOL_DESCRIPTIONS.keys())

    def test_all_tools_have_annotations(self):
        assert SPARK_TOOL_NAMES == set(CLIENT_TOOL_ANNOTATIONS.keys())

    def test_all_tools_in_tool_phases(self):
        orphan = SPARK_TOOL_NAMES - set(TOOL_TO_PHASE.keys())
        assert not orphan, f"Spark TOOLS entries missing phase map: {orphan}"

    def test_annotation_schema(self):
        required = {"title", "readOnlyHint", "destructiveHint", "idempotentHint", "tags"}
        for name, ann in CLIENT_TOOL_ANNOTATIONS.items():
            assert not (required - set(ann.keys())), f"'{name}' missing annotation keys"

    def test_module_status_implemented(self):
        assert MODULE_INFO["status"] == "implemented"

    def test_read_tools_marked_readonly(self):
        for name in ("list_spark_sessions", "get_spark_session", "get_spark_session_logs"):
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is True

    def test_write_tools_not_readonly(self):
        for name in ("create_spark_session", "delete_spark_session"):
            assert CLIENT_TOOL_ANNOTATIONS[name]["readOnlyHint"] is False


# ─── Destructive + phase + persona wiring ───────────────────────────────────


class TestWiring:
    def test_delete_is_destructive(self):
        assert "delete_spark_session" in DESTRUCTIVE_TOOLS
        assert CLIENT_TOOL_ANNOTATIONS["delete_spark_session"]["destructiveHint"] is True

    def test_spark_phases_present(self):
        for phase in ("spark_discovery", "spark_sessions", "spark_monitoring"):
            assert phase in TOOL_PHASES

    def test_module_registered(self):
        assert CLIENT_MODULES["spark"] == "kubeflow_mcp.spark"

    def test_phase_to_section_maps_to_valid_slots(self):
        # Sections must be one of the server's fixed slots (or None).
        valid = {"planning", "monitoring", "training", "platform", None}
        for phase, section in PHASE_TO_SECTION.items():
            assert section in valid, f"{phase} -> {section} is not a valid section slot"

    def test_instruction_sections_have_full_tier(self):
        for _section, tiers in INSTRUCTION_SECTIONS.items():
            assert "full" in tiers and len(tiers["full"]) > 10

    def test_readonly_persona_has_read_tools_only(self):
        tools = get_allowed_tools("readonly")
        assert {"list_spark_sessions", "get_spark_session", "get_spark_session_logs"} <= tools
        assert "create_spark_session" not in tools
        assert "delete_spark_session" not in tools

    def test_data_scientist_has_lifecycle_tools(self):
        tools = get_allowed_tools("data-scientist")
        assert {"create_spark_session", "delete_spark_session"} <= tools


# ─── Resources ──────────────────────────────────────────────────────────────


class TestResources:
    def test_resource_files_exist(self):
        from pathlib import Path

        import kubeflow_mcp.spark as spark_module

        base = Path(spark_module.__file__).parent
        assert len(CLIENT_RESOURCES) >= 1
        for uri, (filename, _desc) in CLIENT_RESOURCES.items():
            assert uri.startswith("spark://"), f"URI '{uri}' should use spark:// scheme"
            path = base / filename
            assert path.exists(), f"Resource file missing: {path}"
            assert len(path.read_text(encoding="utf-8")) > 50


# ─── Serialization ──────────────────────────────────────────────────────────


class TestSerialization:
    def test_session_info_to_dict(self):
        data = session_info_to_dict(_fake_session())
        assert data["name"] == "spark-connect-ab12"
        assert data["state"] == "Ready"  # enum .value extracted
        assert data["service_name"] == "spark-connect-ab12-svc"
        assert data["creation_timestamp"] == "2026-07-09T12:00:00"  # datetime -> ISO

    def test_session_info_plain_string_state(self):
        info = _fake_session()
        info.state = "Failed"  # tolerate a plain string state
        assert session_info_to_dict(info)["state"] == "Failed"

    def test_driver_pod_name_from_released_pod_name_field(self):
        # Released kubeflow[spark] 0.4.0/0.4.1 exposes the driver pod as
        # `pod_name`; the serializer must surface it under `driver_pod_name`.
        info = types.SimpleNamespace(
            name="s",
            namespace="default",
            state=types.SimpleNamespace(value="Ready"),
            pod_name="server-pod-0",
            pod_ip=None,
            service_name=None,
            creation_timestamp=None,
        )
        assert session_info_to_dict(info)["driver_pod_name"] == "server-pod-0"

    def test_driver_pod_name_fallback_to_main_field(self):
        # Unreleased SDK `main` renamed the field to `driver_pod_name`; the
        # serializer falls back to it when `pod_name` is absent.
        info = types.SimpleNamespace(
            name="s",
            namespace="default",
            state=types.SimpleNamespace(value="Ready"),
            driver_pod_name="server-pod-1",
            pod_ip=None,
            service_name=None,
            creation_timestamp=None,
        )
        assert session_info_to_dict(info)["driver_pod_name"] == "server-pod-1"


# ─── Tool behavior (mocked SparkClient) ─────────────────────────────────────


class TestListSessions:
    def test_list_and_serialize(self):
        client = MagicMock()
        client.list_sessions.return_value = [_fake_session("a"), _fake_session("b", state="Failed")]
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.list_spark_sessions()
        assert out["success"] is True
        assert out["data"]["total"] == 2

    def test_state_filter(self):
        client = MagicMock()
        client.list_sessions.return_value = [_fake_session("a"), _fake_session("b", state="Failed")]
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.list_spark_sessions(state="failed")  # case-insensitive
        assert out["data"]["total"] == 1
        assert out["data"]["sessions"][0]["name"] == "b"

    def test_invalid_limit(self):
        out = discovery.list_spark_sessions(limit=0)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_missing_extra_friendly_error(self):
        with patch.object(
            discovery,
            "get_spark_client_for_namespace",
            side_effect=ImportError("SparkClient requires the Spark extra."),
        ):
            out = discovery.list_spark_sessions()
        assert out["success"] is False
        assert "Spark extra" in out["error"]


class TestGetSession:
    def test_get_ok(self):
        client = MagicMock()
        client.get_session.return_value = _fake_session("x")
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("x")
        assert out["success"] is True
        assert out["data"]["name"] == "x"

    def test_not_found_maps_to_resource_not_found(self):
        from kubernetes.client.exceptions import ApiException

        exc = RuntimeError("SparkConnect not found: default/missing")
        exc.__cause__ = ApiException(status=404)
        client = MagicMock()
        client.get_session.side_effect = exc
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("missing")
        assert out["success"] is False
        assert out["error_code"] == "RESOURCE_NOT_FOUND"

    def test_failed_state_has_next_steps(self):
        client = MagicMock()
        client.get_session.return_value = _fake_session("x", state="Failed")
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("x")
        assert "next_steps" in out["data"]


class TestLogs:
    def test_logs_bounded_and_truncation_flagged(self):
        client = MagicMock()
        client.get_session_logs.return_value = iter([f"line{i}" for i in range(10)])
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x", tail_lines=3)
        assert out["success"] is True
        assert out["data"]["lines"] == 3
        assert out["data"]["truncated"] is True
        assert out["data"]["logs"].splitlines() == ["line7", "line8", "line9"]

    def test_no_driver_pod_is_validation_error(self):
        client = MagicMock()
        client.get_session_logs.side_effect = RuntimeError("No driver pod for SparkConnect: d/x")
        with patch.object(monitoring, "get_spark_client_for_namespace", return_value=client):
            out = monitoring.get_spark_session_logs("x")
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"


class TestCreateSession:
    def test_preview_when_not_confirmed(self):
        out = sessions.create_spark_session(name="etl", num_executors=2)
        assert out["status"] == "preview"
        assert out["config"]["name"] == "etl"

    def test_invalid_name_rejected(self):
        out = sessions.create_spark_session(name="Bad_Name", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_negative_executors_rejected(self):
        out = sessions.create_spark_session(num_executors=-1, confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_auto_generated_name_in_preview(self):
        out = sessions.create_spark_session()
        assert out["config"]["name"].startswith("spark-connect-")


class TestDeleteSession:
    def test_preview_when_not_confirmed(self):
        out = sessions.delete_spark_session("x")
        assert out["status"] == "preview"

    def test_delete_confirmed(self):
        client = MagicMock()
        with patch.object(sessions, "get_spark_client_for_namespace", return_value=client):
            out = sessions.delete_spark_session("x", confirmed=True)
        assert out["success"] is True
        assert out["data"]["deleted"] is True
        client.delete_session.assert_called_once_with("x")

    def test_delete_not_found(self):
        from kubernetes.client.exceptions import ApiException

        exc = RuntimeError("SparkConnect not found: default/missing")
        exc.__cause__ = ApiException(status=404)
        client = MagicMock()
        client.delete_session.side_effect = exc
        with patch.object(sessions, "get_spark_client_for_namespace", return_value=client):
            out = sessions.delete_spark_session("missing", confirmed=True)
        assert out["error_code"] == "RESOURCE_NOT_FOUND"


class TestClientFactory:
    """get_spark_client_for_namespace must align the operated namespace with the
    one check_namespace_allowed validates (no namespace-allowlist bypass)."""

    def test_none_namespace_resolves_effective(self):
        from kubeflow_mcp.common import utils

        fake_cls = MagicMock()
        utils._spark_ns_client_cache.clear()
        with (
            patch.object(utils, "_import_spark_client", return_value=fake_cls),
            patch.object(utils, "get_trainer_effective_namespace", return_value="team-a"),
        ):
            utils.get_spark_client_for_namespace(None)
        utils._spark_ns_client_cache.clear()

        # A namespaced client was built for the *resolved* effective namespace.
        assert fake_cls.call_count == 1
        _args, kwargs = fake_cls.call_args
        assert kwargs["backend_config"].namespace == "team-a"

    def test_none_namespace_falls_back_when_resolution_fails(self):
        from kubeflow_mcp.common import utils

        with (
            patch.object(
                utils,
                "get_trainer_effective_namespace",
                side_effect=RuntimeError("no kubeconfig"),
            ),
            patch.object(utils, "get_spark_client", return_value="UNSCOPED") as gsc,
        ):
            out = utils.get_spark_client_for_namespace(None)
        assert out == "UNSCOPED"
        gsc.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
