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

"""Unit tests for core/health.py - health check and server logs tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client
from tests.common import SDK_ERROR, VALIDATION_ERROR

from kubeflow_mcp.common.utils import K8S_TIMEOUT
from kubeflow_mcp.core.health import (
    HEALTH_TOOL_ANNOTATIONS,
    HEALTH_TOOL_DESCRIPTIONS,
    HEALTH_TOOLS,
    ReadinessDiagnostic,
    get_server_logs,
    health_check,
    run_readiness_diagnostics,
)
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from kubeflow_mcp.core.server import create_server


class TestHealthCheck:
    """Tests for the health_check tool."""

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_health_check_healthy_when_k8s_succeeds(self, mock_get_core_v1_api: MagicMock) -> None:
        mock_v1 = MagicMock()
        mock_get_core_v1_api.return_value = mock_v1

        result = health_check()

        assert result["success"] is True
        data = result["data"]
        assert data["status"] == "healthy"
        assert data["kubernetes"] is True
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0
        assert "timestamp" in data
        mock_v1.list_namespace.assert_called_once_with(limit=1, _request_timeout=K8S_TIMEOUT)

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_health_check_degraded_when_k8s_fails(self, mock_get_core_v1_api: MagicMock) -> None:
        mock_v1 = MagicMock()
        mock_v1.list_namespace.side_effect = ConnectionError("K8s API unreachable")

        mock_get_core_v1_api.return_value = mock_v1
        result = health_check()

        # The tool should not crash even if k8s fails, it should return degraded status
        assert result["success"] is True
        data = result["data"]
        assert data["status"] == "degraded"
        assert data["kubernetes"] is False


class TestGetServerLogs:
    """Tests for the get_server_logs tool."""

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_reject_invalid_level(self, mock_get_logs: MagicMock) -> None:
        result = get_server_logs(level="INVALID")
        assert result["success"] is False
        assert result["error_code"] == VALIDATION_ERROR

        mock_get_logs.assert_not_called()

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_accepts_case_insensitive_level(self, mock_get_logs: MagicMock) -> None:
        mock_get_logs.return_value = [{"level": "ERROR", "message": "failed"}]

        result = get_server_logs(level="error")

        assert result["success"] is True
        assert result["data"]["logs"] == [{"level": "ERROR", "message": "failed"}]

    @pytest.mark.parametrize(
        ("alias", "record_level"),
        [("WARN", "WARNING"), ("FATAL", "CRITICAL")],
    )
    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_accepts_standard_level_aliases(
        self, mock_get_logs: MagicMock, alias: str, record_level: str
    ) -> None:
        mock_get_logs.return_value = [{"level": record_level, "message": "failed"}]
        result = get_server_logs(level=alias)

        assert result["success"] is True
        assert result["data"]["logs"] == [{"level": record_level, "message": "failed"}]

    @pytest.fixture
    def sample_logs(self) -> list[dict[str, Any]]:
        return [
            {"timestamp": "2026-09-05T00:00:001Z", "level": "DEBUG", "message": "debug message"},
            {"timestamp": "2026-09-05T00:00:001Z", "level": "INFO", "message": "info message 1"},
            {
                "timestamp": "2026-09-05T00:00:001Z",
                "level": "WARNING",
                "message": "warning message",
            },
            {"timestamp": "2026-09-05T00:00:001Z", "level": "ERROR", "message": "error message"},
            {
                "timestamp": "2026-09-05T00:00:001Z",
                "level": "CRITICAL",
                "message": "critical message",
            },
            {"timestamp": "2026-09-05T00:00:001Z", "level": "INFO", "message": "info message 2"},
        ]

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_reject_limit_less_than_one(self, mock_get_buffer: MagicMock) -> None:
        for invalid_limit in [0, -1, -50]:
            result = get_server_logs(limit=invalid_limit)
            assert result["success"] is False
            assert result["error_code"] == VALIDATION_ERROR
            assert f"limit must be >= 1, got {invalid_limit}" in result["error"]
        mock_get_buffer.assert_not_called()

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_filter_by_default_level(
        self, mock_get_buffer: MagicMock, sample_logs: list[dict[str, Any]]
    ) -> None:
        mock_get_buffer.return_value = sample_logs

        result = get_server_logs()

        assert result["success"] is True
        data = result["data"]

        assert data["total"] == 5
        assert data["buffer_size"] == 6

        levels = [log["level"] for log in data["logs"]]
        assert "DEBUG" not in levels
        assert levels == ["INFO", "WARNING", "ERROR", "CRITICAL", "INFO"]

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_filter_by_level_warning(
        self, mock_get_buffer: MagicMock, sample_logs: list[dict[str, Any]]
    ) -> None:
        mock_get_buffer.return_value = sample_logs

        result = get_server_logs(level="WARNING")

        assert result["success"] is True
        data = result["data"]
        assert data["total"] == 3
        levels = [log["level"] for log in data["logs"]]
        assert levels == ["WARNING", "ERROR", "CRITICAL"]
        assert data["logs"][0]["message"] == "warning message"
        assert data["logs"][1]["message"] == "error message"
        assert data["logs"][2]["message"] == "critical message"

    @patch("kubeflow_mcp.core.health.get_log_buffer")
    def test_handles_unexpected_exception(self, mock_get_buffer: MagicMock) -> None:
        mock_get_buffer.side_effect = RuntimeError("Buffer Locked")

        result = get_server_logs()

        assert result["success"] is False
        assert result["error_code"] == SDK_ERROR


class TestHealthMetadata:
    """Tests for metadata, registrations, and annotations."""

    def test_health_tools_exported(self) -> None:
        assert health_check in HEALTH_TOOLS
        assert get_server_logs in HEALTH_TOOLS
        assert len(HEALTH_TOOLS) == 2

    def test_descriptions_defined_for_all_tools(self) -> None:
        for tool in HEALTH_TOOLS:
            name = tool.__name__
            assert name in HEALTH_TOOL_DESCRIPTIONS
            assert len(HEALTH_TOOL_DESCRIPTIONS[name]) > 10

    def test_annotations_defined_for_all_tools(self) -> None:
        for tool in HEALTH_TOOLS:
            name = tool.__name__
            assert name in HEALTH_TOOL_ANNOTATIONS
            ann = HEALTH_TOOL_ANNOTATIONS[name]
            assert ann.get("readOnlyHint") is True
            assert ann.get("destructiveHint") is False
            assert ann.get("idempotentHint") is True
            assert ann.get("openWorldHint") is False
            assert "health" in ann.get("tags", [])

    @pytest.mark.asyncio
    async def test_health_tools_registered_on_server(self) -> None:
        """Integration test: verify health tools are exposed via create_server() with metadata."""
        previous_persona = get_effective_persona()
        try:
            mcp = create_server()
            async with Client(mcp) as client:
                tools = await client.list_tools()
            tool_map = {tool.name: tool for tool in tools}

            assert "health_check" in tool_map
            assert "get_server_logs" in tool_map

            assert tool_map["health_check"].description == HEALTH_TOOL_DESCRIPTIONS["health_check"]
            assert (
                tool_map["get_server_logs"].description
                == HEALTH_TOOL_DESCRIPTIONS["get_server_logs"]
            )

            hc_ann = tool_map["health_check"].annotations
            assert hc_ann is not None
            assert hc_ann.read_only_hint is True
            assert hc_ann.destructive_hint is False
            assert hc_ann.idempotent_hint is True
            assert "health" in tool_map["health_check"].meta["fastmcp"]["tags"]

            logs_ann = tool_map["get_server_logs"].annotations
            assert logs_ann is not None
            assert logs_ann.read_only_hint is True
            assert logs_ann.destructive_hint is False
            assert "debug" in tool_map["get_server_logs"].meta["fastmcp"]["tags"]

        finally:
            set_effective_persona(previous_persona)


class TestReadinessDiagnostics:
    """Tests for ReadinessDiagnostic and run_readiness_diagnostics."""

    def test_readiness_diagnostic_to_dict(self) -> None:
        diag = ReadinessDiagnostic(
            check="rbac:list:trainjobs:default",
            status="fail",
            message="Permission denied",
            category="permission_denied",
            resource="trainjobs",
            verb="list",
            namespace="default",
        )
        data = diag.to_dict()
        assert data == {
            "check": "rbac:list:trainjobs:default",
            "status": "fail",
            "message": "Permission denied",
            "category": "permission_denied",
            "resource": "trainjobs",
            "verb": "list",
            "namespace": "default",
        }

    def test_readiness_diagnostic_to_dict_minimal(self) -> None:
        diag = ReadinessDiagnostic(
            check="k8s_api",
            status="pass",
            message="Kubernetes API reachable",
        )
        data = diag.to_dict()
        assert data == {
            "check": "k8s_api",
            "status": "pass",
            "message": "Kubernetes API reachable",
        }

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="kubeflow")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_all_pass_when_healthy(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        # 1. API reachable
        mock_v1 = MagicMock()
        mock_core.return_value = mock_v1

        # 2. CRD exists serving v1alpha1
        mock_crd = MagicMock()
        version_mock = MagicMock()
        version_mock.name = "v1alpha1"
        version_mock.served = True
        mock_crd.spec.versions = [version_mock]
        mock_apiext.return_value.read_custom_resource_definition.return_value = mock_crd

        # 3. RBAC checks allowed
        mock_ssar = MagicMock()
        mock_ssar.status.allowed = True
        mock_auth.return_value.create_self_subject_access_review.return_value = mock_ssar

        ok, diagnostics = run_readiness_diagnostics()

        assert ok is True
        assert len(diagnostics) == 5  # 1 api + 1 crd + 3 rbac
        assert all(d.status == "pass" for d in diagnostics)
        assert diagnostics[0].check == "k8s_api"
        assert diagnostics[1].check == "trainer_crd"
        assert diagnostics[2].check == "rbac:list:trainjobs:kubeflow"
        assert diagnostics[3].check == "rbac:create:trainjobs:kubeflow"
        assert diagnostics[4].check == "rbac:list:clustertrainingruntimes"

    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_api_unreachable_short_circuits(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
    ) -> None:
        mock_core.return_value.list_namespace.side_effect = ConnectionError("Connection refused")

        ok, diagnostics = run_readiness_diagnostics()

        assert ok is False
        assert len(diagnostics) == 1
        assert diagnostics[0].check == "k8s_api"
        assert diagnostics[0].status == "fail"
        assert diagnostics[0].category == "api_unreachable"
        # Verify CRD and Auth APIs were not touched
        mock_apiext.assert_not_called()
        mock_auth.assert_not_called()

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="default")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_crd_missing(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        mock_apiext.return_value.read_custom_resource_definition.side_effect = Exception(
            "404 Not Found"
        )
        mock_auth.return_value.create_self_subject_access_review.return_value = MagicMock(
            status=MagicMock(allowed=True)
        )

        ok, diagnostics = run_readiness_diagnostics()

        assert ok is False
        crd_diag = next(d for d in diagnostics if d.check == "trainer_crd")
        assert crd_diag.status == "fail"
        assert crd_diag.category == "crd_missing"

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="default")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_crd_version_unsupported(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        mock_crd = MagicMock()
        v_mock = MagicMock()
        v_mock.name = "v1beta1"
        v_mock.served = True
        mock_crd.spec.versions = [v_mock]
        mock_apiext.return_value.read_custom_resource_definition.return_value = mock_crd
        mock_auth.return_value.create_self_subject_access_review.return_value = MagicMock(
            status=MagicMock(allowed=True)
        )

        ok, diagnostics = run_readiness_diagnostics()

        assert ok is False
        crd_diag = next(d for d in diagnostics if d.check == "trainer_crd")
        assert crd_diag.status == "fail"
        assert crd_diag.category == "crd_version_unsupported"

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="default")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_rbac_denied(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        mock_crd = MagicMock()
        v_mock = MagicMock()
        v_mock.name = "v1alpha1"
        v_mock.served = True
        mock_crd.spec.versions = [v_mock]
        mock_apiext.return_value.read_custom_resource_definition.return_value = mock_crd

        mock_ssar = MagicMock()
        mock_ssar.status.allowed = False
        mock_ssar.status.reason = "User cannot create trainjobs"
        mock_auth.return_value.create_self_subject_access_review.return_value = mock_ssar

        ok, diagnostics = run_readiness_diagnostics()

        assert ok is False
        rbac_denied = [d for d in diagnostics if d.category == "permission_denied"]
        assert len(rbac_denied) == 3
        assert rbac_denied[0].status == "fail"
        assert "Permission denied: list trainjobs" in rbac_denied[0].message
        assert rbac_denied[0].resource == "trainjobs"
        assert rbac_denied[0].verb == "list"
        assert rbac_denied[0].namespace == "default"

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="default")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_rbac_api_init_fails_warns_without_blocking(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        mock_crd = MagicMock()
        v_mock = MagicMock()
        v_mock.name = "v1alpha1"
        v_mock.served = True
        mock_crd.spec.versions = [v_mock]
        mock_apiext.return_value.read_custom_resource_definition.return_value = mock_crd

        mock_auth.side_effect = Exception("Authorization API init failure")

        ok, diagnostics = run_readiness_diagnostics()

        # Fail-safe: warning on SSAR initialization does not mark server not-ready
        assert ok is True
        rbac_warn = next(d for d in diagnostics if d.check == "rbac")
        assert rbac_warn.status == "warn"
        assert rbac_warn.category == "unknown"

    @patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace", return_value="default")
    @patch("kubeflow_mcp.common.utils.get_authorization_v1_api")
    @patch("kubeflow_mcp.common.utils.get_apiextensions_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_rbac_check_exception_warns_without_blocking(
        self,
        mock_core: MagicMock,
        mock_apiext: MagicMock,
        mock_auth: MagicMock,
        mock_ns: MagicMock,
    ) -> None:
        mock_crd = MagicMock()
        v_mock = MagicMock()
        v_mock.name = "v1alpha1"
        v_mock.served = True
        mock_crd.spec.versions = [v_mock]
        mock_apiext.return_value.read_custom_resource_definition.return_value = mock_crd

        mock_auth.return_value.create_self_subject_access_review.side_effect = Exception(
            "ServiceAccount cannot create selfsubjectaccessreviews"
        )

        ok, diagnostics = run_readiness_diagnostics()

        # Fail-safe: warning does not mark server not-ready
        assert ok is True
        rbac_warns = [d for d in diagnostics if d.status == "warn"]
        assert len(rbac_warns) == 3
        assert all(d.category == "unknown" for d in rbac_warns)
