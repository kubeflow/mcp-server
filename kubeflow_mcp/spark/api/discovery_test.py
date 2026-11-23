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

"""Tests for spark discovery tools (list / get SparkConnect sessions).

The spark module imports the SDK lazily, so these run without the
``kubeflow[spark]`` extra installed; the SparkClient is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.conftest import make_spark_session_info
from kubeflow_mcp.spark.api import discovery


class TestListTruncation:
    """`limit` silently cut the list, so an agent could not tell more existed."""

    def test_truncated_flag_set_when_limit_cuts(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info(f"s{i}") for i in range(5)
        ]
        out = discovery.list_spark_sessions(limit=2)
        assert out["data"]["truncated"] is True
        assert len(out["data"]["sessions"]) == 2
        assert out["data"]["total"] == 5

    def test_not_truncated_when_everything_fits(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info("a"),
            make_spark_session_info("b"),
        ]
        out = discovery.list_spark_sessions(limit=10)
        assert out["data"]["truncated"] is False
        assert out["data"]["total"] == 2


class TestNextStepsRespectConfirmGate:
    def test_failed_session_hint_previews_before_deleting(self, mock_spark_client):
        """The hint must not tell an agent to delete with confirmed=True outright."""
        mock_spark_client.get_session.return_value = make_spark_session_info("x", state="Failed")
        out = discovery.get_spark_session("x")
        delete_hints = [s for s in out["data"]["next_steps"] if "delete_spark_session" in s]
        assert delete_hints
        assert all("confirmed=True) —" not in h for h in delete_hints)
        assert any("preview" in h for h in delete_hints)


class TestListSessions:
    def test_list_and_serialize(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info("a"),
            make_spark_session_info("b", state="Failed"),
        ]
        out = discovery.list_spark_sessions()
        assert out["success"] is True
        assert out["data"]["total"] == 2

    def test_state_filter(self, mock_spark_client):
        mock_spark_client.list_sessions.return_value = [
            make_spark_session_info("a"),
            make_spark_session_info("b", state="Failed"),
        ]
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
    def test_invalid_session_name_is_rejected_before_sdk_call(self):
        client = MagicMock()
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("Not a valid/name")
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"
        client.get_session.assert_not_called()

    def test_get_ok(self, mock_spark_client):
        mock_spark_client.get_session.return_value = make_spark_session_info("x")
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

    def test_failed_state_has_next_steps(self, mock_spark_client):
        mock_spark_client.get_session.return_value = make_spark_session_info("x", state="Failed")
        out = discovery.get_spark_session("x")
        assert "next_steps" in out["data"]


def _forbidden() -> Exception:
    """A Kubernetes 403 wrapped the way the SDK raises it."""
    from kubernetes.client.exceptions import ApiException

    exc = RuntimeError("Failed to list SparkConnect sessions")
    exc.__cause__ = ApiException(status=403, reason="Forbidden")
    return exc


class TestForbiddenIsPermissionDenied:
    def test_list_forbidden_is_permission_denied(self):
        client = MagicMock()
        client.list_sessions.side_effect = _forbidden()
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.list_spark_sessions(namespace="team-b")
        assert out["success"] is False
        assert out["error_code"] == "PERMISSION_DENIED"
        assert "team-b" in out["error"]
        assert "RBAC" in out["hint"]

    def test_get_forbidden_is_permission_denied(self):
        client = MagicMock()
        client.get_session.side_effect = _forbidden()
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.get_spark_session("etl", namespace="team-b")
        assert out["error_code"] == "PERMISSION_DENIED"

    def test_other_failures_stay_sdk_error(self):
        client = MagicMock()
        client.list_sessions.side_effect = RuntimeError("connection refused")
        with patch.object(discovery, "get_spark_client_for_namespace", return_value=client):
            out = discovery.list_spark_sessions(namespace="team-a")
        assert out["error_code"] == "SDK_ERROR"


class TestForbiddenDoesNotOpenSharedBreaker:
    """The breaker is keyed by tool name only, so it is shared by every user and
    namespace. One user's denied requests must not lock the tool for others."""

    @staticmethod
    async def _call(client, mode: str, namespace: str) -> dict:
        if mode == "full":
            out = await client.call_tool(
                "list_spark_sessions", {"namespace": namespace}, raise_on_error=False
            )
        else:
            out = await client.call_tool(
                "execute_tool",
                {"tool_name": "list_spark_sessions", "arguments": {"namespace": namespace}},
                raise_on_error=False,
            )
        return out.structured_content

    @staticmethod
    def _clients_by_namespace(denied: Exception):
        from kubeflow_mcp.conftest import make_spark_session_info

        denied_client = MagicMock()
        denied_client.list_sessions.side_effect = denied
        ok_client = MagicMock()
        ok_client.list_sessions.return_value = [make_spark_session_info("a1")]
        return lambda ns: denied_client if ns == "team-b" else ok_client

    @pytest.mark.parametrize("mode", ["full", "progressive"])
    async def test_denied_namespace_does_not_lock_tool_for_others(self, spark_persona, mode):
        from fastmcp import Client

        from kubeflow_mcp.core.resilience import CircuitState, get_breaker
        from kubeflow_mcp.core.server import create_server

        spark_persona("data-scientist")
        threshold = get_breaker("list_spark_sessions").failure_threshold
        with patch.object(
            discovery,
            "get_spark_client_for_namespace",
            side_effect=self._clients_by_namespace(_forbidden()),
        ):
            async with Client(
                create_server(clients=["spark"], persona="data-scientist", mode=mode)
            ) as client:
                for _ in range(threshold + 1):
                    denied = await self._call(client, mode, "team-b")
                    assert denied["error_code"] == "PERMISSION_DENIED"
                allowed = await self._call(client, mode, "team-a")

        assert get_breaker("list_spark_sessions").state == CircuitState.CLOSED
        assert allowed["success"] is True
        assert allowed["data"]["sessions"][0]["name"] == "a1"

    async def test_real_outage_still_opens_breaker(self, spark_persona):
        """Guard against the fix simply switching the breaker off."""
        from fastmcp import Client

        from kubeflow_mcp.core.resilience import get_breaker
        from kubeflow_mcp.core.server import create_server

        spark_persona("data-scientist")
        threshold = get_breaker("list_spark_sessions").failure_threshold
        with patch.object(
            discovery,
            "get_spark_client_for_namespace",
            side_effect=self._clients_by_namespace(RuntimeError("connection refused")),
        ):
            async with Client(
                create_server(clients=["spark"], persona="data-scientist", mode="full")
            ) as client:
                for _ in range(threshold):
                    failed = await self._call(client, "full", "team-b")
                    assert failed["error_code"] == "SDK_ERROR"
                blocked = await self._call(client, "full", "team-a")

        assert blocked["error_code"] == "CIRCUIT_OPEN"
