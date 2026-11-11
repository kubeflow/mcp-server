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

from kubeflow_mcp.conftest import make_spark_job, make_spark_session_info
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


class TestListJobs:
    def test_list_serializes_jobs(self, mock_spark_client):
        mock_spark_client.list_jobs.return_value = [
            make_spark_job("a"),
            make_spark_job("b", status="Failed"),
        ]
        out = discovery.list_spark_jobs()
        assert out["success"] is True
        assert out["data"]["total"] == 2
        assert out["data"]["jobs"][0] == {
            "name": "a",
            "namespace": "default",
            "status": "Running",
            "num_executors": 2,
            "driver_pod_name": "etl-nightly-driver",
            "creation_timestamp": "2026-07-09T12:00:00",
        }

    def test_status_filter_is_case_insensitive(self, mock_spark_client):
        mock_spark_client.list_jobs.return_value = [
            make_spark_job("a"),
            make_spark_job("b", status="Failed"),
        ]
        out = discovery.list_spark_jobs(status="failed")
        assert [j["name"] for j in out["data"]["jobs"]] == ["b"]

    def test_unknown_status_is_refused_before_sdk_call(self, mock_spark_client):
        """A status the SDK never reports would silently return no jobs."""
        out = discovery.list_spark_jobs(status="Succeeded")
        assert out["error_code"] == "VALIDATION_ERROR"
        assert "Completed" in out["error"]
        mock_spark_client.list_jobs.assert_not_called()

    def test_truncated_flag_set_when_limit_cuts(self, mock_spark_client):
        mock_spark_client.list_jobs.return_value = [make_spark_job(f"j{i}") for i in range(5)]
        out = discovery.list_spark_jobs(limit=2)
        assert len(out["data"]["jobs"]) == 2
        assert out["data"]["total"] == 5
        assert out["data"]["truncated"] is True

    def test_invalid_limit(self):
        out = discovery.list_spark_jobs(limit=0)
        assert out["error_code"] == "VALIDATION_ERROR"


class TestGetJob:
    def test_get_ok(self, mock_spark_client):
        mock_spark_client.get_job.return_value = make_spark_job("x", status="Completed")
        out = discovery.get_spark_job("x")
        assert out["success"] is True
        assert out["data"]["status"] == "Completed"
        assert "next_steps" not in out["data"]

    @pytest.mark.parametrize(
        ("status", "tool"),
        [
            ("Failed", "get_spark_job_logs"),
            ("Running", "get_spark_job"),
            ("Created", "get_spark_job"),
        ],
    )
    def test_next_steps_point_to_existing_read_tools(self, mock_spark_client, status, tool):
        mock_spark_client.get_job.return_value = make_spark_job("x", status=status)
        out = discovery.get_spark_job("x")
        assert out["data"]["next_steps"][0].startswith(f"{tool}(name='x')")

    def test_invalid_name_is_rejected_before_sdk_call(self, mock_spark_client):
        out = discovery.get_spark_job("Not a valid/name")
        assert out["error_code"] == "VALIDATION_ERROR"
        mock_spark_client.get_job.assert_not_called()

    def test_not_found_maps_to_resource_not_found(self, mock_spark_client):
        from kubernetes.client.exceptions import ApiException

        exc = RuntimeError("Spark job not found: default/missing")
        exc.__cause__ = ApiException(status=404)
        mock_spark_client.get_job.side_effect = exc
        out = discovery.get_spark_job("missing")
        assert out["error_code"] == "RESOURCE_NOT_FOUND"
        assert "list_spark_jobs" in out["hint"]

    def test_other_failure_stays_sdk_error(self, mock_spark_client):
        mock_spark_client.get_job.side_effect = RuntimeError("Failed to get Spark job")
        out = discovery.get_spark_job("x")
        assert out["error_code"] == "SDK_ERROR"
