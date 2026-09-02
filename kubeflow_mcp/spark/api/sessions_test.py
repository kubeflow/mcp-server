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

"""Tests for SparkConnect session lifecycle tools (create / delete).

Covers the confirm gate, name validation, the MCP ownership label written on
create, and the ownership guard enforced on delete.
"""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE
from kubeflow_mcp.spark.api import sessions


class TestCreateSession:
    def test_preview_when_not_confirmed(self):
        out = sessions.create_spark_session(name="etl", num_executors=2)
        assert out["status"] == "preview"
        assert out["config"]["name"] == "etl"

    def test_invalid_name_rejected(self):
        out = sessions.create_spark_session(name="Bad_Name", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_name_too_long_rejected(self):
        out = sessions.create_spark_session(name="a" * 60, confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_negative_executors_rejected(self):
        out = sessions.create_spark_session(num_executors=-1, confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_auto_generated_name_in_preview(self):
        out = sessions.create_spark_session()
        assert out["config"]["name"].startswith("spark-connect-")

    def test_created_session_carries_mcp_ownership_label(self, mock_spark_client):
        """The CR must be labelled at creation, or the delete guard can never
        recognise it as MCP-owned."""
        pytest.importorskip("kubeflow.spark")  # asserts on real SDK option objects
        out = sessions.create_spark_session(name="etl", confirmed=True)
        assert out["success"] is True

        _args, kwargs = mock_spark_client.connect.call_args
        label_opts = [o for o in kwargs["options"] if hasattr(o, "labels")]
        assert label_opts, "create_spark_session must pass a Labels option"
        assert label_opts[0].labels[MCP_MANAGED_LABEL] == MCP_MANAGED_VALUE

    def test_session_name_option_still_passed(self, mock_spark_client):
        pytest.importorskip("kubeflow.spark")  # asserts on real SDK option objects
        sessions.create_spark_session(name="etl", confirmed=True)
        _args, kwargs = mock_spark_client.connect.call_args
        name_opts = [o for o in kwargs["options"] if hasattr(o, "name")]
        assert name_opts and name_opts[0].name == "etl"


class TestDeleteOwnershipGuard:
    """CONVENTIONS.md: non-admin personas may only mutate MCP-labelled resources,
    and a missing resource must be distinguishable from an unowned one."""

    def test_unmanaged_session_is_validation_error(self, spark_persona):
        spark_persona("data-scientist")
        with patch.object(sessions, "get_spark_session_ownership", return_value="unmanaged"):
            out = sessions.delete_spark_session("x", namespace="default", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_missing_session_is_resource_not_found(self, spark_persona):
        spark_persona("data-scientist")
        with patch.object(sessions, "get_spark_session_ownership", return_value="not_found"):
            out = sessions.delete_spark_session("gone", namespace="default", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "RESOURCE_NOT_FOUND"

    def test_unverifiable_ownership_is_sdk_error(self, spark_persona):
        spark_persona("data-scientist")
        with patch.object(sessions, "get_spark_session_ownership", return_value="unknown"):
            out = sessions.delete_spark_session("x", namespace="default", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "SDK_ERROR"

    def test_guard_runs_before_preview(self, spark_persona):
        """An unowned session must be refused at preview time, not after the
        user has already confirmed."""
        spark_persona("data-scientist")
        with patch.object(sessions, "get_spark_session_ownership", return_value="unmanaged"):
            out = sessions.delete_spark_session("x", namespace="default")
        assert out.get("status") != "preview"
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_platform_admin_bypasses_guard(self, spark_persona, mock_spark_client):
        spark_persona("platform-admin")
        with patch.object(sessions, "get_spark_session_ownership") as ownership:
            out = sessions.delete_spark_session("x", namespace="default", confirmed=True)
        ownership.assert_not_called()
        assert out["success"] is True

    def test_managed_session_deletes(self, spark_persona, mock_spark_client):
        spark_persona("data-scientist")
        with patch.object(sessions, "get_spark_session_ownership", return_value="managed"):
            out = sessions.delete_spark_session("x", namespace="default", confirmed=True)
        assert out["success"] is True
        assert out["data"]["deleted"] is True
        mock_spark_client.delete_session.assert_called_once_with("x")


class TestDeleteSession:
    def test_preview_when_not_confirmed(self, spark_persona):
        spark_persona("platform-admin")
        out = sessions.delete_spark_session("x")
        assert out["status"] == "preview"

    def test_invalid_name_rejected(self, spark_persona):
        spark_persona("platform-admin")
        out = sessions.delete_spark_session("Bad_Name", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_delete_confirmed(self, spark_persona, mock_spark_client):
        spark_persona("platform-admin")
        out = sessions.delete_spark_session("x", confirmed=True)
        assert out["success"] is True
        assert out["data"]["deleted"] is True
        mock_spark_client.delete_session.assert_called_once_with("x")

    def test_delete_not_found(self, spark_persona):
        from kubernetes.client.exceptions import ApiException

        spark_persona("platform-admin")
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
