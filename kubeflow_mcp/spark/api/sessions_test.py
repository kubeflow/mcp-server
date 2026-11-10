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
from kubeflow_mcp.conftest import make_spark_session_info
from kubeflow_mcp.spark.api import sessions


@pytest.fixture
def provisioned_session(mock_spark_client):
    """Model a real CR carrying the labels passed by this invocation."""
    mock_spark_client.get_session.return_value = make_spark_session_info("etl")

    def read_cr(**kwargs):
        from kubeflow.spark.backends.kubernetes.utils import build_spark_connect_cr

        assert kwargs["name"] == "etl"
        assert kwargs["namespace"] == "default"
        options = mock_spark_client.connect.call_args.kwargs["options"]
        labels = next(option.labels for option in options if hasattr(option, "labels"))
        cr = build_spark_connect_cr(name="etl", namespace="default").to_dict()
        cr["metadata"]["labels"] = dict(labels)
        cr["status"] = {"state": "Ready"}
        return cr

    with patch("kubeflow_mcp.common.utils.get_custom_objects_api") as api:
        api.return_value.get_namespaced_custom_object.side_effect = read_cr
        yield mock_spark_client


class TestConfirmationType:
    @pytest.mark.parametrize("tool", [sessions.create_spark_session, sessions.delete_spark_session])
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
    def test_non_boolean_confirmation_cannot_write(self, mock_spark_client, tool, value):
        out = tool(name="etl", namespace="default", confirmed=value)
        assert out["error_code"] == "VALIDATION_ERROR"
        mock_spark_client.connect.assert_not_called()
        mock_spark_client.delete_session.assert_not_called()

    # Under #484, execute_tool coerces "false" to False in lax mode (previewing without
    # writing). Non-boolean strings such as "maybe" are rejected with VALIDATION_ERROR.
    @pytest.mark.parametrize("mode", ["progressive", "semantic"])
    @pytest.mark.parametrize("tool_name", ["create_spark_session", "delete_spark_session"])
    async def test_non_boolean_confirmation_is_refused_over_mcp(
        self, mock_spark_client, spark_persona, mode, tool_name
    ):
        from fastmcp import Client

        from kubeflow_mcp.core.server import create_server

        spark_persona("data-scientist")
        async with Client(
            create_server(clients=["spark"], persona="data-scientist", mode=mode)
        ) as client:
            out = await client.call_tool(
                "execute_tool",
                {
                    "tool_name": tool_name,
                    "arguments": {"name": "etl", "namespace": "default", "confirmed": "maybe"},
                },
                raise_on_error=False,
            )
        assert out.structured_content["error_code"] == "VALIDATION_ERROR"
        mock_spark_client.connect.assert_not_called()
        mock_spark_client.delete_session.assert_not_called()


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


class TestCreateGuardsInput:
    """num_executors and spark_conf reach the cluster, so both are bounded."""

    def test_executor_count_above_max_rejected(self):
        out = sessions.create_spark_session(
            num_executors=sessions._MAX_EXECUTORS + 1, confirmed=True
        )
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_executor_count_at_max_allowed(self):
        # The ceiling itself must stay usable.
        out = sessions.create_spark_session(num_executors=sessions._MAX_EXECUTORS)
        assert out["status"] == "preview"

    @pytest.mark.parametrize(
        "key",
        [
            "spark.kubernetes.container.image",
            "spark.kubernetes.driver.container.image",
            "spark.kubernetes.authenticate.driver.serviceAccountName",
            "spark.kubernetes.namespace",
            "spark.kubernetes.driver.podTemplateFile",
            "spark.kubernetes.driver.volumes.hostPath.data.mount.path",
            "spark.kubernetes.executor.secrets.my-secret",
        ],
    )
    def test_server_managed_conf_keys_rejected(self, key):
        """These would change the image, identity, namespace or mounts without
        appearing in the confirm-gate preview."""
        out = sessions.create_spark_session(
            name="etl", spark_conf={key: "x"}, namespace="default", confirmed=True
        )
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"
        assert key in out["error"]

    def test_restricted_key_rejected_before_the_preview(self):
        # Refused up front, not after the user has already confirmed.
        out = sessions.create_spark_session(
            name="etl", spark_conf={"spark.kubernetes.container.image": "evil:latest"}
        )
        assert out.get("status") != "preview"
        assert out["error_code"] == "VALIDATION_ERROR"

    def test_ordinary_spark_conf_still_allowed(self):
        out = sessions.create_spark_session(
            name="etl", spark_conf={"spark.sql.shuffle.partitions": "8"}
        )
        assert out["status"] == "preview"
        assert out["config"]["spark_conf"] == {"spark.sql.shuffle.partitions": "8"}


class TestCreateBoundsTimeouts:
    """connect() blocks a server thread for the whole timeout, so both are bounded."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("timeout", sessions._MAX_TIMEOUT + 1),
            ("timeout", 0),
            ("timeout", -1),
            ("connect_timeout", sessions._MAX_CONNECT_TIMEOUT + 1),
            ("connect_timeout", 0),
            ("connect_timeout", -5),
        ],
    )
    def test_out_of_range_rejected(self, field, value):
        out = sessions.create_spark_session(name="etl", confirmed=True, **{field: value})
        assert out["success"] is False
        assert out["error_code"] == "VALIDATION_ERROR"
        assert field in out["error"]

    def test_ceiling_values_allowed(self):
        out = sessions.create_spark_session(
            name="etl",
            timeout=sessions._MAX_TIMEOUT,
            connect_timeout=sessions._MAX_CONNECT_TIMEOUT,
        )
        assert out["status"] == "preview"


class TestPreviewShowsEffectiveNamespace:
    def test_omitted_namespace_resolved_in_preview(self):
        with patch.object(sessions, "get_spark_effective_namespace", return_value="team-a"):
            out = sessions.create_spark_session(name="etl")
        assert out["config"]["namespace"] == "team-a"
        assert "team-a" in out["message"]

    def test_explicit_namespace_kept(self):
        out = sessions.create_spark_session(name="etl", namespace="default")
        assert out["config"]["namespace"] == "default"

    def test_unresolvable_namespace_still_previews(self):
        with patch.object(
            sessions, "get_spark_effective_namespace", side_effect=RuntimeError("no kubeconfig")
        ):
            out = sessions.create_spark_session(name="etl")
        assert out["status"] == "preview"
        assert out["config"]["namespace"] is None


class TestDeleteUsesSparkResolver:
    """The ownership guard must resolve the namespace through the same function
    the client factory uses, or it could check one namespace and delete in another."""

    def test_guard_checks_the_resolved_namespace(self, spark_persona):
        spark_persona("data-scientist")
        with (
            patch.object(sessions, "get_spark_effective_namespace", return_value="team-a"),
            patch.object(
                sessions, "get_spark_session_ownership", return_value="unmanaged"
            ) as ownership,
        ):
            sessions.delete_spark_session("x", confirmed=True)
        ownership.assert_called_once_with("x", "team-a")


class TestCreateFailsAfterProvisioning:
    """``connect()`` can fail after the CR exists. Reporting only the error would
    leave a running, MCP-labelled session the caller never hears about."""

    def test_import_error_after_create_reports_the_live_session(self, provisioned_session):
        # pyspark imports pandas only when building the SparkSession, so a missing
        # or incompatible dependency surfaces *after* the CR has been created.
        provisioned_session.connect.side_effect = ImportError(
            "[PACKAGE_NOT_INSTALLED] Pandas >= 2.2.0 must be installed"
        )
        out = sessions.create_spark_session(name="etl", namespace="default", confirmed=True)
        assert out["success"] is True
        assert "warning" in out["data"]
        assert out["data"]["name"] == "etl"

    def test_generic_failure_after_create_reports_the_live_session(self, provisioned_session):
        provisioned_session.connect.side_effect = RuntimeError("driver unreachable")
        out = sessions.create_spark_session(name="etl", namespace="default", confirmed=True)
        assert out["success"] is True
        assert "warning" in out["data"]

    def test_missing_extra_with_no_session_still_errors(self):
        # Nothing was provisioned, so the friendly install hint must survive.
        err = ImportError("SparkClient requires the Spark extra.")
        with patch.object(sessions, "get_spark_client_for_namespace", side_effect=err):
            out = sessions.create_spark_session(name="etl", namespace="default", confirmed=True)
        assert out["success"] is False
        assert out["error_code"] == "SDK_ERROR"
        assert "Spark extra" in out["error"]


class TestCreateConflictRecovery:
    @pytest.mark.parametrize("managed", [False, True])
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_existing_name_preserves_conflict(self, mock_spark_client, managed, wrapped):
        from kubernetes.client.exceptions import ApiException

        conflict = ApiException(status=409, reason="AlreadyExists")
        error = RuntimeError("Failed to create SparkConnect") if wrapped else conflict
        if wrapped:
            error.__cause__ = conflict
        mock_spark_client.connect.side_effect = error
        labels = {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE} if managed else {}
        with patch("kubeflow_mcp.common.utils.get_custom_objects_api") as api:
            api.return_value.get_namespaced_custom_object.return_value = {
                "metadata": {"labels": labels}
            }
            out = sessions.create_spark_session("etl", namespace="default", confirmed=True)
            api.assert_not_called()
        assert out["success"] is False
        assert out["error_code"] == "SDK_ERROR"
        assert out["hint"]
        assert "409" in str(out["details"])
        mock_spark_client.get_session.assert_not_called()
        mock_spark_client.delete_session.assert_not_called()

    @pytest.mark.parametrize(
        "labels",
        [
            {},
            {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE},
            {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE, sessions.SPARK_CREATION_LABEL: "old-attempt"},
            {sessions.SPARK_CREATION_LABEL: "same-attempt"},
        ],
    )
    def test_unrelated_existing_session_does_not_mask_failure(self, mock_spark_client, labels):
        mock_spark_client.connect.side_effect = RuntimeError("driver unreachable")
        with (
            patch("kubeflow_mcp.common.utils.get_custom_objects_api") as api,
            patch.object(sessions.uuid, "uuid4", return_value=MagicMock(hex="same-attempt")),
        ):
            api.return_value.get_namespaced_custom_object.return_value = {
                "metadata": {"labels": labels}
            }
            out = sessions.create_spark_session("etl", namespace="default", confirmed=True)
        assert out["success"] is False
        mock_spark_client.get_session.assert_not_called()

    def test_failed_recovery_lookup_keeps_original_error(self, mock_spark_client):
        mock_spark_client.connect.side_effect = RuntimeError("original create failure")
        with patch(
            "kubeflow_mcp.common.utils.get_custom_objects_api", side_effect=OSError("offline")
        ):
            out = sessions.create_spark_session("etl", namespace="default", confirmed=True)
        assert out["error"] == "original create failure"

    def test_recovery_uses_checked_snapshot_without_second_lookup(self, provisioned_session):
        provisioned_session.connect.side_effect = RuntimeError("driver unreachable")
        provisioned_session.get_session.side_effect = AssertionError("second lookup could race")
        out = sessions.create_spark_session("etl", namespace="default", confirmed=True)
        assert out["success"] is True
        assert out["data"]["name"] == "etl"
        provisioned_session.get_session.assert_not_called()

    @pytest.mark.parametrize("field,value", [("name", "another"), ("namespace", "other")])
    def test_wrong_snapshot_metadata_cannot_be_recovered(self, provisioned_session, field, value):
        from kubeflow.spark.backends.kubernetes.utils import build_spark_connect_cr

        provisioned_session.connect.side_effect = RuntimeError("driver unreachable")
        snapshot = build_spark_connect_cr(name="etl", namespace="default").to_dict()
        snapshot["metadata"][field] = value
        snapshot["metadata"]["labels"] = {
            MCP_MANAGED_LABEL: MCP_MANAGED_VALUE,
            sessions.SPARK_CREATION_LABEL: "attempt",
        }
        with (
            patch.object(sessions.uuid, "uuid4", return_value=MagicMock(hex="attempt")),
            patch("kubeflow_mcp.common.utils.get_custom_objects_api") as api,
        ):
            api.return_value.get_namespaced_custom_object.return_value = snapshot
            out = sessions.create_spark_session("etl", namespace="default", confirmed=True)
        assert out["success"] is False


class TestResourceKeys:
    @pytest.mark.parametrize("field", ["driver_resources", "executor_resources"])
    @pytest.mark.parametrize(
        "key,value",
        [
            ("cpu", "-1"),
            ("cpu", "0"),
            ("cpu", "nan"),
            ("cpu", "inf"),
            ("cpu", "1025"),
            ("cpu", "1.5m"),
            ("cpu", "bad"),
            ("cpu", ""),
            ("memory", "0Gi"),
            ("memory", "-1Gi"),
            ("memory", "nonsense"),
            ("memory", "\u0660\u0661Gi"),
            ("memory", "\uff11Gi"),
            ("memory", "nan"),
            ("memory", "1.5"),
            ("memory", ""),
        ],
    )
    @pytest.mark.parametrize("confirmed", [False, True])
    def test_invalid_quantities_rejected_before_namespace(self, field, key, value, confirmed):
        with patch.object(sessions, "check_namespace_allowed") as check:
            out = sessions.create_spark_session(**{field: {key: value}}, confirmed=confirmed)
        assert out["error_code"] == "VALIDATION_ERROR"
        assert f"{field}.{key}" in out["error"]
        check.assert_not_called()

    @pytest.mark.parametrize(
        "cpu,expected", [("100m", "1"), ("1500m", "2"), ("1.2", "2"), ("1024", "1024")]
    )
    def test_preview_shows_sdk_cpu_rounding_without_mutating_input(self, cpu, expected):
        resources = {"cpu": cpu, "memory": "1Gi"}
        out = sessions.create_spark_session(driver_resources=resources)
        assert out["config"]["driver_resources"]["cpu"] == expected
        assert resources["cpu"] == cpu

    @pytest.mark.parametrize("memory", ["512Mi", "2Gi", "1.5Gi", "1g", "1GB", "1024", "1Ei"])
    def test_supported_memory_values_still_preview(self, memory):
        assert (
            sessions.create_spark_session(driver_resources={"memory": memory})["status"]
            == "preview"
        )

    def test_empty_name_is_not_silently_replaced(self):
        with patch.object(sessions, "check_namespace_allowed") as check:
            out = sessions.create_spark_session(name="")
        assert out["error_code"] == "VALIDATION_ERROR"
        check.assert_not_called()

    @pytest.mark.parametrize("field", ["driver_resources", "executor_resources"])
    @pytest.mark.parametrize("key", ["nvidia.com/gpu", "amd.com/gpu", "ephemeral-storage", "typo"])
    @pytest.mark.parametrize("confirmed", [False, True])
    def test_unsupported_keys_rejected_before_sdk(self, mock_spark_client, field, key, confirmed):
        out = sessions.create_spark_session(**{field: {key: "1"}}, confirmed=confirmed)
        assert out["error_code"] == "VALIDATION_ERROR"
        assert field in out["error"] and key in out["error"]
        mock_spark_client.connect.assert_not_called()


class TestDriverServiceAccount:
    @pytest.mark.parametrize("name", ["spark", "spark.driver", ("a" * 63 + ".") * 3 + "a" * 61])
    def test_preview_shows_valid_account_without_writes(self, mock_spark_client, name):
        out = sessions.create_spark_session(service_account_name=name)
        assert out["status"] == "preview"
        assert out["config"]["service_account_name"] == name
        mock_spark_client.connect.assert_not_called()

    @pytest.mark.parametrize("name", ["", "Bad_Name", "spark/driver", "a" * 64, "a" * 254])
    def test_invalid_account_rejected_before_namespace_lookup(self, name):
        with patch.object(sessions, "check_namespace_allowed") as check:
            out = sessions.create_spark_session(service_account_name=name)
        assert out["error_code"] == "VALIDATION_ERROR"
        check.assert_not_called()

    @pytest.mark.parametrize("resources", [None, {"cpu": "2", "memory": "2Gi"}])
    def test_account_reaches_generated_spec(self, mock_spark_client, resources):
        pytest.importorskip("kubeflow.spark")
        from kubeflow.spark.backends.kubernetes.utils import build_spark_connect_cr

        out = sessions.create_spark_session(
            "etl", service_account_name="spark.driver", driver_resources=resources, confirmed=True
        )
        assert out["success"] is True
        driver = mock_spark_client.connect.call_args.kwargs["driver"]
        cr = build_spark_connect_cr(name="etl", namespace="default", driver=driver)
        assert cr.spec.server.template.spec.service_account_name == "spark.driver"
        assert cr.spec.server.cores == (2 if resources else 1)

    def test_omitted_account_keeps_sdk_default(self, mock_spark_client):
        sessions.create_spark_session("etl", confirmed=True)
        assert mock_spark_client.connect.call_args.kwargs["driver"] is None

    def test_admission_denial_is_permission_error(self, mock_spark_client):
        from kubernetes.client.exceptions import ApiException

        error = RuntimeError("create denied")
        error.__cause__ = ApiException(status=403)
        mock_spark_client.connect.side_effect = error
        out = sessions.create_spark_session(
            "etl", service_account_name="spark", namespace="default", confirmed=True
        )
        assert out["error_code"] == "PERMISSION_DENIED"
        assert "ServiceAccount" in out["hint"]
        mock_spark_client.get_session.assert_not_called()


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
