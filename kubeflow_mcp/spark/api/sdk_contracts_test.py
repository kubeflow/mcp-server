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

"""Contract tests between the spark module and the ``kubeflow[spark]`` SDK.

These pin the SDK surface that ``SDK_COMPATIBILITY`` claims to cover, plus the
serializer that adapts SDK types to MCP responses. They exist because field
names differed between the released SDK and its ``main`` branch (the driver pod
was ``pod_name`` on release, ``driver_pod_name`` on ``main``) — a divergence that
silently returned ``None`` until it was pinned here.
"""

import inspect
from types import SimpleNamespace

import pytest

from kubeflow_mcp.common.constants import SDK_COMPATIBILITY
from kubeflow_mcp.conftest import make_spark_session_info
from kubeflow_mcp.spark import TOOLS
from kubeflow_mcp.spark.types import session_info_to_dict

sdk_spark = pytest.importorskip("kubeflow.spark", reason="requires the kubeflow[spark] extra")

SPARK_COMPAT = SDK_COMPATIBILITY["clients"]["spark"]


# ─── Serializer contract ────────────────────────────────────────────────────


class TestSerialization:
    def test_session_info_to_dict(self):
        data = session_info_to_dict(make_spark_session_info())
        assert data["name"] == "spark-connect-ab12"
        assert data["state"] == "Ready"  # enum .value extracted
        assert data["service_name"] == "spark-connect-ab12-svc"
        assert data["creation_timestamp"] == "2026-07-09T12:00:00"  # datetime -> ISO

    def test_session_info_plain_string_state(self):
        info = make_spark_session_info()
        info.state = "Failed"  # tolerate a plain string state
        assert session_info_to_dict(info)["state"] == "Failed"

    def test_driver_pod_name_from_released_pod_name_field(self):
        # Released kubeflow[spark] 0.4.x exposes the driver pod as `pod_name`;
        # the serializer must surface it under `driver_pod_name`.
        info = SimpleNamespace(
            name="s",
            namespace="default",
            state=SimpleNamespace(value="Ready"),
            pod_name="server-pod-0",
            pod_ip=None,
            service_name=None,
            creation_timestamp=None,
        )
        assert session_info_to_dict(info)["driver_pod_name"] == "server-pod-0"

    def test_driver_pod_name_fallback_to_main_field(self):
        # Unreleased SDK `main` renamed the field to `driver_pod_name`; the
        # serializer falls back to it when `pod_name` is absent.
        info = SimpleNamespace(
            name="s",
            namespace="default",
            state=SimpleNamespace(value="Ready"),
            driver_pod_name="server-pod-1",
            pod_ip=None,
            service_name=None,
            creation_timestamp=None,
        )
        assert session_info_to_dict(info)["driver_pod_name"] == "server-pod-1"

    def test_serializer_reads_a_field_the_installed_sdk_actually_has(self):
        """Guard against the whole class of bug above: at least one of the two
        driver-pod field names must exist on the installed SparkConnectInfo."""
        fields = set(getattr(sdk_spark.SparkConnectInfo, "__annotations__", {}))
        assert fields & {"pod_name", "driver_pod_name"}, (
            f"Installed SDK SparkConnectInfo exposes neither driver-pod field: {sorted(fields)}"
        )


# ─── SDK surface contract ───────────────────────────────────────────────────


class TestSDKSurface:
    def test_declared_client_is_importable(self):
        assert SPARK_COMPAT["sdk_client"] == "kubeflow.spark.SparkClient"
        assert hasattr(sdk_spark, "SparkClient")

    def test_covered_methods_exist_on_client(self):
        for method in SPARK_COMPAT["covered_methods"]:
            base = method.split("(")[0]
            assert hasattr(sdk_spark.SparkClient, base), (
                f"SDK_COMPATIBILITY claims '{base}' but SparkClient has no such method"
            )

    def test_options_used_by_create_are_available(self):
        # create_spark_session passes both of these to client.connect().
        assert hasattr(sdk_spark, "Name")
        assert hasattr(sdk_spark, "Labels")

    def test_labels_option_carries_a_labels_mapping(self):
        labels = sdk_spark.Labels({"a": "b"})
        assert labels.labels == {"a": "b"}

    def test_connect_accepts_the_parameters_create_passes(self):
        params = inspect.signature(sdk_spark.SparkClient.connect).parameters
        for name in ("spark_conf", "driver", "executor", "options", "timeout", "connect_timeout"):
            assert name in params, f"SparkClient.connect lost parameter '{name}'"

    def test_driver_and_executor_types_available(self):
        assert hasattr(sdk_spark, "Driver")
        assert hasattr(sdk_spark, "Executor")


# ─── Compatibility bookkeeping ──────────────────────────────────────────────


class TestCompatibilityDeclaration:
    def test_every_tool_is_backed_by_declared_coverage(self):
        assert SPARK_COMPAT["status"] == "implemented"
        assert TOOLS, "spark module declares no tools"
        assert SPARK_COMPAT["covered_methods"], "implemented module must declare covered methods"

    def test_uncovered_methods_are_documented_not_empty_placeholders(self):
        for entry in SPARK_COMPAT.get("uncovered_methods", []):
            assert entry.strip(), "uncovered_methods entries must name a real SDK method"
