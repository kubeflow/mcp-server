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

"""Unit tests for ResourceQuota pre-flight verification (quota.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubernetes.client.rest import ApiException

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.trainer.api.quota import (
    check_resource_quota,
    get_namespace_resource_quotas,
)


def _make_resource_quota(
    name: str,
    hard: dict[str, str],
    used: dict[str, str],
    scopes: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(scopes=scopes),
        status=SimpleNamespace(hard=hard, used=used),
    )


class TestResourceQuotaChecks:
    def test_check_resource_quota_no_quotas(self) -> None:
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[])
        result = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "2", "memory": "4Gi"},
            num_nodes=2,
            core_v1_api=mock_api,
        )
        assert result is None

    def test_check_resource_quota_api_forbidden(self) -> None:
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.side_effect = ApiException(
            status=403, reason="Forbidden"
        )
        result = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "2"},
            core_v1_api=mock_api,
        )
        # Fails open so missing quota read permissions do not block users
        assert result is None

    def test_check_resource_quota_api_network_error(self) -> None:
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.side_effect = RuntimeError("Connection timeout")
        result = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "2"},
            core_v1_api=mock_api,
        )
        assert result is None

    def test_check_resource_quota_cpu_within_limits(self) -> None:
        quota = _make_resource_quota(
            name="compute-quota",
            hard={"requests.cpu": "16", "requests.memory": "64Gi"},
            used={"requests.cpu": "4", "requests.memory": "16Gi"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        result = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "4", "memory": "16Gi"},
            num_nodes=2,
            core_v1_api=mock_api,
        )
        # Requested 8 CPU (available 12), 32Gi memory (available 48Gi) -> passes
        assert result is None

    def test_check_resource_quota_cpu_exceeded(self) -> None:
        quota = _make_resource_quota(
            name="team-quota",
            hard={"requests.cpu": "8"},
            used={"requests.cpu": "6"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # 2 nodes * 2 CPUs = 4 CPUs requested, but only 2 available
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "2"},
            num_nodes=2,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'team-quota' exceeded" in err.error
        assert err.details["requested"] == "4"
        assert err.details["available"] == "2"
        assert err.details["hard"] == "8"
        assert err.details["used"] == "6"
        assert "Reduce requested cpu" in err.details["remediation"]

    def test_check_resource_quota_memory_exceeded(self) -> None:
        quota = _make_resource_quota(
            name="mem-quota",
            hard={"requests.memory": "32Gi"},
            used={"requests.memory": "24Gi"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # 1 node * 16Gi requested, but only 8Gi available
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"memory": "16Gi"},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'mem-quota' exceeded" in err.error
        assert err.details["resource"] == "requests.memory"

    def test_check_resource_quota_gpu_exceeded(self) -> None:
        quota = _make_resource_quota(
            name="gpu-quota",
            hard={"requests.nvidia.com/gpu": "4"},
            used={"requests.nvidia.com/gpu": "3"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # 2 nodes * 1 GPU = 2 GPUs requested, only 1 available
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=2,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'gpu-quota' exceeded" in err.error
        assert err.details["requested"] == "2"
        assert err.details["available"] == "1"

    def test_check_resource_quota_pods_exceeded(self) -> None:
        quota = _make_resource_quota(
            name="pod-quota",
            hard={"count/pods": "10"},
            used={"count/pods": "9"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # 2 nodes = 2 pods requested, only 1 available
        err = check_resource_quota(
            namespace="kubeflow-user",
            num_nodes=2,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "requested 2 pod(s), but only 1 available" in err.error
        assert err.details["resource"] == "count/pods"

    def test_check_resource_quota_best_effort_scope_skipped_with_cpu(self) -> None:
        quota = _make_resource_quota(
            name="best-effort-quota",
            hard={"pods": "0"},
            used={"pods": "0"},
            scopes=["BestEffort"],
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # Workload requesting CPU has Burstable QoS, skipping BestEffort quota
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"cpu": "2"},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is None

    def test_check_resource_quota_best_effort_scope_enforced_with_gpu_only(self) -> None:
        quota = _make_resource_quota(
            name="best-effort-quota",
            hard={"pods": "1"},
            used={"pods": "1"},
            scopes=["BestEffort"],
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        # Workload requesting GPU without CPU or memory maintains BestEffort QoS in k8s;
        # BestEffort pod quota must be evaluated and block the request.
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'best-effort-quota' exceeded" in err.error

    def test_check_resource_quota_best_effort_scope_skipped_with_limit_range_default(self) -> None:
        quota = _make_resource_quota(
            name="best-effort-quota",
            hard={"pods": "1"},
            used={"pods": "1"},
            scopes=["BestEffort"],
        )
        limit_range = SimpleNamespace(
            spec=SimpleNamespace(
                limits=[
                    SimpleNamespace(
                        default_request={"cpu": "500m", "memory": "512Mi"},
                    )
                ]
            )
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])
        mock_api.list_namespaced_limit_range.return_value = SimpleNamespace(items=[limit_range])

        # Even though workload only requested GPU, the namespace LimitRange injects default CPU/memory requests,
        # changing effective QoS to Burstable and skipping the BestEffort quota.
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is None

    def test_check_resource_quota_best_effort_scope_skipped_when_limit_range_forbidden(
        self,
    ) -> None:
        quota = _make_resource_quota(
            name="best-effort-quota",
            hard={"pods": "1"},
            used={"pods": "1"},
            scopes=["BestEffort"],
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])
        mock_api.list_namespaced_limit_range.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        # When LimitRange lookup is forbidden, pre-flight cannot know if admission mutates the pod,
        # so it skips the BestEffort quota to avoid false rejections.
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is None

    def test_check_resource_quota_defaulted_cpu_exceeds_quota(self) -> None:
        quota = _make_resource_quota(
            name="cpu-quota",
            hard={"requests.cpu": "1"},
            used={"requests.cpu": "800m"},
        )
        limit_range = SimpleNamespace(
            spec=SimpleNamespace(
                limits=[
                    SimpleNamespace(
                        default_request={"cpu": "500m"},
                    )
                ]
            )
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])
        mock_api.list_namespaced_limit_range.return_value = SimpleNamespace(items=[limit_range])

        # Workload specifies only GPU, but LimitRange default adds 500m CPU.
        # Remaining CPU is 200m (1 - 800m), so 500m exceeds available quota.
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'cpu-quota' exceeded" in err.error
        assert "CPU(s)" in err.error

    def test_check_resource_quota_defaulted_memory_exceeds_quota(self) -> None:
        quota = _make_resource_quota(
            name="mem-quota",
            hard={"requests.memory": "2Gi"},
            used={"requests.memory": "1.5Gi"},
        )
        limit_range = SimpleNamespace(
            spec=SimpleNamespace(
                limits=[
                    SimpleNamespace(
                        default_request={"memory": "1Gi"},
                    )
                ]
            )
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])
        mock_api.list_namespaced_limit_range.return_value = SimpleNamespace(items=[limit_range])

        # Workload specifies only GPU, but LimitRange default adds 1Gi memory.
        # Remaining memory is 0.5Gi, so 1Gi exceeds available quota.
        err = check_resource_quota(
            namespace="kubeflow-user",
            resources_per_node={"gpu": 1},
            num_nodes=1,
            core_v1_api=mock_api,
        )
        assert err is not None
        assert err.error_code == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'mem-quota' exceeded" in err.error
        assert "bytes memory" in err.error


class TestGetNamespaceResourceQuotas:
    def test_get_namespace_resource_quotas_success(self) -> None:
        quota = _make_resource_quota(
            name="cluster-quota",
            hard={"requests.cpu": "10", "requests.memory": "20Gi"},
            used={"requests.cpu": "4", "requests.memory": "10Gi"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        quotas = get_namespace_resource_quotas("kubeflow-dev", core_v1_api=mock_api)
        assert len(quotas) == 1
        entry = quotas[0]
        assert entry["name"] == "cluster-quota"
        assert entry["namespace"] == "kubeflow-dev"
        assert entry["hard"]["requests.cpu"] == "10"
        assert entry["used"]["requests.cpu"] == "4"
        assert entry["available"]["requests.cpu"] == "6"

    def test_get_namespace_resource_quotas_forbidden(self) -> None:
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.side_effect = ApiException(
            status=403, reason="Forbidden"
        )
        quotas = get_namespace_resource_quotas("restricted-ns", core_v1_api=mock_api)
        assert quotas == []


class TestTrainingSubmissionQuotaIntegration:
    def test_fine_tune_blocked_when_quota_exceeded(self) -> None:
        from kubeflow_mcp.trainer.api.training import fine_tune

        quota = _make_resource_quota(
            name="strict-quota",
            hard={"requests.cpu": "4"},
            used={"requests.cpu": "4"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        with (
            patch("kubeflow_mcp.trainer.api.quota.get_core_v1_api", return_value=mock_api),
            patch("kubeflow_mcp.trainer.api.training.check_namespace_allowed", return_value=None),
        ):
            res = fine_tune(
                model="hf://google/gemma-2b",
                dataset="hf://tatsu-lab/alpaca",
                runtime="torchtune-llama3.2-1b",
                namespace="default",
                resources_per_node={"cpu": "2"},
                num_nodes=1,
                confirmed=False,
            )

        assert res["success"] is False
        assert res["error_code"] == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'strict-quota' exceeded" in res["error"]

    def test_run_custom_training_blocked_when_quota_exceeded(self) -> None:
        from kubeflow_mcp.trainer.api.training import run_custom_training

        quota = _make_resource_quota(
            name="strict-quota",
            hard={"count/pods": "2"},
            used={"count/pods": "2"},
        )
        mock_api = MagicMock()
        mock_api.list_namespaced_resource_quota.return_value = SimpleNamespace(items=[quota])

        with (
            patch("kubeflow_mcp.trainer.api.quota.get_core_v1_api", return_value=mock_api),
            patch("kubeflow_mcp.trainer.api.training.check_namespace_allowed", return_value=None),
        ):
            res = run_custom_training(
                script="print('hello')",
                namespace="default",
                num_nodes=1,
                confirmed=False,
            )

        assert res["success"] is False
        assert res["error_code"] == ErrorCode.QUOTA_EXCEEDED
        assert "Resource quota 'strict-quota' exceeded" in res["error"]
