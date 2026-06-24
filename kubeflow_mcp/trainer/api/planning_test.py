"""Tests for planning tools: get_cluster_resources, estimate_resources."""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.planning import (
    _estimate_from_params,
    _get_model_info_from_hf,
    estimate_resources,
    get_cluster_resources,
)


class TestGetClusterResources:
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_success(self, mock_get_v1):
        mock_node = MagicMock()
        mock_node.metadata.name = "node-1"
        mock_node.status.allocatable = {
            "nvidia.com/gpu": "2",
            "memory": "64Gi",
            "cpu": "16",
        }

        mock_v1 = MagicMock()
        mock_v1.list_node.return_value.items = [mock_node]
        mock_get_v1.return_value = mock_v1

        result = get_cluster_resources()

        assert result["success"] is True
        assert result["data"]["gpu_total"] == 2
        assert result["data"]["nodes_with_gpu"] == 1
        assert result["data"]["node_count"] == 1

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_no_gpus(self, mock_get_v1):
        mock_node = MagicMock()
        mock_node.metadata.name = "cpu-node"
        mock_node.status.allocatable = {"memory": "32Gi", "cpu": "8"}

        mock_v1 = MagicMock()
        mock_v1.list_node.return_value.items = [mock_node]
        mock_get_v1.return_value = mock_v1

        result = get_cluster_resources()

        assert result["success"] is True
        assert result["data"]["gpu_total"] == 0
        assert result["data"]["nodes_with_gpu"] == 0

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_multiple_nodes(self, mock_get_v1):
        mock_nodes = []
        for i, gpus in enumerate([4, 2, 0]):
            node = MagicMock()
            node.metadata.name = f"node-{i}"
            node.status.allocatable = {
                "nvidia.com/gpu": str(gpus),
                "memory": "64Gi",
                "cpu": "16",
            }
            mock_nodes.append(node)

        mock_v1 = MagicMock()
        mock_v1.list_node.return_value.items = mock_nodes
        mock_get_v1.return_value = mock_v1

        result = get_cluster_resources()

        assert result["success"] is True
        assert result["data"]["gpu_total"] == 6
        assert result["data"]["nodes_with_gpu"] == 2
        assert result["data"]["node_count"] == 3

    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_k8s_unavailable(self, mock_get_v1):
        mock_get_v1.side_effect = Exception("No cluster")

        result = get_cluster_resources()

        assert result["success"] is False
        assert "No cluster" in result["error"]
        assert result["error_code"] == "KUBERNETES_ERROR"


class TestEstimateResources:
    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_small_model(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
            "library": "transformers",
        }

        result = estimate_resources(model="google/gemma-2b")

        assert result["success"] is True
        assert result["data"]["model"] == "google/gemma-2b"
        assert result["data"]["params_billions"] == 2.0
        assert result["data"]["gpu_per_worker"] == 1
        assert (
            "8GB" in result["data"]["gpu_type_recommended"]
            or "16GB" in result["data"]["gpu_type_recommended"]
        )

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_large_model(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "meta-llama/Llama-3-70B",
            "params": 70_000_000_000,
        }

        result = estimate_resources(model="meta-llama/Llama-3-70B")

        assert result["success"] is True
        assert result["data"]["params_billions"] == 70.0
        assert result["data"]["total_gpu"] >= 2 or "80GB" in result["data"]["gpu_type_recommended"]

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_hf_prefix_stripped(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
        }

        estimate_resources(model="hf://google/gemma-2b")

        mock_hf_info.assert_called_once()

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_with_multiple_workers(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
        }

        result = estimate_resources(model="google/gemma-2b", num_workers=4)

        assert result["success"] is True
        assert result["data"]["num_workers"] == 4
        assert result["data"]["total_gpu"] == result["data"]["gpu_per_worker"] * 4

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_batch_size_increases_memory(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
        }

        small = estimate_resources(model="google/gemma-2b", batch_size=1)
        large = estimate_resources(model="google/gemma-2b", batch_size=16)

        small_mem = int(small["data"]["gpu_memory_required"].replace("GB", ""))
        large_mem = int(large["data"]["gpu_memory_required"].replace("GB", ""))
        assert large_mem >= small_mem

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_hf_api_error(self, mock_hf_info):
        mock_hf_info.return_value = {"error": "Model not found"}

        result = estimate_resources(model="nonexistent/model")

        assert result["success"] is False
        assert "HuggingFace" in result["error"]

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_no_param_count(self, mock_hf_info):
        mock_hf_info.return_value = {"model_id": "some/model", "params": None}

        result = estimate_resources(model="some/model")

        assert result["success"] is False
        assert "parameter count" in result["error"].lower()

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_includes_recommendation(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
        }

        result = estimate_resources(model="google/gemma-2b")

        assert result["success"] is True
        assert "recommendation" in result["data"]
        assert "GPU" in result["data"]["recommendation"]

    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    def test_training_type_lora(self, mock_hf_info):
        mock_hf_info.return_value = {
            "model_id": "google/gemma-2b",
            "params": 2_000_000_000,
        }

        result = estimate_resources(model="google/gemma-2b")

        assert result["data"]["training_type"] == "LoRA (bf16)"


class TestGetModelInfoFromHF:
    def test_success(self):
        mock_info = MagicMock()
        mock_info.id = "meta-llama/Llama-2-7b"
        mock_info.safetensors = MagicMock(total=7_000_000_000)
        mock_info.card_data = None
        mock_info.library_name = "transformers"
        mock_info.pipeline_tag = "text-generation"

        with patch("huggingface_hub.model_info", return_value=mock_info):
            result = _get_model_info_from_hf("meta-llama/Llama-2-7b")

        assert result["model_id"] == "meta-llama/Llama-2-7b"
        assert result["params"] == 7_000_000_000
        assert result["library"] == "transformers"

    def test_uses_card_data_when_no_safetensors(self):
        mock_info = MagicMock()
        mock_info.id = "some/model"
        mock_info.safetensors = None
        mock_info.card_data = MagicMock(num_parameters=3_000_000_000)
        mock_info.library_name = None
        mock_info.pipeline_tag = None

        with patch("huggingface_hub.model_info", return_value=mock_info):
            result = _get_model_info_from_hf("some/model")

        assert result["params"] == 3_000_000_000

    def test_api_error(self):
        with patch("huggingface_hub.model_info", side_effect=Exception("rate limited")):
            result = _get_model_info_from_hf("some/model")

        assert "error" in result


class TestEstimateFromParams:
    def test_1b_model(self):
        result = _estimate_from_params(1e9)
        assert result["gpu_count"] == 1
        assert result["gpu_memory_gb"] <= 16

    def test_7b_model(self):
        result = _estimate_from_params(7e9)
        assert result["gpu_count"] == 1
        assert result["gpu_memory_gb"] > 4

    def test_13b_model(self):
        result = _estimate_from_params(13e9)
        assert result["gpu_count"] == 1

    def test_70b_model(self):
        result = _estimate_from_params(70e9)
        assert result["gpu_count"] >= 1

    def test_batch_size_increases_memory(self):
        result_bs1 = _estimate_from_params(7e9, batch_size=1)
        result_bs8 = _estimate_from_params(7e9, batch_size=8)
        assert result_bs8["gpu_memory_gb"] >= result_bs1["gpu_memory_gb"]

    def test_quantization_int4_reduces_memory(self):
        bf16 = _estimate_from_params(7e9, quantization="bf16")
        int4 = _estimate_from_params(7e9, quantization="int4")
        assert int4["gpu_memory_gb"] < bf16["gpu_memory_gb"]
        assert int4["quantization"] == "int4"

    def test_quantization_fp32_increases_memory(self):
        bf16 = _estimate_from_params(7e9, quantization="bf16")
        fp32 = _estimate_from_params(7e9, quantization="fp32")
        assert fp32["gpu_memory_gb"] > bf16["gpu_memory_gb"]

    def test_breakdown_present(self):
        result = _estimate_from_params(7e9)
        assert "breakdown" in result
        bd = result["breakdown"]
        assert "weights_gb" in bd
        assert "lora_adapters_gb" in bd
        assert "activations_gb" in bd
        assert "overhead_gb" in bd
        assert bd["weights_gb"] > 0
        assert bd["lora_adapters_gb"] > 0


# ─── check_compatibility ──────────────────────────────────────────────────────


class TestCheckCompatibility:
    @patch("kubeflow_mcp.common.utils.get_version_api")
    @patch("kubeflow_mcp.common.utils.get_custom_objects_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_compatible_environment(self, mock_core, mock_custom, mock_version):
        from kubeflow_mcp.trainer.api.planning import check_compatibility

        mock_version_api = MagicMock()
        mock_version_api.get_code.return_value.git_version = "v1.28.0"
        mock_version.return_value = mock_version_api

        mock_custom_api = MagicMock()
        mock_crd = MagicMock()
        mock_crd.metadata.name = "trainjobs.trainer.kubeflow.org"
        mock_ver = MagicMock()
        mock_ver.name = "v1alpha1"
        mock_ver.served = True
        mock_crd.spec.versions = [mock_ver]
        mock_custom_api.list_custom_resource_definition.return_value.items = [mock_crd]
        mock_custom.return_value = mock_custom_api

        mock_core_api = MagicMock()
        mock_core_api.list_node.return_value.items = []
        mock_core.return_value = mock_core_api

        result = check_compatibility()

        assert result["success"] is True
        assert "compatible" in result["data"]
        assert "checks" in result["data"]
        assert "platform" in result["data"]

    @patch("kubeflow_mcp.common.utils.get_version_api")
    @patch("kubeflow_mcp.common.utils.get_custom_objects_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_returns_blockers_when_crd_missing(self, mock_core, mock_custom, mock_version):
        from kubeflow_mcp.trainer.api.planning import check_compatibility

        mock_version_api = MagicMock()
        mock_version_api.get_code.return_value.git_version = "v1.28.0"
        mock_version.return_value = mock_version_api

        mock_custom_api = MagicMock()
        mock_custom_api.list_custom_resource_definition.return_value.items = []
        mock_custom.return_value = mock_custom_api

        mock_core_api = MagicMock()
        mock_core_api.list_node.return_value.items = []
        mock_core.return_value = mock_core_api

        result = check_compatibility()

        assert result["success"] is True
        assert result["data"]["compatible"] is False
        assert len(result["data"]["blockers"]) > 0

    @patch("kubeflow_mcp.common.utils.get_version_api")
    @patch("kubeflow_mcp.common.utils.get_custom_objects_api")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_k8s_api_error_recorded_as_blocker(self, mock_core, mock_custom, mock_version):
        """K8s version check failures are recorded as blockers, not exceptions."""
        from kubeflow_mcp.trainer.api.planning import check_compatibility

        mock_version.side_effect = Exception("connection refused")
        mock_custom.return_value.list_custom_resource_definition.return_value.items = []
        mock_core.return_value.list_node.return_value.items = []

        result = check_compatibility()

        assert result["success"] is True
        assert result["data"]["compatible"] is False
        assert len(result["data"]["blockers"]) > 0


# ─── pre_flight ───────────────────────────────────────────────────────────────


class TestPreFlight:
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    def test_returns_all_sections(self, mock_runtimes, mock_cluster, mock_compat):
        from kubeflow_mcp.trainer.api.planning import pre_flight

        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes", "checks": {}},
        }
        mock_cluster.return_value = {
            "success": True,
            "data": {"gpu_total": 4, "nodes_with_gpu": 2, "node_count": 2, "nodes": []},
        }
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune-llama3"}], "count": 1},
        }

        result = pre_flight()

        assert result["success"] is True
        data = result["data"]
        assert "compatibility" in data
        assert "cluster" in data
        assert "runtimes" in data
        assert "next_steps" in data

    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_stops_early_on_blockers(self, mock_compat):
        from kubeflow_mcp.trainer.api.planning import pre_flight

        mock_compat.return_value = {
            "success": True,
            "data": {
                "compatible": False,
                "blockers": ["Trainer CRD not installed"],
                "platform": "kubernetes",
                "checks": {},
            },
        }

        result = pre_flight()

        assert result["success"] is True
        data = result["data"]
        assert "cluster" not in data
        assert any("Blocker" in s for s in data["next_steps"])

    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.estimate_resources")
    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    def test_includes_estimate_when_model_given(
        self, mock_runtimes, mock_estimate, mock_cluster, mock_compat
    ):
        from kubeflow_mcp.trainer.api.planning import pre_flight

        mock_compat.return_value = {
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes", "checks": {}}
        }
        mock_cluster.return_value = {"data": {"gpu_total": 2, "nodes_with_gpu": 1, "nodes": []}}
        mock_estimate.return_value = {
            "data": {"params_billions": 1.0, "gpu_memory_gb": 3.0, "breakdown": {}}
        }
        mock_runtimes.return_value = {"data": {"runtimes": [], "count": 0}}

        result = pre_flight(model="meta-llama/Llama-3.2-1B")

        assert result["success"] is True
        assert "estimate" in result["data"]
        mock_estimate.assert_called_once()
