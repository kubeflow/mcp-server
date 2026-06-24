"""Tests for training tools: fine_tune, run_custom_training, run_container_training."""

from unittest.mock import MagicMock, patch

from kubeflow.trainer.options import Annotations, Labels, RuntimePatch
from kubeflow.trainer.types.types import (
    BuiltinTrainer,
    CustomTrainer,
    CustomTrainerContainer,
    DataType,
    Initializer,
    LoraConfig,
    S3DatasetInitializer,
    S3ModelInitializer,
    TorchTuneConfig,
)

from kubeflow_mcp.trainer.api.training import (
    _build_runtime_patch,
    _sdk_error,
    fine_tune,
    run_container_training,
    run_custom_training,
)

# ---------------------------------------------------------------------------
# _build_runtime_patch
# ---------------------------------------------------------------------------


class TestBuildRuntimePatch:
    def test_empty_returns_empty(self):
        assert _build_runtime_patch() == []

    def test_node_selector(self):
        result = _build_runtime_patch(node_selector={"gpu": "true"})
        assert len(result) == 1
        assert isinstance(result[0], RuntimePatch)

    def test_tolerations(self):
        result = _build_runtime_patch(
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
        )
        assert len(result) == 1

    def test_env(self):
        result = _build_runtime_patch(env=[{"name": "HF_TOKEN", "value": "secret"}])
        assert len(result) == 1

    def test_volumes_and_mounts(self):
        result = _build_runtime_patch(
            volumes=[{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}],
            volume_mounts=[{"name": "data", "mountPath": "/data"}],
        )
        assert len(result) == 1

    def test_affinity(self):
        affinity = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}}
        result = _build_runtime_patch(affinity=affinity)
        assert len(result) == 1
        pod_spec = (
            result[0]
            .training_runtime_spec.template.spec.replicated_jobs[0]
            .template.spec.template.spec
        )
        assert pod_spec.affinity == affinity

    def test_service_account(self):
        result = _build_runtime_patch(service_account_name="trainer-sa")
        pod_spec = (
            result[0]
            .training_runtime_spec.template.spec.replicated_jobs[0]
            .template.spec.template.spec
        )
        assert pod_spec.service_account_name == "trainer-sa"

    def test_image_pull_secrets(self):
        secrets = [{"name": "regcred"}]
        result = _build_runtime_patch(image_pull_secrets=secrets)
        pod_spec = (
            result[0]
            .training_runtime_spec.template.spec.replicated_jobs[0]
            .template.spec.template.spec
        )
        assert pod_spec.image_pull_secrets == secrets

    def test_labels_only(self):
        """Labels alone → Labels option, no RuntimePatch."""
        result = _build_runtime_patch(labels={"team": "ml"})
        assert len(result) == 1
        assert isinstance(result[0], Labels)
        assert result[0].labels == {"team": "ml"}

    def test_annotations_only(self):
        result = _build_runtime_patch(annotations={"note": "experiment-42"})
        assert len(result) == 1
        assert isinstance(result[0], Annotations)

    def test_labels_and_pod_patch_returns_both(self):
        """Pod patch + labels/annotations → RuntimePatch + Labels + Annotations."""
        result = _build_runtime_patch(
            node_selector={"gpu": "true"},
            labels={"team": "ml"},
            annotations={"run": "exp1"},
        )
        assert len(result) == 3
        types = {type(o) for o in result}
        assert RuntimePatch in types
        assert Labels in types
        assert Annotations in types

    def test_all_pod_options(self):
        result = _build_runtime_patch(
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            env=[{"name": "TOKEN", "value": "abc"}],
            volumes=[{"name": "data", "emptyDir": {}}],
            volume_mounts=[{"name": "data", "mountPath": "/tmp/data"}],
            affinity={"nodeAffinity": {}},
            service_account_name="sa",
            image_pull_secrets=[{"name": "reg"}],
        )
        assert len(result) == 1
        assert isinstance(result[0], RuntimePatch)

    def test_sdk_unavailable_returns_empty(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            assert _build_runtime_patch(node_selector={"gpu": "true"}) == []


# ---------------------------------------------------------------------------
# _sdk_error
# ---------------------------------------------------------------------------


class TestSdkError:
    def test_plain_exception(self):
        result = _sdk_error(Exception("connection refused"))
        assert result["success"] is False
        assert result["error"] == "connection refused"
        assert result["details"] is None

    def test_exception_with_cause(self):
        err = RuntimeError("wrapped")
        err.__cause__ = ValueError("root cause")
        result = _sdk_error(err)
        assert result["details"] == {"cause": "root cause"}

    def test_exception_with_response_attribute(self):
        err = Exception("api error")
        mock_resp = MagicMock()
        mock_resp.text = "404 Not Found"
        err.response = mock_resp
        result = _sdk_error(err)
        assert result["details"] == {"response": "404 Not Found"}

    def test_hint_propagated(self):
        result = _sdk_error(Exception("oops"), hint="check resource_planning")
        assert result["hint"] == "check resource_planning"


# ---------------------------------------------------------------------------
# fine_tune
# ---------------------------------------------------------------------------


class TestFineTune:
    # --- preview ---

    def test_preview_mode(self):
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["model"] == "hf://google/gemma-2b"
        assert result["config"]["dataset"] == "hf://tatsu-lab/alpaca"

    def test_preview_includes_all_params(self):
        result = fine_tune(
            model="hf://meta-llama/Llama-3.2-1B",
            dataset="hf://squad",
            batch_size=8,
            epochs=3,
            num_nodes=2,
            lora_rank=16,
            lora_alpha=32,
            confirmed=False,
        )
        cfg = result["config"]
        assert cfg["batch_size"] == 8
        assert cfg["epochs"] == 3
        assert cfg["num_nodes"] == 2
        assert cfg["lora_rank"] == 16
        assert cfg["lora_alpha"] == 32

    def test_preview_dtype_and_lora_advanced(self):
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            dtype="bf16",
            lora_dropout=0.05,
            use_dora=True,
            quantize_base=True,
            confirmed=False,
        )
        cfg = result["config"]
        assert cfg["dtype"] == "bf16"
        assert cfg["lora_dropout"] == 0.05
        assert cfg["use_dora"] is True
        assert cfg["quantize_base"] is True

    def test_preview_s3_sources_masks_credentials(self):
        result = fine_tune(
            model="s3://my-bucket/models/llama-3b",
            dataset="s3://my-bucket/datasets/alpaca",
            s3_endpoint="https://minio.internal:9000",
            s3_access_key_id="AKID",
            s3_secret_access_key="SECRET",
            confirmed=False,
        )
        cfg = result["config"]
        assert cfg["model"] == "s3://my-bucket/models/llama-3b"
        assert cfg["s3_endpoint"] == "https://minio.internal:9000"
        assert cfg.get("s3_access_key_id") == "***"
        assert cfg.get("s3_secret_access_key") == "***"

    def test_preview_with_runtime_patches(self):
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            node_selector={"node-type": "gpu"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
            confirmed=False,
        )
        cfg = result["config"]
        assert cfg["node_selector"] == {"node-type": "gpu"}
        assert cfg["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists"}]

    def test_preview_masks_hf_token(self):
        result = fine_tune(
            model="hf://meta-llama/Llama-3.2-1B",
            dataset="hf://squad",
            hf_token="hf_secret_token_12345",
            confirmed=False,
        )
        assert result["config"]["hf_token"] == "***"

    def test_invalid_dtype_rejected(self):
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            dtype="fp16",  # not a valid DataType value
            confirmed=False,
        )
        assert result["success"] is False
        assert "fp16" in result["error"]

    # --- SDK not available ---

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False)
    def test_sdk_not_available(self):
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            confirmed=True,
        )
        assert result["success"] is False
        assert "Required training libraries not available" in result["error"]

    # --- submit: real SDK types ---

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_builtin_trainer(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.return_value = "trainjob-abc123"
        mock_get_client.return_value = mock_client

        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-tune",
            batch_size=8,
            lora_rank=16,
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["job_name"] == "trainjob-abc123"
        assert result["data"]["status"] == "Created"

        kw = mock_client.train.call_args.kwargs
        assert isinstance(kw["trainer"], BuiltinTrainer)
        assert isinstance(kw["initializer"], Initializer)
        assert kw["initializer"].model.storage_uri == "hf://google/gemma-2b"
        assert kw["initializer"].dataset.storage_uri == "hf://tatsu-lab/alpaca"
        assert kw["trainer"].config.batch_size == 8
        assert kw["trainer"].config.peft_config.lora_rank == 16

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_builtin_trainer_advanced_lora(self, mock_get_client):
        """LoraConfig advanced fields reach the SDK."""
        mock_client = MagicMock()
        mock_client.train.return_value = "trainjob-lora"
        mock_get_client.return_value = mock_client

        fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-tune",
            lora_dropout=0.05,
            use_dora=True,
            quantize_base=True,
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, BuiltinTrainer)
        lora: LoraConfig = trainer.config.peft_config
        assert lora.lora_dropout == 0.05
        assert lora.use_dora is True
        assert lora.quantize_base is True

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_builtin_trainer_dtype(self, mock_get_client):
        """dtype='bf16' produces TorchTuneConfig.dtype == DataType.BF16."""
        mock_client = MagicMock()
        mock_client.train.return_value = "trainjob-dtype"
        mock_get_client.return_value = mock_client

        fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-tune",
            dtype="bf16",
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, BuiltinTrainer)
        assert isinstance(trainer.config, TorchTuneConfig)
        assert trainer.config.dtype == DataType.BF16

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_s3_initializers(self, mock_get_client):
        """s3:// URIs produce S3ModelInitializer / S3DatasetInitializer."""
        mock_client = MagicMock()
        mock_client.train.return_value = "trainjob-s3"
        mock_get_client.return_value = mock_client

        fine_tune(
            model="s3://bucket/models/gemma",
            dataset="s3://bucket/datasets/alpaca",
            runtime="torch-tune",
            s3_endpoint="https://minio:9000",
            s3_access_key_id="AKID",
            s3_secret_access_key="SECRET",
            s3_region="us-east-1",
            confirmed=True,
        )

        initializer = mock_client.train.call_args.kwargs["initializer"]
        assert isinstance(initializer.model, S3ModelInitializer)
        assert initializer.model.storage_uri == "s3://bucket/models/gemma"
        assert initializer.model.endpoint == "https://minio:9000"
        assert initializer.model.access_key_id == "AKID"
        assert initializer.model.region == "us-east-1"
        assert isinstance(initializer.dataset, S3DatasetInitializer)
        assert initializer.dataset.storage_uri == "s3://bucket/datasets/alpaca"

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_labels_and_annotations_in_options(self, mock_get_client):
        """labels/annotations appear in the options list."""
        mock_client = MagicMock()
        mock_client.train.return_value = "trainjob-meta"
        mock_get_client.return_value = mock_client

        fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-tune",
            labels={"team": "ml", "env": "prod"},
            annotations={"owner": "alice"},
            confirmed=True,
        )

        opts = mock_client.train.call_args.kwargs["options"]
        assert opts is not None
        opt_types = {type(o) for o in opts}
        assert Labels in opt_types
        assert Annotations in opt_types
        labels_opt = next(o for o in opts if isinstance(o, Labels))
        assert "team" in labels_opt.labels
        assert "env" in labels_opt.labels
        assert labels_opt.labels["team"] == "ml"
        assert labels_opt.labels["env"] == "prod"

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    def test_non_torchtune_runtime_rejected(self):
        """Non-torchtune runtimes are rejected with a validation error."""
        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-distributed",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.side_effect = Exception("Quota exceeded")
        mock_get_client.return_value = mock_client

        result = fine_tune(
            model="hf://google/gemma-2b",
            dataset="hf://tatsu-lab/alpaca",
            runtime="torch-tune",
            confirmed=True,
        )

        assert result["success"] is False
        assert "Quota exceeded" in result["error"]


# ---------------------------------------------------------------------------
# run_custom_training
# ---------------------------------------------------------------------------


class TestRunCustomTraining:
    # --- preview ---

    def test_preview_mode(self):
        result = run_custom_training(
            script="print('hello world')",
            num_nodes=2,
            gpu_per_node=4,
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["num_nodes"] == 2
        assert result["config"]["gpu_per_node"] == 4

    def test_script_truncated_in_preview(self):
        result = run_custom_training(script="x = 1\n" * 100, confirmed=False)
        assert len(result["config"]["script"]) <= 203  # 200 + "..."

    def test_preview_with_node_selector(self):
        result = run_custom_training(
            script="print('hello')",
            node_selector={"gpu-type": "a100"},
            confirmed=False,
        )
        assert result["config"]["node_selector"] == {"gpu-type": "a100"}

    def test_preview_with_tolerations(self):
        tolerations = [{"key": "nvidia.com/gpu", "operator": "Exists"}]
        result = run_custom_training(
            script="print('hello')", tolerations=tolerations, confirmed=False
        )
        assert result["config"]["tolerations"] == tolerations

    def test_preview_env_is_dict(self):
        """env is dict[str, str] — goes to CustomTrainer, not pod-level patch."""
        env = {"MY_VAR": "123", "SEED": "42"}
        result = run_custom_training(script="print('hi')", env=env, confirmed=False)
        assert result["config"]["env"] == env

    def test_preview_image_and_pip_index_urls(self):
        result = run_custom_training(
            script="print('hi')",
            image="myrepo/trainer:v2",
            pip_index_urls=["https://pypi.internal.corp/simple"],
            confirmed=False,
        )
        assert result["config"]["image"] == "myrepo/trainer:v2"
        assert result["config"]["pip_index_urls"] == ["https://pypi.internal.corp/simple"]

    def test_preview_custom_resources_per_node(self):
        result = run_custom_training(
            script="print('hi')",
            resources_per_node={"cpu": "8", "memory": "32Gi"},
            confirmed=False,
        )
        assert result["config"]["resources_per_node"] == {"cpu": "8", "memory": "32Gi"}

    def test_preview_gpu_per_node_default_resources(self):
        result = run_custom_training(script="print('hi')", gpu_per_node=2, confirmed=False)
        assert result["config"]["resources_per_node"] == {"gpu": 2}

    def test_preview_cpu_only_no_resources(self):
        result = run_custom_training(script="print('hi')", gpu_per_node=0, confirmed=False)
        assert result["config"]["resources_per_node"] is None

    def test_accepts_valid_job_name(self):
        result = run_custom_training(
            script="print('hello')", name="my-training-job", confirmed=False
        )
        assert result["status"] == "preview"
        assert result["config"]["name"] == "my-training-job"

    # --- script safety advisory (preview warnings, not submission blockers) ---

    def test_preview_warns_on_dangerous_os_system(self):
        result = run_custom_training(script="import os\nos.system('rm -rf /')", confirmed=False)
        assert result["status"] == "preview"
        assert len(result["config"]["safety_warnings"]) > 0

    def test_preview_warns_on_subprocess(self):
        result = run_custom_training(
            script="import subprocess\nsubprocess.run(['ls'])", confirmed=False
        )
        assert result["status"] == "preview"
        assert len(result["config"]["safety_warnings"]) > 0

    def test_preview_warns_on_eval(self):
        result = run_custom_training(script="eval('print(1)')", confirmed=False)
        assert len(result["config"]["safety_warnings"]) > 0

    def test_preview_warns_on_exec(self):
        result = run_custom_training(script="exec('x = 1')", confirmed=False)
        assert len(result["config"]["safety_warnings"]) > 0

    def test_safe_script_no_warnings(self):
        script = """
import torch
import torch.distributed as dist

def train():
    print("Training...")
    model = torch.nn.Linear(10, 10)
    return model

train()
"""
        result = run_custom_training(script=script, confirmed=False)
        assert result["status"] == "preview"
        assert result["config"]["safety_warnings"] == []

    def test_rejects_invalid_job_name(self):
        result = run_custom_training(script="print('hello')", name="INVALID_NAME", confirmed=True)
        assert result["success"] is False
        assert "validation" in result["error"].lower() or "name" in result["error"].lower()

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False)
    def test_sdk_not_available(self):
        result = run_custom_training(script="print('hello')", confirmed=True)
        assert result["success"] is False
        assert "Required training libraries not available" in result["error"]

    # --- submit: real SDK types ---

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-job-123"
        mock_get_client.return_value = mock_client

        result = run_custom_training(
            script="print('training')",
            packages=["torch", "transformers"],
            num_nodes=2,
            gpu_per_node=4,
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["job_name"] == "custom-job-123"
        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainer)
        assert trainer.packages_to_install == ["torch", "transformers"]
        assert trainer.num_nodes == 2
        assert trainer.resources_per_node == {"gpu": 4}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_env_on_trainer_not_patch(self, mock_get_client):
        """env dict must be on CustomTrainer.env, not injected via RuntimePatch."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-env"
        mock_get_client.return_value = mock_client

        run_custom_training(script="print('hi')", env={"TOKEN": "abc"}, confirmed=True)

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainer)
        assert trainer.env == {"TOKEN": "abc"}
        # env goes on CustomTrainer, not as node_selector/tolerations patches
        assert trainer.env == {"TOKEN": "abc"}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_image_and_pip_urls_on_trainer(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.return_value = "job-img"
        mock_get_client.return_value = mock_client

        run_custom_training(
            script="print('hi')",
            image="myrepo/base:cuda12",
            pip_index_urls=["https://nexus.corp/pypi/simple"],
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainer)
        assert trainer.image == "myrepo/base:cuda12"
        assert trainer.pip_index_urls == ["https://nexus.corp/pypi/simple"]

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_resources_per_node_override(self, mock_get_client):
        """Explicit resources_per_node overrides gpu_per_node logic."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-res"
        mock_get_client.return_value = mock_client

        run_custom_training(
            script="print('hi')",
            gpu_per_node=4,
            resources_per_node={"cpu": "16", "memory": "64Gi"},
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert trainer.resources_per_node == {"cpu": "16", "memory": "64Gi"}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_node_selector_goes_to_patch(self, mock_get_client):
        """node_selector must appear in RuntimePatch options."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-pe"
        mock_get_client.return_value = mock_client

        run_custom_training(
            script="print('hi')",
            node_selector={"gpu-type": "a100"},
            confirmed=True,
        )

        opts = mock_client.train.call_args.kwargs.get("options") or []
        assert opts is not None and len(opts) > 0

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_train_func_isolated_namespace(self, mock_get_client):
        """train_func closure must exec script in an isolated namespace."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-xyz"
        mock_get_client.return_value = mock_client

        run_custom_training(script="isolated_sentinel = 'was_executed'", confirmed=True)

        trainer = mock_client.train.call_args.kwargs["trainer"]
        trainer.func()  # call the closure

        import kubeflow_mcp.trainer.api.training as _mod

        assert not hasattr(_mod, "isolated_sentinel"), "Script leaked into module namespace"
        assert "isolated_sentinel" not in globals(), "Script leaked into test globals"

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_env_based_iteration_pattern(self, mock_get_client):
        """Agents can keep a fixed script and iterate by changing env only."""
        script = "import os; lr = float(os.environ['LR']); print(f'LR={lr}')"

        mock_client = MagicMock()
        mock_client.train.return_value = "iter-job-1"
        mock_get_client.return_value = mock_client

        r1 = run_custom_training(script=script, env={"LR": "1e-4"}, confirmed=True)
        assert r1["success"] is True

        mock_client.train.return_value = "iter-job-2"
        r2 = run_custom_training(script=script, env={"LR": "5e-5"}, confirmed=True)
        assert r2["success"] is True
        assert r2["data"]["job_name"] == "iter-job-2"

        t1 = mock_client.train.call_args_list[0].kwargs["trainer"]
        t2 = mock_client.train.call_args_list[1].kwargs["trainer"]
        assert t1.env == {"LR": "1e-4"}
        assert t2.env == {"LR": "5e-5"}


# ---------------------------------------------------------------------------
# run_container_training
# ---------------------------------------------------------------------------


class TestRunContainerTraining:
    # --- preview ---

    def test_preview_mode(self):
        result = run_container_training(
            image="pytorch/pytorch:2.0-cuda11.8",
            num_nodes=2,
            gpu_per_node=4,
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["image"] == "pytorch/pytorch:2.0-cuda11.8"
        assert result["config"]["num_nodes"] == 2
        assert result["config"]["gpu_per_node"] == 4

    def test_preview_with_env(self):
        result = run_container_training(
            image="myorg/trainer:v1",
            env={"BATCH_SIZE": "32", "LR": "0.001"},
            confirmed=False,
        )
        assert result["config"]["env"] == {"BATCH_SIZE": "32", "LR": "0.001"}

    def test_preview_with_volumes(self):
        volumes = [{"name": "data", "persistentVolumeClaim": {"claimName": "my-pvc"}}]
        volume_mounts = [{"name": "data", "mountPath": "/data"}]
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            volumes=volumes,
            volume_mounts=volume_mounts,
            confirmed=False,
        )
        assert result["config"]["volumes"] == volumes
        assert result["config"]["volume_mounts"] == volume_mounts

    def test_preview_with_node_selector(self):
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            node_selector={"nvidia.com/gpu.product": "A100"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            confirmed=False,
        )
        assert result["config"]["node_selector"] == {"nvidia.com/gpu.product": "A100"}
        assert result["config"]["tolerations"] == [{"key": "gpu", "operator": "Exists"}]

    def test_preview_custom_resources_per_node(self):
        result = run_container_training(
            image="myrepo/trainer:v1",
            resources_per_node={"cpu": "8", "memory": "32Gi"},
            confirmed=False,
        )
        assert result["config"]["resources_per_node"] == {"cpu": "8", "memory": "32Gi"}

    def test_preview_cpu_only(self):
        result = run_container_training(image="python:3.10", gpu_per_node=0, confirmed=False)
        assert result["config"]["gpu_per_node"] == 0
        assert result["config"]["resources_per_node"] is None

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False)
    def test_sdk_not_available(self):
        result = run_container_training(image="pytorch/pytorch:2.0", confirmed=True)
        assert result["success"] is False
        assert "Required training libraries not available" in result["error"]

    # --- submit: real SDK types ---

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.return_value = "container-job-456"
        mock_get_client.return_value = mock_client

        result = run_container_training(
            image="ghcr.io/myorg/trainer:v1",
            num_nodes=2,
            gpu_per_node=4,
            env={"EPOCHS": "5"},
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["job_name"] == "container-job-456"
        assert result["data"]["status"] == "Created"

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainerContainer)
        assert trainer.image == "ghcr.io/myorg/trainer:v1"
        assert trainer.num_nodes == 2
        assert trainer.resources_per_node == {"gpu": 4}
        assert trainer.env == {"EPOCHS": "5"}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training._build_runtime_patch")
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_env_not_in_runtime_patch(self, mock_get_client, mock_patch):
        """env must reach CustomTrainerContainer.env, NOT _build_runtime_patch."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-abc"
        mock_get_client.return_value = mock_client
        mock_patch.return_value = []

        run_container_training(
            image="pytorch/pytorch:2.0",
            env={"HF_TOKEN": "secret", "BATCH": "32"},
            node_selector={"gpu": "true"},
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainerContainer)
        assert trainer.env == {"HF_TOKEN": "secret", "BATCH": "32"}

        patch_kwargs = mock_patch.call_args.kwargs
        assert "env" not in patch_kwargs
        assert patch_kwargs.get("node_selector") == {"gpu": "true"}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_custom_resources_per_node(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.return_value = "job-res"
        mock_get_client.return_value = mock_client

        run_container_training(
            image="myrepo/trainer:v1",
            resources_per_node={"cpu": "8", "memory": "32Gi"},
            confirmed=True,
        )

        trainer = mock_client.train.call_args.kwargs["trainer"]
        assert isinstance(trainer, CustomTrainerContainer)
        assert trainer.resources_per_node == {"cpu": "8", "memory": "32Gi"}

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_affinity_in_patch(self, mock_get_client):
        """affinity must appear in the RuntimePatch pod spec."""
        mock_client = MagicMock()
        mock_client.train.return_value = "job-aff"
        mock_get_client.return_value = mock_client

        affinity = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}}
        run_container_training(image="myrepo/trainer:v1", affinity=affinity, confirmed=True)

        opts = mock_client.train.call_args.kwargs["options"]
        assert opts is not None
        patch = opts[0]
        assert isinstance(patch, RuntimePatch)
        pod_spec = patch.training_runtime_spec.template.spec.replicated_jobs[
            0
        ].template.spec.template.spec
        assert pod_spec.affinity == affinity

    @patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", True)
    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_submit_handles_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.train.side_effect = Exception("Image pull failed")
        mock_get_client.return_value = mock_client

        result = run_container_training(image="invalid/image:notfound", confirmed=True)

        assert result["success"] is False
        assert "Image pull failed" in result["error"]
