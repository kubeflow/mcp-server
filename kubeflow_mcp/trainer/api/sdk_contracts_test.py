"""SDK Contract Tests - Validate MCP tools against actual Kubeflow SDK APIs.

These tests ensure our MCP tool parameters and logic are compatible with
the actual Kubeflow SDK without requiring a cluster connection.

Tests validate:
- Verify SDK type instantiation (LoraConfig, TorchTuneConfig, etc.)
- Validate API signatures haven't changed
- Test MCP-to-SDK parameter conversions
- Ensure dataclass fields match our usage
"""

import builtins
import inspect
import textwrap
from dataclasses import fields
from typing import get_type_hints

import pytest
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.types import types as sdk_types


class TestSDKTypeInstantiation:
    """Test that SDK types can be instantiated with MCP-provided parameters."""

    def test_lora_config_defaults(self):
        """Test LoraConfig instantiation with default values."""
        config = sdk_types.LoraConfig()

        assert config.lora_rank is None
        assert config.lora_alpha is None
        assert config.lora_dropout is None
        assert config.quantize_base is None
        assert config.use_dora is None
        assert config.apply_lora_to_mlp is None
        assert config.apply_lora_to_output is None
        assert config.lora_attn_modules == ["q_proj", "v_proj", "output_proj"]

    def test_lora_config_with_mcp_params(self):
        """Test LoraConfig with parameters our MCP tools provide."""
        config = sdk_types.LoraConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            quantize_base=True,
            use_dora=False,
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_attn_modules=["q_proj", "k_proj", "v_proj"],
        )

        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.quantize_base is True
        assert config.use_dora is False
        assert config.apply_lora_to_mlp is True

    def test_torchtune_config_with_mcp_params(self):
        """Test TorchTuneConfig with parameters our fine_tune() provides."""
        lora = sdk_types.LoraConfig(lora_rank=8, lora_alpha=16)

        config = sdk_types.TorchTuneConfig(
            dtype=sdk_types.DataType.BF16,
            batch_size=4,
            epochs=3,
            num_nodes=2,
            peft_config=lora,
            resources_per_node={"gpu": 1, "cpu": 4, "memory": "16Gi"},
        )

        assert config.batch_size == 4
        assert config.epochs == 3
        assert config.num_nodes == 2
        assert config.dtype == sdk_types.DataType.BF16
        assert config.peft_config.lora_rank == 8

    def test_builtin_trainer_with_torchtune_config(self):
        """Test BuiltinTrainer instantiation as used by fine_tune()."""
        config = sdk_types.TorchTuneConfig(
            batch_size=4,
            epochs=2,
            peft_config=sdk_types.LoraConfig(lora_rank=8),
        )
        trainer = sdk_types.BuiltinTrainer(config=config)

        assert trainer.config.batch_size == 4
        assert trainer.config.peft_config.lora_rank == 8

    def test_custom_trainer_with_callable(self):
        """Test CustomTrainer with a callable as used by run_custom_training()."""

        def train_func():
            print("training")

        trainer = sdk_types.CustomTrainer(
            func=train_func,
            func_args={"lr": 0.001},
            num_nodes=2,
            resources_per_node={"gpu": 1},
            packages_to_install=["torch", "transformers"],
            env={"DEBUG": "1"},
        )

        assert trainer.func == train_func
        assert trainer.num_nodes == 2
        assert trainer.packages_to_install == ["torch", "transformers"]
        assert trainer.env == {"DEBUG": "1"}

    def test_custom_trainer_container(self):
        """Test CustomTrainerContainer as used by run_container_training()."""
        trainer = sdk_types.CustomTrainerContainer(
            image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
            num_nodes=2,
            resources_per_node={"gpu": 2, "cpu": 8, "memory": "32Gi"},
            env={"LEARNING_RATE": "0.001"},
        )

        assert trainer.image == "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
        assert trainer.num_nodes == 2
        assert trainer.env == {"LEARNING_RATE": "0.001"}

    def test_torchtune_config_with_loss_and_dataset_preprocess(self):
        """Test TorchTuneConfig with loss and dataset preprocessing."""
        ds_config = sdk_types.TorchTuneInstructDataset(
            source=sdk_types.DataFormat.JSON,
            split="train",
            train_on_input=False,
            new_system_prompt="You are a helpful assistant.",
            column_map={"input": "question", "output": "answer"},
        )

        config = sdk_types.TorchTuneConfig(
            batch_size=4,
            epochs=1,
            loss=sdk_types.Loss.CEWithChunkedOutputLoss,
            dataset_preprocess_config=ds_config,
            resources_per_node={"gpu": 1},
        )

        assert config.loss == sdk_types.Loss.CEWithChunkedOutputLoss
        assert config.dataset_preprocess_config.source == sdk_types.DataFormat.JSON
        assert config.dataset_preprocess_config.split == "train"
        assert config.resources_per_node == {"gpu": 1}

    def test_lora_config_with_advanced_targeting(self):
        """Test LoraConfig with apply_lora_to_mlp/output and custom attn modules."""
        config = sdk_types.LoraConfig(
            lora_rank=16,
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_attn_modules=["q_proj", "k_proj", "v_proj"],
        )

        assert config.apply_lora_to_mlp is True
        assert config.apply_lora_to_output is False
        assert config.lora_attn_modules == ["q_proj", "k_proj", "v_proj"]

    def test_hf_model_initializer_with_ignore_patterns(self):
        """Test HuggingFaceModelInitializer with ignore_patterns."""
        init = sdk_types.HuggingFaceModelInitializer(
            storage_uri="hf://meta-llama/Llama-3.2-1B",
            ignore_patterns=["*.bin", "*.h5"],
            access_token="hf_token123",
        )

        assert init.ignore_patterns == ["*.bin", "*.h5"]

    def test_custom_trainer_with_func_args(self):
        """Test CustomTrainer with func_args."""
        trainer = sdk_types.CustomTrainer(
            func=lambda lr=None, epochs=None: None,
            func_args={"lr": 0.001, "epochs": 5},
        )

        assert trainer.func_args == {"lr": 0.001, "epochs": 5}

    def test_huggingface_model_initializer(self):
        """Test HuggingFaceModelInitializer with hf:// prefix."""
        init = sdk_types.HuggingFaceModelInitializer(
            storage_uri="hf://meta-llama/Llama-3.2-1B",
            access_token="hf_token123",
        )

        assert init.storage_uri == "hf://meta-llama/Llama-3.2-1B"
        assert init.access_token == "hf_token123"

    def test_huggingface_model_initializer_validates_prefix(self):
        """Test that HuggingFaceModelInitializer validates hf:// prefix."""
        with pytest.raises(ValueError, match="must start with 'hf://'"):
            sdk_types.HuggingFaceModelInitializer(storage_uri="https://huggingface.co/model")

    def test_huggingface_dataset_initializer(self):
        """Test HuggingFaceDatasetInitializer with hf:// prefix."""
        init = sdk_types.HuggingFaceDatasetInitializer(
            storage_uri="hf://tatsu-lab/alpaca",
            access_token="hf_token123",
        )

        assert init.storage_uri == "hf://tatsu-lab/alpaca"

    def test_s3_model_initializer(self):
        """Test S3ModelInitializer for S3 data sources."""
        init = sdk_types.S3ModelInitializer(
            storage_uri="s3://my-bucket/models/llama",
            endpoint="https://s3.amazonaws.com",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="secret",
            region="us-east-1",
        )

        assert init.storage_uri == "s3://my-bucket/models/llama"
        assert init.region == "us-east-1"

    def test_s3_initializer_validates_prefix(self):
        """Test that S3ModelInitializer validates s3:// prefix."""
        with pytest.raises(ValueError, match="must start with 's3://'"):
            sdk_types.S3ModelInitializer(storage_uri="gs://bucket/path")

    def test_initializer_with_hf_sources(self):
        """Test Initializer combining model and dataset sources."""
        init = sdk_types.Initializer(
            model=sdk_types.HuggingFaceModelInitializer(
                storage_uri="hf://google/gemma-2b",
                access_token="token",
            ),
            dataset=sdk_types.HuggingFaceDatasetInitializer(storage_uri="hf://tatsu-lab/alpaca"),
        )

        assert init.model.storage_uri == "hf://google/gemma-2b"
        assert init.dataset.storage_uri == "hf://tatsu-lab/alpaca"


class TestSDKAPISignatures:
    """Verify SDK API signatures match what our MCP tools expect."""

    def test_trainer_client_train_signature(self):
        """Verify TrainerClient.train() accepts our expected parameters."""
        sig = inspect.signature(TrainerClient.train)
        params = list(sig.parameters.keys())

        assert "runtime" in params
        assert "initializer" in params
        assert "trainer" in params
        assert "options" in params

    def test_trainer_client_train_accepts_string_runtime(self):
        """Verify train() accepts runtime as string (not just Runtime object)."""
        hints = get_type_hints(TrainerClient.train)
        runtime_hint = str(hints.get("runtime", ""))

        assert "str" in runtime_hint or "Runtime" in runtime_hint

    def test_trainer_client_list_jobs_signature(self):
        """Verify list_jobs() takes optional Runtime (not a runtime name string)."""
        sig = inspect.signature(TrainerClient.list_jobs)
        params = list(sig.parameters.keys())

        assert "runtime" in params
        hints = get_type_hints(TrainerClient.list_jobs)
        assert hints.get("runtime") == sdk_types.Runtime | None

    def test_trainer_client_get_job_signature(self):
        """Verify get_job() accepts name parameter."""
        sig = inspect.signature(TrainerClient.get_job)
        params = list(sig.parameters.keys())

        assert "name" in params

    def test_trainer_client_get_job_logs_signature(self):
        """Verify get_job_logs() signature matches our usage."""
        sig = inspect.signature(TrainerClient.get_job_logs)
        params = list(sig.parameters.keys())

        assert "name" in params
        assert "step" in params
        assert "follow" in params

    def test_trainer_client_get_job_events_signature(self):
        """Verify get_job_events() signature."""
        sig = inspect.signature(TrainerClient.get_job_events)
        params = list(sig.parameters.keys())

        assert "name" in params

    def test_trainer_client_wait_for_job_status_signature(self):
        """Verify wait_for_job_status() signature."""
        sig = inspect.signature(TrainerClient.wait_for_job_status)
        params = list(sig.parameters.keys())

        assert "name" in params
        assert "status" in params
        assert "timeout" in params
        assert "polling_interval" in params

    def test_trainer_client_delete_job_signature(self):
        """Verify delete_job() signature."""
        sig = inspect.signature(TrainerClient.delete_job)
        params = list(sig.parameters.keys())

        assert "name" in params

    def test_trainer_client_list_runtimes_signature(self):
        """Verify list_runtimes() exists and takes no required args beyond self."""
        sig = inspect.signature(TrainerClient.list_runtimes)
        required = [
            p
            for p in sig.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
        ]
        assert len(required) == 0

    def test_trainer_client_get_runtime_signature(self):
        """Verify get_runtime() accepts name parameter."""
        sig = inspect.signature(TrainerClient.get_runtime)
        params = list(sig.parameters.keys())

        assert "name" in params

    def test_trainer_client_get_runtime_packages_signature(self):
        """Verify get_runtime_packages() expects Runtime object, not string."""
        sig = inspect.signature(TrainerClient.get_runtime_packages)
        params = list(sig.parameters.keys())

        assert "runtime" in params

        hints = get_type_hints(TrainerClient.get_runtime_packages)
        runtime_hint = str(hints.get("runtime", ""))
        assert "Runtime" in runtime_hint


class TestSDKDataclassFields:
    """Verify SDK dataclass fields match our expectations."""

    def test_lora_config_has_expected_fields(self):
        """Verify LoraConfig has all fields our MCP tools reference."""
        field_names = {f.name for f in fields(sdk_types.LoraConfig)}

        expected = {
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "quantize_base",
            "use_dora",
            "apply_lora_to_mlp",
            "apply_lora_to_output",
            "lora_attn_modules",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_torchtune_config_has_expected_fields(self):
        """Verify TorchTuneConfig has all fields we use."""
        field_names = {f.name for f in fields(sdk_types.TorchTuneConfig)}

        expected = {
            "dtype",
            "batch_size",
            "epochs",
            "loss",
            "num_nodes",
            "peft_config",
            "dataset_preprocess_config",
            "resources_per_node",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_torchtune_instruct_dataset_has_expected_fields(self):
        """Verify TorchTuneInstructDataset has all fields we use."""
        field_names = {f.name for f in fields(sdk_types.TorchTuneInstructDataset)}

        expected = {
            "source",
            "split",
            "train_on_input",
            "new_system_prompt",
            "column_map",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_hf_model_initializer_has_ignore_patterns(self):
        """Verify HuggingFaceModelInitializer has ignore_patterns field."""
        field_names = {f.name for f in fields(sdk_types.HuggingFaceModelInitializer)}
        assert "ignore_patterns" in field_names

    def test_hf_dataset_initializer_has_ignore_patterns(self):
        """Verify HuggingFaceDatasetInitializer has ignore_patterns field."""
        field_names = {f.name for f in fields(sdk_types.HuggingFaceDatasetInitializer)}
        assert "ignore_patterns" in field_names

    def test_custom_trainer_has_expected_fields(self):
        """Verify CustomTrainer has all fields we use."""
        field_names = {f.name for f in fields(sdk_types.CustomTrainer)}

        expected = {
            "func",
            "func_args",
            "num_nodes",
            "resources_per_node",
            "packages_to_install",
            "pip_index_urls",
            "env",
            "image",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_custom_trainer_container_has_expected_fields(self):
        """Verify CustomTrainerContainer has all fields we use."""
        field_names = {f.name for f in fields(sdk_types.CustomTrainerContainer)}

        expected = {"image", "num_nodes", "resources_per_node", "env"}

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_trainjob_has_expected_fields(self):
        """Verify TrainJob response type has fields we extract."""
        field_names = {f.name for f in fields(sdk_types.TrainJob)}

        expected = {"name", "status", "runtime", "steps", "creation_timestamp"}

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_runtime_has_expected_fields(self):
        """Verify Runtime type has fields we extract."""
        field_names = {f.name for f in fields(sdk_types.Runtime)}

        expected = {"name", "trainer"}

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_event_has_expected_fields(self):
        """Verify Event type has fields we extract."""
        field_names = {f.name for f in fields(sdk_types.Event)}

        expected = {
            "involved_object_kind",
            "involved_object_name",
            "message",
            "reason",
            "event_time",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_s3_dataset_initializer_has_expected_fields(self):
        """Verify S3DatasetInitializer has fields MCP tools reference."""
        field_names = {f.name for f in fields(sdk_types.S3DatasetInitializer)}

        expected = {
            "storage_uri",
            "endpoint",
            "access_key_id",
            "secret_access_key",
            "region",
        }

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_step_has_expected_fields(self):
        """Verify Step type (returned in TrainJob.steps) has fields we extract."""
        field_names = {f.name for f in fields(sdk_types.Step)}

        expected = {"name", "status", "pod_name"}

        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_trainjob_has_num_nodes(self):
        """Verify TrainJob includes num_nodes (used in monitoring summary)."""
        field_names = {f.name for f in fields(sdk_types.TrainJob)}
        assert "num_nodes" in field_names


class TestSDKEnums:
    """Verify SDK enums are available and have expected values."""

    def test_data_type_enum(self):
        """Verify DataType enum has bf16 and fp32."""
        assert hasattr(sdk_types.DataType, "BF16")
        assert hasattr(sdk_types.DataType, "FP32")
        assert sdk_types.DataType.BF16.value == "bf16"
        assert sdk_types.DataType.FP32.value == "fp32"

    def test_loss_enum(self):
        """Verify Loss enum exists."""
        assert hasattr(sdk_types.Loss, "CEWithChunkedOutputLoss")

    def test_data_format_enum(self):
        """Verify DataFormat enum has values MCP fine_tune uses for dataset_source."""
        assert hasattr(sdk_types.DataFormat, "JSON")
        assert hasattr(sdk_types.DataFormat, "CSV")
        assert hasattr(sdk_types.DataFormat, "PARQUET")
        assert sdk_types.DataFormat.JSON.value == "json"


class TestMCPToSDKConversions:
    """Test conversions from MCP tool parameters to SDK types."""

    def test_fine_tune_params_to_builtin_trainer(self):
        """Test converting fine_tune() params to BuiltinTrainer."""
        mcp_params = {
            "batch_size": 4,
            "epochs": 3,
            "lora_rank": 16,
            "lora_alpha": 32,
            "num_nodes": 2,
        }

        lora_config = sdk_types.LoraConfig(
            lora_rank=mcp_params["lora_rank"],
            lora_alpha=mcp_params["lora_alpha"],
        )

        tune_config = sdk_types.TorchTuneConfig(
            batch_size=mcp_params["batch_size"],
            epochs=mcp_params["epochs"],
            num_nodes=mcp_params["num_nodes"],
            peft_config=lora_config,
        )

        trainer = sdk_types.BuiltinTrainer(config=tune_config)

        assert trainer.config.batch_size == 4
        assert trainer.config.peft_config.lora_rank == 16

    def test_fine_tune_hf_sources_to_initializer(self):
        """Test converting hf:// model/dataset to Initializer."""
        model_uri = "hf://google/gemma-2b"
        dataset_uri = "hf://tatsu-lab/alpaca"
        hf_token = "hf_secret"

        initializer = sdk_types.Initializer(
            model=sdk_types.HuggingFaceModelInitializer(
                storage_uri=model_uri,
                access_token=hf_token,
            ),
            dataset=sdk_types.HuggingFaceDatasetInitializer(
                storage_uri=dataset_uri,
                access_token=hf_token,
            ),
        )

        assert initializer.model.storage_uri == model_uri
        assert initializer.model.access_token == hf_token
        assert initializer.dataset.storage_uri == dataset_uri

    def test_custom_training_script_to_callable(self):
        """Test that script string can become a callable for CustomTrainer."""
        script = textwrap.dedent("""
                def train_func():
                    import torch
                    print("Training...")
                """)
        local_vars = {}
        exec(script, {}, local_vars)
        train_func = local_vars["train_func"]

        trainer = sdk_types.CustomTrainer(
            func=train_func,
            num_nodes=1,
            resources_per_node={"gpu": 1},
        )

        assert callable(trainer.func)

    def _run_wrapped_train(
        self,
        script: str,
        func_args: dict | None = None,
        **kwargs,
    ):
        from kubeflow_mcp.trainer.api.training import _make_train_func

        builtins._kubeflow_mcp_train_marker = []
        try:
            train_func = _make_train_func(script, func_args=func_args)
            train_func(**kwargs)
            return builtins._kubeflow_mcp_train_marker
        finally:
            del builtins._kubeflow_mcp_train_marker

    def test_make_train_func_calls_user_defined_train(self):
        """A script-defined train() should run when the generated wrapper runs."""
        marker = self._run_wrapped_train(
            """
                import builtins
                def train():
                    builtins._kubeflow_mcp_train_marker.append("ran")
            """
        )

        assert marker == ["ran"]

    def test_make_train_func_does_not_double_call_user_defined_train(self):
        """A script that already calls train() should not run twice."""
        marker = self._run_wrapped_train(
            """
                import builtins
                def train():
                    builtins._kubeflow_mcp_train_marker.append("ran")
                train()
            """
        )

        assert marker == ["ran"]

    @pytest.mark.parametrize(
        "train_call",
        [
            "result = train()",
            "if True:\n    train()",
            "print(train())",
        ],
    )
    def test_make_train_func_detects_existing_train_calls(self, train_call):
        """Existing train() calls in module-level code should not be duplicated."""
        script = textwrap.dedent("""
                import builtins
                def train():
                    builtins._kubeflow_mcp_train_marker.append("ran")
        """) + train_call + "\n"
        marker = self._run_wrapped_train(script)

        assert marker == ["ran"]

    def test_make_train_func_forwards_func_args_to_user_defined_train(self):
        """Generated train() should forward matching func_args to a user-defined train()."""
        marker = self._run_wrapped_train(
            """
                import builtins
                def train(lr=None, epochs=None):
                    builtins._kubeflow_mcp_train_marker.append((lr, epochs))
            """,
            func_args={"lr": 0.01, "epochs": 3},
            lr=0.01,
            epochs=3,
        )

        assert marker == [(0.01, 3)]

    def test_make_train_func_raises_value_error_for_required_params(self):
        """train(required_param) with no func_args must raise a ValueError."""
        with pytest.raises(ValueError, match=r"User-defined train\(\) requires parameters but no func_args were provided"):
            self._run_wrapped_train(
                """
                    import builtins
                    def train(required_param):
                        builtins._kubeflow_mcp_train_marker.append("ran")
                """
            )

    def test_make_train_func_raises_syntax_error_for_invalid_script(self):
        """SyntaxError in the user script should be surfaced."""
        from kubeflow_mcp.trainer.api.training import _make_train_func

        with pytest.raises(SyntaxError, match="Invalid Python script"):
            _make_train_func("def train():\n    pass\n  invalid syntax")

    def test_make_train_func_raises_value_error_for_mismatched_func_args(self):
        """train() with no args but func_args provided should raise ValueError."""
        with pytest.raises(ValueError, match=r"User-defined train\(\) signature does not accept"):
            self._run_wrapped_train(
                """
                    def train():
                        pass
                """,
                func_args={"lr": 0.01}
            )

    def test_make_train_func_adds_pass_for_empty_script(self):
        """An empty or whitespace-only script should generate a valid def train(): pass."""
        from kubeflow_mcp.trainer.api.training import _make_train_func

        train_func = _make_train_func("   \n  \t  \n")
        assert callable(train_func)
        train_func()  # Should run without Error

    def test_make_train_func_async_train_executes(self):
        """async def train() should be invoked via asyncio.run()."""
        marker = self._run_wrapped_train(
            """
                import builtins
                async def train():
                    builtins._kubeflow_mcp_train_marker.append("async_ran")
            """
        )
        assert marker == ["async_ran"]

    def test_resources_dict_format(self):
        """Test resources_per_node dict format is valid for SDK."""
        resources = {"gpu": 2, "cpu": 8, "memory": "32Gi"}

        trainer = sdk_types.CustomTrainerContainer(
            image="test:latest",
            resources_per_node=resources,
        )

        assert trainer.resources_per_node == resources

    def test_env_dict_format(self):
        """Test env dict format for CustomTrainer."""
        env = {"LEARNING_RATE": "0.001", "BATCH_SIZE": "32"}

        trainer = sdk_types.CustomTrainer(
            func=lambda: None,
            env=env,
        )

        assert trainer.env == env


class TestSDKOptionsImport:
    """Test that SDK options module is available for runtime patches."""

    def test_options_module_available(self):
        """Verify options module can be imported."""
        from kubeflow.trainer import options

        assert hasattr(options, "kubernetes")

    def test_runtime_patch_available(self):
        """Verify RuntimePatch and PodSpecPatch classes exist for node_selector, tolerations, etc."""
        from kubeflow.trainer.options import kubernetes

        assert hasattr(kubernetes, "RuntimePatch")
        assert hasattr(kubernetes, "PodSpecPatch")
        assert hasattr(kubernetes, "ContainerPatch")

    def test_pod_spec_patch_has_scheduling_fields(self):
        """Verify PodSpecPatch has fields MCP tools use for node scheduling."""
        from kubeflow.trainer.options.kubernetes import PodSpecPatch

        field_names = {f.name for f in fields(PodSpecPatch)}
        expected = {
            "service_account_name",
            "volumes",
            "image_pull_secrets",
            "node_selector",
            "tolerations",
        }
        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_container_patch_has_expected_fields(self):
        """Verify ContainerPatch has fields MCP tools use."""
        from kubeflow.trainer.options.kubernetes import ContainerPatch

        field_names = {f.name for f in fields(ContainerPatch)}
        expected = {"name", "env", "volume_mounts"}
        assert expected.issubset(field_names), f"Missing: {expected - field_names}"

    def test_trainer_command_and_args_available(self):
        """Verify TrainerCommand and TrainerArgs options exist."""
        from kubeflow.trainer.options import kubernetes

        assert hasattr(kubernetes, "TrainerCommand")
        assert hasattr(kubernetes, "TrainerArgs")


class TestK8sClientContracts:
    """Validate kubernetes client APIs used by utils.py and lifecycle.py.

    These catch signature changes when the ``kubernetes`` package is upgraded.
    """

    def test_core_v1_api_list_node(self):
        """list_node() must be callable (used by get_cluster_resources)."""
        from kubernetes.client import CoreV1Api

        assert callable(getattr(CoreV1Api, "list_node", None))
        sig = inspect.signature(CoreV1Api.list_node)
        # _request_timeout is passed via **kwargs in newer client versions
        assert "kwargs" in sig.parameters or "_request_timeout" in sig.parameters

    def test_core_v1_api_list_namespace(self):
        """list_namespace() must be callable (used by health_check)."""
        from kubernetes.client import CoreV1Api

        assert callable(getattr(CoreV1Api, "list_namespace", None))
        sig = inspect.signature(CoreV1Api.list_namespace)
        assert "kwargs" in sig.parameters or "_request_timeout" in sig.parameters

    def test_custom_objects_api_patch(self):
        """patch_namespaced_custom_object() must accept our positional kwargs."""
        from kubernetes.client import CustomObjectsApi

        sig = inspect.signature(CustomObjectsApi.patch_namespaced_custom_object)
        for param in ["group", "version", "namespace", "plural", "name", "body"]:
            assert param in sig.parameters, f"Missing param: {param}"
        # _request_timeout passed via **kwargs
        assert "kwargs" in sig.parameters or "_request_timeout" in sig.parameters

    def test_api_client_instantiation(self):
        """ApiClient can be instantiated with a Configuration object."""
        from kubernetes.client import ApiClient, Configuration

        config = Configuration()
        client = ApiClient(configuration=config)
        assert client is not None
        client.close()

    def test_load_config_exists(self):
        """kubernetes.config.load_config is importable."""
        from kubernetes.config import load_config

        assert callable(load_config)

    def test_custom_objects_api_cluster_operations(self):
        """Cluster-scoped CR operations used by platform.py."""
        from kubernetes.client import CustomObjectsApi

        for method_name in [
            "list_cluster_custom_object",
            "get_cluster_custom_object",
            "create_cluster_custom_object",
            "patch_cluster_custom_object",
            "delete_cluster_custom_object",
        ]:
            assert callable(getattr(CustomObjectsApi, method_name, None)), f"Missing: {method_name}"


class TestSDKConstants:
    """Verify SDK constants used by MCP server match expected values."""

    def test_trainer_group_and_version(self):
        """Verify API group/version used in platform.py CRD operations."""
        from kubeflow.trainer.constants import constants

        assert constants.GROUP == "trainer.kubeflow.org"
        assert constants.VERSION == "v1alpha1"

    def test_trainjob_status_strings(self):
        """Verify status constants used in monitoring/discovery status mapping."""
        from kubeflow.trainer.constants import constants

        assert constants.TRAINJOB_COMPLETE == "Complete"
        assert constants.TRAINJOB_FAILED == "Failed"
        assert constants.TRAINJOB_RUNNING == "Running"
        assert constants.TRAINJOB_CREATED == "Created"

    def test_resource_plurals(self):
        """Verify CRD plural names used by platform.py and utils.py."""
        from kubeflow.trainer.constants import constants

        assert constants.TRAINJOB_PLURAL == "trainjobs"
        assert constants.CLUSTER_TRAINING_RUNTIME_PLURAL == "clustertrainingruntimes"

    def test_default_log_step(self):
        """Verify NODE constant used as default log step prefix."""
        from kubeflow.trainer.constants import constants

        assert constants.NODE == "node"

    def test_kubernetes_backend_config_fields(self):
        """Verify KubernetesBackendConfig has namespace field used by utils.py."""
        from kubeflow.common.types import KubernetesBackendConfig

        config = KubernetesBackendConfig(namespace="test-ns")
        assert config.namespace == "test-ns"


class TestMCPToolSignatures:
    """Verify MCP tool function signatures include expected SDK-aligned parameters."""

    FINE_TUNE_EXPECTED_PARAMS = {
        "model",
        "dataset",
        "runtime",
        "namespace",
        "hf_token",
        "batch_size",
        "epochs",
        "num_nodes",
        "dtype",
        "loss",
        "resources_per_node",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "use_dora",
        "quantize_base",
        "apply_lora_to_mlp",
        "apply_lora_to_output",
        "lora_attn_modules",
        "dataset_source",
        "dataset_split",
        "dataset_train_on_input",
        "dataset_system_prompt",
        "dataset_column_map",
        "model_ignore_patterns",
        "dataset_ignore_patterns",
        "confirmed",
    }

    CUSTOM_TRAINING_EXPECTED_PARAMS = {
        "script",
        "name",
        "namespace",
        "num_nodes",
        "gpu_per_node",
        "func_args",
        "packages",
        "pip_index_urls",
        "image",
        "runtime",
        "resources_per_node",
        "env",
        "confirmed",
    }

    CONTAINER_TRAINING_EXPECTED_PARAMS = {
        "image",
        "command",
        "args",
        "name",
        "namespace",
        "num_nodes",
        "gpu_per_node",
        "resources_per_node",
        "env",
        "confirmed",
    }

    def test_fine_tune_signature(self):
        """fine_tune() has all expected SDK-aligned parameters."""
        from kubeflow_mcp.trainer.api.training import fine_tune

        sig = inspect.signature(fine_tune)
        params = set(sig.parameters.keys())
        missing = self.FINE_TUNE_EXPECTED_PARAMS - params
        assert not missing, f"fine_tune missing params: {missing}"

    def test_run_custom_training_signature(self):
        """run_custom_training() has all expected SDK-aligned parameters."""
        from kubeflow_mcp.trainer.api.training import run_custom_training

        sig = inspect.signature(run_custom_training)
        params = set(sig.parameters.keys())
        missing = self.CUSTOM_TRAINING_EXPECTED_PARAMS - params
        assert not missing, f"run_custom_training missing params: {missing}"

    def test_run_container_training_signature(self):
        """run_container_training() has all expected SDK-aligned parameters."""
        from kubeflow_mcp.trainer.api.training import run_container_training

        sig = inspect.signature(run_container_training)
        params = set(sig.parameters.keys())
        missing = self.CONTAINER_TRAINING_EXPECTED_PARAMS - params
        assert not missing, f"run_container_training missing params: {missing}"

    def test_fine_tune_sdk_type_params_match(self):
        """fine_tune LoRA params align with SDK LoraConfig fields."""
        from kubeflow_mcp.trainer.api.training import fine_tune

        sig = inspect.signature(fine_tune)
        lora_fields = {f.name for f in fields(sdk_types.LoraConfig)}
        mcp_lora_params = {
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "use_dora",
            "quantize_base",
            "apply_lora_to_mlp",
            "apply_lora_to_output",
            "lora_attn_modules",
        }
        for p in mcp_lora_params:
            assert p in sig.parameters, f"MCP fine_tune missing LoRA param: {p}"
            assert p in lora_fields, f"SDK LoraConfig missing field: {p}"

    def test_should_apply_hf_dataset_workaround(self):
        """Workaround helper correctly detects top-level HuggingFace dataset URIs."""
        from kubeflow_mcp.trainer.api.training import _should_apply_hf_dataset_workaround

        assert _should_apply_hf_dataset_workaround("hf://org/ds") is True
        assert _should_apply_hf_dataset_workaround("hf://org/ds/") is True
        assert _should_apply_hf_dataset_workaround("hf://ds") is True
        assert _should_apply_hf_dataset_workaround("hf://") is False
        assert _should_apply_hf_dataset_workaround("hf://org/ds/subpath") is False
        assert _should_apply_hf_dataset_workaround("s3://bucket/dataset") is False

