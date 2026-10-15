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

"""Tests for trainer/api/training.py — fine_tune, run_custom_training, run_container_training."""

from __future__ import annotations

import ast
import inspect
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubeflow.trainer.options import Labels, Name
from tests.common import FAILED, PREVIEW, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE
from kubeflow_mcp.trainer.api.training import (
    _build_dataset_preprocess_cfg,
    _build_fine_tune_config,
    _build_initializer,
    _build_lora_cfg,
    _check_gpu_available,
    _forwarded_train_args,
    _get_client,
    _has_required_params,
    _has_train_call,
    _inject_ownership_label,
    _inject_trainer_hf_home,
    _make_train_func,
    _sdk_error,
    _should_apply_hf_dataset_workaround,
    _uncalled_train_call,
    _validate_fine_tune_params,
    _validate_resources_per_node,
    fine_tune,
    run_container_training,
    run_custom_training,
)

PATCH_CLIENT = "kubeflow_mcp.trainer.api.training._get_client"
PATCH_NS_CHECK = "kubeflow_mcp.trainer.api.training.check_namespace_allowed"
PATCH_GPU_CHECK = "kubeflow_mcp.trainer.api.training._check_gpu_available"

# ─── fine_tune validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="preview accepts any model URI",
            expected_status=PREVIEW,
            config={"model": "not-a-model", "dataset": "hf://org/ds"},
        ),
        TestCase(
            name="preview includes model in config",
            expected_status=PREVIEW,
            config={"model": "hf://org/model", "dataset": "hf://org/ds"},
            expected_output={"model": "hf://org/model"},
        ),
        TestCase(
            name="invalid name format rejected",
            expected_status=FAILED,
            config={"model": "hf://org/model", "dataset": "hf://org/ds", "name": "INVALID_NAME"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="batch_size out of bounds rejected",
            expected_status=FAILED,
            config={"model": "hf://org/model", "dataset": "hf://org/ds", "batch_size": 9999},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_fine_tune_validation(test_case):
    with patch(PATCH_GPU_CHECK, return_value=None):
        assert_test_case(test_case, fine_tune)


class TestFineTuneConfirmed:
    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-gemma-abc"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "train-gemma-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_hf_token_passed_to_sdk(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            hf_token="hf_secret_token",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert initializer.model.access_token == "hf_secret_token"

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_dataset_parameter_handling(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/dataset",
            runtime="torchtune-llama",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert "hf://org/dataset" in initializer.dataset.storage_uri

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_lora_config(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.config.peft_config.lora_rank == 16
        assert trainer.config.peft_config.lora_alpha == 32
        assert trainer.config.peft_config.lora_dropout == 0.1

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_s3_model_source(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="s3://bucket/model",
            dataset="s3://bucket/dataset",
            runtime="torchtune-llama",
            s3_access_key_id="AKIA",
            s3_secret_access_key="secret",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert initializer.model.storage_uri == "s3://bucket/model"
        assert initializer.dataset.storage_uri == "s3://bucket/dataset"

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_gpu_validation_blocks_on_zero_gpus(self):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        gpu_err = ToolErrorModel(
            error="fine_tune() requires GPUs", error_code="VALIDATION_ERROR"
        ).model_dump()
        with patch(PATCH_GPU_CHECK, return_value=gpu_err):
            result = fine_tune(
                model="hf://org/model",
                dataset="hf://org/ds",
            )
        assert result["success"] is False
        assert "GPU" in result["error"]

    @patch(PATCH_GPU_CHECK, return_value=None)
    def test_non_torchtune_runtime_rejected(self, _gpu):
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torch-distributed",
        )
        assert result["success"] is False
        assert "torchtune" in result["error"]


# ─── run_custom_training validation ─────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="empty script rejected",
            expected_status=FAILED,
            config={"script": "   ", "runtime": "torch-distributed"},
        ),
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"script": "print('hello')", "runtime": "torch-distributed", "name": "BadName"},
        ),
        TestCase(
            name="unsafe script blocked on confirmed",
            expected_status=FAILED,
            config={
                "script": "import os\nos.system('rm -rf /')",
                "runtime": "torch-distributed",
                "confirmed": True,
            },
        ),
        TestCase(
            name="unsafe script preview includes warnings",
            expected_status=PREVIEW,
            config={
                "script": "import os\nos.system('rm -rf /')",
                "runtime": "torch-distributed",
                "confirmed": False,
            },
        ),
    ],
)
def test_run_custom_training_validation(test_case):
    if test_case.expected_status == PREVIEW:
        result = run_custom_training(**test_case.config)
        assert result["status"] == "preview"
        assert len(result["config"]["safety_warnings"]) > 0
    else:
        assert_test_case(test_case, run_custom_training)


@pytest.mark.parametrize(
    ("tool", "config"),
    [
        (run_custom_training, {"script": "print('hello')", "runtime": "torch-distributed"}),
        (run_container_training, {"image": "pytorch/pytorch:2.0"}),
        (
            fine_tune,
            {
                "model": "hf://org/model",
                "dataset": "hf://org/dataset",
                "runtime": "torchtune-llama",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "bad_resources",
    [
        {"cpu": "4cores"},
        {"memory": "256MB"},
        {"gpu": -1},
        {"gpu": "two"},
        {"gpu": 1.5},
        {"cpu": ""},
        {"memory": None},
        {"amd.com/gpu": "lots"},
        {},
        ["gpu", 1],
    ],
)
def test_training_tools_reject_invalid_resources_before_sdk_call(tool, config, bad_resources):
    with patch(PATCH_CLIENT) as mock_client:
        result = tool(**config, resources_per_node=bad_resources, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_client.assert_not_called()


@pytest.mark.parametrize(
    "resources",
    [
        {"cpu": 8},
        {"cpu": 0.5},
        {"cpu": "500m"},
        {"memory": "10G"},
        {"memory": "1.5Gi"},
        {"gpu": 2},
        {"amd.com/gpu": 2},
    ],
)
def test_validate_resources_per_node_accepts_kubernetes_quantities(resources):
    assert _validate_resources_per_node(resources) is None


def test_fine_tune_validates_resources_before_gpu_preflight():
    with patch(PATCH_GPU_CHECK) as gpu_check:
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/dataset",
            runtime="torchtune-llama",
            resources_per_node={"cpu": "4cores"},
        )

    assert result["error_code"] == VALIDATION_ERROR
    gpu_check.assert_not_called()


class TestRunCustomTrainingConfirmed:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-train-abc"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "custom-train-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_env_parameter_passed(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            env={"MY_VAR": "value"},
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.env == {"MY_VAR": "value"}

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_packages_parameter(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="import torch; print(torch.__version__)",
            runtime="torch-distributed",
            packages=["torch", "transformers"],
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.packages_to_install == ["torch", "transformers"]

    def test_preview_returns_config(self):
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["runtime"] == "torch-distributed"
        assert "print('hello')" in result["config"]["script"]

    def test_auto_generated_name_format(self):
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["name"] is None

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_unsafe_scripts_override(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        with patch.dict("os.environ", {"KUBEFLOW_MCP_UNSAFE_SCRIPTS": "true"}):
            result = run_custom_training(
                script="import os\nos.system('ls')",
                runtime="torch-distributed",
                confirmed=True,
            )
        assert result["success"] is True

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = run_custom_training(
            script="print('hello')",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"


# ─── run_container_training validation ──────────────────────────────────────


def test_run_container_training_invalid_name():
    result = run_container_training(
        image="ghcr.io/kubeflow/trainer/torch:latest",
        name="BadName",
    )
    assert result["success"] is False


class TestRunContainerTrainingConfirmed:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "container-train-abc"
        mock_client_fn.return_value = mock_client
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "container-train-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_command_and_args_override(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            command=["python", "train.py"],
            args=["--epochs", "5"],
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        options = call_kwargs.kwargs["options"]
        from kubeflow.trainer.options import TrainerArgs, TrainerCommand

        cmd_opts = [o for o in options if isinstance(o, TrainerCommand)]
        args_opts = [o for o in options if isinstance(o, TrainerArgs)]
        assert len(cmd_opts) == 1
        assert cmd_opts[0].command == ["python", "train.py"]
        assert len(args_opts) == 1
        assert args_opts[0].args == ["--epochs", "5"]

    def test_preview_returns_config(self):
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["image"] == "pytorch/pytorch:2.0"

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"


# ─── _make_train_func ────────────────────────────────────────────────────────


def test_make_train_func_wraps_script():
    func = _make_train_func("x = 1 + 1\nprint(x)")
    assert callable(func)
    assert func.__name__ == "train"


def test_make_train_func_accepts_args():
    func = _make_train_func("print('hi')", func_args={"lr": 0.001})
    assert callable(func)


def test_make_train_func_inspect_getsource():
    func = _make_train_func("x = 42\nprint(x)")
    source = inspect.getsource(func)
    assert "x = 42" in source
    assert "def train" in source


def test_make_train_func_syntax_error():
    with pytest.raises(SyntaxError):
        _make_train_func("def broken(:\n    pass")


def test_make_train_func_args_appear_as_params():
    func = _make_train_func("print(lr)", func_args={"lr": 0.001, "epochs": 5})
    source = inspect.getsource(func)
    assert "lr=None" in source
    assert "epochs=None" in source


# ─── Pure helpers (AST / builders) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "script,expected",
    [
        ("def train():\n    pass\ntrain()", True),
        ("def train():\n    pass", False),
    ],
)
def test_has_train_call(script, expected):
    tree = ast.parse(script)
    assert _has_train_call(tree.body) is expected


def test_has_required_params_detects_missing_defaults():
    tree = ast.parse("def train(lr, batch_size=32):\n    pass")
    train_def = tree.body[0]
    assert _has_required_params(train_def) is True


def test_has_required_params_all_optional():
    tree = ast.parse("def train(lr=0.01):\n    pass")
    train_def = tree.body[0]
    assert _has_required_params(train_def) is False


def test_forwarded_train_args_builds_kwargs():
    tree = ast.parse("def train(lr, batch_size):\n    pass")
    train_def = tree.body[0]
    assert _forwarded_train_args(train_def, {"lr": 0.01, "batch_size": 4}) == (
        "lr=lr, batch_size=batch_size"
    )


def test_forwarded_train_args_rejects_unknown_keys():
    tree = ast.parse("def train(lr):\n    pass")
    train_def = tree.body[0]
    with pytest.raises(ValueError, match="func_args"):
        _forwarded_train_args(train_def, {"lr": 0.01, "extra": 1})


def test_uncalled_train_call_appends_invocation():
    lines = _uncalled_train_call("def train():\n    print('hi')")
    assert lines == ["train()"]


def test_uncalled_train_call_skips_when_already_called():
    script = "def train():\n    pass\ntrain()"
    assert _uncalled_train_call(script) == []


@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("hf://org/dataset", True),
        ("hf://org/dataset/subpath", False),
        ("s3://bucket/ds", False),
    ],
)
def test_should_apply_hf_dataset_workaround(dataset, expected):
    assert _should_apply_hf_dataset_workaround(dataset) is expected


def test_build_fine_tune_config_masks_secrets():
    config = _build_fine_tune_config(
        model="hf://org/model",
        dataset="hf://org/ds",
        runtime="torchtune-llama",
        name="train-gemma",
        hf_token="hf_secret",
        batch_size=4,
        epochs=1,
        num_nodes=1,
        dtype=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=None,
        use_dora=None,
        quantize_base=None,
        s3_access_key_id="AKIA",
        s3_secret_access_key="secret",
        optional_fields=[],
    )
    assert config["hf_token"] == "***"
    assert config["s3_access_key_id"] == "***"
    assert config["s3_secret_access_key"] == "***"


def test_inject_trainer_hf_home_concurrent_isolation():
    """Verify concurrent _inject_trainer_hf_home calls maintain context isolation and restore original function."""
    import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
    from kubeflow.trainer.types import types as sdk_types

    original_func = k8s_utils.get_trainer_cr_from_builtin_trainer
    runtime_trainer = sdk_types.RuntimeTrainer(
        trainer_type=sdk_types.TrainerType.BUILTIN_TRAINER,
        framework="torchtune",
        image="test:latest",
    )
    runtime_trainer.set_command(("torchrun",))
    runtime = sdk_types.Runtime(
        name="test-rt",
        trainer=runtime_trainer,
        kind=sdk_types.RuntimeKind.CLUSTER_TRAINING_RUNTIME,
    )
    builtin = sdk_types.BuiltinTrainer(config=sdk_types.TorchTuneConfig(batch_size=4, epochs=1))

    barrier = threading.Barrier(4)
    errors: list[str] = []
    completed: list[int | str] = []

    def worker(worker_id: int) -> None:
        try:
            target_path = f"/workspace/.hf_{worker_id}"
            with _inject_trainer_hf_home(target_path):
                barrier.wait(timeout=5.0)
                cr = k8s_utils.get_trainer_cr_from_builtin_trainer(runtime, builtin)
                if cr.env is None:
                    errors.append(f"Worker {worker_id}: cr.env is None")
                    return
                hf_env = next((e for e in cr.env if e.name == "HF_HOME"), None)
                if not hf_env or hf_env.value != target_path:
                    errors.append(f"Worker {worker_id}: expected {target_path}, got {hf_env}")
                    return
                completed.append(worker_id)
        except Exception as exc:
            errors.append(f"Worker {worker_id} raised: {exc!r}")

    def uninjected_worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            cr = k8s_utils.get_trainer_cr_from_builtin_trainer(runtime, builtin)
            if cr.env is not None and any(e.name == "HF_HOME" for e in cr.env):
                errors.append("Uninjected worker unexpectedly got HF_HOME")
                return
            completed.append("uninjected")
        except Exception as exc:
            errors.append(f"Uninjected worker raised: {exc!r}")

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(2,)),
        threading.Thread(target=worker, args=(3,)),
        threading.Thread(target=uninjected_worker),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "Worker thread timed out"

    assert not errors, f"Concurrency errors encountered: {errors}"
    assert len(completed) == 4, f"Expected 4 completed workers, got {len(completed)}"
    assert k8s_utils.get_trainer_cr_from_builtin_trainer is original_func, (
        "Monkey-patch function identity was not restored to original"
    )


def test_inject_trainer_hf_home_exception_handling():
    """Verify refcount and monkey-patch cleanly revert if an exception occurs within yield."""
    import kubeflow.trainer.backends.kubernetes.utils as k8s_utils

    original_func = k8s_utils.get_trainer_cr_from_builtin_trainer
    with pytest.raises(RuntimeError, match="simulated failure"):
        with _inject_trainer_hf_home("/workspace/.hf_err"):
            assert k8s_utils.get_trainer_cr_from_builtin_trainer is not original_func
            raise RuntimeError("simulated failure")

    assert k8s_utils.get_trainer_cr_from_builtin_trainer is original_func, (
        "Monkey-patch function identity was not restored after exception"
    )


# ─── Script parsing & uncalled train edge cases ─────────────────────────────


class TestUncalledTrainCallEdgeCases:
    def test_uncalled_train_with_required_params_and_no_func_args_raises(self):
        script = "def train(lr, batch_size):\n    pass\n"
        with pytest.raises(
            ValueError,
            match="User-defined train\\(\\) requires parameters but no func_args were provided.",
        ):
            _uncalled_train_call(script, func_args=None)

    def test_has_required_params_kwonly_arg_without_default(self):
        code_without_default = "def train(*, lr):\n    pass\n"
        def_node_without_default = ast.parse(code_without_default).body[0]
        assert _has_required_params(def_node_without_default) is True

        code_with_default = "def train(*, lr=0.01):\n    pass\n"
        def_node_with_default = ast.parse(code_with_default).body[0]
        assert _has_required_params(def_node_with_default) is False

    def test_uncalled_train_async_function_wraps_in_asyncio_run(self):
        script = "async def train():\n    pass\n"
        lines = _uncalled_train_call(script)
        assert lines == ["import asyncio", "asyncio.run(train())"]

        func = _make_train_func(script)
        assert callable(func)

    def test_forwarded_train_args_with_var_kwargs(self):
        code = "def train(**kwargs):\n    pass\n"
        def_node = ast.parse(code).body[0]
        args_str = _forwarded_train_args(def_node, {"lr": 0.01, "epochs": 5})
        assert "lr=lr" in args_str
        assert "epochs=epochs" in args_str


# ─── Client resolution & Ownership labels ───────────────────────────────────


class TestClientAndOwnershipLabel:
    def test_get_client_singleton_vs_namespace(self):
        with (
            patch("kubeflow_mcp.trainer.api.training.get_trainer_client") as mock_singleton,
            patch(
                "kubeflow_mcp.trainer.api.training.get_trainer_client_for_namespace"
            ) as mock_scoped,
        ):
            mock_singleton.return_value = "singleton-client"
            mock_scoped.return_value = "scoped-client"

            assert _get_client(None) == "singleton-client"
            mock_singleton.assert_called_once()
            mock_scoped.assert_not_called()

            assert _get_client("custom-ns") == "scoped-client"
            mock_scoped.assert_called_once_with("custom-ns")

    def test_inject_ownership_label_merges_into_existing_labels(self):
        existing_label = Labels(labels={"app": "training-job"})
        options = [existing_label]
        _inject_ownership_label(options)
        assert len(options) == 1
        assert options[0].labels["app"] == "training-job"
        assert options[0].labels[MCP_MANAGED_LABEL] == MCP_MANAGED_VALUE

    def test_inject_ownership_label_appends_when_no_labels_present(self):
        options = []
        _inject_ownership_label(options)
        assert len(options) == 1
        assert isinstance(options[0], Labels)
        assert options[0].labels == {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}


# ─── SDK Error wrapper details ──────────────────────────────────────────────


class TestSdkErrorHandling:
    def test_sdk_error_with_cause(self):
        cause = ValueError("original connection error")
        err = RuntimeError("wrapper error")
        err.__cause__ = cause
        result = _sdk_error(err, hint="check connection")
        assert result["success"] is False
        assert result["error_code"] == "SDK_ERROR"
        assert result["details"] == {"cause": "original connection error"}
        assert result["hint"] == "check connection"

    def test_sdk_error_with_response_text(self):
        err = Exception("k8s api call failed")
        err.response = SimpleNamespace(text='{"message": "resource not found"}')
        result = _sdk_error(err)
        assert result["details"] == {"response": '{"message": "resource not found"}'}

    def test_sdk_error_with_failing_response_text(self):
        class BrokenResponse:
            @property
            def text(self):
                raise RuntimeError("unreadable stream")

        err = Exception("read failed")
        err.response = BrokenResponse()
        result = _sdk_error(err)
        assert result["details"] is None


# ─── GPU availability & Resource validation ─────────────────────────────────


class TestGpuAvailabilityAndResourceValidation:
    def test_check_gpu_available_zero_gpus_returns_error(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"data": {"gpu_total": 0}},
        ):
            err = _check_gpu_available()
            assert err is not None
            assert err["error_code"] == "VALIDATION_ERROR"
            assert "fine_tune() requires GPUs" in err["error"]
            assert err["details"]["tool_selection"]["recommended"] == "run_custom_training"

    def test_check_gpu_available_positive_gpus_returns_none(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"data": {"gpu_total": 8}},
        ):
            assert _check_gpu_available() is None

    def test_check_gpu_available_handles_exception_gracefully(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            side_effect=Exception("cluster API unreachable"),
        ):
            assert _check_gpu_available() is None

    def test_validate_resources_per_node_rejects_bool(self):
        err = _validate_resources_per_node({"cpu": True})
        assert err is not None
        assert "must be a string or a number" in err.error


# ─── fine_tune parameter validation and helpers ─────────────────────────────


class TestFineTuneValidationAndConfigExtended:
    def test_validate_fine_tune_params_invalid_dtype(self):
        err = _validate_fine_tune_params(
            namespace=None,
            name=None,
            dtype="fp16",
            loss=None,
            dataset_source=None,
            batch_size=4,
            epochs=1,
            num_nodes=1,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=None,
            confirmed=True,
        )
        assert err is not None
        assert err["error_code"] == "VALIDATION_ERROR"
        assert "Must be 'bf16' or 'fp32'" in err["error"]

    def test_validate_fine_tune_params_invalid_loss(self):
        err = _validate_fine_tune_params(
            namespace=None,
            name=None,
            dtype="bf16",
            loss="INVALID_LOSS_NAME",
            dataset_source=None,
            batch_size=4,
            epochs=1,
            num_nodes=1,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=None,
            confirmed=True,
        )
        assert err is not None
        assert err["error_code"] == "VALIDATION_ERROR"
        assert "Invalid loss 'INVALID_LOSS_NAME'" in err["error"]

    def test_validate_fine_tune_params_invalid_dataset_source(self):
        err = _validate_fine_tune_params(
            namespace=None,
            name=None,
            dtype="bf16",
            loss=None,
            dataset_source="unknown_format",
            batch_size=4,
            epochs=1,
            num_nodes=1,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=None,
            confirmed=True,
        )
        assert err is not None
        assert err["error_code"] == "VALIDATION_ERROR"
        assert "Invalid dataset_source 'unknown_format'" in err["error"]

    def test_validate_fine_tune_params_when_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            err = _validate_fine_tune_params(
                namespace=None,
                name=None,
                dtype=None,
                loss=None,
                dataset_source=None,
                batch_size=4,
                epochs=1,
                num_nodes=1,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=None,
                confirmed=True,
            )
            assert err is not None
            assert err["error_code"] == "SDK_ERROR"
            assert "Required training libraries not available" in err["error"]

    def test_build_lora_cfg_extended_options(self):
        cfg = _build_lora_cfg(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            use_dora=True,
            quantize_base=True,
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_attn_modules=["q_proj", "v_proj"],
        )
        assert cfg.apply_lora_to_mlp is True
        assert cfg.apply_lora_to_output is True
        assert cfg.lora_attn_modules == ["q_proj", "v_proj"]

    def test_build_dataset_preprocess_cfg_all_none_returns_none(self):
        assert _build_dataset_preprocess_cfg(None, None, None, None, None) is None

    def test_build_dataset_preprocess_cfg_with_arguments(self):
        ds = _build_dataset_preprocess_cfg(
            dataset_source="json",
            dataset_split="train",
            dataset_train_on_input=True,
            dataset_system_prompt="system prompt",
            dataset_column_map={"prompt": "instruction"},
        )
        assert ds is not None
        assert ds.split == "train"
        assert ds.train_on_input is True
        assert ds.new_system_prompt == "system prompt"
        assert ds.column_map == {"prompt": "instruction"}

    def test_build_initializer_s3_with_credentials_and_region(self):
        init = _build_initializer(
            model="s3://my-bucket/model",
            dataset="s3://my-bucket/dataset",
            s3_endpoint="https://s3.custom.io",
            s3_access_key_id="KEY",
            s3_secret_access_key="SECRET",
            s3_region="us-west-2",
            s3_role_arn="arn:aws:iam::123:role/trainer",
        )
        assert init.model.storage_uri == "s3://my-bucket/model"
        assert init.dataset.storage_uri == "s3://my-bucket/dataset"
        assert init.model.endpoint == "https://s3.custom.io"
        assert init.model.region == "us-west-2"
        assert init.model.role_arn == "arn:aws:iam::123:role/trainer"

    def test_fine_tune_preview_includes_optional_fields(self):
        with patch(PATCH_GPU_CHECK, return_value=None):
            result = fine_tune(
                model="hf://org/model",
                dataset="hf://org/ds",
                name="fine-tune-preview",
                dtype="bf16",
                loss="CEWithChunkedOutputLoss",
                dataset_source="json",
                dataset_split="train",
                apply_lora_to_mlp=True,
                apply_lora_to_output=True,
                lora_attn_modules=["q_proj"],
                s3_endpoint="https://s3.local",
                confirmed=False,
            )
            assert result["status"] == "preview"
            cfg = result["config"]
            assert cfg["name"] == "fine-tune-preview"
            assert cfg["loss"] == "CEWithChunkedOutputLoss"
            assert cfg["dataset_source"] == "json"
            assert cfg["apply_lora_to_mlp"] is True
            assert cfg["s3_endpoint"] == "https://s3.local"


# ─── fine_tune submission & error handling ──────────────────────────────────


class TestFineTuneConfirmedExtended:
    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fine_tune_confirmed_with_name_and_all_options(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-named-job"
        mock_client_fn.return_value = mock_client

        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune",
            name="custom-named-job",
            dtype="bf16",
            loss="CEWithChunkedOutputLoss",
            resources_per_node={"cpu": 4},
            dataset_source="json",
            dataset_split="train",
            apply_lora_to_mlp=True,
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "custom-named-job"

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "custom-named-job"

        trainer = call_kwargs["trainer"]
        assert trainer.config.loss is not None
        assert trainer.config.dtype is not None
        assert trainer.config.dataset_preprocess_config is not None

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fine_tune_confirmed_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("SDK client backend connection failure")
        mock_client_fn.return_value = mock_client

        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "SDK_ERROR"
        assert "SDK client backend connection failure" in result["error"]
        assert "Read trainer://guides/troubleshooting" in result["hint"]


# ─── run_custom_training extended tests ─────────────────────────────────────


class TestRunCustomTrainingExtended:
    def test_run_custom_training_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            res = run_custom_training(script="print(1)", confirmed=True)
            assert res["success"] is False
            assert res["error_code"] == "SDK_ERROR"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_workspace_volume_injection(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            volumes=[{"name": "custom-data", "persistentVolumeClaim": {"claimName": "pvc"}}],
            volume_mounts=[{"name": "custom-data", "mountPath": "/data"}],
            confirmed=True,
        )
        assert res["success"] is True

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        runtime_patch = options[0]
        node_spec = runtime_patch.training_runtime_spec.template.spec.replicated_jobs[
            0
        ].template.spec.template.spec
        volume_names = [v["name"] for v in node_spec.volumes]
        assert "workspace" in volume_names
        assert "custom-data" in volume_names

        mount_paths = [m["mountPath"] for m in node_spec.containers[0].volume_mounts]
        assert "/workspace" in mount_paths
        assert "/data" in mount_paths

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_workspace_volume_already_present_no_duplicate(
        self, mock_client_fn, _ns
    ):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            volumes=[{"name": "my-workspace", "emptyDir": {}}],
            volume_mounts=[{"name": "my-workspace", "mountPath": "/workspace"}],
            confirmed=True,
        )
        assert res["success"] is True

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        runtime_patch = options[0]
        node_spec = runtime_patch.training_runtime_spec.template.spec.replicated_jobs[
            0
        ].template.spec.template.spec
        assert len(node_spec.volumes) == 1
        assert node_spec.volumes[0]["name"] == "my-workspace"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_with_explicit_name(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "explicit-train-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            name="explicit-train-job",
            confirmed=True,
        )
        assert res["success"] is True
        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "explicit-train-job"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("Kubernetes rejected custom job")
        mock_client_fn.return_value = mock_client

        res = run_custom_training(script="print('hello')", confirmed=True)
        assert res["success"] is False
        assert res["error_code"] == "SDK_ERROR"
        assert "Kubernetes rejected custom job" in res["error"]
        assert "Read trainer://guides/troubleshooting" in res["hint"]


# ─── run_container_training extended tests ──────────────────────────────────


class TestRunContainerTrainingExtended:
    def test_run_container_training_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            res = run_container_training(image="torch:latest", confirmed=True)
            assert res["success"] is False
            assert res["error_code"] == "SDK_ERROR"

    def test_run_container_training_bounds_error(self):
        res = run_container_training(image="torch:latest", num_nodes=9999)
        assert res["success"] is False
        assert res["error_code"] == "VALIDATION_ERROR"

    def test_run_container_training_resource_error(self):
        res = run_container_training(image="torch:latest", resources_per_node={"cpu": True})
        assert res["success"] is False
        assert res["error_code"] == "VALIDATION_ERROR"

    def test_run_container_training_preview_with_extra_configs(self):
        res = run_container_training(
            image="torch:latest",
            name="container-preview-test",
            labels={"team": "ml"},
            annotations={"env": "prod"},
            volumes=[{"name": "vol", "emptyDir": {}}],
            volume_mounts=[{"name": "vol", "mountPath": "/data"}],
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            affinity={"nodeAffinity": {}},
            service_account_name="trainer-sa",
            image_pull_secrets=[{"name": "regcred"}],
            confirmed=False,
        )
        assert res["status"] == "preview"
        cfg = res["config"]
        assert cfg["name"] == "container-preview-test"
        assert cfg["labels"] == {"team": "ml"}
        assert cfg["annotations"] == {"env": "prod"}
        assert cfg["node_selector"] == {"gpu": "true"}
        assert cfg["service_account_name"] == "trainer-sa"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_container_training_confirmed_with_name_and_options(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "named-container-job"
        mock_client_fn.return_value = mock_client

        res = run_container_training(
            image="torch:latest",
            name="named-container-job",
            labels={"team": "ml"},
            confirmed=True,
        )
        assert res["success"] is True
        assert res["data"]["job_name"] == "named-container-job"

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "named-container-job"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_container_training_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("Container runtime unavailable")
        mock_client_fn.return_value = mock_client

        res = run_container_training(image="torch:latest", confirmed=True)
        assert res["success"] is False
        assert res["error_code"] == "SDK_ERROR"
        assert "Container runtime unavailable" in res["error"]
        assert "Read trainer://guides/troubleshooting" in res["hint"]
