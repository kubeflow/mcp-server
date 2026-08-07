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
from unittest.mock import MagicMock, patch

import pytest
from tests.common import FAILED, PREVIEW, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.trainer.api.training import (
    _build_fine_tune_config,
    _forwarded_train_args,
    _has_required_params,
    _has_train_call,
    _make_train_func,
    _should_apply_hf_dataset_workaround,
    _uncalled_train_call,
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

    def test_non_torchtune_runtime_rejected(self):
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
