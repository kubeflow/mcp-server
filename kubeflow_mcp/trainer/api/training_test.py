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

"""Tests for trainer/api/training.py — fine_tune, run_custom_training, run_container_training.

Covers validation, preview (confirmed=False), and error paths.
Submission paths require mocking the Kubeflow SDK and are marked as TODOs.
"""

from __future__ import annotations

import ast

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


# TODO(test): test confirmed=True with mock SDK creates job
# TODO(test): test hf_token passed through to SDK
# TODO(test): test dataset parameter handling
# TODO(test): test LoRA config (lora_rank, lora_alpha, lora_dropout)
# TODO(test): test custom volumes and tolerations
# TODO(test): test S3 model/dataset sources
# TODO(test): test namespace policy enforcement
# TODO(test): test GPU validation against cluster resources


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


# TODO(test): test preview returns script + config without submitting
# TODO(test): test confirmed=True with mock SDK creates job
# TODO(test): test env parameter passed to training job
# TODO(test): test packages parameter installs dependencies
# TODO(test): test auto-generated name format
# TODO(test): test KUBEFLOW_MCP_UNSAFE_SCRIPTS=true override
# TODO(test): test namespace policy enforcement


# ─── run_container_training validation ──────────────────────────────────────


def test_run_container_training_invalid_name():
    result = run_container_training(
        image="ghcr.io/kubeflow/trainer/torch:latest",
        name="BadName",
    )
    assert result["success"] is False


# TODO(test): test preview returns config without submitting
# TODO(test): test confirmed=True with mock SDK creates job
# TODO(test): test command and args override
# TODO(test): test custom image configuration
# TODO(test): test namespace policy enforcement


# ─── _make_train_func ────────────────────────────────────────────────────────


def test_make_train_func_wraps_script():
    func = _make_train_func("x = 1 + 1\nprint(x)")
    assert callable(func)
    assert func.__name__ == "train"


def test_make_train_func_accepts_args():
    func = _make_train_func("print('hi')", func_args={"lr": 0.001})
    assert callable(func)


# TODO(test): test inspect.getsource works on returned function
# TODO(test): test syntax error in script raises SyntaxError
# TODO(test): test func_args appear as keyword parameters


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
