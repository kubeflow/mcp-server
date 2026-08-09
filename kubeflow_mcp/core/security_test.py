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

"""Tests for core/security.py — validation, AST safety, masking."""

from __future__ import annotations

import pytest
from tests.common import FAILED, SUCCESS, TestCase, assert_test_case

from kubeflow_mcp.core.security import (
    is_safe_python_code,
    mask_sensitive_data,
    truncate_log_output,
    validate_k8s_name,
    validate_namespace,
    validate_resource_limits,
    validate_training_bounds,
)

# ─── K8s name validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid lowercase name",
            expected_status=SUCCESS,
            config={"name": "train-gemma-abc"},
        ),
        TestCase(
            name="empty name rejected",
            expected_status=FAILED,
            config={"name": ""},
        ),
        TestCase(
            name="too long name rejected",
            expected_status=FAILED,
            config={"name": "a" * 64},
        ),
        TestCase(
            name="uppercase rejected",
            expected_status=FAILED,
            config={"name": "TrainJob"},
        ),
        TestCase(
            name="starts with hyphen rejected",
            expected_status=FAILED,
            config={"name": "-bad-name"},
        ),
        TestCase(
            name="path traversal attempt rejected",
            expected_status=FAILED,
            config={"name": "../../etc"},
        ),
        # TODO(test): test single character name "a"
        # TODO(test): test max length name (63 chars)
        # TODO(test): test ends with hyphen rejected
    ],
)
def test_validate_k8s_name(test_case):
    assert_test_case(test_case, validate_k8s_name)


def test_validate_k8s_name_custom_field():
    err = validate_k8s_name("", field="runtime")
    assert "runtime" in err.error


def test_validate_namespace_delegates():
    assert validate_namespace("default") is None
    err = validate_namespace("Bad_NS")
    assert err is not None


# TODO(test): test check_namespace_allowed with policy allowing namespace
# TODO(test): test check_namespace_allowed with policy denying namespace
# TODO(test): test check_namespace_allowed with None resolving to default
# TODO(test): test check_namespace_allowed fail-closed when resolution errors


# ─── Resource limits validation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid cpu millis",
            expected_status=SUCCESS,
            config={"cpu": "100m", "memory": None, "gpu": None},
        ),
        TestCase(
            name="valid cpu cores",
            expected_status=SUCCESS,
            config={"cpu": "4", "memory": None, "gpu": None},
        ),
        TestCase(
            name="valid memory",
            expected_status=SUCCESS,
            config={"cpu": None, "memory": "256Mi", "gpu": None},
        ),
        TestCase(
            name="all None is valid",
            expected_status=SUCCESS,
            config={"cpu": None, "memory": None, "gpu": None},
        ),
        TestCase(
            name="invalid cpu format",
            expected_status=FAILED,
            config={"cpu": "4cores", "memory": None, "gpu": None},
        ),
        TestCase(
            name="invalid memory format",
            expected_status=FAILED,
            config={"cpu": None, "memory": "256MB", "gpu": None},
        ),
        TestCase(
            name="negative gpu rejected",
            expected_status=FAILED,
            config={"cpu": None, "memory": None, "gpu": -1},
        ),
    ],
)
def test_validate_resource_limits(test_case):
    assert_test_case(test_case, validate_resource_limits)


# ─── Training bounds validation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid bounds",
            expected_status=SUCCESS,
            config={"batch_size": 32, "epochs": 10},
        ),
        TestCase(
            name="batch_size too large",
            expected_status=FAILED,
            config={"batch_size": 2048},
        ),
        TestCase(
            name="batch_size zero",
            expected_status=FAILED,
            config={"batch_size": 0},
        ),
        TestCase(
            name="epochs too large",
            expected_status=FAILED,
            config={"epochs": 5000},
        ),
        TestCase(
            name="lora_dropout out of range",
            expected_status=FAILED,
            config={"lora_dropout": 1.5},
        ),
        TestCase(
            name="empty script rejected",
            expected_status=FAILED,
            config={"script": "   "},
        ),
        TestCase(
            name="too many packages",
            expected_status=FAILED,
            config={"packages": ["pkg"] * 51},
        ),
        TestCase(
            name="script too large",
            expected_status=FAILED,
            config={"script": "x" * 1_000_001},
        ),
        # TODO(test): test exact boundary values (batch_size=1, batch_size=1024)
        # TODO(test): test lora_rank boundary (1 and 256)
        # TODO(test): test num_nodes boundary (1 and 100)
        # TODO(test): test gpu_per_node boundary (0 and 16)
    ],
)
def test_validate_training_bounds(test_case):
    result = validate_training_bounds(**test_case.config)
    if test_case.expected_status == SUCCESS:
        assert result is None
    else:
        assert result is not None


# ─── Script safety (AST) ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="safe ML script",
            expected_status=SUCCESS,
            config={
                "code": (
                    "import torch\n"
                    "from transformers import AutoModelForCausalLM\n"
                    'model = AutoModelForCausalLM.from_pretrained("bert-base")\n'
                )
            },
        ),
        TestCase(
            name="detects eval",
            expected_status=FAILED,
            config={"code": "eval('1+1')"},
            expected_output="eval",
        ),
        TestCase(
            name="detects exec",
            expected_status=FAILED,
            config={"code": "exec('import os')"},
            expected_output="exec",
        ),
        TestCase(
            name="detects os.system",
            expected_status=FAILED,
            config={"code": "import os\nos.system('rm -rf /')"},
            expected_output="os.system",
        ),
        TestCase(
            name="detects subprocess",
            expected_status=FAILED,
            config={"code": "import subprocess\nsubprocess.run(['ls'])"},
            expected_output="subprocess",
        ),
        TestCase(
            name="detects ctypes import",
            expected_status=FAILED,
            config={"code": "import ctypes"},
            expected_output="ctypes",
        ),
        TestCase(
            name="detects dunder access",
            expected_status=FAILED,
            config={"code": "x.__builtins__"},
            expected_output="__builtins__",
        ),
        TestCase(
            name="syntax error caught",
            expected_status=FAILED,
            config={"code": "def (invalid"},
            expected_output="Syntax error",
        ),
        # TODO(test): test __import__ direct call
        # TODO(test): test compile() call
        # TODO(test): test shutil.rmtree detection
        # TODO(test): test socket import detection
        # TODO(test): test from ctypes import detection (ImportFrom)
        # TODO(test): test __subclasses__ dunder access
        # TODO(test): bypass via getattr is NOT caught (document limitation)
    ],
)
def test_is_safe_python_code(test_case):
    safe, reason = is_safe_python_code(test_case.config["code"])
    if test_case.expected_status == SUCCESS:
        assert safe is True
        assert reason == "OK"
    else:
        assert safe is False
        if test_case.expected_output:
            assert test_case.expected_output in reason


# ─── Sensitive data masking ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="masks hf_token",
            config={"data": {"hf_token": "hf_abc123"}, "key": "hf_token", "expected": "***"},
        ),
        TestCase(
            name="masks password substring",
            config={"data": {"db_password": "secret123"}, "key": "db_password", "expected": "***"},
        ),
        TestCase(
            name="preserves public_key (safe key)",
            config={
                "data": {"public_key": "ssh-rsa ..."},
                "key": "public_key",
                "expected": "ssh-rsa ...",
            },
        ),
        TestCase(
            name="preserves keyword (safe key)",
            config={"data": {"keyword": "train"}, "key": "keyword", "expected": "train"},
        ),
        TestCase(
            name="preserves normal keys",
            config={"data": {"model": "gemma"}, "key": "model", "expected": "gemma"},
        ),
        TestCase(
            name="masks _token suffix",
            config={"data": {"auth_token": "tok123"}, "key": "auth_token", "expected": "***"},
        ),
        TestCase(
            name="masks _key suffix",
            config={"data": {"api_key": "key123"}, "key": "api_key", "expected": "***"},
        ),
        # TODO(test): test all _SENSITIVE_EXACT keys are masked
        # TODO(test): test all _SENSITIVE_SUBSTRINGS are masked
        # TODO(test): test all _SAFE_KEYS are preserved
        # TODO(test): test BufferingHandler redaction patterns
    ],
)
def test_mask_sensitive_data(test_case):
    result = mask_sensitive_data(test_case.config["data"])
    assert result[test_case.config["key"]] == test_case.config["expected"]


def test_mask_sensitive_data_recurses_into_nested_dicts():
    result = mask_sensitive_data({"config": {"secret_key": "x"}})
    assert result["config"]["secret_key"] == "***"


def test_mask_sensitive_data_recurses_into_lists():
    result = mask_sensitive_data({"items": [{"hf_token": "x"}, {"model": "y"}]})
    assert result["items"][0]["hf_token"] == "***"
    assert result["items"][1]["model"] == "y"


def test_mask_sensitive_data_masks_sensitive_exact_fields() -> None:
    """Regression from main: bare token / api_token / S3 secret keys must mask."""
    data = {
        "model": "llama",
        "token": "ghp_xxxxxxxxxxxx",
        "hf_token": "hf_secret",
        "access_token": "at_secret",
        "secret_access_key": "sak_secret",
        "s3_secret_access_key": "s3_secret",
        "api_token": "api_secret",
    }
    assert mask_sensitive_data(data) == {
        "model": "llama",
        "token": "***",
        "hf_token": "***",
        "access_token": "***",
        "secret_access_key": "***",
        "s3_secret_access_key": "***",
        "api_token": "***",
    }


# ─── Log truncation ──────────────────────────────────────────────────────────


def test_truncate_short_output_unchanged():
    assert truncate_log_output("hello") == "hello"


def test_truncate_long_output():
    result = truncate_log_output("x" * 20000, max_length=100)
    assert len(result) < 20000
    assert "truncated" in result


# TODO(test): test exact boundary (max_length characters)
