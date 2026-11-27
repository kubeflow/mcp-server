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

"""Tests for common/failures.py — workload log failure signature classification."""

from __future__ import annotations

import pytest

from kubeflow_mcp.common.failures import FAILURE_PATTERNS, extract_failure_hint


@pytest.mark.parametrize(
    ("log_text", "expected_category"),
    [
        ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "OOM"),
        ("OutOfMemoryError: CUDA backend out of memory", "OOM"),
        ("RuntimeError: CUDA error: invalid device ordinal", "CUDA_ERROR"),
        ("ModuleNotFoundError: No module named 'transformers'", "MISSING_MODULE"),
        ("ImportError: cannot import name 'AutoModelForCausalLM'", "IMPORT_ERROR"),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: '/models/weights.bin'",
            "FILE_NOT_FOUND",
        ),
        ("Permission denied: '/root/.cache/huggingface/hub/models'", "HF_CACHE_WRITE_ERROR"),
        ("HF_HOME directory cannot be written: Permission denied", "HF_CACHE_WRITE_ERROR"),
        ("PermissionError: [Errno 13] Permission denied: '/.local/bin'", "OPENSHIFT_PIP_ERROR"),
        ("Permission denied: '/.local/lib/python3.11/site-packages'", "OPENSHIFT_PIP_ERROR"),
        ("PermissionError: [Errno 13] Permission denied: '/etc/secret'", "PERMISSION_ERROR"),
        ("Access Denied to S3 bucket my-training-bucket", "PERMISSION_ERROR"),
        ("ConnectionError: Failed to establish a new connection", "NETWORK_ERROR"),
        ("ConnectionRefusedError: [Errno 111] Connection refused", "NETWORK_ERROR"),
        ("ConnectionResetError: [Errno 104] Connection reset by peer", "NETWORK_ERROR"),
        ("Traceback (most recent call last):\n  File 'train.py', line 12", "PYTHON_EXCEPTION"),
    ],
)
def test_extract_failure_hint_detects_categories(log_text: str, expected_category: str):
    hint = extract_failure_hint(log_text)
    assert hint is not None
    assert hint["category"] == expected_category
    assert len(hint["suggestion"]) > 0


def test_extract_failure_hint_returns_none_when_no_pattern_matches():
    clean_logs = "Epoch 1/10 complete. Loss: 0.423. Accuracy: 92.1%."
    assert extract_failure_hint(clean_logs) is None


def test_hf_cache_error_takes_precedence_over_generic_permission_error():
    log_text = "PermissionError: [Errno 13] Permission denied: '/root/.cache/huggingface'"
    hint = extract_failure_hint(log_text)
    assert hint is not None
    assert hint["category"] == "HF_CACHE_WRITE_ERROR"


def test_openshift_pip_error_takes_precedence_over_generic_permission_error():
    log_text = "PermissionError: [Errno 13] Permission denied: '/.local/lib'"
    hint = extract_failure_hint(log_text)
    assert hint is not None
    assert hint["category"] == "OPENSHIFT_PIP_ERROR"


def test_failure_patterns_are_compiled_regex_tuples():
    assert len(FAILURE_PATTERNS) > 0
    for pattern, category, suggestion in FAILURE_PATTERNS:
        assert hasattr(pattern, "search")
        assert isinstance(category, str) and category
        assert isinstance(suggestion, str) and suggestion
