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

"""Tests for trainer/api/monitoring.py — log analysis, failure pattern detection.

Covers failure pattern regex matching (including OpenShift packages=/pip
/.local hints). K8s API interaction tests (get_training_logs,
get_training_events, wait_for_training) require mocking and are marked as TODOs.
"""

from __future__ import annotations

import pytest
from tests.common import TestCase

from kubeflow_mcp.trainer.api.monitoring import _FAILURE_PATTERNS, _extract_failure_hint


def test_extract_failure_hint_openshift_pip_error():
    # Test exact PermissionError trace
    logs = (
        "Installing collected packages: torch\n"
        "PermissionError: [Errno 13] Permission denied: '/.local'\n"
        "ERROR: Job failed"
    )
    hint = _extract_failure_hint(logs)
    assert hint is not None
    assert hint["category"] == "OPENSHIFT_PIP_ERROR"
    assert "On OpenShift under a restricted SCC" in hint["suggestion"]
    assert "Do NOT use the 'packages' parameter" in hint["suggestion"]

    # Test generic permission denied on /.local
    logs_generic = "Permission denied: '/.local/bin/pip'"
    hint_generic = _extract_failure_hint(logs_generic)
    assert hint_generic is not None
    assert hint_generic["category"] == "OPENSHIFT_PIP_ERROR"


def test_extract_failure_hint_generic_permission_error():
    # Test a generic permission error does not trigger OpenShift pip error
    logs = "PermissionError: [Errno 13] Permission denied: '/workspace/data.csv'"
    hint = _extract_failure_hint(logs)
    assert hint is not None
    assert hint["category"] == "PERMISSION_ERROR"
    assert "Check service account permissions" in hint["suggestion"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="CUDA OOM detected",
            config={
                "log_line": "CUDA out of memory. Tried to allocate 2.00 GiB",
                "expected_category": "OOM",
            },
        ),
        TestCase(
            name="CUDA runtime error detected",
            config={
                "log_line": "RuntimeError: CUDA error: device-side assert triggered",
                "expected_category": "CUDA_ERROR",
            },
        ),
        TestCase(
            name="missing module detected",
            config={
                "log_line": "ModuleNotFoundError: No module named 'torchtune'",
                "expected_category": "MISSING_MODULE",
            },
        ),
        TestCase(
            name="import error detected",
            config={
                "log_line": "ImportError: cannot import name 'Foo'",
                "expected_category": "IMPORT_ERROR",
            },
        ),
        TestCase(
            name="file not found detected",
            config={
                "log_line": "FileNotFoundError: [Errno 2] No such file",
                "expected_category": "FILE_NOT_FOUND",
            },
        ),
        TestCase(
            name="torch OOM detected",
            config={
                "log_line": "torch.cuda.OutOfMemoryError: allocator",
                "expected_category": "OOM",
            },
        ),
    ],
)
def test_failure_pattern_matches(test_case):
    log_line = test_case.config["log_line"]
    expected = test_case.config["expected_category"]
    matched = False
    for pattern, category, hint in _FAILURE_PATTERNS:
        if pattern.search(log_line):
            assert category == expected
            assert len(hint) > 0
            matched = True
            break
    assert matched, f"No pattern matched: {log_line}"


def test_normal_log_not_matched():
    normal = "Epoch 1/10: loss=2.34, lr=0.001"
    for pattern, _category, _hint in _FAILURE_PATTERNS:
        assert not pattern.search(normal)
    assert _extract_failure_hint("Training completed successfully. Epoch 5/5 finished.") is None


# TODO(test): test get_training_logs returns truncated output
# TODO(test): test get_training_logs attaches failure hints
# TODO(test): test get_training_logs with invalid job name
# TODO(test): test get_training_logs namespace policy enforcement
# TODO(test): test get_training_events returns K8s events
# TODO(test): test get_training_events with invalid job name
# TODO(test): test wait_for_training returns on completion
# TODO(test): test wait_for_training times out
# TODO(test): test wait_for_training returns on failure
