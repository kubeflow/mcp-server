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

"""Unit tests for log monitoring and failure pattern extraction."""

from kubeflow_mcp.trainer.api.monitoring import _extract_failure_hint


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


def test_extract_failure_hint_other_patterns():
    # Test CUDA OOM
    oom_logs = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    oom_hint = _extract_failure_hint(oom_logs)
    assert oom_hint is not None
    assert oom_hint["category"] == "OOM"

    # Test Missing Module
    module_logs = "ModuleNotFoundError: No module named 'peft'"
    module_hint = _extract_failure_hint(module_logs)
    assert module_hint is not None
    assert module_hint["category"] == "MISSING_MODULE"

    # Test no match
    clean_logs = "Training completed successfully. Epoch 5/5 finished."
    assert _extract_failure_hint(clean_logs) is None
