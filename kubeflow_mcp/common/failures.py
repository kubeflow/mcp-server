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

"""Failure-pattern detection for workload logs.

Shared by every client module that surfaces pod logs. Katib trials run as
TrainJobs, so trainer and optimizer classify the same failure signatures —
keeping the table here avoids one client module importing another's internals.
"""

import re

FAILURE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"CUDA out of memory", re.IGNORECASE),
        "OOM",
        "Reduce batch_size, enable quantization (int8/int4), or request a larger GPU.",
    ),
    (
        re.compile(r"OutOfMemoryError", re.IGNORECASE),
        "OOM",
        "Reduce batch_size, enable quantization (int8/int4), or request a larger GPU.",
    ),
    (
        re.compile(r"RuntimeError: CUDA error", re.IGNORECASE),
        "CUDA_ERROR",
        "Check GPU driver compatibility and CUDA version in the runtime image.",
    ),
    (
        re.compile(r"ModuleNotFoundError: No module named", re.IGNORECASE),
        "MISSING_MODULE",
        "Add the missing package to the 'packages' list.",
    ),
    (
        re.compile(r"ImportError", re.IGNORECASE),
        "IMPORT_ERROR",
        "Verify package versions; add missing package to 'packages'.",
    ),
    (
        re.compile(r"FileNotFoundError", re.IGNORECASE),
        "FILE_NOT_FOUND",
        "Check dataset/model paths and volume mounts.",
    ),
    # HF cache pattern MUST come before the generic PermissionError catch-all
    # so that /.cache/huggingface failures are classified correctly.
    (
        re.compile(
            r"Permission denied.*(?:huggingface|HF_HOME)|(?:huggingface|HF_HOME).*Permission denied",
            re.IGNORECASE,
        ),
        "HF_CACHE_WRITE_ERROR",
        "Set env var HF_HOME=/workspace to store HuggingFace cache on a writable volume mount.",
    ),
    (
        re.compile(
            r"PermissionError: \[Errno 13\] Permission denied: '/\.local|Permission denied: '/\.local",
            re.IGNORECASE,
        ),
        "OPENSHIFT_PIP_ERROR",
        "On OpenShift under a restricted SCC, pip install --user fails on read-only /.local. Do NOT use the 'packages' parameter in run_custom_training(). Instead, install packages inside your script to /workspace/lib using subprocess and append to sys.path. Read trainer://guides/platform-fixes for details.",
    ),
    (
        re.compile(r"PermissionError|Access Denied", re.IGNORECASE),
        "PERMISSION_ERROR",
        "Check service account permissions and storage credentials.",
    ),
    (
        re.compile(r"Connection(Error|Refused|Reset)", re.IGNORECASE),
        "NETWORK_ERROR",
        "Check network policies, DNS, and endpoint reachability.",
    ),
    (
        re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
        "PYTHON_EXCEPTION",
        "Review the traceback above for the root cause.",
    ),
]


def extract_failure_hint(logs: str) -> dict[str, str] | None:
    """Pattern-match common failure signatures and return an actionable hint."""
    for pattern, category, suggestion in FAILURE_PATTERNS:
        if pattern.search(logs):
            return {"category": category, "suggestion": suggestion}
    return None
