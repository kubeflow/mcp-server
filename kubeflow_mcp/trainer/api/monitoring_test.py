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

"""Tests for trainer/api/monitoring.py — logs, events, wait, failure hints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from tests.common import TestCase

from kubeflow_mcp.trainer.api.monitoring import (
    _FAILURE_PATTERNS,
    MAX_LOG_LINES,
    _extract_failure_hint,
    _is_pod_for_step,
    get_training_events,
    get_training_logs,
    wait_for_training,
)


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
        TestCase(
            name="network connection error detected",
            config={
                "log_line": "ConnectionError: HTTPSConnectionPool(host='huggingface.co')",
                "expected_category": "NETWORK_ERROR",
            },
        ),
        TestCase(
            name="python traceback detected",
            config={
                "log_line": "Traceback (most recent call last):\n  File 'train.py', line 42",
                "expected_category": "PYTHON_EXCEPTION",
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


def test_extract_failure_hint_suggestion_text():
    cuda = _extract_failure_hint("RuntimeError: CUDA error: device-side assert triggered")
    assert cuda is not None
    assert "GPU driver compatibility" in cuda["suggestion"]

    import_err = _extract_failure_hint("ImportError: cannot import name 'AutoModel'")
    assert import_err is not None
    assert "package versions" in import_err["suggestion"]

    missing = _extract_failure_hint("FileNotFoundError: [Errno 2] No such file")
    assert missing is not None
    assert "dataset/model paths" in missing["suggestion"]

    network = _extract_failure_hint("ConnectionRefused: could not connect to master")
    assert network is not None
    assert network["category"] == "NETWORK_ERROR"
    assert "network policies" in network["suggestion"]


# ─── _is_pod_for_step (helper) ─────────────────────────────────────────


def _make_pod(labels=None):
    return SimpleNamespace(metadata=SimpleNamespace(labels=labels))


class TestIsPodForStep:
    def test_matches_replicated_job_name(self):
        pod = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "node-0"})
        assert _is_pod_for_step(pod, "node-0") is True

    def test_matches_replicated_job_with_index(self):
        pod = _make_pod(
            {
                "jobset.sigs.k8s.io/replicatedjob-name": "worker",
                "jobset.sigs.k8s.io/job-index": "2",
            }
        )
        assert _is_pod_for_step(pod, "worker-2") is True

    def test_no_match(self):
        pod = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "node-1"})
        assert _is_pod_for_step(pod, "node-0") is False

    def test_no_labels(self):
        pod = _make_pod(None)
        assert _is_pod_for_step(pod, "node-0") is False


# ─── get_training_logs (tool-level) ──────────────────────────────────────

PATCH_CLIENT = "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace"
PATCH_NS_CHECK = "kubeflow_mcp.trainer.api.monitoring.check_namespace_allowed"
PATCH_EFF_NS = "kubeflow_mcp.trainer.api.monitoring.get_trainer_effective_namespace"
PATCH_CORE_V1 = "kubeflow_mcp.trainer.api.monitoring.get_core_v1_api"
PATCH_VALIDATE_NAME = "kubeflow_mcp.trainer.api.monitoring.validate_k8s_name"


def _make_mock_client(**method_returns):
    client = MagicMock()
    for method, value in method_returns.items():
        if isinstance(value, Exception):
            getattr(client, method).side_effect = value
        else:
            getattr(client, method).return_value = value
    return client


class TestGetTrainingLogs:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_success(self, mock_client_fn, _ns):
        mock_client_fn.return_value = _make_mock_client(
            get_job_logs=["epoch 1/3 loss=0.5", "epoch 2/3 loss=0.3", "epoch 3/3 loss=0.1"]
        )
        result = get_training_logs("my-job", step="node-0")
        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["step"] == "node-0"
        assert "epoch 1/3" in result["data"]["logs"]
        assert result["data"]["lines"] >= 3

    @patch(PATCH_NS_CHECK, return_value=None)
    def test_follow_true_returns_early(self, _ns):
        result = get_training_logs("my-job", follow=True)
        assert result["success"] is True
        assert "Streaming not supported" in result["data"]["logs"]

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_failure_hint_injected(self, mock_client_fn, _ns):
        mock_client_fn.return_value = _make_mock_client(
            get_job_logs=["RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB"]
        )
        result = get_training_logs("oom-job")
        assert result["success"] is True
        assert result["data"]["failure_hint"]["category"] == "OOM"
        assert len(result["data"]["next_steps"]) == 2

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_log_truncation(self, mock_client_fn, _ns):
        lines = [f"line {i}" for i in range(MAX_LOG_LINES + 500)]
        mock_client_fn.return_value = _make_mock_client(get_job_logs=lines)
        result = get_training_logs("big-job")
        assert result["success"] is True
        assert "line 499" not in result["data"]["logs"]
        assert f"line {MAX_LOG_LINES + 499}" in result["data"]["logs"]
        assert result["data"]["lines"] == MAX_LOG_LINES

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_log_truncation_with_generator(self, mock_client_fn, _ns):
        def _log_generator():
            for i in range(MAX_LOG_LINES + 500):
                yield f"g{i}"

        mock_client_fn.return_value = _make_mock_client(get_job_logs=_log_generator())
        result = get_training_logs("big-generator-job")
        assert result["success"] is True
        assert "g499\n" not in result["data"]["logs"]
        assert f"g{MAX_LOG_LINES + 499}" in result["data"]["logs"]
        assert result["data"]["lines"] == MAX_LOG_LINES

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_not_found_error(self, mock_client_fn, _ns):
        from kubernetes.client.exceptions import ApiException

        mock_client_fn.return_value = _make_mock_client(
            get_job_logs=ApiException(status=404, reason="Not Found")
        )
        result = get_training_logs("missing-job")
        assert result["success"] is False
        assert result["error_code"] == "RESOURCE_NOT_FOUND"
        assert "missing-job" in result["error"]

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_generic_sdk_error(self, mock_client_fn, _ns):
        mock_client_fn.return_value = _make_mock_client(
            get_job_logs=RuntimeError("connection reset")
        )
        result = get_training_logs("err-job")
        assert result["success"] is False
        assert result["error_code"] == "SDK_ERROR"

    @patch(PATCH_NS_CHECK)
    def test_namespace_denied(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = get_training_logs("my-job", namespace="forbidden-ns")
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"

    @patch(PATCH_VALIDATE_NAME, return_value=None)
    @patch(PATCH_CORE_V1)
    @patch(PATCH_EFF_NS, return_value="default")
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fallback_to_previous_pod_logs(
        self, mock_client_fn, _ns, _eff_ns, mock_v1_fn, _validate
    ):
        mock_client_fn.return_value = _make_mock_client(get_job_logs=[])
        pod = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "node-0"})
        pod.metadata.name = "crash-job-node-0-0"
        mock_v1 = MagicMock()
        mock_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
        mock_v1.read_namespaced_pod_log.return_value = "previous crash output"
        mock_v1_fn.return_value = mock_v1
        result = get_training_logs("crash-job", step="node-0")
        assert result["success"] is True
        assert "previous crash output" in result["data"]["logs"]
        mock_v1.read_namespaced_pod_log.assert_called_once_with(
            name="crash-job-node-0-0",
            namespace="default",
            previous=True,
            tail_lines=MAX_LOG_LINES,
        )

    @patch(PATCH_VALIDATE_NAME, return_value=None)
    @patch(PATCH_CORE_V1)
    @patch(PATCH_EFF_NS, return_value="default")
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fallback_pod_log_read_error_is_swallowed(
        self, mock_client_fn, _ns, _eff_ns, mock_v1_fn, _validate
    ):
        mock_client_fn.return_value = _make_mock_client(get_job_logs=[])
        pod = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "node-0"})
        pod.metadata.name = "crash-job-node-0-0"
        mock_v1 = MagicMock()
        mock_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[pod])
        mock_v1.read_namespaced_pod_log.side_effect = RuntimeError("container not found")
        mock_v1_fn.return_value = mock_v1
        result = get_training_logs("crash-job", step="node-0")
        assert result["success"] is True
        assert result["data"]["logs"] == ""

    @patch(PATCH_VALIDATE_NAME)
    @patch(PATCH_EFF_NS, return_value="default")
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fallback_invalid_name_returns_error(self, mock_client_fn, _ns, _eff_ns, mock_validate):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_client_fn.return_value = _make_mock_client(get_job_logs=[])
        mock_validate.return_value = ToolErrorModel(
            error="invalid name", error_code="VALIDATION_ERROR"
        )
        result = get_training_logs("bad..name", step="node-0")
        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch(PATCH_VALIDATE_NAME, return_value=None)
    @patch(PATCH_CORE_V1)
    @patch(PATCH_EFF_NS, return_value="default")
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fallback_skips_non_matching_pods(
        self, mock_client_fn, _ns, _eff_ns, mock_v1_fn, _validate
    ):
        mock_client_fn.return_value = _make_mock_client(get_job_logs=[])
        non_matching = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "worker"})
        non_matching.metadata.name = "job-worker-0"
        matching = _make_pod({"jobset.sigs.k8s.io/replicatedjob-name": "node-0"})
        matching.metadata.name = "job-node-0-0"
        mock_v1 = MagicMock()
        mock_v1.list_namespaced_pod.return_value = SimpleNamespace(items=[non_matching, matching])
        mock_v1.read_namespaced_pod_log.return_value = "recovered logs"
        mock_v1_fn.return_value = mock_v1
        result = get_training_logs("my-job", step="node-0")
        assert result["success"] is True
        assert "recovered logs" in result["data"]["logs"]
        mock_v1.read_namespaced_pod_log.assert_called_once()

    @patch(PATCH_EFF_NS, side_effect=RuntimeError("kubeconfig missing"))
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fallback_outer_error_is_swallowed(self, mock_client_fn, _ns, _eff_ns):
        mock_client_fn.return_value = _make_mock_client(get_job_logs=[])
        result = get_training_logs("crash-job", step="node-0")
        assert result["success"] is True
        assert result["data"]["logs"] == ""


# ─── get_training_events (tool-level) ────────────────────────────────────


class TestGetTrainingEvents:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_success_with_events(self, mock_client_fn, _ns):
        from datetime import datetime, timezone

        event = SimpleNamespace(
            involved_object_kind="Pod",
            involved_object_name="my-job-node-0",
            reason="Scheduled",
            message="Successfully assigned",
            event_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        mock_client_fn.return_value = _make_mock_client(get_job_events=[event])
        result = get_training_events("my-job")
        assert result["success"] is True
        assert result["data"]["job"] == "my-job"
        assert result["data"]["total"] == 1
        assert result["data"]["events"][0]["reason"] == "Scheduled"
        assert "2026" in result["data"]["events"][0]["event_time"]

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_event_missing_attributes(self, mock_client_fn, _ns):
        event = SimpleNamespace(reason="Pulled", message="Image pulled")
        mock_client_fn.return_value = _make_mock_client(get_job_events=[event])
        result = get_training_events("my-job")
        assert result["success"] is True
        parsed = result["data"]["events"][0]
        assert parsed["involved_object_kind"] == ""
        assert parsed["involved_object_name"] == ""
        assert parsed["event_time"] == ""

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_event_time_without_isoformat(self, mock_client_fn, _ns):
        event = SimpleNamespace(
            involved_object_kind="Pod",
            involved_object_name="pod-0",
            reason="Started",
            message="Started container",
            event_time="2026-01-01T00:00:00Z",
        )
        mock_client_fn.return_value = _make_mock_client(get_job_events=[event])
        result = get_training_events("my-job")
        assert result["data"]["events"][0]["event_time"] == "2026-01-01T00:00:00Z"

    @patch(PATCH_NS_CHECK, return_value=None)
    def test_limit_less_than_one(self, _ns):
        result = get_training_events("my-job", limit=0)
        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_limit_capped_at_max(self, mock_client_fn, _ns):
        events = [
            SimpleNamespace(reason=f"r{i}", message=f"m{i}", event_time=None) for i in range(600)
        ]
        mock_client_fn.return_value = _make_mock_client(get_job_events=events)
        result = get_training_events("my-job", limit=1000)
        assert result["success"] is True
        assert len(result["data"]["events"]) == 500
        assert result["data"]["total"] == 600

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_sdk_error(self, mock_client_fn, _ns):
        mock_client_fn.return_value = _make_mock_client(
            get_job_events=RuntimeError("api unavailable")
        )
        result = get_training_events("err-job")
        assert result["success"] is False
        assert result["error_code"] == "SDK_ERROR"

    @patch(PATCH_NS_CHECK)
    def test_namespace_denied(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = get_training_events("my-job", namespace="forbidden-ns")
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"


# ─── wait_for_training (tool-level) ──────────────────────────────────────


class TestWaitForTraining:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_success(self, mock_client_fn, _ns):
        job = SimpleNamespace(status="Complete")
        mock_client_fn.return_value = _make_mock_client(wait_for_job_status=job)
        result = wait_for_training("my-job")
        assert result["success"] is True
        assert result["data"]["reached"] is True
        assert result["data"]["status"] == "Complete"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_timeout(self, mock_client_fn, _ns):
        client = MagicMock()
        client.wait_for_job_status.side_effect = TimeoutError("timed out")
        mock_client_fn.return_value = client
        result = wait_for_training("my-job", timeout_seconds=10)
        assert result["success"] is True
        assert result["data"]["reached"] is False
        assert "Timeout" in result["data"]["message"]

    @patch(PATCH_NS_CHECK, return_value=None)
    def test_timeout_seconds_less_than_one(self, _ns):
        result = wait_for_training("my-job", timeout_seconds=0)
        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch(PATCH_NS_CHECK, return_value=None)
    def test_polling_interval_less_than_min(self, _ns):
        result = wait_for_training("my-job", polling_interval=0)
        assert result["success"] is False
        assert result["error_code"] == "VALIDATION_ERROR"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_succeeded_alias(self, mock_client_fn, _ns):
        job = SimpleNamespace(status="Complete")
        client = MagicMock()
        client.wait_for_job_status.return_value = job
        mock_client_fn.return_value = client
        wait_for_training("my-job", target_statuses="Succeeded")
        call_kwargs = client.wait_for_job_status.call_args
        assert "Complete" in call_kwargs.kwargs["status"]

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_target_statuses_list(self, mock_client_fn, _ns):
        job = SimpleNamespace(status="Failed")
        client = MagicMock()
        client.wait_for_job_status.return_value = job
        mock_client_fn.return_value = client
        result = wait_for_training("my-job", target_statuses=["Complete", "Failed"])
        assert result["success"] is True
        call_kwargs = client.wait_for_job_status.call_args
        assert call_kwargs.kwargs["status"] == {"Complete", "Failed"}

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_generic_sdk_error(self, mock_client_fn, _ns):
        mock_client_fn.return_value = _make_mock_client(wait_for_job_status=RuntimeError("broken"))
        result = wait_for_training("err-job")
        assert result["success"] is False
        assert result["error_code"] == "SDK_ERROR"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_job_without_status_attribute(self, mock_client_fn, _ns):
        job = SimpleNamespace()
        mock_client_fn.return_value = _make_mock_client(wait_for_job_status=job)
        result = wait_for_training("my-job")
        assert result["success"] is True
        assert result["data"]["status"] == "Unknown"
        assert result["data"]["reached"] is True

    @patch(PATCH_NS_CHECK)
    def test_namespace_denied(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = wait_for_training("my-job", namespace="forbidden-ns")
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"
