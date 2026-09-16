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

"""Unit tests for optimizer monitoring tools (mocked OptimizerClient)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.optimizer.api import monitoring

_MON = "kubeflow_mcp.optimizer.api.monitoring"


def _metric(name="accuracy", latest="0.95"):
    return SimpleNamespace(name=name, min="0.5", max=latest, latest=latest)


def _trial(name, status="Complete"):
    return SimpleNamespace(
        name=name,
        parameters={"lr": "0.01"},
        trainjob=SimpleNamespace(name=f"{name}-job", status=status),
        metrics=[_metric()],
    )


def _job(name="exp-1", status="Running", trials=None):
    return SimpleNamespace(
        name=name,
        status=status,
        creation_timestamp=None,
        trials=trials if trials is not None else [],
        objectives=[],
        algorithm=SimpleNamespace(random_state=None),
        search_space={},
        trial_config=SimpleNamespace(num_trials=10, parallel_trials=2, max_failed_trials=None),
    )


def _not_found():
    """Build the exception shape the real SDK raises for a missing resource.

    The kubernetes backend wraps a 404 as ``raise RuntimeError(...) from
    ApiException(404)`` — is_k8s_not_found must detect it via ``__cause__``.
    """
    cause = ApiException(status=404, reason="Not Found")
    err = RuntimeError("Failed to get OptimizationJob: default/missing")
    err.__cause__ = cause
    return err


# ─── get_experiment_trials ─────────────────────────────────────────────────


def test_get_experiment_trials_happy_path():
    client = MagicMock()
    client.get_job.return_value = _job(trials=[_trial("t1"), _trial("t2", "Failed")])
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trials("exp-1")
    assert result["success"] is True
    assert result["data"]["total"] == 2
    assert result["data"]["summary"]["failed_trials"] == 1


def test_get_experiment_trials_status_filter():
    client = MagicMock()
    client.get_job.return_value = _job(trials=[_trial("t1", "Complete"), _trial("t2", "Failed")])
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trials("exp-1", status="Failed")
    assert result["data"]["total"] == 1
    assert result["data"]["trials"][0]["name"] == "t2"


def test_get_experiment_trials_not_found():
    client = MagicMock()
    client.get_job.side_effect = _not_found()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trials("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_best_trial ────────────────────────────────────────────────────────


def test_get_best_trial_happy_path():
    client = MagicMock()
    client.get_best_results.return_value = SimpleNamespace(
        parameters={"lr": "0.01"}, metrics=[_metric()]
    )
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_best_trial("exp-1")
    assert result["success"] is True
    assert result["data"]["parameters"] == {"lr": "0.01"}
    assert result["data"]["metrics"][0]["name"] == "accuracy"


def test_get_best_trial_no_result():
    client = MagicMock()
    client.get_best_results.return_value = None
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_best_trial("exp-1")
    assert result["success"] is True
    assert result["data"]["best_trial"] is None


def test_get_best_trial_not_found():
    client = MagicMock()
    client.get_best_results.side_effect = _not_found()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_best_trial("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_suggestion ────────────────────────────────────────────────────────


def test_get_suggestion_happy_path():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = {
        "spec": {"algorithm": {"algorithmName": "random"}, "requests": 3},
        "status": {
            "suggestionCount": 3,
            "conditions": [{"type": "Running", "status": "True", "reason": "Ready"}],
        },
    }
    with (
        patch(f"{_MON}.get_custom_objects_api", return_value=api),
        patch(f"{_MON}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = monitoring.get_suggestion("exp-1")
    assert result["success"] is True
    assert result["data"]["algorithm"] == "random"
    assert result["data"]["conditions"][0]["type"] == "Running"


def test_get_suggestion_not_found():
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = _not_found()
    with (
        patch(f"{_MON}.get_custom_objects_api", return_value=api),
        patch(f"{_MON}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = monitoring.get_suggestion("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── wait_for_experiment ───────────────────────────────────────────────────


def test_wait_for_experiment_reaches_terminal():
    client = MagicMock()
    client.wait_for_job_status.return_value = _job(status="Complete", trials=[_trial("t1")])
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.wait_for_experiment("exp-1", timeout_seconds=30)
    assert result["data"]["reached"] is True
    assert result["data"]["status"] == "Complete"
    # status set passed to SDK is the terminal set
    _, kwargs = client.wait_for_job_status.call_args
    assert kwargs["status"] == {"Complete", "Failed"}


def test_wait_for_experiment_caps_timeout():
    client = MagicMock()
    client.wait_for_job_status.return_value = _job(status="Complete")
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        monitoring.wait_for_experiment("exp-1", timeout_seconds=99999, polling_interval=15)
    _, kwargs = client.wait_for_job_status.call_args
    assert kwargs["timeout"] == monitoring.MAX_WAIT_TIMEOUT
    assert kwargs["polling_interval"] == 15


def test_wait_for_experiment_rejects_too_frequent_polling():
    """Matches wait_for_training: too low is an error, not a silent adjustment.

    Clamping upward would poll slower than the caller asked without saying so.
    """
    client = MagicMock()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.wait_for_experiment("exp-1", polling_interval=1)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    client.wait_for_job_status.assert_not_called()


def test_wait_for_experiment_timeout():
    client = MagicMock()
    client.wait_for_job_status.side_effect = TimeoutError()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.wait_for_experiment("exp-1", timeout_seconds=5)
    assert result["success"] is True
    assert result["data"]["reached"] is False


def test_wait_for_experiment_invalid_timeout():
    result = monitoring.wait_for_experiment("exp-1", timeout_seconds=0)
    assert result["error_code"] == "VALIDATION_ERROR"


# ─── get_experiment_trial_logs ─────────────────────────────────────────────


def test_get_experiment_trial_logs_happy_path():
    client = MagicMock()
    client.get_job_logs.return_value = iter(["epoch 1", "epoch 2", "done"])
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trial_logs("exp-1", trial="t1")
    assert result["success"] is True
    assert "epoch 1" in result["data"]["logs"]
    _, kwargs = client.get_job_logs.call_args
    assert kwargs["trial_name"] == "t1"
    assert kwargs["follow"] is False


def test_get_experiment_trial_logs_failure_hint():
    client = MagicMock()
    client.get_job_logs.return_value = iter(
        ["Traceback (most recent call last):", "RuntimeError: CUDA out of memory"]
    )
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trial_logs("exp-1")
    assert result["data"]["failure_hint"]["category"] == "OOM"


def test_get_experiment_trial_logs_not_found():
    client = MagicMock()
    client.get_job_logs.side_effect = _not_found()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trial_logs("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_experiment_events ─────────────────────────────────────────────────


def test_get_experiment_events_happy_path():
    client = MagicMock()
    client.get_job_events.return_value = [
        SimpleNamespace(
            involved_object_kind="Pod",
            involved_object_name="exp-1-trial-0",
            reason="Scheduled",
            message="assigned to node",
            event_time=None,
        )
    ]
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_events("exp-1")
    assert result["success"] is True
    assert result["data"]["total"] == 1
    assert result["data"]["events"][0]["reason"] == "Scheduled"


def test_get_experiment_events_invalid_limit():
    result = monitoring.get_experiment_events("exp-1", limit=0)
    assert result["error_code"] == "VALIDATION_ERROR"


def test_get_experiment_events_not_found():
    client = MagicMock()
    client.get_job_events.side_effect = _not_found()
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_events("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_get_experiment_trials_accepts_the_succeeded_alias():
    """Katib's CRD vocabulary says "Succeeded"; the SDK reports "Complete".
    Both must select the same trials, as they do for list_experiments."""
    client = MagicMock()
    client.get_job.return_value = _job(trials=[_trial("t1", "Complete"), _trial("t2", "Failed")])
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        aliased = monitoring.get_experiment_trials("exp-1", status="Succeeded")
        native = monitoring.get_experiment_trials("exp-1", status="Complete")

    assert aliased["data"]["total"] == 1
    assert aliased["data"]["trials"][0]["name"] == "t1"
    assert aliased["data"]["trials"] == native["data"]["trials"]


def test_get_suggestion_not_found_names_the_right_kind():
    """It must not report a missing Suggestion as a missing Experiment."""
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = _not_found()
    with (
        patch(f"{_MON}.get_custom_objects_api", return_value=api),
        patch(f"{_MON}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        result = monitoring.get_suggestion("exp-1")

    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Suggestion 'exp-1' not found" in result["error"]
    assert "experiment starts running" in result["hint"]
