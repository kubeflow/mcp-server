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

"""Unit tests for optimizer discovery tools (mocked OptimizerClient)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.optimizer.api import discovery
from kubeflow_mcp.optimizer.types import trial_counts_from_cr

_DISC = "kubeflow_mcp.optimizer.api.discovery"


def _metric(name="accuracy", latest="0.95"):
    return SimpleNamespace(name=name, min="0.5", max=latest, latest=latest)


def _trial(name, status="Complete", params=None, metrics=None):
    trainjob = SimpleNamespace(name=f"{name}-job", status=status)
    return SimpleNamespace(
        name=name,
        parameters=params or {"lr": "0.01"},
        trainjob=trainjob,
        metrics=metrics if metrics is not None else [_metric()],
    )


def _job(name="exp-1", status="Running", trials=None):
    return SimpleNamespace(
        name=name,
        status=status,
        creation_timestamp=datetime(2026, 7, 21, tzinfo=timezone.utc),
        trials=trials if trials is not None else [],
        objectives=[SimpleNamespace(metric="accuracy", direction="maximize")],
        algorithm=SimpleNamespace(random_state=42),
        search_space={},
        trial_config=SimpleNamespace(num_trials=10, parallel_trials=2, max_failed_trials=3),
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


# ─── list_experiments ──────────────────────────────────────────────────────


def test_list_experiments_happy_path():
    client = MagicMock()
    client.list_jobs.return_value = [_job("a", "Running"), _job("b", "Complete")]
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.list_experiments()
    assert result["success"] is True
    assert result["data"]["total"] == 2
    names = {e["name"] for e in result["data"]["experiments"]}
    assert names == {"a", "b"}


def test_list_experiments_status_filter():
    client = MagicMock()
    client.list_jobs.return_value = [_job("a", "Running"), _job("b", "Complete")]
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.list_experiments(status="Complete")
    assert result["data"]["total"] == 1
    assert result["data"]["experiments"][0]["name"] == "b"


def test_list_experiments_sdk_error():
    client = MagicMock()
    client.list_jobs.side_effect = RuntimeError("boom")
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.list_experiments()
    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


# ─── get_experiment ────────────────────────────────────────────────────────


def test_get_experiment_happy_path():
    client = MagicMock()
    client.get_job.return_value = _job(
        "exp-1", "Running", trials=[_trial("t1"), _trial("t2", status="Failed")]
    )
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_experiment("exp-1")
    assert result["success"] is True
    data = result["data"]
    assert data["name"] == "exp-1"
    assert data["total_trials"] == 2
    assert data["succeeded_trials"] == 1
    assert data["failed_trials"] == 1
    assert len(data["trials"]) == 2
    assert "trial_config" in data


def test_get_experiment_not_found():
    client = MagicMock()
    client.get_job.side_effect = _not_found()
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_experiment("missing")
    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_get_experiment_invalid_name():
    result = discovery.get_experiment("Invalid_Name!")
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


# ─── get_experiment_status ─────────────────────────────────────────────────


def _status_cr(**status):
    """Experiment CR carrying only status, as get_experiment_status reads it."""
    return {"metadata": {"name": "exp-1"}, "spec": {}, "status": status}


def _patch_cr(cr):
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = cr
    return (
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    )


def test_get_experiment_status_counts():
    cr = _status_cr(trials=3, trialsRunning=1, trialsSucceeded=1, trialsFailed=1)
    api_p, ns_p = _patch_cr(cr)
    with api_p, ns_p:
        data = discovery.get_experiment_status("exp-1")["data"]
    assert data["status"] == "Running"
    assert data["total_trials"] == 3
    assert data["running_trials"] == 1
    assert data["succeeded_trials"] == 1
    assert data["failed_trials"] == 1


def test_get_experiment_status_reads_the_cr_not_the_sdk():
    """It must not call get_job(), which also resolves every trial's TrainJob."""
    client = MagicMock()
    api_p, ns_p = _patch_cr(_status_cr(trials=1))
    with api_p, ns_p, patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        discovery.get_experiment_status("exp-1")
    client.get_job.assert_not_called()


def test_get_experiment_status_surfaces_early_stopped_trials():
    """The SDK's trial view has no EarlyStopped state; the CR does."""
    cr = _status_cr(trials=4, trialsSucceeded=2, trialsEarlyStopped=2)
    api_p, ns_p = _patch_cr(cr)
    with api_p, ns_p:
        data = discovery.get_experiment_status("exp-1")["data"]
    assert data["early_stopped_trials"] == 2


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"conditions": [{"type": "Succeeded", "status": "True"}]}, "Complete"),
        ({"conditions": [{"type": "Failed", "status": "True"}]}, "Failed"),
        ({"conditions": [{"type": "Succeeded", "status": "False"}]}, "Created"),
        ({"trialsRunning": 2}, "Running"),
        ({}, "Created"),
    ],
)
def test_get_experiment_status_derives_phase_like_the_sdk(status, expected):
    """Mirrors the SDK's own derivation from Experiment conditions."""
    api_p, ns_p = _patch_cr(_status_cr(**status))
    with api_p, ns_p:
        data = discovery.get_experiment_status("exp-1")["data"]
    assert data["status"] == expected


def test_get_experiment_and_status_report_identical_counts():
    """Regression: the two tools previously disagreed, because one derived
    counts from TrainJob statuses and the other read the CR."""
    cr = _experiment_cr()
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1", trials=[_trial("t1")])))
    api_p, ns_p = _patch_cr(cr)
    with api_p, ns_p, patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        full = discovery.get_experiment("exp-1")["data"]
        light = discovery.get_experiment_status("exp-1")["data"]

    shared = set(trial_counts_from_cr(cr["status"]))
    assert shared, "fixture should carry counters"
    for key in shared:
        assert full[key] == light[key], f"{key}: {full[key]} != {light[key]}"


def test_get_experiment_status_not_found():
    client = MagicMock()
    client.get_job.side_effect = _not_found()
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_experiment_status("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_trial ─────────────────────────────────────────────────────────────


def test_get_trial_found():
    client = MagicMock()
    client.get_job.return_value = _job("exp-1", trials=[_trial("t1"), _trial("t2")])
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_trial("t2", experiment="exp-1")
    assert result["success"] is True
    assert result["data"]["name"] == "t2"
    assert result["data"]["experiment"] == "exp-1"


def test_get_trial_missing_trial():
    client = MagicMock()
    client.get_job.return_value = _job("exp-1", trials=[_trial("t1")])
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_trial("nope", experiment="exp-1")
    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_get_trial_missing_experiment():
    client = MagicMock()
    client.get_job.side_effect = _not_found()
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_trial("t1", experiment="missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_successful_trials ─────────────────────────────────────────────────


def test_get_successful_trials_filters():
    client = MagicMock()
    client.get_job.return_value = _job(
        "exp-1",
        trials=[
            _trial("t1", "Complete"),
            _trial("t2", "Failed"),
            _trial("t3", "Complete"),
        ],
    )
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_successful_trials("exp-1")
    assert result["data"]["total"] == 2
    assert {t["name"] for t in result["data"]["trials"]} == {"t1", "t3"}


# ─── list_suggestions ──────────────────────────────────────────────────────


def test_list_suggestions_happy_path():
    api = MagicMock()
    api.list_namespaced_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": "exp-1"},
                "spec": {"algorithm": {"algorithmName": "random"}, "requests": 5},
                "status": {"suggestionCount": 5, "conditions": [{"type": "Succeeded"}]},
            }
        ]
    }
    with (
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = discovery.list_suggestions()
    assert result["success"] is True
    assert result["data"]["total"] == 1
    sugg = result["data"]["suggestions"][0]
    assert sugg["algorithm"] == "random"
    assert sugg["condition"] == "Succeeded"


def test_list_suggestions_sdk_error():
    api = MagicMock()
    api.list_namespaced_custom_object.side_effect = RuntimeError("boom")
    with (
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = discovery.list_suggestions()
    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


# ─── get_experiment: Experiment CR enrichment ──────────────────────────────
# OptimizationJob omits conditions, the current optimal trial and the
# early-stopping spec, so get_experiment reads them from the CR. Field names
# below follow kubeflow_katib_api v0.19.0 (camelCase wire format).


def _experiment_cr(**status):
    base = {
        "trials": 6,
        "trialsSucceeded": 3,
        "trialsFailed": 1,
        "trialsRunning": 1,
        "trialsEarlyStopped": 1,
        "conditions": [
            {
                "type": "Failed",
                "status": "True",
                "reason": "ExperimentFailed",
                "message": "max failed trials exceeded",
                "lastTransitionTime": "2026-08-10T00:00:00Z",
            }
        ],
        "currentOptimalTrial": {
            "bestTrialName": "exp-1-abc",
            "parameterAssignments": [{"name": "lr", "value": "0.03"}],
            "observation": {
                "metrics": [{"name": "accuracy", "latest": "0.93", "max": "0.93", "min": "0.41"}]
            },
        },
    }
    base.update(status)
    return {
        "metadata": {"name": "exp-1"},
        "spec": {
            "earlyStopping": {
                "algorithmName": "medianstop",
                "algorithmSettings": [{"name": "min_trials_required", "value": "3"}],
            }
        },
        "status": base,
    }


def _with_cr(cr):
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = cr
    return api


def test_get_experiment_surfaces_conditions():
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1", "Failed")))
    with (
        patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_DISC}.get_custom_objects_api", return_value=_with_cr(_experiment_cr())),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        data = discovery.get_experiment("exp-1")["data"]

    assert data["conditions"][0]["type"] == "Failed"
    assert data["conditions"][0]["reason"] == "ExperimentFailed"
    assert "max failed trials" in data["conditions"][0]["message"]


def test_get_experiment_surfaces_best_trial_and_early_stopping():
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1")))
    with (
        patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_DISC}.get_custom_objects_api", return_value=_with_cr(_experiment_cr())),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        data = discovery.get_experiment("exp-1")["data"]

    assert data["best_trial"]["name"] == "exp-1-abc"
    assert data["best_trial"]["parameters"] == {"lr": "0.03"}
    assert data["best_trial"]["metrics"][0]["name"] == "accuracy"
    assert data["early_stopping"]["algorithm"] == "medianstop"
    assert data["early_stopping"]["settings"]["min_trials_required"] == "3"


def test_cr_counts_supersede_sdk_counts_and_include_early_stopped():
    """The SDK derives counts from TrainJob status, which has no EarlyStopped
    state — the CR is authoritative."""
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1", trials=[_trial("t1")])))
    with (
        patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_DISC}.get_custom_objects_api", return_value=_with_cr(_experiment_cr())),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        data = discovery.get_experiment("exp-1")["data"]

    assert data["total_trials"] == 6  # CR value, not the 1 embedded trial
    assert data["succeeded_trials"] == 3
    assert data["early_stopped_trials"] == 1


def test_get_experiment_reports_no_best_trial_before_metrics_exist():
    cr = _experiment_cr(currentOptimalTrial={"parameterAssignments": []})
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1")))
    with (
        patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_DISC}.get_custom_objects_api", return_value=_with_cr(cr)),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        data = discovery.get_experiment("exp-1")["data"]
    assert data["best_trial"] is None


def test_get_experiment_degrades_when_cr_read_fails():
    """The CR read is supplementary — losing it must not fail the whole call."""
    client = MagicMock(get_job=MagicMock(return_value=_job("exp-1")))
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = RuntimeError("forbidden")
    with (
        patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="kubeflow"),
    ):
        result = discovery.get_experiment("exp-1")

    assert result["success"] is True
    assert result["data"]["detail_unavailable"] is True
    assert result["data"]["name"] == "exp-1"
