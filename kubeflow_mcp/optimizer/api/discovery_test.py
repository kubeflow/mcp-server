# Copyright The Kubeflow Authors.
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

"""Unit tests for optimizer discovery tools, against a fake Katib cluster."""

import pytest

from kubeflow_mcp.optimizer.api import discovery
from kubeflow_mcp.optimizer.api.optimization import ALGORITHMS
from kubeflow_mcp.optimizer.conftest import experiment_cr, metric, trial_cr

_COMPLETE = [{"type": "Succeeded", "status": "True", "reason": "ExperimentGoalReached"}]
_FAILED = [
    {
        "type": "Failed",
        "status": "True",
        "reason": "ExperimentFailed",
        "message": "max failed trials exceeded",
        "lastTransitionTime": "2026-08-10T00:00:00Z",
    }
]
_OPTIMAL = {
    "bestTrialName": "exp-1-abc",
    "parameterAssignments": [{"name": "lr", "value": "0.03"}],
    "observation": {
        "metrics": [{"name": "accuracy", "latest": "0.93", "max": "0.93", "min": "0.41"}]
    },
}


# ─── list_hpo_experiments ──────────────────────────────────────────────────


def test_list_experiments_happy_path(katib):
    katib.add_experiment(experiment_cr("a", status={"trials": 1, "trialsRunning": 1}))
    katib.add_experiment(experiment_cr("b", conditions=_COMPLETE, status={"trials": 1}))
    result = discovery.list_hpo_experiments()
    assert result["success"] is True
    assert result["data"]["total"] == 2
    assert {e["name"] for e in result["data"]["experiments"]} == {"a", "b"}


def test_list_experiments_reads_the_cr_in_one_call(katib):
    """Listing must cost one API call however many experiments and trials exist."""
    for i in range(30):
        katib.add_experiment(experiment_cr(f"exp-{i}", status={"trials": 50}))
    result = discovery.list_hpo_experiments()
    assert result["data"]["total"] == 30
    assert [c[0] for c in katib.custom.calls] == ["list"]


def test_list_experiments_reports_counters_even_when_katib_omits_them(katib):
    """A fresh experiment has no counters on its CR, but the keys must persist."""
    katib.add_experiment(experiment_cr("fresh"))
    exp = discovery.list_hpo_experiments()["data"]["experiments"][0]
    assert exp["status"] == "Created"
    for key in ("total_trials", "running_trials", "succeeded_trials", "failed_trials"):
        assert exp[key] == 0


@pytest.mark.parametrize("wanted", ["Complete", "Succeeded"])
def test_list_experiments_status_filter_and_alias(katib, wanted):
    katib.add_experiment(experiment_cr("a", status={"trialsRunning": 1}))
    katib.add_experiment(experiment_cr("b", conditions=_COMPLETE))
    result = discovery.list_hpo_experiments(status=wanted)
    assert [e["name"] for e in result["data"]["experiments"]] == ["b"]


def test_list_experiments_api_error(katib, monkeypatch):
    def boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(katib.custom, "list_namespaced_custom_object", boom)
    result = discovery.list_hpo_experiments()
    assert result["error_code"] == "SDK_ERROR"


# ─── get_hpo_experiment ────────────────────────────────────────────────────


def test_get_experiment_reads_spec_status_and_trials(katib):
    katib.add_experiment(
        experiment_cr(
            conditions=_FAILED,
            status={
                "trials": 2,
                "trialsSucceeded": 1,
                "trialsRunning": 1,
                "trialsEarlyStopped": 0,
                "currentOptimalTrial": _OPTIMAL,
            },
            earlyStopping={
                "algorithmName": "medianstop",
                "algorithmSettings": [{"name": "min_trials_required", "value": "3"}],
            },
        )
    )
    katib.add_trials(trial_cr("t1", "Complete", metrics=[metric()]), trial_cr("t2", "Running"))

    data = discovery.get_hpo_experiment("exp-1")["data"]

    assert data["status"] == "Failed"
    assert data["objectives"] == [{"metric": "accuracy", "direction": "maximize"}]
    assert data["algorithm"] == {"name": "random", "settings": {}}
    assert data["search_space"]["lr"] == {
        "type": "double",
        "min": "0.001",
        "max": "0.1",
        "distribution": "logUniform",
    }
    assert data["trial_config"] == {
        "num_trials": 10,
        "parallel_trials": 2,
        "max_failed_trials": None,
    }
    assert [t["name"] for t in data["trials"]] == ["t1", "t2"]
    assert data["trials"][0]["metrics"][0]["latest"] == "0.95"
    assert data["conditions"][0]["reason"] == "ExperimentFailed"
    assert data["best_trial"] == {
        "name": "exp-1-abc",
        "parameters": {"lr": "0.03"},
        "metrics": [{"name": "accuracy", "min": "0.41", "max": "0.93", "latest": "0.93"}],
    }
    assert data["early_stopping"] == {
        "algorithm": "medianstop",
        "settings": {"min_trials_required": "3"},
    }
    assert data["succeeded_trials"] == 1
    assert data["trials_truncated"] is False


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_get_experiment_works_for_every_advertised_algorithm(katib, algorithm):
    """Regression: SDK 0.5's reader rejects everything but random and grid.

    A Bayesian experiment that completed in cluster testing could not be read.
    """
    katib.add_experiment(experiment_cr(algorithm=algorithm, conditions=_COMPLETE))
    katib.add_trials(trial_cr("t1", "Complete", metrics=[metric()]))
    result = discovery.get_hpo_experiment("exp-1")
    assert result["success"] is True, result
    assert result["data"]["algorithm"]["name"] == algorithm


def test_get_experiment_describes_job_trials(katib):
    """Regression: SDK 0.5 resolves every trial as a TrainJob and reported a
    batch/v1 Job trial's experiment as missing."""
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Complete", kind="Job", metrics=[metric()]))
    trial = discovery.get_hpo_experiment("exp-1")["data"]["trials"][0]
    assert trial["status"] == "Complete"
    assert trial["workload_kind"] == "Job"
    assert trial["trainjob"] is None


def test_get_experiment_truncates_embedded_trials(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(*(trial_cr(f"t{i}", created=f"2026-11-01T00:00:{i:02d}Z") for i in range(5)))
    data = discovery.get_hpo_experiment("exp-1", limit=2)["data"]
    assert [t["name"] for t in data["trials"]] == ["t0", "t1"]
    assert data["trials_truncated"] is True


def test_get_experiment_reports_no_best_trial_before_metrics_exist(katib):
    katib.add_experiment(experiment_cr(status={"currentOptimalTrial": {"bestTrialName": ""}}))
    assert discovery.get_hpo_experiment("exp-1")["data"]["best_trial"] is None


def test_get_experiment_not_found(katib):
    result = discovery.get_hpo_experiment("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_get_experiment_invalid_name(katib):
    result = discovery.get_hpo_experiment("Invalid_Name")
    assert result["error_code"] == "VALIDATION_ERROR"
    assert katib.custom.calls == []


# ─── get_hpo_experiment_status ─────────────────────────────────────────────


def test_get_experiment_status_counts_include_early_stopped(katib):
    katib.add_experiment(
        experiment_cr(status={"trials": 3, "trialsSucceeded": 2, "trialsEarlyStopped": 1})
    )
    data = discovery.get_hpo_experiment_status("exp-1")["data"]
    assert data["total_trials"] == 3
    assert data["succeeded_trials"] == 2
    assert data["early_stopped_trials"] == 1
    assert [c[0] for c in katib.custom.calls] == ["get"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"conditions": _COMPLETE}, "Complete"),
        ({"conditions": _FAILED}, "Failed"),
        ({"conditions": [{"type": "Running", "status": "True"}]}, "Created"),
        ({"trialsRunning": 1}, "Running"),
        ({}, "Created"),
    ],
)
def test_get_experiment_status_derives_phase_like_the_sdk(katib, status, expected):
    # Copy rather than pop: the parametrized dict is shared by every run of the case.
    counters = {k: v for k, v in status.items() if k != "conditions"}
    katib.add_experiment(experiment_cr(conditions=status.get("conditions"), status=counters))
    assert discovery.get_hpo_experiment_status("exp-1")["data"]["status"] == expected


def test_get_experiment_and_status_report_identical_counts(katib):
    katib.add_experiment(
        experiment_cr(status={"trials": 4, "trialsSucceeded": 2, "trialsEarlyStopped": 2})
    )
    full = discovery.get_hpo_experiment("exp-1")["data"]
    status = discovery.get_hpo_experiment_status("exp-1")["data"]
    for key, value in status.items():
        if key != "name":
            assert full[key] == value, key


def test_get_experiment_status_not_found(katib):
    assert discovery.get_hpo_experiment_status("missing")["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_trial / get_successful_trials ─────────────────────────────────────


def test_get_trial_found(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Complete", params={"lr": "0.02"}), trial_cr("t2"))
    data = discovery.get_trial("exp-1", "t1")["data"]
    assert data["name"] == "t1"
    assert data["experiment"] == "exp-1"
    assert data["parameters"] == {"lr": "0.02"}


def test_get_trial_only_sees_its_own_experiment(katib):
    katib.add_experiment(experiment_cr())
    katib.add_experiment(experiment_cr("exp-2"))
    katib.add_trials(trial_cr("t1", experiment="exp-2"))
    assert discovery.get_trial("exp-1", "t1")["error_code"] == "RESOURCE_NOT_FOUND"


def test_get_trial_missing_trial(katib):
    katib.add_experiment(experiment_cr())
    result = discovery.get_trial("exp-1", "nope")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Trial 'nope'" in result["error"]


def test_get_trial_missing_experiment(katib):
    result = discovery.get_trial("missing", "t1")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Experiment 'missing'" in result["error"]


def test_get_successful_trials_returns_only_succeeded(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(
        trial_cr("ok", "Complete", metrics=[metric()]),
        trial_cr("stopped", "EarlyStopped", metrics=[metric()]),
        trial_cr("bad", "Failed"),
        trial_cr("live", "Running"),
    )
    data = discovery.get_successful_trials("exp-1")["data"]
    assert [t["name"] for t in data["trials"]] == ["ok"]
    assert data["total"] == 1


def test_get_successful_trials_missing_experiment(katib):
    result = discovery.get_successful_trials("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── list_suggestions ──────────────────────────────────────────────────────


def test_list_suggestions_happy_path(katib):
    katib.custom.suggestions["exp-1"] = {
        "metadata": {"name": "exp-1"},
        "spec": {"algorithm": {"algorithmName": "random"}, "requests": 5},
        "status": {"suggestionCount": 5, "conditions": [{"type": "Succeeded"}]},
    }
    data = discovery.list_suggestions()["data"]
    assert data["total"] == 1
    assert data["suggestions"][0]["algorithm"] == "random"
    assert [c["type"] for c in data["suggestions"][0]["conditions"]] == ["Succeeded"]


def test_list_suggestions_api_error(katib, monkeypatch):
    def boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(katib.custom, "list_namespaced_custom_object", boom)
    assert discovery.list_suggestions()["error_code"] == "SDK_ERROR"
