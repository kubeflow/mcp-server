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

"""Unit tests for the optimizer serializers (Katib custom resources -> dicts)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kubeflow_mcp.optimizer import types as ser
from kubeflow_mcp.optimizer.conftest import experiment_cr, metric, trial_cr


@pytest.mark.parametrize(
    "state",
    ["Created", "Running", "Complete", "Failed", "Killed", "EarlyStopped", "MetricsUnavailable"],
)
def test_trial_status_reads_the_trial_conditions(state):
    assert ser.trial_status(trial_cr("t", state)) == state


def test_trial_status_prefers_the_terminal_condition():
    """Katib keeps Running set alongside a later terminal condition."""
    trial = trial_cr("t", "Running")
    trial["status"]["conditions"].append({"type": "Succeeded", "status": "True"})
    assert ser.trial_status(trial) == "Complete"


def test_a_trial_without_a_true_condition_is_pending_as_katib_counts_it():
    """Katib's status summary counts every unmatched trial as pending, including
    a freshly created one with no conditions yet."""
    trial = trial_cr("t", "Running")
    trial["status"]["conditions"] = [{"type": "Running", "status": "False"}]
    assert ser.trial_status(trial) == "Created"
    trial["status"]["conditions"] = []
    assert ser.trial_status(trial) == "Created"
    assert ser.trial_status({}) == "Created"


@pytest.mark.parametrize(
    ("conditions", "state"),
    [
        (["Running", "MetricsUnavailable"], "Running"),
        (["Succeeded", "Killed"], "Killed"),
        (["Failed", "EarlyStopped"], "Failed"),
    ],
)
def test_trial_status_follows_katibs_order_when_conditions_overlap(conditions, state):
    trial = trial_cr("t", "Created")
    trial["status"]["conditions"] = [{"type": c, "status": "True"} for c in conditions]
    assert ser.trial_status(trial) == state


@pytest.mark.parametrize(("kind", "trainjob"), [("TrainJob", "t1"), ("Job", None)])
def test_trial_to_dict(kind, trainjob):
    data = ser.trial_to_dict(
        trial_cr("t1", "Complete", kind=kind, params={"lr": "0.02"}, metrics=[metric()])
    )
    assert data == {
        "name": "t1",
        "status": "Complete",
        "parameters": {"lr": "0.02"},
        "metrics": [{"name": "accuracy", "min": "0.5", "max": "0.95", "latest": "0.95"}],
        "trainjob": trainjob,
        "workload_kind": kind,
        "creation_timestamp": "2026-11-01T00:00:01Z",
    }


def test_trial_counts_from_trials_match_the_cr_counter_keys():
    trials = [
        ser.trial_to_dict(trial_cr(n, s))
        for n, s in [
            ("a", "Complete"),
            ("b", "Running"),
            ("c", "Created"),
            ("d", "Killed"),
            ("e", "MetricsUnavailable"),
        ]
    ]
    counts = ser.trial_counts_from_trials(trials)
    assert counts == {
        "total_trials": 5,
        "running_trials": 1,
        "succeeded_trials": 1,
        "failed_trials": 0,
        "pending_trials": 1,
        "killed_trials": 1,
        "early_stopped_trials": 0,
        "metrics_unavailable_trials": 1,
    }
    cr_keys = set(ser.trial_counts_from_cr({f: 1 for f in ser._CR_TRIAL_COUNTERS}))
    assert set(counts) == cr_keys


def test_experiment_to_dict_covers_spec_and_status():
    cr = experiment_cr(
        algorithm="tpe",
        max_failed=3,
        status={"trials": 2, "trialsSucceeded": 2},
    )
    cr["spec"]["algorithm"]["algorithmSettings"] = [{"name": "n_startup_trials", "value": "5"}]
    cr["spec"]["objective"]["goal"] = 0.99
    cr["spec"]["parameters"].append(
        {"name": "opt", "parameterType": "categorical", "feasibleSpace": {"list": ["sgd", "adam"]}}
    )

    data = ser.experiment_to_dict(cr)

    assert data["name"] == "exp-1"
    assert data["status"] == "Created"
    assert data["objectives"] == [{"metric": "accuracy", "direction": "maximize", "goal": 0.99}]
    assert data["algorithm"] == {"name": "tpe", "settings": {"n_startup_trials": "5"}}
    assert data["search_space"]["opt"] == {"type": "categorical", "choices": ["sgd", "adam"]}
    assert data["trial_config"] == {"num_trials": 10, "parallel_trials": 2, "max_failed_trials": 3}
    assert data["succeeded_trials"] == 2
    assert data["best_trial"] is None
    assert data["early_stopping"] is None


def test_event_to_dict_prefers_the_latest_timestamp():
    when = datetime(2026, 11, 1, tzinfo=timezone.utc)
    ev = SimpleNamespace(
        type="Warning",
        involved_object=SimpleNamespace(kind="Pod", name="p"),
        reason="BackOff",
        message="Back-off restarting",
        last_timestamp=when,
        event_time=None,
        first_timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    assert ser.event_to_dict(ev) == {
        "type": "Warning",
        "involved_object_kind": "Pod",
        "involved_object_name": "p",
        "reason": "BackOff",
        "message": "Back-off restarting",
        "event_time": when.isoformat(),
    }


def test_event_to_dict_uses_the_latest_occurrence_of_an_event_series():
    """An events.k8s.io/v1 series keeps its first occurrence in eventTime, so
    sorting on that would drop a warning that is still recurring."""
    first, latest = (datetime(2026, 11, d, tzinfo=timezone.utc) for d in (1, 3))
    ev = SimpleNamespace(
        last_timestamp=None,
        event_time=first,
        first_timestamp=None,
        series=SimpleNamespace(last_observed_time=latest, count=7),
    )
    assert ser.event_to_dict(ev)["event_time"] == latest.isoformat()


def test_event_to_dict_tolerates_missing_fields():
    assert ser.event_to_dict(SimpleNamespace())["event_time"] == ""
