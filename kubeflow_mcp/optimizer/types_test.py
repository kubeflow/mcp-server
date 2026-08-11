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

"""Unit tests for optimizer SDK→dict serializers."""

from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace

from kubeflow_mcp.optimizer import types as ser


class _Direction(Enum):
    MAXIMIZE = "maximize"


def _metric(name="acc", latest="0.9"):
    return SimpleNamespace(name=name, min="0.5", max="0.9", latest=latest)


def _trial(name, status="Complete"):
    return SimpleNamespace(
        name=name,
        parameters={"lr": "0.01"},
        trainjob=SimpleNamespace(name=f"{name}-job", status=status),
        metrics=[_metric()],
    )


def test_metric_to_dict():
    assert ser.metric_to_dict(_metric()) == {
        "name": "acc",
        "min": "0.5",
        "max": "0.9",
        "latest": "0.9",
    }


def test_trial_status_from_trainjob():
    assert ser.trial_status(_trial("t1", "Running")) == "Running"
    # Missing trainjob → Unknown, not a crash.
    assert ser.trial_status(SimpleNamespace(trainjob=None)) == "Unknown"


def test_trial_to_dict():
    d = ser.trial_to_dict(_trial("t1", "Complete"))
    assert d["name"] == "t1"
    assert d["status"] == "Complete"
    assert d["parameters"] == {"lr": "0.01"}
    assert d["trainjob"] == "t1-job"
    assert d["metrics"][0]["name"] == "acc"


def test_objective_to_dict_unwraps_enum():
    obj = SimpleNamespace(metric="accuracy", direction=_Direction.MAXIMIZE)
    assert ser.objective_to_dict(obj) == {"metric": "accuracy", "direction": "maximize"}


def test_search_space_continuous():
    space = SimpleNamespace(min=0.001, max=0.1, distribution=SimpleNamespace(value="uniform"))
    out = ser.search_space_entry_to_dict(space)
    assert out == {"type": "continuous", "min": 0.001, "max": 0.1, "distribution": "uniform"}


def test_search_space_categorical():
    space = SimpleNamespace(choices=[16, 32, 64])
    assert ser.search_space_entry_to_dict(space) == {
        "type": "categorical",
        "choices": [16, 32, 64],
    }


def test_search_space_to_dict_map():
    spaces = {"lr": SimpleNamespace(choices=[1, 2])}
    assert ser.search_space_to_dict(spaces) == {"lr": {"type": "categorical", "choices": [1, 2]}}
    assert ser.search_space_to_dict(None) == {}


def test_algorithm_to_dict():
    assert ser.algorithm_to_dict(SimpleNamespace(random_state=42)) == {
        "name": "SimpleNamespace",
        "random_state": 42,
    }
    # random_state None is omitted.
    assert ser.algorithm_to_dict(SimpleNamespace(random_state=None)) == {"name": "SimpleNamespace"}
    assert ser.algorithm_to_dict(None) == {}


def test_trial_counts():
    job = SimpleNamespace(
        trials=[
            _trial("a", "Complete"),
            _trial("b", "Running"),
            _trial("c", "Failed"),
            _trial("d", "Created"),  # neither running/succeeded/failed
        ]
    )
    assert ser.trial_counts(job) == {
        "total_trials": 4,
        "running_trials": 1,
        "succeeded_trials": 1,
        "failed_trials": 1,
    }


def test_trial_config_to_dict():
    tc = SimpleNamespace(num_trials=10, parallel_trials=2, max_failed_trials=3)
    assert ser.trial_config_to_dict(tc) == {
        "num_trials": 10,
        "parallel_trials": 2,
        "max_failed_trials": 3,
    }
    assert ser.trial_config_to_dict(None) == {}


def test_experiment_summary_and_full():
    job = SimpleNamespace(
        name="exp",
        status="Running",
        creation_timestamp=datetime(2026, 7, 21, tzinfo=timezone.utc),
        trials=[_trial("t1")],
        objectives=[SimpleNamespace(metric="acc", direction=_Direction.MAXIMIZE)],
        algorithm=SimpleNamespace(random_state=None),
        search_space={},
        trial_config=SimpleNamespace(num_trials=1, parallel_trials=1, max_failed_trials=None),
    )
    summary = ser.experiment_summary(job)
    assert summary["name"] == "exp"
    assert summary["total_trials"] == 1
    assert summary["creation_timestamp"].startswith("2026-07-21")

    full = ser.experiment_to_dict(job)
    assert full["objectives"][0]["direction"] == "maximize"
    assert "trial_config" in full
    assert "search_space" in full


def test_result_to_dict():
    assert ser.result_to_dict(None) is None
    result = SimpleNamespace(parameters={"lr": "0.01"}, metrics=[_metric()])
    out = ser.result_to_dict(result)
    assert out["parameters"] == {"lr": "0.01"}
    assert out["metrics"][0]["name"] == "acc"


def test_event_to_dict():
    ev = SimpleNamespace(
        involved_object_kind="Pod",
        involved_object_name="exp-trial-0",
        reason="Scheduled",
        message="assigned",
        event_time=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    out = ser.event_to_dict(ev)
    assert out["involved_object_kind"] == "Pod"
    assert out["reason"] == "Scheduled"
    assert out["event_time"].startswith("2026-07-21")


def test_is_success_status():
    assert ser.is_success_status("Complete") is True
    assert ser.is_success_status("Succeeded") is True
    assert ser.is_success_status("Failed") is False
    assert ser.is_success_status(None) is False
