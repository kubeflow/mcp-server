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

"""Unit tests for optimizer monitoring tools, against a fake Katib cluster."""

import itertools

import pytest

from kubeflow_mcp.optimizer.api import monitoring
from kubeflow_mcp.optimizer.api.optimization import ALGORITHMS
from kubeflow_mcp.optimizer.conftest import (
    RawResponse,
    event,
    experiment_cr,
    metric,
    pod,
    trial_cr,
)

_MON = "kubeflow_mcp.optimizer.api.monitoring"
_COMPLETE = [{"type": "Succeeded", "status": "True", "reason": "ExperimentMaxTrialsReached"}]
_FAILED = [{"type": "Failed", "status": "True", "reason": "ExperimentFailed"}]
_OPTIMAL = {
    "bestTrialName": "t1",
    "parameterAssignments": [{"name": "lr", "value": "0.03"}],
    "observation": {"metrics": [metric(latest="0.93")]},
}
_COLLECTOR = "metrics-logger-and-collector"


def _trial_pod(trial: str, *containers: str, katib_label: bool = True, **labels: str):
    if katib_label:
        labels["katib.kubeflow.org/trial"] = trial
    labels.setdefault("katib.kubeflow.org/experiment", "exp-1")
    return pod(f"{trial}-pod", labels, list(containers) or ["node", _COLLECTOR])


# ─── get_hpo_experiment_trials ─────────────────────────────────────────────


def test_get_experiment_trials_lists_every_state(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(
        trial_cr("t1", "Complete", metrics=[metric()], created="2026-11-01T00:00:01Z"),
        trial_cr("t2", "Running", created="2026-11-01T00:00:02Z"),
        trial_cr("t3", "EarlyStopped", created="2026-11-01T00:00:03Z"),
        trial_cr("t4", "Created", created="2026-11-01T00:00:04Z"),
    )
    data = monitoring.get_hpo_experiment_trials("exp-1")["data"]
    assert [(t["name"], t["status"]) for t in data["trials"]] == [
        ("t1", "Complete"),
        ("t2", "Running"),
        ("t3", "EarlyStopped"),
        ("t4", "Created"),
    ]
    assert data["summary"] == {
        "total_trials": 4,
        "running_trials": 1,
        "succeeded_trials": 1,
        "failed_trials": 0,
        "pending_trials": 1,
        "killed_trials": 0,
        "early_stopped_trials": 1,
        "metrics_unavailable_trials": 0,
    }


@pytest.mark.parametrize("wanted", ["Complete", "Succeeded"])
def test_get_experiment_trials_status_filter_and_alias(katib, wanted):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("ok", "Complete"), trial_cr("live", "Running"))
    data = monitoring.get_hpo_experiment_trials("exp-1", status=wanted)["data"]
    assert [t["name"] for t in data["trials"]] == ["ok"]
    assert data["total"] == 1
    # The summary still describes every trial, not just the filtered ones.
    assert data["summary"]["total_trials"] == 2


@pytest.mark.parametrize("bad_status", ["", "failed", "Error", "Suspended"])
def test_get_experiment_trials_rejects_unknown_status_before_api_call(katib, bad_status):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Failed"))
    result = monitoring.get_hpo_experiment_trials("exp-1", status=bad_status)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["details"] == {
        "supported_statuses": [
            "Complete",
            "Created",
            "EarlyStopped",
            "Failed",
            "Killed",
            "MetricsUnavailable",
            "Running",
        ],
        "aliases": {"Succeeded": "Complete"},
    }
    assert katib.custom.calls == []


def test_get_experiment_trials_not_found(katib):
    result = monitoring.get_hpo_experiment_trials("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_best_trial ────────────────────────────────────────────────────────


def test_get_best_trial_reads_the_current_optimal_trial(katib):
    katib.add_experiment(experiment_cr(status={"currentOptimalTrial": _OPTIMAL}))
    data = monitoring.get_best_trial("exp-1")["data"]
    assert data["trial"] == "t1"
    assert data["parameters"] == {"lr": "0.03"}
    assert data["metrics"][0]["latest"] == "0.93"


@pytest.mark.parametrize("kind", ["Job", "TrainJob"])
def test_get_best_trial_works_for_any_trial_workload(katib, kind):
    """Regression: the SDK looked up a TrainJob for the best trial and reported
    a batch/v1 Job trial's experiment as missing."""
    katib.add_experiment(experiment_cr(status={"currentOptimalTrial": _OPTIMAL}))
    katib.add_trials(trial_cr("t1", "Complete", kind=kind, metrics=[metric()]))
    assert monitoring.get_best_trial("exp-1")["success"] is True


def test_get_best_trial_no_result_yet(katib):
    katib.add_experiment(experiment_cr())
    data = monitoring.get_best_trial("exp-1")["data"]
    assert data["best_trial"] is None


def test_get_best_trial_not_found(katib):
    assert monitoring.get_best_trial("missing")["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_suggestion ────────────────────────────────────────────────────────


def test_get_suggestion_happy_path(katib):
    katib.custom.suggestions["exp-1"] = {
        "spec": {"algorithm": {"algorithmName": "random"}, "requests": 3},
        "status": {
            "suggestionCount": 3,
            "conditions": [{"type": "Running", "status": "True", "reason": "Ready"}],
        },
    }
    data = monitoring.get_suggestion("exp-1")["data"]
    assert data["algorithm"] == "random"
    assert data["conditions"][0]["type"] == "Running"
    assert data["namespace"] == "kubeflow"


def test_get_suggestion_not_found_names_the_right_kind(katib):
    result = monitoring.get_suggestion("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Suggestion 'missing'" in result["error"]


# ─── wait_for_hpo_experiment ───────────────────────────────────────────────


@pytest.fixture
def clock(monkeypatch):
    """A fake clock: sleeping advances it instead of blocking."""
    now = {"t": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(f"{_MON}.time.monotonic", lambda: now["t"])
    monkeypatch.setattr(f"{_MON}.time.sleep", sleep)
    return sleeps


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_wait_reaches_a_terminal_phase_for_every_algorithm(katib, clock, algorithm):
    katib.add_experiment(
        experiment_cr(
            algorithm=algorithm, conditions=_COMPLETE, status={"trials": 2, "trialsSucceeded": 2}
        )
    )
    data = monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=30)["data"]
    assert data["reached"] is True
    assert data["status"] == "Complete"
    assert data["succeeded_trials"] == 2
    assert clock == []


def test_wait_polls_until_the_experiment_finishes(katib, clock, monkeypatch):
    running = experiment_cr(status={"trialsRunning": 1})
    failed = experiment_cr(conditions=_FAILED)
    states = itertools.chain([running, running], itertools.repeat(failed))
    monkeypatch.setattr(katib.custom, "get_namespaced_custom_object", lambda **_: next(states))
    data = monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=60, polling_interval=5)[
        "data"
    ]
    assert data["status"] == "Failed"
    assert clock == [5, 5]


def test_wait_times_out_with_the_last_seen_phase(katib, clock):
    katib.add_experiment(experiment_cr(status={"trialsRunning": 1}))
    data = monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=20, polling_interval=5)[
        "data"
    ]
    assert data["reached"] is False
    assert data["status"] == "Running"
    assert sum(clock) <= 20


def test_wait_caps_the_timeout(katib, clock):
    katib.add_experiment(experiment_cr(status={"trialsRunning": 1}))
    monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=99999, polling_interval=600)
    assert sum(clock) <= monitoring.MAX_WAIT_TIMEOUT


def test_wait_rejects_too_frequent_polling(katib):
    """Matches wait_for_training: too low is an error, not a silent adjustment."""
    result = monitoring.wait_for_hpo_experiment("exp-1", polling_interval=1)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert katib.custom.calls == []


@pytest.mark.parametrize(
    ("timeout_seconds", "polling_interval"),
    [(15, 15), (10, 15), (99999, 5000)],
)
def test_wait_rejects_interval_not_below_timeout(katib, timeout_seconds, polling_interval):
    result = monitoring.wait_for_hpo_experiment(
        "exp-1", timeout_seconds=timeout_seconds, polling_interval=polling_interval
    )
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "strictly less" in result["error"]
    assert katib.custom.calls == []


def test_wait_rejects_invalid_timeout(katib):
    assert monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=0)["error_code"] == (
        "VALIDATION_ERROR"
    )


def test_wait_not_found(katib, clock):
    result = monitoring.wait_for_hpo_experiment("missing", timeout_seconds=30)
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── get_hpo_experiment_trial_logs ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "workload_label"),
    [("TrainJob", "jobset.sigs.k8s.io/jobset-name"), ("Job", "batch.kubernetes.io/job-name")],
)
def test_logs_come_from_the_metrics_collector_of_any_workload(katib, kind, workload_label):
    """Regression: the SDK only found TrainJob pods, so Job trials had no logs."""
    katib.add_experiment(experiment_cr(status={"currentOptimalTrial": _OPTIMAL}))
    katib.add_trials(trial_cr("t1", "Complete", kind=kind))
    katib.core.pods.append(_trial_pod("t1", **{workload_label: "t1"}))
    katib.core.logs[("t1-pod", _COLLECTOR)] = "epoch 1\naccuracy=0.93"

    data = monitoring.get_hpo_experiment_trial_logs("exp-1")["data"]

    assert data["trial"] == "t1"
    assert data["container"] == _COLLECTOR
    assert data["logs"] == "epoch 1\naccuracy=0.93"
    assert data["lines"] == 2


def test_logs_are_bounded_on_the_api_server(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1"))
    katib.core.pods.append(_trial_pod("t1"))
    monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")
    (call,) = [kw for name, kw in katib.core.calls if name == "read_log"]
    assert call["tail_lines"] == monitoring.MAX_LOG_LINES
    assert call["limit_bytes"] == monitoring.MAX_LOG_BYTES


@pytest.mark.parametrize(("kind", "label"), [("Job", "batch.kubernetes.io/job-name")])
def test_logs_fall_back_to_the_workload_label_without_injection(katib, kind, label):
    """Without metrics-collector injection Katib never labels the pods."""
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", kind=kind))
    katib.core.pods.append(_trial_pod("t1", "train", katib_label=False, **{label: "t1"}))
    katib.core.logs[("t1-pod", "train")] = "hello"
    data = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")["data"]
    assert data["container"] == "train"
    assert data["logs"] == "hello"


def test_logs_default_to_the_oldest_trial_without_a_best_one(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(
        trial_cr("old", created="2026-11-01T00:00:01Z"),
        trial_cr("new", created="2026-11-01T00:00:09Z"),
    )
    katib.core.pods.extend([_trial_pod("old"), _trial_pod("new")])
    assert monitoring.get_hpo_experiment_trial_logs("exp-1")["data"]["trial"] == "old"


def test_logs_failure_hint(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Failed"))
    katib.core.pods.append(_trial_pod("t1"))
    katib.core.logs[("t1-pod", _COLLECTOR)] = "torch.cuda.OutOfMemoryError: CUDA out of memory"
    data = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")["data"]
    assert data["failure_hint"]["category"]
    assert data["next_steps"]


def test_logs_report_a_trial_without_pods(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Created"))
    data = monitoring.get_hpo_experiment_trial_logs("exp-1")["data"]
    assert data["logs"] == ""
    assert "No pods" in data["message"]


def test_logs_with_no_trials_yet(katib):
    katib.add_experiment(experiment_cr())
    data = monitoring.get_hpo_experiment_trial_logs("exp-1")["data"]
    assert data["trial"] is None
    assert data["lines"] == 0


def test_logs_unknown_trial(katib):
    katib.add_experiment(experiment_cr())
    result = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="nope")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_logs_missing_experiment(katib):
    result = monitoring.get_hpo_experiment_trial_logs("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "Experiment 'missing'" in result["error"]


# ─── get_hpo_experiment_events ─────────────────────────────────────────────


def test_events_cover_experiment_trials_workloads_and_pods(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", kind="Job"))
    katib.core.pods.append(_trial_pod("t1"))
    katib.core.events.extend(
        [
            event("exp-1", "Experiment", "ExperimentCreated", "2026-11-01T00:00:00Z"),
            event("t1", "Trial", "TrialCreated", "2026-11-01T00:00:01Z"),
            event("t1", "Job", "SuccessfulCreate", "2026-11-01T00:00:02Z"),
            event("t1-pod", "Pod", "FailedScheduling", "2026-11-01T00:00:03Z", type_="Warning"),
            event("other", "Pod", "Pulled", "2026-11-01T00:00:04Z"),
        ]
    )
    data = monitoring.get_hpo_experiment_events("exp-1")["data"]
    assert [(e["involved_object_kind"], e["reason"]) for e in data["events"]] == [
        ("Experiment", "ExperimentCreated"),
        ("Trial", "TrialCreated"),
        ("Job", "SuccessfulCreate"),
        ("Pod", "FailedScheduling"),
    ]
    assert data["events"][-1]["type"] == "Warning"
    assert data["total"] == 4


def test_events_keep_the_most_recent_when_limited(katib):
    katib.add_experiment(experiment_cr())
    katib.core.events.extend(
        event("exp-1", "Experiment", f"R{i}", f"2026-11-01T00:00:{i:02d}Z") for i in range(5)
    )
    data = monitoring.get_hpo_experiment_events("exp-1", limit=2)["data"]
    assert [e["reason"] for e in data["events"]] == ["R3", "R4"]
    assert data["total"] == 5


def test_events_invalid_limit(katib):
    assert monitoring.get_hpo_experiment_events("exp-1", limit=0)["error_code"] == (
        "VALIDATION_ERROR"
    )


def test_events_missing_experiment(katib):
    result = monitoring.get_hpo_experiment_events("missing")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


@pytest.mark.parametrize(
    ("collector", "container", "expected"),
    [
        (None, _COLLECTOR, "stdout via sidecar"),
        ("StdOut", _COLLECTOR, "stdout via sidecar"),
        ("File", "node", "Traceback: boom"),
        ("TensorFlowEvent", "node", "Traceback: boom"),
    ],
)
def test_logs_follow_the_metrics_collector_kind(katib, collector, container, expected):
    """Only the StdOut collector redirects training output to its sidecar; with a
    File collector the traceback stays in the primary container."""
    katib.add_experiment(experiment_cr())
    trial = trial_cr("t1", "Failed")
    if collector:
        trial["spec"]["metricsCollector"] = {"collector": {"kind": collector}}
    katib.add_trials(trial)
    katib.core.pods.append(_trial_pod("t1"))
    katib.core.logs[("t1-pod", _COLLECTOR)] = "stdout via sidecar"
    katib.core.logs[("t1-pod", "node")] = "Traceback: boom"
    data = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")["data"]
    assert (data["container"], data["logs"]) == (container, expected)


def test_logs_keep_the_newest_lines_when_the_byte_cap_applies(katib, monkeypatch):
    """The API server caps bytes from the start of the tail window, which would
    cut off the newest lines; the read shrinks the window until they fit."""
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Failed"))
    katib.core.pods.append(_trial_pod("t1"))
    lines = [f"step {i} " + "x" * 2000 for i in range(1000)] + ["OOMKilled: the last line"]
    limit = monitoring.MAX_LOG_BYTES
    tails = []

    def api_server(**kw):
        tails.append(kw["tail_lines"])
        return RawResponse("\n".join(lines[-kw["tail_lines"] :]).encode()[: kw["limit_bytes"]])

    monkeypatch.setattr(katib.core, "read_namespaced_pod_log", api_server)
    data = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")["data"]
    assert data["logs"].endswith("OOMKilled: the last line")
    assert tails[0] == monitoring.MAX_LOG_LINES
    assert len(tails) <= 6
    assert len("\n".join(lines[-tails[-1] :]).encode()) < limit


def test_logs_survive_the_byte_cap_cutting_a_character_in_half(katib, monkeypatch):
    """The API server caps bytes, not characters. Strict decoding of a body cut
    inside a multibyte character raised before the read could be retried."""
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", "Failed"))
    katib.core.pods.append(_trial_pod("t1"))
    lines = ["é" * 1500 for _ in range(1000)] + ["Traceback: the last line"]
    responses = []

    def api_server(**kw):
        responses.append(RawResponse("\n".join(lines[-kw["tail_lines"] :]).encode()[:2999]))
        # The client decodes strictly unless told to hand over the raw body.
        return responses[-1] if kw.get("_preload_content") is False else responses[-1].data.decode()

    monkeypatch.setattr(katib.core, "read_namespaced_pod_log", api_server)
    # Each é is two bytes, so an odd cap inside the first line splits one.
    monkeypatch.setattr(monitoring, "MAX_LOG_BYTES", 2999)
    result = monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")

    with pytest.raises(UnicodeDecodeError):
        responses[0].data.decode()
    assert result["success"] is True
    assert result["data"]["logs"].endswith("Traceback: the last line")
    assert all(r.released for r in responses)


def test_log_reads_release_their_connection(katib):
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1"))
    katib.core.pods.append(_trial_pod("t1"))
    katib.core.logs[("t1-pod", _COLLECTOR)] = "accuracy=0.93"
    monitoring.get_hpo_experiment_trial_logs("exp-1", trial="t1")
    assert katib.core.log_responses
    assert all(r.released for r in katib.core.log_responses)


def test_events_match_kind_as_well_as_name(katib):
    """An unrelated Deployment named like the experiment must not show up, while
    a TrainJob trial's JobSet, which shares the trial's name, must."""
    katib.add_experiment(experiment_cr())
    katib.add_trials(trial_cr("t1", kind="TrainJob"))
    katib.core.events.extend(
        [
            event("exp-1", "Deployment", "ScalingReplicaSet", "2026-11-01T00:00:01Z"),
            event("exp-1", "Suggestion", "SuggestionCreated", "2026-11-01T00:00:02Z"),
            event("t1", "JobSet", "SuccessfulCreate", "2026-11-01T00:00:03Z"),
        ]
    )
    data = monitoring.get_hpo_experiment_events("exp-1")["data"]
    assert [e["involved_object_kind"] for e in data["events"]] == ["Suggestion", "JobSet"]


def test_wait_checks_once_more_at_the_deadline(katib, clock, monkeypatch):
    """With a 20s timeout and a 15s interval, an experiment finishing at 18s must be
    reported as finished, not as a timeout after 15s."""
    running = experiment_cr(status={"trialsRunning": 1})
    done = experiment_cr(conditions=_COMPLETE)
    states = iter([running, running, done])
    monkeypatch.setattr(katib.custom, "get_namespaced_custom_object", lambda **_: next(states))
    data = monitoring.wait_for_hpo_experiment("exp-1", timeout_seconds=20, polling_interval=15)[
        "data"
    ]
    assert data["reached"] is True
    assert clock == [15, 5]
