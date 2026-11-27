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

"""End-to-end optimizer workflows against the fake Katib cluster.

Regressions from cluster testing: a Bayesian experiment, and an experiment whose
trials run batch/v1 Jobs, both completed in Katib but could not be read back
through the SDK. Each flow creates the experiment through the tool, lets the
fake play Katib's part (trials, metrics, status), then reads it back through
every reader an agent would use.
"""

import pytest

from kubeflow_mcp.optimizer.api import discovery, monitoring, optimization
from kubeflow_mcp.optimizer.conftest import event, metric, pod, trial_cr

_JOB_TEMPLATE = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "train",
                        "image": "python:3.12-slim",
                        "command": ["python", "train.py", "--lr=${trialParameters.lr}"],
                    }
                ],
                "restartPolicy": "Never",
            }
        }
    },
}
_TRAINJOB_TEMPLATE = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"image": "my/trainer:1", "args": ["--lr=${trialParameters.lr}"]},
    },
}


def _run_katib(katib, name, kind):
    """Play Katib: two trials finish, the second is optimal, the experiment succeeds."""
    katib.add_trials(
        trial_cr("t1", "Complete", experiment=name, kind=kind, metrics=[metric(latest="0.81")]),
        trial_cr(
            "t2",
            "Complete",
            experiment=name,
            kind=kind,
            created="2026-11-01T00:00:02Z",
            params={"lr": "0.03"},
            metrics=[metric(latest="0.93")],
        ),
    )
    katib.custom.experiments[name]["status"] = {
        "conditions": [
            {"type": "Succeeded", "status": "True", "reason": "ExperimentMaxTrialsReached"}
        ],
        "trials": 2,
        "trialsSucceeded": 2,
        "currentOptimalTrial": {
            "bestTrialName": "t2",
            "parameterAssignments": [{"name": "lr", "value": "0.03"}],
            "observation": {"metrics": [metric(latest="0.93")]},
        },
    }
    workload_label = (
        "batch.kubernetes.io/job-name" if kind == "Job" else "jobset.sigs.k8s.io/jobset-name"
    )
    for trial in ("t1", "t2"):
        katib.core.pods.append(
            pod(
                f"{trial}-pod",
                {
                    "katib.kubeflow.org/trial": trial,
                    "katib.kubeflow.org/experiment": name,
                    workload_label: trial,
                },
                ["train" if kind == "Job" else "node", "metrics-logger-and-collector"],
            )
        )
        katib.core.logs[(f"{trial}-pod", "metrics-logger-and-collector")] = f"{trial}: accuracy"
    katib.core.events.extend(
        [
            event(name, "Experiment", "ExperimentSucceeded", "2026-11-01T00:01:00Z"),
            event("t2", kind, "Completed", "2026-11-01T00:00:50Z"),
            event("t2-pod", "Pod", "Started", "2026-11-01T00:00:10Z"),
        ]
    )


@pytest.mark.parametrize(
    ("algorithm", "template", "kind"),
    [
        ("bayesianoptimization", _TRAINJOB_TEMPLATE, "TrainJob"),
        ("tpe", _JOB_TEMPLATE, "Job"),
        ("random", _JOB_TEMPLATE, "Job"),
    ],
)
def test_create_then_read_back_through_every_reader(katib, monkeypatch, algorithm, template, kind):
    monkeypatch.setattr(monitoring.time, "sleep", lambda _: None)
    name = f"{algorithm}-{kind.lower()}"

    created = optimization.create_hpo_experiment(
        name=name,
        objective_metric="accuracy",
        search_space={"lr": {"min": 0.001, "max": 0.1, "type": "loguniform"}},
        trial_template=template,
        algorithm=algorithm,
        max_trial_count=2,
        parallel_trial_count=1,
        confirmed=True,
    )
    assert created["success"] is True, created
    _run_katib(katib, name, kind)

    detail = discovery.get_hpo_experiment(name)["data"]
    assert detail["algorithm"]["name"] == algorithm
    assert detail["status"] == "Complete"
    assert {t["workload_kind"] for t in detail["trials"]} == {kind}

    trials = monitoring.get_hpo_experiment_trials(name, status="Succeeded")["data"]
    assert trials["total"] == 2

    best = monitoring.get_best_trial(name)["data"]
    assert best["trial"] == "t2"
    assert best["parameters"] == {"lr": "0.03"}

    logs = monitoring.get_hpo_experiment_trial_logs(name)["data"]
    assert logs["trial"] == "t2"
    assert logs["logs"] == "t2: accuracy"

    events = monitoring.get_hpo_experiment_events(name)["data"]
    assert [e["involved_object_kind"] for e in events["events"]] == ["Pod", kind, "Experiment"]

    waited = monitoring.wait_for_hpo_experiment(name, timeout_seconds=30)["data"]
    assert waited["reached"] is True
    assert waited["status"] == "Complete"
