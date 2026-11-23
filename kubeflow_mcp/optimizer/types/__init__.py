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

"""Optimizer-specific type helpers.

Serializers that turn Katib's own resources (Experiment, Trial and Suggestion
custom resources, and Kubernetes events) into plain JSON-serializable dicts for
MCP tool responses.

They read the custom resources rather than the SDK's ``OptimizationJob``
dataclasses because the SDK can only describe experiments it could have
created itself: its reader rejects every algorithm but random and grid, and
assumes each trial runs a TrainJob. The resources describe every algorithm and
workload kind the creation tools accept.

Kept here (rather than inline in api/*.py) so discovery, monitoring and
lifecycle tools produce identical field names for the same resource.
"""

from typing import Any

from kubeflow_mcp.common.constants import JobStatus
from kubeflow_mcp.optimizer.constants import (
    OPTIMIZATION_JOB_COMPLETE,
    OPTIMIZATION_JOB_CREATED,
    OPTIMIZATION_JOB_FAILED,
    OPTIMIZATION_JOB_RUNNING,
)

# Katib spells the success condition on the Experiment CR "Succeeded", while the
# SDK reports the resulting job status as "Complete".
_CR_CONDITION_SUCCEEDED = "Succeeded"

# Trial states, from the Trial CR's true conditions, checked in the order
# Katib's own status summary checks them (updateTrialsSummary). Katib leaves
# earlier conditions in place as a trial progresses, so the order decides, and
# a trial matching none of them, including one with no conditions yet, is
# pending. A succeeded trial is reported as "Complete", the word the trial
# status filter and the TrainJob tools already use, so a filter written for one
# client works for the other.
_TRIAL_STATES: tuple[tuple[str, str], ...] = (
    ("Killed", "Killed"),
    ("Failed", JobStatus.FAILED),
    ("Succeeded", JobStatus.COMPLETE),
    ("EarlyStopped", "EarlyStopped"),
    ("Running", JobStatus.RUNNING),
    ("MetricsUnavailable", "MetricsUnavailable"),
)
TRIAL_SUCCEEDED = JobStatus.COMPLETE
TRIAL_PENDING = JobStatus.CREATED
# Katib counts a trial as active (it occupies a parallelTrialCount slot) while
# it is pending or running.
ACTIVE_TRIAL_STATES = frozenset({TRIAL_PENDING, JobStatus.RUNNING})

# trial status -> the response counter it feeds
_TRIAL_STATE_COUNTERS = {
    JobStatus.RUNNING: "running_trials",
    TRIAL_SUCCEEDED: "succeeded_trials",
    JobStatus.FAILED: "failed_trials",
    TRIAL_PENDING: "pending_trials",
    "Killed": "killed_trials",
    "EarlyStopped": "early_stopped_trials",
}


def _isoformat(value: Any) -> str | None:
    """Best-effort ISO-8601 string for datetime-like values."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _true_conditions(status: dict[str, Any]) -> set[str]:
    """Types of the conditions currently ``True`` on a CR status."""
    return {c.get("type") for c in (status.get("conditions") or []) if c.get("status") == "True"}


def metrics_from_observation(observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Serialize ``observation.metrics`` from a Trial or the optimal trial."""
    return [
        {
            "name": m.get("name"),
            "min": m.get("min"),
            "max": m.get("max"),
            "latest": m.get("latest"),
        }
        for m in ((observation or {}).get("metrics") or [])
    ]


def parameters_from_assignments(assignments: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Serialize ``parameterAssignments`` (Katib stores every value as a string)."""
    return {pa.get("name"): pa.get("value") for pa in (assignments or []) if pa.get("name")}


def trial_status(trial: dict[str, Any]) -> str:
    """Derive a trial's state from the Trial CR's conditions, as Katib classifies it."""
    true = _true_conditions(trial.get("status") or {})
    for condition, state in _TRIAL_STATES:
        if condition in true:
            return state
    return TRIAL_PENDING


def trial_to_dict(trial: dict[str, Any]) -> dict[str, Any]:
    """Serialize a Trial CR (name, status, parameters, metrics, workload).

    ``trainjob`` names the trial's TrainJob when it runs one and is ``None``
    otherwise; ``workload_kind`` says what it runs (a TrainJob, a batch/v1 Job,
    ...), which is what logs and events resolve against.
    """
    metadata = trial.get("metadata") or {}
    spec = trial.get("spec") or {}
    kind = (spec.get("runSpec") or {}).get("kind")
    name = metadata.get("name")
    return {
        "name": name,
        "status": trial_status(trial),
        "parameters": parameters_from_assignments(spec.get("parameterAssignments")),
        "metrics": metrics_from_observation((trial.get("status") or {}).get("observation")),
        # Katib names a trial's workload after the trial.
        "trainjob": name if kind == "TrainJob" else None,
        "workload_kind": kind,
        "creation_timestamp": metadata.get("creationTimestamp"),
    }


def trial_counts_from_trials(trials: list[dict[str, Any]]) -> dict[str, int]:
    """Count serialized trials by state, with the same keys as the CR counters."""
    counts = {"total_trials": len(trials), **dict.fromkeys(_TRIAL_STATE_COUNTERS.values(), 0)}
    for trial in trials:
        key = _TRIAL_STATE_COUNTERS.get(trial.get("status"))
        if key:
            counts[key] += 1
    return counts


def _search_space_entry(parameter: dict[str, Any]) -> dict[str, Any]:
    """Serialize one ``spec.parameters`` entry as Katib defines it."""
    feasible = parameter.get("feasibleSpace") or {}
    entry: dict[str, Any] = {"type": parameter.get("parameterType")}
    if feasible.get("list") is not None:
        entry["choices"] = list(feasible["list"])
    for key in ("min", "max", "step", "distribution"):
        if feasible.get(key) is not None:
            entry[key] = feasible[key]
    return entry


def experiment_to_dict(experiment: dict[str, Any]) -> dict[str, Any]:
    """Serialize an Experiment CR to full detail (for get views).

    Everything comes from the one resource, so the phase, the counters, the
    conditions and the best trial cannot disagree with each other.
    """
    spec = experiment.get("spec") or {}
    status = experiment.get("status") or {}
    objective = spec.get("objective") or {}
    algorithm = spec.get("algorithm") or {}

    data = experiment_summary_from_cr(experiment)
    data["objectives"] = [
        {"metric": objective.get("objectiveMetricName"), "direction": objective.get("type")}
    ]
    if objective.get("goal") is not None:
        data["objectives"][0]["goal"] = objective["goal"]
    data["algorithm"] = {
        "name": algorithm.get("algorithmName"),
        "settings": {
            s.get("name"): s.get("value") for s in (algorithm.get("algorithmSettings") or [])
        },
    }
    data["search_space"] = {
        p.get("name"): _search_space_entry(p) for p in (spec.get("parameters") or [])
    }
    data["trial_config"] = {
        "num_trials": spec.get("maxTrialCount"),
        "parallel_trials": spec.get("parallelTrialCount"),
        "max_failed_trials": spec.get("maxFailedTrialCount"),
    }
    data["conditions"] = conditions_to_list(status)
    data["best_trial"] = optimal_trial_to_dict(status)
    data["early_stopping"] = early_stopping_to_dict(spec)
    return data


def event_to_dict(event: Any) -> dict[str, Any]:
    """Serialize a Kubernetes ``CoreV1Event``."""
    involved = getattr(event, "involved_object", None)
    when = (
        getattr(event, "last_timestamp", None)
        or getattr(event, "event_time", None)
        or getattr(event, "first_timestamp", None)
    )
    return {
        "type": getattr(event, "type", "") or "",
        "involved_object_kind": getattr(involved, "kind", "") or "",
        "involved_object_name": getattr(involved, "name", "") or "",
        "reason": getattr(event, "reason", "") or "",
        "message": getattr(event, "message", "") or "",
        "event_time": _isoformat(when) or "",
    }


# ─── Experiment CR ─────────────────────────────────────────────────────────
# Field names follow the v1beta1 wire format (camelCase) as generated in
# kubeflow_katib_api.models.

# status counters -> the response key each maps to
_CR_TRIAL_COUNTERS = {
    "trials": "total_trials",
    "trialsSucceeded": "succeeded_trials",
    "trialsFailed": "failed_trials",
    "trialsRunning": "running_trials",
    "trialsPending": "pending_trials",
    "trialsKilled": "killed_trials",
    "trialsEarlyStopped": "early_stopped_trials",
}


def experiment_phase(status: dict[str, Any]) -> str:
    """Derive the OptimizationJob status string from Experiment CR status.

    Mirrors the SDK's own derivation in
    ``kubeflow.optimizer.backends.kubernetes.backend.__get_optimization_job_from_cr``
    so a CR-sourced status matches what ``OptimizerClient.get_job()`` reports:
    a true ``Succeeded`` condition means Complete, a true ``Failed`` condition
    means Failed, otherwise Running if any trial is running, else Created.

    The SDK checks each trial's TrainJob for the running case; the CR's own
    ``trialsRunning`` counter carries the same information without fetching
    every trial.
    """
    for condition in status.get("conditions") or []:
        if condition.get("status") != "True":
            continue
        if condition.get("type") == _CR_CONDITION_SUCCEEDED:
            return OPTIMIZATION_JOB_COMPLETE
        if condition.get("type") == OPTIMIZATION_JOB_FAILED:
            return OPTIMIZATION_JOB_FAILED

    if status.get("trialsRunning"):
        return OPTIMIZATION_JOB_RUNNING
    return OPTIMIZATION_JOB_CREATED


def conditions_to_list(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Serialize ``status.conditions`` from an Experiment CR.

    Conditions carry the reason an experiment failed or stalled, which is the
    detail agents need and which ``OptimizationJob`` drops entirely.
    """
    return [
        {
            "type": c.get("type"),
            "status": c.get("status"),
            "reason": c.get("reason"),
            "message": c.get("message"),
            "last_transition_time": c.get("lastTransitionTime"),
        }
        for c in (status.get("conditions") or [])
    ]


def optimal_trial_to_dict(status: dict[str, Any]) -> dict[str, Any] | None:
    """Serialize ``status.currentOptimalTrial``, or None when unset.

    Katib leaves the name empty until a trial has produced metrics, so an
    object with no ``bestTrialName`` is reported as "no best trial yet".
    """
    optimal = status.get("currentOptimalTrial") or {}
    name = optimal.get("bestTrialName")
    if not name:
        return None
    return {
        "name": name,
        "parameters": parameters_from_assignments(optimal.get("parameterAssignments")),
        "metrics": metrics_from_observation(optimal.get("observation")),
    }


def trial_counts_from_cr(status: dict[str, Any]) -> dict[str, int]:
    """Trial counts as reported by the Experiment CR itself.

    Preferred over deriving counts from ``OptimizationJob.trials``: Katib tracks
    ``trialsEarlyStopped`` here, a state that never appears on the underlying
    TrainJob and is therefore invisible to the SDK view.

    Counters Katib has not published are omitted rather than reported as zero,
    so callers that need a stable set of keys should seed their own defaults
    (see :func:`experiment_summary_from_cr`).
    """
    return {
        key: int(status[field])
        for field, key in _CR_TRIAL_COUNTERS.items()
        if isinstance(status.get(field), int)
    }


def experiment_summary_from_cr(item: dict[str, Any]) -> dict[str, Any]:
    """Summarize a raw Experiment CR, matching :func:`experiment_summary`'s shape.

    Lets ``list_hpo_experiments`` read the Experiment list in a single API call
    instead of going through ``OptimizerClient.list_jobs()``, which resolves
    every trial's TrainJob individually. Everything a summary needs is already
    on the CR.

    The four SDK-derived counters are seeded to 0 before the CR's own counters
    are applied: Katib omits them entirely until trials exist, and dropping keys
    the SDK path always emitted would be a silent contract change. Counters only
    the CR knows about (pending, killed, early stopped) are additive.
    """
    metadata = item.get("metadata") or {}
    status = item.get("status") or {}

    data: dict[str, Any] = {
        "name": metadata.get("name"),
        "status": experiment_phase(status),
        # Left as the raw RFC3339 string Kubernetes emits, as with the
        # lastTransitionTime values in conditions_to_list.
        "creation_timestamp": metadata.get("creationTimestamp"),
        "total_trials": 0,
        "running_trials": 0,
        "succeeded_trials": 0,
        "failed_trials": 0,
    }
    data.update(trial_counts_from_cr(status))
    return data


def suggestion_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Serialize a raw Suggestion CR from ``CustomObjectsApi``.

    Shared by ``list_suggestions`` and ``get_suggestion`` so both describe the
    same resource with the same field names. They previously carried separate
    serializers that disagreed: one emitted a single ``condition``, the other a
    ``conditions`` list.
    """
    metadata = item.get("metadata") or {}
    spec = item.get("spec") or {}
    status = item.get("status") or {}
    return {
        "name": metadata.get("name"),
        "algorithm": (spec.get("algorithm") or {}).get("algorithmName"),
        "requests": spec.get("requests"),
        "suggestion_count": status.get("suggestionCount"),
        "conditions": [
            {"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason")}
            for c in (status.get("conditions") or [])
        ],
    }


def early_stopping_to_dict(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Serialize ``spec.earlyStopping``, or None when not configured."""
    early = spec.get("earlyStopping") or {}
    if not early:
        return None
    return {
        "algorithm": early.get("algorithmName"),
        "settings": {s.get("name"): s.get("value") for s in (early.get("algorithmSettings") or [])},
    }
