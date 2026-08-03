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

Serializers that convert kubeflow.optimizer SDK dataclasses
(``OptimizationJob``, ``Trial``, ``Result``, ``Metric``, ``Objective``,
search-space objects, ``Event``) into plain JSON-serializable dicts for
MCP tool responses.

Kept here (rather than inline in api/*.py) so discovery and monitoring
tools produce identical field names for the same SDK objects.
"""

from typing import Any

# Terminal SDK statuses for an OptimizationJob (see optimizer/constants).
_SUCCESS_STATUSES = {"Complete", "Succeeded"}
_FAILED_STATUSES = {"Failed"}
_RUNNING_STATUSES = {"Running"}


def _enum_value(value: Any) -> Any:
    """Return ``value.value`` for enums, otherwise the value unchanged."""
    return getattr(value, "value", value)


def _isoformat(value: Any) -> str | None:
    """Best-effort ISO-8601 string for datetime-like values."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def metric_to_dict(metric: Any) -> dict[str, Any]:
    """Serialize an optimizer ``Metric`` (name, min, max, latest)."""
    return {
        "name": getattr(metric, "name", None),
        "min": getattr(metric, "min", None),
        "max": getattr(metric, "max", None),
        "latest": getattr(metric, "latest", None),
    }


def metrics_to_list(metrics: Any) -> list[dict[str, Any]]:
    """Serialize a list of ``Metric`` objects."""
    return [metric_to_dict(m) for m in (metrics or [])]


def trial_status(trial: Any) -> str:
    """Derive a trial's status from its underlying TrainJob.

    The optimizer ``Trial`` dataclass has no status field of its own; the
    execution state lives on ``trial.trainjob.status``.
    """
    trainjob = getattr(trial, "trainjob", None)
    return getattr(trainjob, "status", None) or "Unknown"


def trial_to_dict(trial: Any) -> dict[str, Any]:
    """Serialize an optimizer ``Trial`` (name, parameters, metrics, status)."""
    trainjob = getattr(trial, "trainjob", None)
    return {
        "name": getattr(trial, "name", None),
        "status": trial_status(trial),
        "parameters": dict(getattr(trial, "parameters", {}) or {}),
        "metrics": metrics_to_list(getattr(trial, "metrics", None)),
        "trainjob": getattr(trainjob, "name", None),
    }


def objective_to_dict(objective: Any) -> dict[str, Any]:
    """Serialize an ``Objective`` (metric + direction enum)."""
    return {
        "metric": getattr(objective, "metric", None),
        "direction": _enum_value(getattr(objective, "direction", None)),
    }


def search_space_entry_to_dict(space: Any) -> dict[str, Any]:
    """Serialize a single search-space entry.

    Handles both ``ContinuousSearchSpace`` (min/max/distribution) and
    ``CategoricalSearchSpace`` (choices).
    """
    if hasattr(space, "choices"):
        return {"type": "categorical", "choices": list(space.choices or [])}
    if hasattr(space, "min") and hasattr(space, "max"):
        return {
            "type": "continuous",
            "min": space.min,
            "max": space.max,
            "distribution": _enum_value(getattr(space, "distribution", None)),
        }
    return {"type": "unknown", "value": str(space)}


def search_space_to_dict(search_space: Any) -> dict[str, Any]:
    """Serialize the ``dict[str, SearchSpace]`` map on an OptimizationJob."""
    if not search_space:
        return {}
    return {name: search_space_entry_to_dict(space) for name, space in search_space.items()}


def algorithm_to_dict(algorithm: Any) -> dict[str, Any]:
    """Serialize an algorithm object (RandomSearch / GridSearch)."""
    if algorithm is None:
        return {}
    data: dict[str, Any] = {"name": type(algorithm).__name__}
    random_state = getattr(algorithm, "random_state", None)
    if random_state is not None:
        data["random_state"] = random_state
    return data


def trial_counts(job: Any) -> dict[str, int]:
    """Aggregate trial counts by status for an OptimizationJob."""
    trials = getattr(job, "trials", None) or []
    running = succeeded = failed = 0
    for trial in trials:
        status = trial_status(trial)
        if status in _SUCCESS_STATUSES:
            succeeded += 1
        elif status in _FAILED_STATUSES:
            failed += 1
        elif status in _RUNNING_STATUSES:
            running += 1
    return {
        "total_trials": len(trials),
        "running_trials": running,
        "succeeded_trials": succeeded,
        "failed_trials": failed,
    }


def trial_config_to_dict(trial_config: Any) -> dict[str, Any]:
    """Serialize a ``TrialConfig`` (num_trials, parallel_trials, max_failed_trials)."""
    if trial_config is None:
        return {}
    return {
        "num_trials": getattr(trial_config, "num_trials", None),
        "parallel_trials": getattr(trial_config, "parallel_trials", None),
        "max_failed_trials": getattr(trial_config, "max_failed_trials", None),
    }


def experiment_summary(job: Any) -> dict[str, Any]:
    """Serialize an OptimizationJob to a compact summary (for list views)."""
    data: dict[str, Any] = {
        "name": getattr(job, "name", None),
        "status": getattr(job, "status", None) or "Unknown",
        "creation_timestamp": _isoformat(getattr(job, "creation_timestamp", None)),
    }
    data.update(trial_counts(job))
    return data


def experiment_to_dict(job: Any) -> dict[str, Any]:
    """Serialize an OptimizationJob to full detail (for get views)."""
    data = experiment_summary(job)
    data["objectives"] = [objective_to_dict(o) for o in (getattr(job, "objectives", None) or [])]
    data["algorithm"] = algorithm_to_dict(getattr(job, "algorithm", None))
    data["search_space"] = search_space_to_dict(getattr(job, "search_space", None))
    data["trial_config"] = trial_config_to_dict(getattr(job, "trial_config", None))
    return data


def result_to_dict(result: Any) -> dict[str, Any] | None:
    """Serialize a ``Result`` (best parameters + metrics), or None."""
    if result is None:
        return None
    return {
        "parameters": dict(getattr(result, "parameters", {}) or {}),
        "metrics": metrics_to_list(getattr(result, "metrics", None)),
    }


def event_to_dict(event: Any) -> dict[str, Any]:
    """Serialize a trainer/optimizer ``Event`` object."""
    return {
        "involved_object_kind": getattr(event, "involved_object_kind", "") or "",
        "involved_object_name": getattr(event, "involved_object_name", "") or "",
        "reason": getattr(event, "reason", "") or "",
        "message": getattr(event, "message", "") or "",
        "event_time": _isoformat(getattr(event, "event_time", None)) or "",
    }


def is_success_status(status: str | None) -> bool:
    """Return True if a status string denotes a successful/complete trial."""
    return status in _SUCCESS_STATUSES
