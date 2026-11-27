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

"""Monitoring tools for Katib experiment tracking and debugging.

Every tool reads Katib's custom resources and the trial pods directly (see
``optimizer/api/_katib.py`` for why the SDK's ``OptimizerClient`` readers are
not used):

    - Experiment CR       → get_best_trial, wait_for_hpo_experiment
    - Trial CRs           → get_hpo_experiment_trials
    - Trial pod logs      → get_hpo_experiment_trial_logs
    - Namespace events    → get_hpo_experiment_events
    - Suggestion CR       → get_suggestion
"""

import logging
import time
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.failures import extract_failure_hint
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_custom_objects_api,
    get_optimizer_effective_namespace,
)
from kubeflow_mcp.core.security import (
    truncate_log_output,
    validate_k8s_name,
)
from kubeflow_mcp.optimizer.api import _katib
from kubeflow_mcp.optimizer.api._common import (
    DEFAULT_LIST_LIMIT,
    check_optimizer_namespace,
    clamp_limit,
    experiment_error,
    resolve_status_filter,
    resource_error,
)
from kubeflow_mcp.optimizer.constants import (
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    OPTIMIZATION_JOB_COMPLETE,
    OPTIMIZATION_JOB_FAILED,
    SUGGESTION_PLURAL,
)
from kubeflow_mcp.optimizer.types import (
    event_to_dict,
    experiment_summary_from_cr,
    optimal_trial_to_dict,
    suggestion_to_dict,
    trial_counts_from_trials,
    trial_to_dict,
)

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 1000
# Upper bound on the bytes the API server sends for one log read, so a trial
# that logs very long lines cannot make the server hold an unbounded response.
# The reply itself is truncated further by truncate_log_output.
MAX_LOG_BYTES = 1024 * 1024
MAX_WAIT_TIMEOUT = 3600
MIN_POLLING_INTERVAL = 5
_TERMINAL_PHASES = frozenset({OPTIMIZATION_JOB_COMPLETE, OPTIMIZATION_JOB_FAILED})


def get_hpo_experiment_trials(
    name: str,
    namespace: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """List trials for a Katib experiment with optional status filter.

    Trials are read from the experiment's Trial CRs, oldest first.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        status: Optional status filter: Created, Running, Complete (alias
            Succeeded), Failed, Killed, EarlyStopped or MetricsUnavailable.
        limit: Maximum trials to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``trials`` (up to ``limit``), ``total``
        matching trials, and a ``summary`` of counts by status.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    limit, limit_err = clamp_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        # Read the experiment first so a missing one is not reported as empty.
        _katib.read_experiment(name, ns)
        all_trials = [trial_to_dict(t) for t in _katib.list_trials(name, ns)]
        trials = all_trials
        if status:
            want = resolve_status_filter(status)
            trials = [t for t in all_trials if t.get("status") == want]

        data: dict[str, Any] = {
            "experiment": name,
            "trials": trials[:limit],
            "total": len(trials),
            "summary": trial_counts_from_trials(all_trials),
        }
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_best_trial(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get the best trial with optimal hyperparameters and metrics.

    Reads ``status.currentOptimalTrial`` from the Experiment CR, which Katib
    keeps up to date for every algorithm and workload kind.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing best hyperparameters and metrics, or a note
              when no successful trial exists yet.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        experiment = _katib.read_experiment(name, ns)
        best = optimal_trial_to_dict(experiment.get("status") or {})
        if best is None:
            return ToolResponse(
                data={
                    "experiment": name,
                    "best_trial": None,
                    "message": "No optimal result available yet — no successful trials.",
                }
            ).model_dump()

        return ToolResponse(
            data={
                "experiment": name,
                "trial": best["name"],
                "parameters": best["parameters"],
                "metrics": best["metrics"],
                "next_steps": [
                    f"fine_tune(...) — retrain with these hyperparameters from '{name}'",
                ],
            }
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_suggestion(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get suggestion algorithm status for an experiment.

    The Suggestion resource shares the experiment's name. Uses CustomObjectsApi
    directly (no SDK method available).

    Args:
        name: Experiment name (suggestion shares the same name).
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing algorithm name, status, and request counts.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        api = get_custom_objects_api()
        obj = api.get_namespaced_custom_object(
            group=KATIB_API_GROUP,
            version=KATIB_API_VERSION,
            namespace=ns,
            plural=SUGGESTION_PLURAL,
            name=name,
            _request_timeout=K8S_TIMEOUT,
        )

        # Same serializer as list_suggestions, so both views of a Suggestion
        # describe it with the same field names. Only the namespace is extra.
        data = suggestion_to_dict(obj)
        data["namespace"] = ns
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return resource_error(e, name, kind="Suggestion").model_dump()


def wait_for_hpo_experiment(
    name: str,
    namespace: str | None = None,
    timeout_seconds: int = 600,
    polling_interval: int = 15,
) -> dict[str, Any]:
    """Block until experiment reaches terminal status (Complete/Failed).

    Polls the Experiment CR's phase. Timeout is capped at 3600 seconds and the
    polling interval has a minimum of 5 seconds.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        timeout_seconds: Maximum seconds to wait. Capped at 3600.
        polling_interval: Seconds between status polls. Minimum 5.

    Returns:
        dict: Response containing final experiment status and trial summary.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if timeout_seconds < 1:
        return ToolError(
            error=f"timeout_seconds must be >= 1, got {timeout_seconds}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    # Rejected rather than clamped upward, matching wait_for_training: silently
    # polling slower than asked looks like the tool ignoring its own arguments.
    if polling_interval < MIN_POLLING_INTERVAL:
        return ToolError(
            error=f"polling_interval must be >= {MIN_POLLING_INTERVAL}, got {polling_interval}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    timeout_seconds = min(timeout_seconds, MAX_WAIT_TIMEOUT)
    # An interval at or above the timeout would poll once and give up, which
    # matches wait_for_training's rule and the SDK's own.
    if polling_interval >= timeout_seconds:
        return ToolError(
            error=(
                "polling_interval must be strictly less than timeout_seconds, got "
                f"polling_interval={polling_interval}, timeout_seconds={timeout_seconds}"
            ),
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        deadline = time.monotonic() + timeout_seconds
        while True:
            summary = experiment_summary_from_cr(_katib.read_experiment(name, ns))
            if summary["status"] in _TERMINAL_PHASES:
                summary.pop("name")
                summary.pop("creation_timestamp")
                return ToolResponse(
                    data={
                        "experiment": name,
                        "reached": True,
                        "message": f"Experiment reached '{summary['status']}'",
                        **summary,
                    }
                ).model_dump()
            # Sleep at most until the deadline, then check once more, so an
            # experiment finishing in the last interval is still reported.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(polling_interval, remaining))

        return ToolResponse(
            data={
                "experiment": name,
                "status": summary["status"],
                "reached": False,
                "message": f"Timeout after {timeout_seconds}s",
                "hint": "Use get_hpo_experiment_events() to check for scheduling issues",
            }
        ).model_dump()
    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_hpo_experiment_trial_logs(
    name: str,
    trial: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get pod logs from an experiment trial.

    Works for any trial workload kind: the trial's pods are found by Katib's
    trial label (or the workload's own label without metrics-collector
    injection). The last ``1000`` lines are read, bounded on the API server.
    If no trial is specified, the current best trial is used, else the oldest.

    Args:
        name: Experiment name.
        trial: Optional trial name. If omitted, uses the best trial.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``logs``, the ``trial``, ``pod`` and
        ``container`` they came from, and ``failure_hint`` when one matches.

    Raises:
        ToolError: If experiment not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()
    if trial is not None:
        trial_err = validate_k8s_name(trial, "trial")
        if trial_err is not None:
            return trial_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        experiment = _katib.read_experiment(name, ns)
        trials = _katib.list_trials(name, ns)
        selected = _select_trial(trials, trial, experiment)
        if selected is None:
            if trial is not None:
                return ToolError(
                    error=f"Trial '{trial}' not found in experiment '{name}'",
                    error_code=ErrorCode.RESOURCE_NOT_FOUND,
                    hint=f"Use get_hpo_experiment_trials('{name}') to list trials",
                ).model_dump()
            return ToolResponse(
                data={
                    "experiment": name,
                    "trial": None,
                    "logs": "",
                    "lines": 0,
                    "message": "Experiment has no trials yet",
                }
            ).model_dump()

        trial_name = selected["metadata"]["name"]
        source = _katib.pick_log_source(selected, _katib.find_trial_pods(selected, ns))
        if source is None:
            return ToolResponse(
                data={
                    "experiment": name,
                    "trial": trial_name,
                    "logs": "",
                    "lines": 0,
                    "message": f"No pods found for trial '{trial_name}'",
                    "hint": f"Check get_hpo_experiment_events('{name}') for scheduling issues",
                }
            ).model_dump()

        pod, container = source
        logs = _katib.read_pod_log(pod, container, ns, MAX_LOG_LINES, MAX_LOG_BYTES) or ""
        sanitized = truncate_log_output(logs)

        data: dict[str, Any] = {
            "experiment": name,
            "trial": trial_name,
            "pod": pod.metadata.name,
            "container": container,
            "logs": sanitized,
            "lines": len(sanitized.split("\n")) if sanitized else 0,
        }

        hint = extract_failure_hint(logs)
        if hint:
            data["failure_hint"] = hint
            data["next_steps"] = [
                f"Detected {hint['category']}: {hint['suggestion']}",
                f"Check get_hpo_experiment_events('{name}') for K8s-level issues",
            ]

        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_hpo_experiment_events(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get K8s events for a Katib experiment.

    Covers the Experiment and its Suggestion, every Trial and its workload
    (which shares the trial's name), and the experiment's pods, where
    scheduling and image-pull failures are reported. When there are more than
    ``limit``, the most recent are returned.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum events to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``events`` (up to ``limit``, oldest first)
        with type, involved object, reason, message and timestamp, plus the
        ``total`` found.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    limit, limit_err = clamp_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        _katib.read_experiment(name, ns)
        objects = _katib.experiment_objects(
            name, _katib.list_trials(name, ns), _katib.list_experiment_pods(name, ns)
        )
        events = sorted(
            (event_to_dict(e) for e in _katib.list_events_for(objects, ns)),
            key=lambda e: e["event_time"],
        )
        # Keep the most recent ones when limiting: those explain the current state.
        return ToolResponse(
            data={"experiment": name, "events": events[-limit:], "total": len(events)}
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def _select_trial(
    trials: list[dict[str, Any]], requested: str | None, experiment: dict[str, Any]
) -> dict[str, Any] | None:
    """The trial to read logs from: the requested one, else the best, else the oldest."""
    by_name = {_katib.name_of(t): t for t in trials}
    if requested is not None:
        return by_name.get(requested)
    best = optimal_trial_to_dict(experiment.get("status") or {})
    if best is not None and best["name"] in by_name:
        return by_name[best["name"]]
    return trials[0] if trials else None
