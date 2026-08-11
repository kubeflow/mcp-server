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

SDK methods used:
    - OptimizerClient.get_job()                → get_experiment_trials
    - OptimizerClient.get_best_results()       → get_best_trial
    - OptimizerClient.wait_for_job_status()    → wait_for_experiment
    - OptimizerClient.get_job_logs()           → get_experiment_trial_logs
    - OptimizerClient.get_job_events()         → get_experiment_events
    - CustomObjectsApi                         → get_suggestion (no SDK method)
"""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.failures import extract_failure_hint
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_custom_objects_api,
    get_optimizer_client_for_namespace,
    get_optimizer_effective_namespace,
)
from kubeflow_mcp.core.security import (
    truncate_log_output,
    validate_k8s_name,
)
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
    result_to_dict,
    trial_counts_from_job,
    trial_to_dict,
)

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 1000
MAX_WAIT_TIMEOUT = 3600
MIN_POLLING_INTERVAL = 5


def get_experiment_trials(
    name: str,
    namespace: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """List trials for a Katib experiment with optional status filter.

    Trials are read from ``OptimizerClient.get_job().trials``.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        status: Optional status filter (e.g. Running, Complete, Failed).
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
        client = get_optimizer_client_for_namespace(namespace)
        job = client.get_job(name=name)

        trials = [trial_to_dict(t) for t in (getattr(job, "trials", None) or [])]
        if status:
            want = resolve_status_filter(status)
            trials = [t for t in trials if t.get("status") == want]

        data: dict[str, Any] = {
            "experiment": name,
            "trials": trials[:limit],
            "total": len(trials),
        }
        data["summary"] = trial_counts_from_job(job)
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_best_trial(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get the best trial with optimal hyperparameters and metrics.

    Uses ``OptimizerClient.get_best_results()``.

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
        client = get_optimizer_client_for_namespace(namespace)
        result = client.get_best_results(name=name)

        best = result_to_dict(result)
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

        spec = obj.get("spec", {})
        status = obj.get("status", {})
        algorithm = spec.get("algorithm", {})
        conditions = status.get("conditions", []) or []
        data = {
            "name": name,
            "namespace": ns,
            "algorithm": algorithm.get("algorithmName"),
            "requests": spec.get("requests"),
            "suggestion_count": status.get("suggestionCount"),
            "conditions": [
                {"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason")}
                for c in conditions
            ],
        }
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return resource_error(e, name, kind="Suggestion").model_dump()


def wait_for_experiment(
    name: str,
    namespace: str | None = None,
    timeout_seconds: int = 600,
    polling_interval: int = 15,
) -> dict[str, Any]:
    """Block until experiment reaches terminal status (Complete/Failed).

    Uses ``OptimizerClient.wait_for_job_status()``. Timeout is capped at 3600
    seconds and the polling interval has a minimum of 5 seconds.

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

    timeout_seconds = min(timeout_seconds, MAX_WAIT_TIMEOUT)
    polling_interval = max(polling_interval, MIN_POLLING_INTERVAL)

    try:
        client = get_optimizer_client_for_namespace(namespace)
        job = client.wait_for_job_status(
            name=name,
            status={OPTIMIZATION_JOB_COMPLETE, OPTIMIZATION_JOB_FAILED},
            timeout=timeout_seconds,
            polling_interval=polling_interval,
        )

        final_status = getattr(job, "status", None) or "Unknown"
        data: dict[str, Any] = {
            "experiment": name,
            "status": final_status,
            "reached": True,
            "message": f"Experiment reached '{final_status}'",
        }
        data.update(trial_counts_from_job(job))
        return ToolResponse(data=data).model_dump()

    except TimeoutError:
        return ToolResponse(
            data={
                "experiment": name,
                "status": "Unknown",
                "reached": False,
                "message": f"Timeout after {timeout_seconds}s",
                "hint": "Use get_experiment_events() to check for scheduling issues",
            }
        ).model_dump()
    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_experiment_trial_logs(
    name: str,
    trial: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get pod logs from an experiment trial.

    Uses ``OptimizerClient.get_job_logs()``. If no trial is specified, the SDK
    returns logs from the current best trial.

    Args:
        name: Experiment name.
        trial: Optional trial name. If omitted, uses the best trial.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``logs``, ``trial`` name, and ``failure_hint``.

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
        client = get_optimizer_client_for_namespace(namespace)
        log_lines = list(client.get_job_logs(name=name, trial_name=trial, follow=False))
        if len(log_lines) > MAX_LOG_LINES:
            log_lines = log_lines[-MAX_LOG_LINES:]

        logs = "\n".join(log_lines)
        sanitized = truncate_log_output(logs)

        data: dict[str, Any] = {
            "experiment": name,
            "trial": trial,
            "logs": sanitized,
            "lines": len(sanitized.split("\n")) if sanitized else 0,
        }

        hint = extract_failure_hint(logs)
        if hint:
            data["failure_hint"] = hint
            data["next_steps"] = [
                f"Detected {hint['category']}: {hint['suggestion']}",
                f"Check get_experiment_events('{name}') for K8s-level issues",
            ]

        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_experiment_events(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get K8s events for a Katib experiment.

    Uses ``OptimizerClient.get_job_events()``.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum events to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``events`` (up to ``limit``) with type,
        reason, message and timestamp, plus the ``total`` found.
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
        client = get_optimizer_client_for_namespace(namespace)
        events = client.get_job_events(name=name)

        event_list = [event_to_dict(e) for e in events[:limit]]
        return ToolResponse(
            data={"experiment": name, "events": event_list, "total": len(events)}
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()
