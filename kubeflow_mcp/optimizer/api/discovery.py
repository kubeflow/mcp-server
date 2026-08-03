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

"""Discovery tools for Katib experiments, trials, and suggestions.

SDK methods used:
    - OptimizerClient.list_jobs()    → list_experiments
    - OptimizerClient.get_job()      → get_experiment, get_experiment_status,
                                       get_trial, get_successful_trials
    - CustomObjectsApi               → list_suggestions (no SDK method)
"""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_custom_objects_api,
    get_optimizer_client_for_namespace,
    get_optimizer_effective_namespace,
)
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name
from kubeflow_mcp.optimizer.constants import (
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    SUGGESTION_PLURAL,
)
from kubeflow_mcp.optimizer.types import (
    experiment_summary,
    experiment_to_dict,
    is_success_status,
    trial_counts,
    trial_to_dict,
)

logger = logging.getLogger(__name__)

# Legacy/CRD status "Succeeded" maps to the SDK's terminal status "Complete".
# Mirrors the trainer's list_training_jobs alias for consistent filtering.
_STATUS_FILTER_ALIASES: dict[str, str] = {"Succeeded": "Complete"}


def list_experiments(
    namespace: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """List Katib optimization experiments.

    Returns experiments in the target namespace with optional status filtering.

    Args:
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        status: Optional status filter (Created, Running, Complete, Failed).

    Returns:
        dict: Response containing list of experiments with name, status, trial counts.
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        jobs = client.list_jobs()

        experiments = [experiment_summary(job) for job in jobs]
        if status:
            want = _STATUS_FILTER_ALIASES.get(status, status)
            experiments = [e for e in experiments if e.get("status") == want]

        return ToolResponse(
            data={"experiments": experiments, "total": len(experiments)}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_experiment(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get full details of a Katib optimization experiment.

    Returns the complete experiment: status, trials, search space, algorithm,
    objectives, and trial configuration.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing full experiment details.

    Raises:
        ToolError: If experiment not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        job = client.get_job(name=name)

        data = experiment_to_dict(job)
        data["trials"] = [trial_to_dict(t) for t in (getattr(job, "trials", None) or [])]
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Experiment '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_experiments() to find available experiments",
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_experiment_status(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Lightweight status-only check for an experiment.

    Returns only the status string and trial counts — faster to consume than
    ``get_experiment()`` for polling.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``status``, ``total_trials``,
              ``running_trials``, ``succeeded_trials``, ``failed_trials``.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        job = client.get_job(name=name)

        data: dict[str, Any] = {
            "name": name,
            "status": getattr(job, "status", None) or "Unknown",
        }
        data.update(trial_counts(job))
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Experiment '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_experiments() to find available experiments",
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_trial(
    name: str,
    experiment: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get details of a specific trial within an experiment.

    Returns trial parameters, metrics, and status for debugging. The trial is
    located within ``OptimizerClient.get_job().trials``.

    Args:
        name: Trial name.
        experiment: Parent experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing trial parameters, metrics, and status.

    Raises:
        ToolError: If experiment or trial not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name, "name")
    if name_err is not None:
        return name_err.model_dump()
    exp_err = validate_k8s_name(experiment, "experiment")
    if exp_err is not None:
        return exp_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        job = client.get_job(name=experiment)

        for trial in getattr(job, "trials", None) or []:
            if getattr(trial, "name", None) == name:
                data = trial_to_dict(trial)
                data["experiment"] = experiment
                return ToolResponse(data=data).model_dump()

        return ToolError(
            error=f"Trial '{name}' not found in experiment '{experiment}'",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            hint=f"Use get_experiment_trials('{experiment}') to list trials",
        ).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Experiment '{experiment}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_experiments() to find available experiments",
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_successful_trials(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get all successful trials with hyperparameters and metrics.

    Returns only trials whose status denotes success, for comparison. Useful
    for finding the best hyperparameter combinations.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing list of successful trials with
              parameters and metrics.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        job = client.get_job(name=name)

        successful = [
            trial_to_dict(t)
            for t in (getattr(job, "trials", None) or [])
            if is_success_status(getattr(getattr(t, "trainjob", None), "status", None))
        ]
        return ToolResponse(
            data={"experiment": name, "trials": successful, "total": len(successful)}
        ).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Experiment '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                hint="Use list_experiments() to find available experiments",
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def list_suggestions(
    namespace: str | None = None,
) -> dict[str, Any]:
    """List Katib suggestion resources in namespace.

    Suggestions manage the optimization algorithm state. Useful for debugging
    when experiments are stuck. Uses CustomObjectsApi directly (no SDK method
    available).

    Args:
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing list of suggestion resources with
              algorithm, status, and request counts.
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        api = get_custom_objects_api()
        resp = api.list_namespaced_custom_object(
            group=KATIB_API_GROUP,
            version=KATIB_API_VERSION,
            namespace=ns,
            plural=SUGGESTION_PLURAL,
            _request_timeout=K8S_TIMEOUT,
        )

        suggestions = [_suggestion_to_dict(item) for item in resp.get("items", [])]
        return ToolResponse(
            data={"suggestions": suggestions, "total": len(suggestions), "namespace": ns}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def _suggestion_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Serialize a raw Suggestion CRD dict from CustomObjectsApi."""
    metadata = item.get("metadata", {})
    spec = item.get("spec", {})
    status = item.get("status", {})
    algorithm = spec.get("algorithm", {})
    conditions = status.get("conditions", []) or []
    latest_condition = conditions[-1].get("type") if conditions else None
    return {
        "name": metadata.get("name"),
        "algorithm": algorithm.get("algorithmName"),
        "requests": spec.get("requests"),
        "suggestion_count": status.get("suggestionCount"),
        "condition": latest_condition,
    }
