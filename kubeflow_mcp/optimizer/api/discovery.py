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
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_custom_objects_api,
    get_optimizer_client_for_namespace,
    get_optimizer_effective_namespace,
)
from kubeflow_mcp.core.security import validate_k8s_name
from kubeflow_mcp.optimizer.api._common import (
    DEFAULT_LIST_LIMIT,
    check_optimizer_namespace,
    clamp_limit,
    collection_error,
    experiment_error,
    resolve_status_filter,
)
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    SUGGESTION_PLURAL,
)
from kubeflow_mcp.optimizer.types import (
    conditions_to_list,
    early_stopping_to_dict,
    experiment_phase,
    experiment_summary,
    experiment_to_dict,
    is_success_status,
    optimal_trial_to_dict,
    trial_counts_from_cr,
    trial_to_dict,
)

logger = logging.getLogger(__name__)


def list_experiments(
    namespace: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """List Katib optimization experiments.

    Returns experiments in the target namespace with optional status filtering.

    Args:
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        status: Optional status filter (Created, Running, Complete, Failed).
        limit: Maximum experiments to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``experiments`` (up to ``limit``) and
        ``total`` (all matches, so truncation is visible).
    """
    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    limit, limit_err = clamp_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    try:
        client = get_optimizer_client_for_namespace(namespace)
        jobs = client.list_jobs()

        experiments = [experiment_summary(job) for job in jobs]
        if status:
            want = resolve_status_filter(status)
            experiments = [e for e in experiments if e.get("status") == want]

        return ToolResponse(
            data={"experiments": experiments[:limit], "total": len(experiments)}
        ).model_dump()

    except Exception as e:
        return collection_error(e).model_dump()


def get_experiment(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get full details of a Katib optimization experiment.

    Combines the SDK's view (search space, algorithm, objectives, trial config,
    embedded trials) with fields only the Experiment CR carries: ``conditions``,
    ``best_trial``, ``early_stopping``, the phase, and the trial counters.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum embedded trials to return. Defaults to 50, capped at 500.
            ``total_trials`` always reports the true count; use
            ``get_experiment_trials()`` to page through them.

    Returns:
        dict: Response containing ``name``, ``status``, ``creation_timestamp``,
        ``objectives``, ``algorithm``, ``search_space``, ``trial_config``,
        ``trials`` (up to ``limit``), ``trials_truncated``, ``conditions``,
        ``best_trial``, ``early_stopping`` and the CR trial counters. When the
        CR read fails, the CR-sourced keys are replaced by
        ``detail_unavailable: True`` and the SDK-derived values remain.

    Raises:
        ToolError: If experiment not found (``RESOURCE_NOT_FOUND``).
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

        data = experiment_to_dict(job)
        trials = getattr(job, "trials", None) or []
        data["trials"] = [trial_to_dict(t) for t in trials[:limit]]
        data["trials_truncated"] = len(trials) > limit
        data.update(_experiment_cr_detail(name, namespace))
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_experiment_status(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Lightweight status-only check for an experiment.

    Reads the Experiment CR directly rather than going through
    ``OptimizerClient.get_job()``, which additionally resolves every trial's
    TrainJob. That makes this both cheaper than ``get_experiment()`` and
    consistent with it: both report the CR's own counters, including
    ``early_stopped_trials``, which the SDK's trial view cannot express.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``name``, ``status`` and the CR trial
        counters (``total_trials``, ``running_trials``, ``succeeded_trials``,
        ``failed_trials``, ``pending_trials``, ``killed_trials``,
        ``early_stopped_trials``), omitting any the CR does not report.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        obj = _read_experiment_cr(name, namespace)
        status = obj.get("status") or {}

        data: dict[str, Any] = {
            "name": name,
            "status": experiment_phase(status),
        }
        data.update(trial_counts_from_cr(status))
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


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

    ns_err = check_optimizer_namespace(namespace)
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
        return experiment_error(e, experiment).model_dump()


def get_successful_trials(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get successful trials with hyperparameters and metrics.

    Returns only trials whose status denotes success, for comparison. Useful
    for finding the best hyperparameter combinations.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum trials to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``trials`` (up to ``limit``) with parameters
        and metrics, and ``total`` successful trials.
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

        successful = [
            trial_to_dict(t)
            for t in (getattr(job, "trials", None) or [])
            if is_success_status(getattr(getattr(t, "trainjob", None), "status", None))
        ]
        return ToolResponse(
            data={"experiment": name, "trials": successful[:limit], "total": len(successful)}
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def list_suggestions(
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """List Katib suggestion resources in namespace.

    Suggestions manage the optimization algorithm state. Useful for debugging
    when experiments are stuck. Uses CustomObjectsApi directly (no SDK method
    available).

    Args:
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum suggestions to return. Defaults to 50, capped at 500.

    Returns:
        dict: Response containing ``suggestions`` (up to ``limit``) with
        algorithm, status and request counts, plus the ``total`` found.
    """
    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    limit, limit_err = clamp_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

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
            data={
                "suggestions": suggestions[:limit],
                "total": len(suggestions),
                "namespace": ns,
            }
        ).model_dump()

    except Exception as e:
        return collection_error(e).model_dump()


def _read_experiment_cr(name: str, namespace: str | None) -> dict[str, Any]:
    """Fetch the raw Experiment custom resource.

    Raises whatever the Kubernetes client raises; callers decide whether that is
    fatal (``get_experiment_status``) or merely degrades the response
    (``get_experiment``).
    """
    ns = get_optimizer_effective_namespace(namespace)
    return get_custom_objects_api().get_namespaced_custom_object(
        group=KATIB_API_GROUP,
        version=KATIB_API_VERSION,
        namespace=ns,
        plural=EXPERIMENT_PLURAL,
        name=name,
        _request_timeout=K8S_TIMEOUT,
    )


def _experiment_cr_detail(name: str, namespace: str | None) -> dict[str, Any]:
    """Read the fields ``OptimizationJob`` omits straight from the Experiment CR.

    Supplies ``conditions`` (why an experiment failed), ``best_trial`` and
    ``early_stopping``, plus authoritative trial counts — Katib tracks
    early-stopped trials here, and that state never reaches the TrainJob the
    SDK derives its trial view from.

    Best-effort: this is supplementary detail, so a failure here degrades the
    response rather than failing the whole call.
    """
    try:
        obj = _read_experiment_cr(name, namespace)
    except Exception as e:
        logger.debug("get_experiment(%s): CR detail unavailable: %s", name, e)
        return {"detail_unavailable": True}

    status = obj.get("status") or {}
    detail: dict[str, Any] = {
        "conditions": conditions_to_list(status),
        "best_trial": optimal_trial_to_dict(status),
        "early_stopping": early_stopping_to_dict(obj.get("spec") or {}),
        # Derive the phase from the same source get_experiment_status uses, so
        # the two tools cannot disagree about a single experiment.
        "status": experiment_phase(status),
    }
    # CR counters are authoritative; they supersede the SDK-derived ones.
    detail.update(trial_counts_from_cr(status))
    return detail


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
