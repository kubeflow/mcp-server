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

Every tool reads Katib's custom resources through ``CustomObjectsApi``:
Experiments, Trials and Suggestions. The SDK's ``OptimizerClient`` readers are
not used because they only describe experiments using the random or grid
algorithm with TrainJob trials (see ``optimizer/api/_katib.py``).
"""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse
from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_custom_objects_api,
    get_optimizer_effective_namespace,
)
from kubeflow_mcp.core.security import validate_k8s_name
from kubeflow_mcp.optimizer.api import _katib
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
    TRIAL_SUCCEEDED,
    experiment_phase,
    experiment_summary_from_cr,
    experiment_to_dict,
    suggestion_to_dict,
    trial_counts_from_cr,
    trial_to_dict,
)

logger = logging.getLogger(__name__)


def list_hpo_experiments(
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
        ns = get_optimizer_effective_namespace(namespace)
        resp = get_custom_objects_api().list_namespaced_custom_object(
            group=KATIB_API_GROUP,
            version=KATIB_API_VERSION,
            namespace=ns,
            plural=EXPERIMENT_PLURAL,
            _request_timeout=K8S_TIMEOUT,
        )

        experiments = [experiment_summary_from_cr(item) for item in resp.get("items", [])]
        if status:
            want = resolve_status_filter(status)
            experiments = [e for e in experiments if e.get("status") == want]

        return ToolResponse(
            data={"experiments": experiments[:limit], "total": len(experiments)}
        ).model_dump()

    except Exception as e:
        return collection_error(e).model_dump()


def get_hpo_experiment(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get full details of a Katib optimization experiment.

    Reads the Experiment CR (search space, algorithm, objectives, trial config,
    ``conditions``, ``best_trial``, ``early_stopping``, phase and trial
    counters) and its Trial CRs, so it works for every algorithm and trial
    workload kind.

    Args:
        name: Experiment name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum embedded trials to return. Defaults to 50, capped at 500.
            ``total_trials`` always reports the true count; use
            ``get_hpo_experiment_trials()`` to page through them.

    Returns:
        dict: Response containing ``name``, ``status``, ``creation_timestamp``,
        ``objectives``, ``algorithm``, ``search_space``, ``trial_config``,
        ``trials`` (up to ``limit``), ``trials_truncated``, ``conditions``,
        ``best_trial``, ``early_stopping`` and the CR trial counters.

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
        ns = get_optimizer_effective_namespace(namespace)
        data = experiment_to_dict(_katib.read_experiment(name, ns))
        trials = _katib.list_trials(name, ns)
        data["trials"] = [trial_to_dict(t) for t in trials[:limit]]
        data["trials_truncated"] = len(trials) > limit
        return ToolResponse(data=data).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_hpo_experiment_status(
    name: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Lightweight status-only check for an experiment.

    Reads only the Experiment CR, so it is cheaper than ``get_hpo_experiment()``
    while reporting the same phase and counters, including
    ``early_stopped_trials``.

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
        obj = _katib.read_experiment(name, get_optimizer_effective_namespace(namespace))
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
    trial: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get details of a specific trial within an experiment.

    Returns trial parameters, metrics, status and workload kind for debugging.

    Args:
        name: Experiment name. Every optimizer tool takes the experiment as
            ``name``; the trial is selected with ``trial``.
        trial: Trial name within that experiment.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing trial parameters, metrics, and status.

    Raises:
        ToolError: If experiment or trial not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()
    trial_err = validate_k8s_name(trial, "trial")
    if trial_err is not None:
        return trial_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = get_optimizer_effective_namespace(namespace)
        # Read the experiment first so a missing experiment is reported as
        # such, rather than as a missing trial.
        _katib.read_experiment(name, ns)
        for candidate in _katib.list_trials(name, ns):
            if (candidate.get("metadata") or {}).get("name") == trial:
                data = trial_to_dict(candidate)
                data["experiment"] = name
                return ToolResponse(data=data).model_dump()

        return ToolError(
            error=f"Trial '{trial}' not found in experiment '{name}'",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            hint=f"Use get_hpo_experiment_trials('{name}') to list trials",
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def get_successful_trials(
    name: str,
    namespace: str | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """Get successful trials with hyperparameters and metrics.

    Returns only trials Katib marked Succeeded (status ``Complete``), for
    comparison. Useful for finding the best hyperparameter combinations.

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
        ns = get_optimizer_effective_namespace(namespace)
        _katib.read_experiment(name, ns)
        trials = [trial_to_dict(t) for t in _katib.list_trials(name, ns)]
        successful = [t for t in trials if t["status"] == TRIAL_SUCCEEDED]
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

        suggestions = [suggestion_to_dict(item) for item in resp.get("items", [])]
        return ToolResponse(
            data={
                "suggestions": suggestions[:limit],
                "total": len(suggestions),
                "namespace": ns,
            }
        ).model_dump()

    except Exception as e:
        return collection_error(e).model_dump()
