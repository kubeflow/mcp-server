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

"""Lifecycle tools for Katib experiment management.

SDK methods used:
    - OptimizerClient.delete_job()                       → delete_experiment
    - CustomObjectsApi.get/patch_namespaced_custom_object()  → update_experiment
"""

import logging
from typing import Any

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.core.security import validate_k8s_name
from kubeflow_mcp.optimizer.api._common import (
    check_optimizer_namespace,
    experiment_error,
    require_mcp_ownership,
)
from kubeflow_mcp.optimizer.api.optimization import (
    MAX_PARALLEL_TRIAL_LIMIT,
    MAX_TRIAL_COUNT_LIMIT,
)
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    OPTIMIZATION_JOB_COMPLETE,
    OPTIMIZATION_JOB_FAILED,
)
from kubeflow_mcp.optimizer.types import experiment_phase

logger = logging.getLogger(__name__)

# The parameters update_experiment accepts, the Experiment spec field each one
# patches, and its bounds. These are exactly the fields Katib's validating
# webhook lets a live Experiment change. Katib has no pause: the webhook
# rejects parallelTrialCount <= 0, so throttling to 1 is the closest there is.
# Upper bounds match create_hpo_experiment's guard rails.
EDITABLE_FIELDS: tuple[tuple[str, str, int, int], ...] = (
    ("parallel_trial_count", "parallelTrialCount", 1, MAX_PARALLEL_TRIAL_LIMIT),
    ("max_trial_count", "maxTrialCount", 1, MAX_TRIAL_COUNT_LIMIT),
    ("max_failed_trial_count", "maxFailedTrialCount", 0, MAX_TRIAL_COUNT_LIMIT),
)

# Katib restarts a finished experiment only when it stopped by reaching
# maxTrialCount and kept its suggestion service (IsCompletedExperimentRestartable).
# The defaulting webhook sets resumePolicy to Never when it is omitted.
_RESTARTABLE_POLICIES = frozenset({"LongRunning", "FromVolume"})
_MAX_TRIALS_REACHED = "ExperimentMaxTrialsReached"
_DEFAULT_RESUME_POLICY = "Never"


def _check_requested(values: dict[str, Any]) -> ToolError | None:
    """Validate the requested values on their own, before any API call."""
    if not values:
        names = ", ".join(param for param, *_ in EDITABLE_FIELDS)
        return ToolError(
            error=f"Nothing to update: pass at least one of {names}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    for param, _, low, high in EDITABLE_FIELDS:
        if param not in values:
            continue
        value = values[param]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            hint = (
                "Katib cannot pause an experiment. Use parallel_trial_count=1 to throttle "
                "it, or delete_experiment() to stop it."
                if param == "parallel_trial_count" and value == 0
                else None
            )
            return ToolError(
                error=f"{param} must be an integer between {low} and {high}, got {value!r}",
                error_code=ErrorCode.VALIDATION_ERROR,
                hint=hint,
            )
    return None


def _succeeded_condition(status: dict[str, Any]) -> dict[str, Any] | None:
    """The Experiment's true ``Succeeded`` condition, if it has one."""
    for condition in status.get("conditions") or []:
        if condition.get("type") == "Succeeded" and condition.get("status") == "True":
            return condition
    return None


def _katib_rule_violation(spec: dict[str, Any], status: dict[str, Any]) -> str | None:
    """Mirror the rules Katib's webhook applies when an Experiment's spec changes.

    ``spec`` is the current spec with the requested values applied. Checking
    here means a preview is only ever shown for an edit Katib will admit.
    """
    parallel = spec.get("parallelTrialCount")
    max_trials = spec.get("maxTrialCount")
    max_failed = spec.get("maxFailedTrialCount")
    trials = status.get("trials") or 0

    if max_trials is not None:
        if parallel is not None and parallel > max_trials:
            return (
                f"parallel_trial_count ({parallel}) must not exceed max_trial_count ({max_trials})"
            )
        if max_failed is not None and max_failed > max_trials:
            return (
                f"max_failed_trial_count ({max_failed}) must not exceed "
                f"max_trial_count ({max_trials})"
            )
        if max_trials <= trials:
            return (
                f"max_trial_count ({max_trials}) must be greater than the {trials} trials "
                "already created"
            )

    phase = experiment_phase(status)
    if phase in (OPTIMIZATION_JOB_COMPLETE, OPTIMIZATION_JOB_FAILED):
        succeeded = _succeeded_condition(status)
        policy = spec.get("resumePolicy") or _DEFAULT_RESUME_POLICY
        restartable = (
            succeeded is not None
            and succeeded.get("reason") == _MAX_TRIALS_REACHED
            and policy in _RESTARTABLE_POLICIES
        )
        if not restartable:
            return (
                "the experiment has finished and Katib only restarts one that stopped at "
                "max_trial_count with resumePolicy LongRunning or FromVolume "
                f"(this one is {phase} with resumePolicy {policy})"
            )
    return None


def _update_response(
    name: str, namespace: str, changes: dict[str, dict[str, Any]], outcome: str
) -> dict[str, Any]:
    """Single response shape for every update_experiment outcome."""
    return ToolResponse(
        data={
            "experiment": name,
            "namespace": namespace,
            "changes": changes,
            "message": f"Experiment '{name}' {outcome}",
        }
    ).model_dump()


def delete_experiment(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a Katib experiment permanently.

    Removes the experiment along with its trials and suggestion. Irreversible,
    so it requires ``confirmed=True``; the first call returns a preview.

    Non-admin personas may only delete experiments created through MCP.

    Args:
        name: Experiment name to delete.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to delete. ``False`` returns a preview.

    Returns:
        dict: Preview, or deletion result with ``deleted`` (bool).

    Raises:
        ToolError: If experiment not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(namespace)

        owner_err = require_mcp_ownership(name, ns, "delete")
        if owner_err is not None:
            return owner_err.model_dump()

        if not confirmed:
            return PreviewResponse(
                message=(
                    f"Will permanently delete experiment '{name}' in '{ns}', "
                    "including its trials and suggestion. Set confirmed=True to proceed."
                ),
                config={"experiment": name, "namespace": ns},
            ).model_dump()

        client = mcp_utils.get_optimizer_client_for_namespace(namespace)
        client.delete_job(name=name)

        return ToolResponse(
            data={
                "experiment": name,
                "namespace": ns,
                "deleted": True,
                "message": f"Experiment '{name}' deleted",
            }
        ).model_dump()

    except Exception as e:
        logger.warning("delete_experiment(%s) failed: %s", name, e, exc_info=True)
        return experiment_error(e, name).model_dump()


def update_experiment(
    name: str,
    parallel_trial_count: int | None = None,
    max_trial_count: int | None = None,
    max_failed_trial_count: int | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Change a running experiment's concurrency or trial budget.

    These are the only fields Katib lets a live experiment change. Katib has no
    pause, so ``parallel_trial_count=1`` is the way to throttle one; raising
    ``max_trial_count`` extends it, and restarts a finished experiment whose
    ``resumePolicy`` allows it. Edits Katib would reject are refused up front.

    Requires ``confirmed=True``; the first call returns a preview of the change.
    Non-admin personas may only update experiments created through MCP.

    Args:
        name: Experiment name.
        parallel_trial_count: Trials to run concurrently (1 to 100).
        max_trial_count: Total trial budget (1 to 1000). Must exceed the
            trials already created.
        max_failed_trial_count: Failed trials tolerated before the experiment fails.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to apply. ``False`` returns a preview.

    Returns:
        dict: Preview, or a response with ``experiment``, ``namespace`` and
        ``changes`` mapping each changed parameter to ``{"from", "to"}``.

    Raises:
        ToolError: If the experiment is not found or the edit is invalid.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    supplied = {
        "parallel_trial_count": parallel_trial_count,
        "max_trial_count": max_trial_count,
        "max_failed_trial_count": max_failed_trial_count,
    }
    requested = {param: value for param, value in supplied.items() if value is not None}
    input_err = _check_requested(requested)
    if input_err is not None:
        return input_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(namespace)

        owner_err = require_mcp_ownership(name, ns, "update")
        if owner_err is not None:
            return owner_err.model_dump()

        api = mcp_utils.get_custom_objects_api()
        target_ref = {
            "group": KATIB_API_GROUP,
            "version": KATIB_API_VERSION,
            "namespace": ns,
            "plural": EXPERIMENT_PLURAL,
            "name": name,
        }
        current = api.get_namespaced_custom_object(
            **target_ref, _request_timeout=mcp_utils.K8S_TIMEOUT
        )
        spec = current.get("spec") or {}

        changes: dict[str, dict[str, Any]] = {}
        patch: dict[str, int] = {}
        for param, field, *_ in EDITABLE_FIELDS:
            if param in requested and spec.get(field) != requested[param]:
                changes[param] = {"from": spec.get(field), "to": requested[param]}
                patch[field] = requested[param]
        if not changes:
            return _update_response(name, ns, changes, "already has the requested settings")

        violation = _katib_rule_violation({**spec, **patch}, current.get("status") or {})
        if violation is not None:
            return ToolError(
                error=f"Katib would reject this update to '{name}': {violation}",
                error_code=ErrorCode.VALIDATION_ERROR,
                hint="Check current values with get_experiment(name).",
            ).model_dump()

        summary = ", ".join(f"{p} {c['from']} -> {c['to']}" for p, c in changes.items())
        if not confirmed:
            return PreviewResponse(
                message=(
                    f"Will update experiment '{name}' in '{ns}': {summary}. "
                    "Set confirmed=True to proceed."
                ),
                config={"experiment": name, "namespace": ns, "changes": changes},
            ).model_dump()

        api.patch_namespaced_custom_object(
            **target_ref,
            body={"spec": patch},
            # Patching an Experiment is admitted by the same webhooks as a
            # create, so it gets the write budget rather than the read timeout.
            _request_timeout=mcp_utils.K8S_WRITE_TIMEOUT,
        )
        return _update_response(name, ns, changes, f"updated ({summary})")

    except Exception as e:
        logger.warning("update_experiment(%s) failed: %s", name, e, exc_info=True)
        return experiment_error(e, name).model_dump()
