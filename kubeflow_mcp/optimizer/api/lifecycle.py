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

Both tools act on the Experiment CR through ``CustomObjectsApi`` with explicit
request timeouts:

    - get + delete_namespaced_custom_object()     → delete_hpo_experiment
    - get + patch_namespaced_custom_object(), plus
      the experiment's Trial CRs                  → update_hpo_experiment

Each reads the experiment once, decides ownership from that read, and writes
with that version as a precondition, so the write applies only to the state
that was checked.

``OptimizerClient.delete_job()`` is not used because it sends no request
timeout, so a confirmed delete could hang indefinitely on an unresponsive API
server.
"""

from typing import Any

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.core.security import validate_k8s_name
from kubeflow_mcp.optimizer.api import _katib
from kubeflow_mcp.optimizer.api._common import (
    check_optimizer_namespace,
    experiment_error,
    is_conflict,
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
from kubeflow_mcp.optimizer.types import ACTIVE_TRIAL_STATES, experiment_phase, trial_status

# The parameters update_hpo_experiment accepts, the Experiment spec field each one
# patches, and its bounds. These are exactly the fields Katib's validating
# webhook lets a live Experiment change. Katib has no pause: the webhook
# rejects parallelTrialCount <= 0, and lowering it deletes trials (see
# _trials_katib_will_delete). Upper bounds match create_hpo_experiment's guard
# rails.
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
                "Katib cannot pause an experiment. Lowering parallel_trial_count makes "
                "Katib delete trials, newest first and whatever their state, until the "
                "active ones fit the new value. Use delete_hpo_experiment() to stop it."
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


def _trials_katib_will_delete(
    trials: list[dict[str, Any]], parallel: int
) -> tuple[list[dict[str, Any]], bool]:
    """The trials Katib deletes after ``parallelTrialCount`` drops to ``parallel``.

    Replays Katib's experiment controller (``ReconcileTrials`` and
    ``deleteTrials``). Whenever more trials are active (pending or running) than
    the parallelism allows, it deletes that excess from all of the experiment's
    trials, newest first, whatever their state, and repeats on the next
    reconcile until the active trials fit. A completed trial newer than a
    running one is therefore deleted first, along with its results, and the
    running one in a later round.

    ``trials`` is oldest first, as :func:`_katib.list_trials` returns it. The
    result is newest first, plus whether it is exact. Katib orders by creation
    time alone, which Kubernetes records to the second, and does not break ties.
    When a cut falls between trials created in the same second, which of them
    Katib deletes is undetermined, so the list is one possible outcome.
    """
    remaining = list(trials)
    doomed: list[dict[str, Any]] = []
    exact = True
    while True:
        active = sum(trial_status(t) in ACTIVE_TRIAL_STATES for t in remaining)
        excess = active - parallel
        if excess <= 0:
            return [{"name": _katib.name_of(t), "status": trial_status(t)} for t in doomed], exact
        cut, kept = remaining[-excess:], remaining[:-excess]
        # Katib breaks no ties, so a cut between same-second trials is undetermined.
        if kept and _katib.created_at(kept[-1]) == _katib.created_at(cut[0]):
            exact = False
        doomed.extend(reversed(cut))
        remaining = kept


def _deletion_warning(lowering: bool, doomed: list[dict[str, Any]], exact: bool = True) -> str:
    """Sentence appended to update_hpo_experiment messages when parallelism drops.

    Always present for a reduction, even when no trial is expected to go: the
    count is an estimate from the trials at the time of the call, and a trial
    Katib starts before the update lands could still be deleted.
    """
    if not lowering:
        return ""
    if not doomed:
        return (
            " Lowering parallel_trial_count makes Katib delete trials above the new "
            "value without letting them finish; none would be deleted now, but a trial "
            "started before the update applies could be."
        )
    listed = ", ".join(f"{t['name']} ({t['status']})" for t in doomed)
    completed = sum(t["status"] not in ACTIVE_TRIAL_STATES for t in doomed)
    note = (
        f", including {completed} that already finished, whose results are lost"
        if completed
        else ""
    )
    ties = (
        ""
        if exact
        else (
            " Some of these were created in the same second as trials not listed, and Katib's "
            "choice among those is undetermined, so the trials it deletes may differ."
        )
    )
    return (
        " Lowering parallel_trial_count makes Katib delete trials newest first, whatever "
        "their state, until the active ones fit, without letting them finish. Based on the "
        f"trials now, it would delete {len(doomed)}: {listed}{note}.{ties} Deleted trials are "
        "not counted as killed."
    )


def _changed_since_read(name: str, action: str) -> dict[str, Any]:
    """The refusal for a write whose precondition no longer matches the experiment."""
    return ToolError(
        error=(
            f"Experiment '{name}' changed after it was checked, so it was not {action}d; "
            "nothing was changed"
        ),
        error_code=ErrorCode.VALIDATION_ERROR,
        hint="Call the tool again to re-check the experiment's current state.",
    ).model_dump()


def _update_response(
    name: str,
    namespace: str,
    changes: dict[str, dict[str, Any]],
    outcome: str,
    trials_to_delete: list[dict[str, Any]] | None = None,
    trials_to_delete_exact: bool = True,
) -> dict[str, Any]:
    """Single response shape for every update_hpo_experiment outcome."""
    return ToolResponse(
        data={
            "experiment": name,
            "namespace": namespace,
            "changes": changes,
            "trials_to_delete": trials_to_delete or [],
            "trials_to_delete_exact": trials_to_delete_exact,
            "message": f"Experiment '{name}' {outcome}",
        }
    ).model_dump()


def delete_hpo_experiment(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a Katib experiment permanently.

    Removes the experiment along with its trials and suggestion. Irreversible,
    so it requires ``confirmed=True``; the first call returns a preview. The
    delete is refused if the experiment changed after it was checked.

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
        current = _katib.read_experiment(name, ns)
        owner_err = require_mcp_ownership(current, name, "delete")
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

        # Preconditions bind the delete to the object checked above: if it was
        # replaced or changed since (its ownership label removed, say), the API
        # server refuses with 409 instead of deleting something not checked.
        metadata = current.get("metadata") or {}
        preconditions = {
            key: metadata[field]
            for key, field in (("uid", "uid"), ("resourceVersion", "resourceVersion"))
            if metadata.get(field)
        }
        try:
            mcp_utils.get_custom_objects_api().delete_namespaced_custom_object(
                group=KATIB_API_GROUP,
                version=KATIB_API_VERSION,
                namespace=ns,
                plural=EXPERIMENT_PLURAL,
                name=name,
                body={"preconditions": preconditions},
                _request_timeout=mcp_utils.K8S_WRITE_TIMEOUT,
            )
        except Exception as e:
            if not is_conflict(e):
                raise
            return _changed_since_read(name, "delete")

        return ToolResponse(
            data={
                "experiment": name,
                "namespace": ns,
                "deleted": True,
                "message": f"Experiment '{name}' deleted",
            }
        ).model_dump()

    except Exception as e:
        return experiment_error(e, name).model_dump()


def update_hpo_experiment(
    name: str,
    parallel_trial_count: int | None = None,
    max_trial_count: int | None = None,
    max_failed_trial_count: int | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Change a running experiment's concurrency or trial budget.

    These are the only fields Katib lets a live experiment change. Katib has no
    pause: lowering ``parallel_trial_count`` makes Katib delete trials, newest
    first and whatever their state, until the active ones fit, without letting
    them finish. The preview lists the trials that would go, by name and state.
    Raising ``max_trial_count`` extends the experiment, and restarts a finished
    one whose ``resumePolicy`` allows it. Edits Katib would reject are refused
    up front, and the update is refused if the experiment changed after it was
    read, so it never applies to a state the caller did not see.

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
        dict: Preview, or a response with ``experiment``, ``namespace``,
        ``changes`` mapping each changed parameter to ``{"from", "to"}``, and
        ``trials_to_delete``, the ``{"name", "status"}`` of each trial Katib is
        expected to delete, estimated from the trials when the call ran, and
        ``trials_to_delete_exact``, false when trials created in the same second
        make Katib's choice among them undetermined.

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
        # One read serves the ownership check, the rule checks and the patch's
        # resourceVersion precondition, so all three describe the same version.
        current = _katib.read_experiment(name, ns)
        owner_err = require_mcp_ownership(current, name, "update")
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
        spec = current.get("spec") or {}

        changes: dict[str, dict[str, Any]] = {}
        patch: dict[str, int] = {}
        for param, field, *_ in EDITABLE_FIELDS:
            if param in requested and spec.get(field) != requested[param]:
                changes[param] = {"from": spec.get(field), "to": requested[param]}
                patch[field] = requested[param]
        if not changes:
            return _update_response(name, ns, changes, "already has the requested settings")

        status = current.get("status") or {}
        violation = _katib_rule_violation({**spec, **patch}, status)
        if violation is not None:
            return ToolError(
                error=f"Katib would reject this update to '{name}': {violation}",
                error_code=ErrorCode.VALIDATION_ERROR,
                hint="Check current values with get_hpo_experiment(name).",
            ).model_dump()

        summary = ", ".join(f"{p} {c['from']} -> {c['to']}" for p, c in changes.items())
        new_parallel = patch.get("parallelTrialCount")
        lowering = new_parallel is not None and new_parallel < (spec.get("parallelTrialCount") or 0)
        doomed, exact = (
            _trials_katib_will_delete(_katib.list_trials(name, ns), new_parallel)
            if lowering
            else ([], True)
        )
        warning = _deletion_warning(lowering, doomed, exact)
        if not confirmed:
            return PreviewResponse(
                message=(
                    f"Will update experiment '{name}' in '{ns}': {summary}.{warning} "
                    "Set confirmed=True to proceed."
                ),
                config={
                    "experiment": name,
                    "namespace": ns,
                    "changes": changes,
                    "trials_to_delete": doomed,
                    "trials_to_delete_exact": exact,
                },
            ).model_dump()

        body: dict[str, Any] = {"spec": patch}
        resource_version = (current.get("metadata") or {}).get("resourceVersion")
        if resource_version:
            # Precondition: the API server rejects the patch with 409 if the
            # experiment changed since it was read, so the rule checks and the
            # trial list above always describe the state being changed.
            body["metadata"] = {"resourceVersion": resource_version}
        try:
            api.patch_namespaced_custom_object(
                **target_ref,
                body=body,
                # Patching an Experiment is admitted by the same webhooks as a
                # create, so it gets the write budget rather than the read timeout.
                _request_timeout=mcp_utils.K8S_WRITE_TIMEOUT,
            )
        except Exception as e:
            if not is_conflict(e):
                raise
            return _changed_since_read(name, "update")
        return _update_response(name, ns, changes, f"updated ({summary}).{warning}", doomed, exact)

    except Exception as e:
        return experiment_error(e, name).model_dump()
