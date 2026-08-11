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
    - CustomObjectsApi.patch_namespaced_custom_object()  → update_experiment
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
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
)

logger = logging.getLogger(__name__)

ACTIONS = ("suspend", "resume")

# Katib has no first-class suspend field, so pausing is done by setting
# spec.parallelTrialCount to 0. The pre-suspend value is stashed in this
# annotation so resume restores the original concurrency instead of guessing.
PARALLELISM_ANNOTATION = "kubeflow-mcp/pre-suspend-parallel-trial-count"
DEFAULT_PARALLELISM = 1


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
    action: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Suspend or resume a Katib experiment.

    Katib exposes no suspend field, so ``"suspend"`` sets
    ``spec.parallelTrialCount`` to 0 — running trials finish but no new ones
    start. The previous value is recorded in an annotation so ``"resume"``
    restores the original concurrency.

    Non-admin personas may only update experiments created through MCP.

    Args:
        name: Experiment name.
        action: ``"suspend"`` to pause, ``"resume"`` to continue.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response with ``experiment``, ``namespace``, ``action`` and the
        resulting ``parallel_trial_count``.

    Raises:
        ToolError: If experiment not found or the action is invalid.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    if action not in ACTIONS:
        return ToolError(
            error=f"Invalid action '{action}'. Must be one of: {ACTIONS}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(namespace)

        owner_err = require_mcp_ownership(name, ns, action)
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
        annotations = (current.get("metadata") or {}).get("annotations") or {}
        parallelism = (current.get("spec") or {}).get("parallelTrialCount")

        if action == "suspend":
            if parallelism == 0:
                return ToolResponse(
                    data={
                        "experiment": name,
                        "namespace": ns,
                        "action": action,
                        "parallel_trial_count": 0,
                        "message": f"Experiment '{name}' is already suspended",
                    }
                ).model_dump()
            target = 0
            # Remember the concurrency we are pausing so resume can restore it.
            annotation_patch = {PARALLELISM_ANNOTATION: str(parallelism or DEFAULT_PARALLELISM)}
        else:
            stashed = annotations.get(PARALLELISM_ANNOTATION)
            target = int(stashed) if stashed and stashed.isdigit() else DEFAULT_PARALLELISM
            target = max(target, DEFAULT_PARALLELISM)
            # Clear the stash so a later suspend records a fresh value.
            annotation_patch = {PARALLELISM_ANNOTATION: None}

        api.patch_namespaced_custom_object(
            **target_ref,
            body={
                "metadata": {"annotations": annotation_patch},
                "spec": {"parallelTrialCount": target},
            },
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        past = "suspended" if action == "suspend" else "resumed"
        return ToolResponse(
            data={
                "experiment": name,
                "namespace": ns,
                "action": action,
                "parallel_trial_count": target,
                "message": f"Experiment '{name}' {past} (parallelTrialCount={target})",
            }
        ).model_dump()

    except Exception as e:
        logger.warning("update_experiment(%s, %s) failed: %s", name, action, e, exc_info=True)
        return experiment_error(e, name).model_dump()
