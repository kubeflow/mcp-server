# Copyright The Kubeflow Authors
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

"""Lifecycle tools for training job management."""

import logging
from typing import Any

from kubeflow.trainer.constants import constants as trainer_constants

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import (
    PreviewResponse,
    ToolError,
    ToolResponse,
    exception_details,
    is_k8s_not_found,
)
from kubeflow_mcp.core.config import get_effective_persona
from kubeflow_mcp.core.security import check_namespace_allowed, validate_k8s_name

logger = logging.getLogger(__name__)


def delete_training_job(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a training job permanently.

    This operation is irreversible - all job resources are removed.
    Requires ``confirmed=True`` to execute. First call returns a preview.

    Args:
        name: TrainJob name to delete.
        namespace: K8s namespace. When omitted, uses the configured default namespace.
        confirmed: Set ``True`` to delete. ``False`` returns a preview.

    Returns:
        dict: Response containing ``job``, ``namespace``, ``deleted`` (bool), ``message``.

    Raises:
        ToolError: If job not found (``RESOURCE_NOT_FOUND``).
    """
    name_err = validate_k8s_name(name)
    if name_err:
        return name_err.model_dump()

    try:
        ns_err = check_namespace_allowed(namespace)
        if ns_err:
            return ns_err.model_dump()

        ns = mcp_utils.get_trainer_effective_namespace(namespace)
        if get_effective_persona() not in ("platform-admin",):
            managed = mcp_utils.is_mcp_managed(name, ns)
            if managed is None:
                return ToolError(
                    error=f"Cannot verify ownership of training job '{name}' (API error)",
                    error_code=ErrorCode.SDK_ERROR,
                    details={"hint": "Retry, or use platform-admin persona to bypass."},
                ).model_dump()
            if not managed:
                return ToolError(
                    error=f"Training job '{name}' was not created by MCP",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={
                        "hint": (
                            "Data scientists can only delete jobs created through MCP tools. "
                            "Use platform-admin persona to delete externally created jobs."
                        ),
                    },
                ).model_dump()

        if not confirmed:
            return PreviewResponse(
                message=f"Will permanently delete training job '{name}'. Set confirmed=True to proceed.",
                config={"job": name, "namespace": ns},
            ).model_dump()

        client = mcp_utils.get_trainer_client_for_namespace(namespace)
        client.delete_job(name=name)

        return ToolResponse(
            data={
                "job": name,
                "namespace": ns,
                "deleted": True,
                "message": f"Training job '{name}' deleted successfully",
            }
        ).model_dump()

    except Exception as e:
        logger.warning("delete_training_job(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def update_training_job(
    name: str,
    action: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Suspend or resume a training job.

    Args:
        name: TrainJob name.
        action: ``"suspend"`` to pause execution, ``"resume"`` to continue.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``job``, ``namespace``, ``action``, ``message``.

    Raises:
        ToolError: If job not found or invalid action.
    """
    name_err = validate_k8s_name(name)
    if name_err:
        return name_err.model_dump()

    valid_actions = ("suspend", "resume")
    if action not in valid_actions:
        return ToolError(
            error=f"Invalid action '{action}'. Must be one of: {valid_actions}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    try:
        ns_err = check_namespace_allowed(namespace)
        if ns_err:
            return ns_err.model_dump()

        ns = mcp_utils.get_trainer_effective_namespace(namespace)
        if get_effective_persona() not in ("platform-admin",):
            managed = mcp_utils.is_mcp_managed(name, ns)
            if managed is None:
                return ToolError(
                    error=f"Cannot verify ownership of training job '{name}' (API error)",
                    error_code=ErrorCode.SDK_ERROR,
                    details={"hint": "Retry, or use platform-admin persona to bypass."},
                ).model_dump()
            if not managed:
                return ToolError(
                    error=f"Training job '{name}' was not created by MCP",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={
                        "hint": (
                            "Non-admin personas can only suspend/resume jobs created through MCP tools. "
                            "Use platform-admin persona for externally created jobs."
                        ),
                    },
                ).model_dump()

        api = mcp_utils.get_trainer_custom_objects_api()
        body = {"spec": {"suspend": action == "suspend"}}

        api.patch_namespaced_custom_object(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            namespace=ns,
            plural=trainer_constants.TRAINJOB_PLURAL,
            name=name,
            body=body,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        past = "suspended" if action == "suspend" else "resumed"
        return ToolResponse(
            data={
                "job": name,
                "namespace": ns,
                "action": action,
                "message": f"Training job '{name}' {past}",
            }
        ).model_dump()

    except Exception as e:
        logger.warning("update_training_job(%s, %s) failed: %s", name, action, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Training job '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
