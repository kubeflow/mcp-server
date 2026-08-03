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

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED = ToolError(
    error="Not yet implemented — planned for Phase 3",
    error_code=ErrorCode.SDK_ERROR,
    hint="This tool is registered but not yet implemented. See KEP-34.",
).model_dump()


def delete_experiment(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a Katib experiment permanently.

    This operation is irreversible. Requires ``confirmed=True``
    to execute. First call returns a preview.

    Args:
        name: Experiment name to delete.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to delete. ``False`` returns a preview.

    Returns:
        dict: Preview or deletion result with ``deleted`` (bool).

    Raises:
        ToolError: If experiment not found (``RESOURCE_NOT_FOUND``).
    """
    return _NOT_IMPLEMENTED


def update_experiment(
    name: str,
    action: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Suspend or resume a Katib experiment.

    Suspend sets ``spec.parallelTrialCount`` to 0. Resume restores it.
    Uses CustomObjectsApi directly (no SDK method available).

    Args:
        name: Experiment name.
        action: ``"suspend"`` to pause, ``"resume"`` to continue.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response with ``action``, ``experiment``, ``namespace``.

    Raises:
        ToolError: If experiment not found or invalid action.
    """
    return _NOT_IMPLEMENTED
