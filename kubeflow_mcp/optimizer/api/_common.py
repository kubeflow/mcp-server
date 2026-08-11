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

"""Shared helpers for optimizer tools.

Holds the two blocks that would otherwise be copy-pasted across every tool in
``optimizer/api``: mapping SDK exceptions onto the right ``ToolError``, and
enforcing MCP ownership before a mutating operation.
"""

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, exception_details, is_k8s_not_found
from kubeflow_mcp.core.policy import get_effective_persona
from kubeflow_mcp.core.security import check_namespace_allowed

_FIND_HINT = "Use list_experiments() to find available experiments"

# Collection responses are truncated so a large experiment cannot exhaust the
# agent's context window. Matches the trainer client's list bounds.
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 500

# Katib's CRD vocabulary calls a finished resource "Succeeded"; the SDK reports
# "Complete". Accept both wherever a caller filters by status, so the same value
# works against experiments and trials alike.
STATUS_FILTER_ALIASES: dict[str, str] = {"Succeeded": "Complete"}


def resolve_status_filter(status: str) -> str:
    """Map a caller-supplied status onto the value the SDK actually reports."""
    return STATUS_FILTER_ALIASES.get(status, status)


def clamp_limit(limit: int) -> tuple[int, ToolError | None]:
    """Validate and cap a caller-supplied result limit.

    Returns the effective limit and, when the input is invalid, the
    ``ToolError`` the tool should return instead of calling the cluster.
    """
    if limit < 1:
        return 0, ToolError(
            error=f"limit must be >= 1, got {limit}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return min(limit, MAX_LIST_LIMIT), None


def check_optimizer_namespace(namespace: str | None) -> ToolError | None:
    """Namespace policy check for optimizer tools.

    Every optimizer tool must go through this rather than calling
    :func:`check_namespace_allowed` directly. The generic helper defaults to
    resolving an omitted namespace through the *trainer* client, which is a
    different backend than the one these tools act on — checking one namespace
    while operating on another would let the allowlist be bypassed.
    """
    return check_namespace_allowed(namespace, resolver=mcp_utils.get_optimizer_effective_namespace)


def _api_status(exc: Exception) -> int | None:
    """HTTP status of the underlying Kubernetes API error, if there is one.

    Mirrors :func:`is_k8s_not_found` by walking the cause chain, because the
    optimizer SDK re-raises API failures as ``RuntimeError(...) from ApiException``.
    """
    try:
        from kubernetes.client.exceptions import ApiException
    except ImportError:  # pragma: no cover - kubernetes is a hard dependency
        return None
    for candidate in (exc, exc.__cause__, exc.__context__):
        if isinstance(candidate, ApiException):
            return candidate.status
    return None


def resource_error(exc: Exception, name: str, kind: str = "Experiment") -> ToolError:
    """Map an exception raised by the SDK or K8s API onto a ``ToolError``.

    The optimizer SDK wraps API failures as ``RuntimeError(...) from
    ApiException``, so a missing resource is only visible through the cause
    chain — :func:`is_k8s_not_found` handles that.

    Args:
        exc: The exception raised by the SDK or Kubernetes client.
        name: Name of the resource the operation targeted.
        kind: Resource kind, used in the message. Defaults to ``"Experiment"``.
    """
    if is_k8s_not_found(exc):
        return ToolError(
            error=f"{kind} '{name}' not found",
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            hint=(
                _FIND_HINT
                if kind == "Experiment"
                else "A Suggestion is created once an experiment starts running."
            ),
        )
    if _api_status(exc) == 409:
        # A name collision is caller-fixable, so report it as a validation
        # error: SDK_ERROR would also count as an infrastructure failure and
        # needlessly trip the circuit breaker for this tool.
        return ToolError(
            error=f"{kind} '{name}' already exists",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint=(
                "Choose a different name, or delete the existing experiment "
                f"with delete_experiment('{name}', confirmed=True)."
            ),
        )
    return ToolError(
        error=str(exc),
        error_code=ErrorCode.SDK_ERROR,
        details=exception_details(exc),
    )


def experiment_error(exc: Exception, name: str) -> ToolError:
    """:func:`resource_error` for the Experiment kind, which most tools target."""
    return resource_error(exc, name)


def collection_error(exc: Exception) -> ToolError:
    """Map an exception from a list operation onto a ``ToolError``.

    List tools have no single resource name to report, so they cannot use
    :func:`resource_error`. A 403 is still surfaced distinctly because a
    namespace the caller cannot read is caller-fixable, not an outage.
    """
    if _api_status(exc) == 403:
        return ToolError(
            error=f"Not authorized to list this resource: {exc}",
            error_code=ErrorCode.PERMISSION_DENIED,
            hint="Check the service account's RBAC for the target namespace.",
        )
    return ToolError(
        error=str(exc),
        error_code=ErrorCode.SDK_ERROR,
        details=exception_details(exc),
    )


def require_mcp_ownership(name: str, namespace: str, action: str) -> ToolError | None:
    """Block non-admin personas from mutating experiments MCP did not create.

    Returns ``None`` when the operation is allowed (admin persona, or the
    experiment carries the MCP ownership label), otherwise the ``ToolError``
    to return to the caller.
    """
    if get_effective_persona() == "platform-admin":
        return None

    managed = mcp_utils.is_optimizer_mcp_managed(name, namespace)
    if managed is None:
        return ToolError(
            error=f"Cannot verify ownership of experiment '{name}' (API error)",
            error_code=ErrorCode.SDK_ERROR,
            hint="Retry, or use the platform-admin persona to bypass the check.",
        )
    if not managed:
        return ToolError(
            error=f"Experiment '{name}' was not created by MCP",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint=(
                f"Non-admin personas can only {action} experiments created through "
                "MCP tools. Use the platform-admin persona for external experiments."
            ),
        )
    return None
