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

"""Shared checks and error mapping for the pipelines tools."""

import logging
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, exception_details
from kubeflow_mcp.common.utils import get_pipelines_effective_namespace
from kubeflow_mcp.core.security import check_namespace_allowed, validate_namespace

logger = logging.getLogger(__name__)

MAX_LIST_LIMIT = 200
MAX_NAME_LENGTH = 256

# PipelinesClient.run() and upload_pipeline() treat a string ending in one of these as
# a path on the server's filesystem and read it. Tools only ever take pipeline names.
_FILE_SUFFIXES = (".yaml", ".yml", ".zip", ".tgz", ".tar.gz")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# KFP v2 run IDs are UUIDs.
_RUN_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
# The SDK reports a missing object as a ValueError; these are its message prefixes.
_NOT_FOUND_RE = re.compile(r"^(Pipeline|Pipeline version|Experiment) not found\b")
_AMBIGUOUS_RE = re.compile(r"^Multiple (pipelines|versions) found\b")

RUN_STATES = frozenset(
    {"pending", "running", "succeeded", "skipped", "failed", "canceling", "canceled", "paused"}
)
TERMINAL_RUN_STATES = frozenset({"succeeded", "skipped", "failed", "canceled"})

INSTALL_HINT = "Install the pipelines extra on the server host: pip install 'kubeflow[pipelines]'"


def validate_display_name(value: str | None, field: str) -> ToolError | None:
    """Check a pipeline, version, experiment or run display name."""
    if not isinstance(value, str) or not value.strip():
        return ToolError(error=f"{field} is required", error_code=ErrorCode.VALIDATION_ERROR)
    if len(value) > MAX_NAME_LENGTH:
        return ToolError(
            error=f"{field} is too long ({len(value)} chars, max {MAX_NAME_LENGTH})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    if _CONTROL_CHARS.search(value):
        return ToolError(
            error=f"{field} contains control characters", error_code=ErrorCode.VALIDATION_ERROR
        )
    if value.strip().lower().endswith(_FILE_SUFFIXES):
        return ToolError(
            error=f"{field} looks like a file path; pass the name of a registered pipeline",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Find registered pipelines with list_pipelines()",
        )
    return None


def validate_run_id(run_id: str | None) -> ToolError | None:
    """Check that *run_id* is a KFP run ID (a UUID)."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        return ToolError(
            error=f"Invalid run_id {run_id!r}: expected a run UUID",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Find run IDs with list_pipeline_runs()",
        )
    return None


def validate_limit(limit: int) -> ToolError | None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIST_LIMIT:
        return ToolError(
            error=f"limit must be between 1 and {MAX_LIST_LIMIT}, got {limit}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None


def resolve_namespace(namespace: str | None) -> tuple[str | None, ToolError | None]:
    """Resolve the namespace a call will run in, then check it against policy.

    The resolved namespace is checked explicitly and handed to the client, so the
    namespace that passes the allowlist is the one the SDK operates in. Fails closed
    when the default cannot be resolved.
    """
    if namespace is not None:
        format_err = validate_namespace(namespace)
        if format_err is not None:
            return None, format_err
    try:
        effective = get_pipelines_effective_namespace(namespace)
    except Exception as e:
        logger.warning("Cannot resolve pipelines namespace: %s", e)
        return None, ToolError(
            error="Cannot resolve the namespace for Kubeflow Pipelines; pass namespace= explicitly",
            error_code=ErrorCode.PERMISSION_DENIED,
        )
    ns_err = check_namespace_allowed(effective)
    if ns_err is not None:
        return None, ns_err
    return effective, None


def check_run_namespace(client: Any, run: Any, namespace: str) -> ToolError | None:
    """Verify that a run belongs to the expected namespace via its experiment."""
    exp_id = getattr(run, "experiment_id", None)
    if exp_id is None and isinstance(run, dict):
        exp_id = run.get("experiment_id")
    if not exp_id:
        return None

    try:
        kfp_c = getattr(client, "kfp_client", None)
        if kfp_c is None:
            return None
        exp = kfp_c.get_experiment(experiment_id=exp_id)
    except Exception as e:
        logger.debug("Could not verify experiment %s namespace: %s", exp_id, e)
        return None

    exp_ns = getattr(exp, "namespace", None)
    if exp_ns is None and isinstance(exp, dict):
        exp_ns = exp.get("namespace")

    if isinstance(exp_ns, str) and exp_ns and exp_ns != namespace:
        run_id = getattr(run, "run_id", None) or (
            run.get("run_id") if isinstance(run, dict) else None
        )
        return ToolError(
            error=f"Run '{run_id}' is not in namespace '{namespace}'",
            error_code=ErrorCode.PERMISSION_DENIED,
        )
    return None


def _http_status(exc: Exception) -> int | None:
    # kfp_server_api and kubernetes both raise an ``ApiException`` with an int status.
    # Matched by class name, including base classes, so this works without importing
    # either package and still catches subclasses.
    if not any(cls.__name__ == "ApiException" for cls in type(exc).__mro__):
        return None
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def pipelines_error(exc: Exception, action: str) -> ToolError:
    """Map an exception from the SDK or the KFP API to a ``ToolError``."""
    if isinstance(exc, ImportError):
        return ToolError(
            error=f"Kubeflow Pipelines support is not installed: {exc}",
            error_code=ErrorCode.SDK_ERROR,
            hint=INSTALL_HINT,
        )
    if isinstance(exc, ValueError):
        message = str(exc)
        if _NOT_FOUND_RE.match(message):
            return ToolError(error=message, error_code=ErrorCode.RESOURCE_NOT_FOUND)
        if _AMBIGUOUS_RE.match(message):
            return ToolError(error=message, error_code=ErrorCode.VALIDATION_ERROR)
    if isinstance(exc, TimeoutError):
        return ToolError(error=str(exc), error_code=ErrorCode.TIMEOUT)

    status = _http_status(exc)
    if status == 404:
        return ToolError(
            error=f"Not found while trying to {action}", error_code=ErrorCode.RESOURCE_NOT_FOUND
        )
    if status in (401, 403):
        return ToolError(
            error=f"Permission denied while trying to {action}",
            error_code=ErrorCode.PERMISSION_DENIED,
            details=exception_details(exc),
        )

    logger.warning("Failed to %s: %s", action, exc, exc_info=True)
    return ToolError(
        error=f"Failed to {action}: {exc}",
        error_code=ErrorCode.SDK_ERROR,
        details=exception_details(exc),
    )


def page_fields(response: Any) -> dict[str, Any]:
    """Paging fields shared by every list response."""
    token = getattr(response, "next_page_token", None) or ""
    return {
        "next_page_token": token,
        "has_more": bool(token),
        "total": getattr(response, "total_size", None),
    }
