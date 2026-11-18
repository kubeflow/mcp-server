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

"""Experiment tools for Kubeflow Pipelines (list / get / create / delete).

An experiment groups runs. ``create_pipeline_run(experiment=...)`` needs an existing
experiment, so agents create or pick one here first.
"""

from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.common.utils import get_pipelines_client
from kubeflow_mcp.pipelines.api._common import (
    _NOT_FOUND_RE,
    check_experiment_namespace,
    page_fields,
    pipelines_error,
    resolve_namespace,
    validate_confirmed,
    validate_display_name,
    validate_limit,
)
from kubeflow_mcp.pipelines.types import experiment_to_dict

MAX_DESCRIPTION_LENGTH = 1000
_DELETE_NOTE = (
    "KFP deletes the experiment but not its runs or recurring runs; they stay in the "
    "namespace without an experiment."
)


def validate_description(description: str | None) -> ToolError | None:
    """Check an optional free-text description."""
    if description is None:
        return None
    if not isinstance(description, str):
        return ToolError(
            error="description must be a string", error_code=ErrorCode.VALIDATION_ERROR
        )
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return ToolError(
            error=f"description is too long ({len(description)} chars, "
            f"max {MAX_DESCRIPTION_LENGTH})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None


def _find_experiment(client: Any, name: str) -> Any | None:
    """Return the experiment called *name*, or None when it does not exist.

    The SDK reports a missing experiment as a ``ValueError``; every other error
    (including an ambiguous name) propagates to the caller.
    """
    try:
        return client.get_experiment(name)
    except ValueError as e:
        if _NOT_FOUND_RE.match(str(e)):
            return None
        raise


def list_experiments(
    limit: int = 50,
    page_token: str = "",
    namespace: str | None = None,
) -> dict[str, Any]:
    """List experiments on the Kubeflow Pipelines server.

    Args:
        limit: Experiments per page, 1 to 200. Defaults to 50.
        page_token: ``next_page_token`` from a previous call, to read the next page.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing ``experiments`` (``experiment_id``, ``name``,
        ``description``, ``namespace``, ``created_at``, ...), ``next_page_token``,
        ``has_more`` and ``total``. Experiments whose namespace cannot be verified
        against the requested one are dropped and counted in ``experiments_filtered``.
    """
    limit_err = validate_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        response = get_pipelines_client(ns).list_experiments(
            page_token=page_token or "", page_size=limit
        )
        experiments: list[dict[str, Any]] = []
        filtered = 0
        for raw in getattr(response, "experiments", None) or []:
            if check_experiment_namespace(raw, ns) is not None:
                filtered += 1
                continue
            experiments.append(experiment_to_dict(raw))
        data: dict[str, Any] = {
            "experiments": experiments,
            "namespace": ns,
            **page_fields(response),
        }
        if filtered:
            data["experiments_filtered"] = filtered
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        return pipelines_error(e, "list experiments").model_dump()


def get_experiment(name: str, namespace: str | None = None) -> dict[str, Any]:
    """Get an experiment by name.

    Args:
        name: Experiment display name, as shown by ``list_experiments()``.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing the experiment fields and ``next_steps``.
    """
    name_err = validate_display_name(name, "name")
    if name_err is not None:
        return name_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        experiment = get_pipelines_client(ns).get_experiment(name)
        exp_ns_err = check_experiment_namespace(experiment, ns)
        if exp_ns_err is not None:
            return exp_ns_err.model_dump()
        data = experiment_to_dict(experiment)
        data["next_steps"] = [
            f"list_pipeline_runs(experiment='{name}') to see its runs",
            f"create_pipeline_run(pipeline=..., experiment='{name}') to preview a run in it",
        ]
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        return pipelines_error(e, f"get experiment '{name}'").model_dump()


def create_experiment(
    name: str,
    description: str | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create an experiment to group pipeline runs (two-phase confirm).

    Call once with ``confirmed=False`` (the default) to preview, then again with
    ``confirmed=True`` to create it. If an experiment with this name already exists,
    it is returned unchanged and nothing new is created.

    Args:
        name: Experiment display name.
        description: Optional description. Ignored when the experiment already exists.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to create the experiment. ``False`` returns a preview.

    Returns:
        dict: Preview, or the experiment (``experiment_id``, ``name``, ...) with
        ``created`` telling whether it is new.
    """
    confirm_err = validate_confirmed(confirmed)
    if confirm_err is not None:
        return confirm_err.model_dump()

    name_err = validate_display_name(name, "name") or validate_description(description)
    if name_err is not None:
        return name_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        existing = _find_experiment(client, name)
    except Exception as e:
        return pipelines_error(e, f"look up experiment '{name}'").model_dump()

    if existing is not None:
        exp_ns_err = check_experiment_namespace(existing, ns)
        if exp_ns_err is not None:
            return exp_ns_err.model_dump()
        data = experiment_to_dict(existing)
        data["created"] = False
        data["message"] = f"Experiment '{name}' already exists; nothing was created"
        data["next_steps"] = [
            f"create_pipeline_run(pipeline=..., experiment='{name}') to preview a run in it"
        ]
        return ToolResponse(data=data).model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will create experiment '{name}' in namespace '{ns}'. "
                "Set confirmed=True to create it."
            ),
            config={"name": name, "description": description, "namespace": ns},
        ).model_dump()

    try:
        experiment = client.create_experiment(name, description=description)
    except Exception as e:
        return pipelines_error(e, f"create experiment '{name}'").model_dump()

    data = experiment_to_dict(experiment)
    data["created"] = True
    data["next_steps"] = [
        f"create_pipeline_run(pipeline=..., experiment='{name}') to preview a run in it"
    ]
    return ToolResponse(data=data).model_dump()


def delete_experiment(
    name: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """[DESTRUCTIVE] Delete an experiment (two-phase confirm).

    Call once with ``confirmed=False`` to preview, then again with ``confirmed=True``.
    KFP keeps the experiment's runs and recurring runs; only the experiment goes away.

    Args:
        name: Experiment display name.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to delete the experiment. ``False`` returns a preview.

    Returns:
        dict: Preview, or a confirmation that the experiment was deleted.
    """
    confirm_err = validate_confirmed(confirmed)
    if confirm_err is not None:
        return confirm_err.model_dump()

    name_err = validate_display_name(name, "name")
    if name_err is not None:
        return name_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        experiment = client.get_experiment(name)
    except Exception as e:
        return pipelines_error(e, f"get experiment '{name}'").model_dump()

    exp_ns_err = check_experiment_namespace(experiment, ns)
    if exp_ns_err is not None:
        return exp_ns_err.model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will delete experiment '{name}' in namespace '{ns}'. {_DELETE_NOTE} "
                "Set confirmed=True to delete it."
            ),
            config={"experiment": experiment_to_dict(experiment), "namespace": ns},
        ).model_dump()

    try:
        client.delete_experiment(name)
    except Exception as e:
        return pipelines_error(e, f"delete experiment '{name}'").model_dump()

    return ToolResponse(
        data={
            "name": name,
            "experiment_id": getattr(experiment, "experiment_id", None),
            "deleted": True,
            "message": _DELETE_NOTE,
            "next_steps": ["list_experiments() to confirm it is gone"],
        }
    ).model_dump()
