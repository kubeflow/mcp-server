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

"""Lifecycle tools for Kubeflow Pipelines runs (create / stop / retry)."""

import json
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.common.utils import get_pipelines_client
from kubeflow_mcp.pipelines.api._common import (
    TERMINAL_RUN_STATES,
    pipelines_error,
    resolve_namespace,
    validate_display_name,
    validate_run_id,
)
from kubeflow_mcp.pipelines.types import pipeline_to_dict, run_to_dict

MAX_PARAMS = 50
MAX_PARAMS_BYTES = 32_768
_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
# KFP can retry a run that failed or was stopped.
_RETRYABLE_STATES = frozenset({"failed", "canceled"})
_OWNERSHIP_NOTE = "KFP runs carry no MCP ownership label, so this acts on any run in the namespace."


def _validate_params(params: dict[str, Any] | None) -> ToolError | None:
    if params is None:
        return None
    if not isinstance(params, dict):
        return ToolError(
            error="params must be an object of parameter name to value",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    if len(params) > MAX_PARAMS:
        return ToolError(
            error=f"Too many params ({len(params)}, max {MAX_PARAMS})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    bad = sorted(str(k) for k in params if not isinstance(k, str) or not _PARAM_NAME_RE.match(k))
    if bad:
        return ToolError(
            error=f"Invalid parameter names: {', '.join(bad)}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    try:
        size = len(json.dumps(params))
    except (TypeError, ValueError):
        return ToolError(
            error="params values must be JSON values (string, number, bool, list or object)",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    if size > MAX_PARAMS_BYTES:
        return ToolError(
            error=f"params are too large ({size} bytes, max {MAX_PARAMS_BYTES})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None


def create_pipeline_run(
    pipeline: str,
    version: str | None = None,
    params: dict[str, Any] | None = None,
    experiment: str | None = None,
    run_name: str | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Start a run of a registered pipeline (two-phase confirm).

    Call once with ``confirmed=False`` (the default) to preview, then again with
    ``confirmed=True`` to start the run. Only registered pipelines can be run, by name;
    file paths are refused.

    Args:
        pipeline: Pipeline display name, as shown by ``list_pipelines()``.
        version: Pipeline version name. Uses the latest version when omitted.
        params: Pipeline parameters, e.g. ``{"learning_rate": 0.01}``.
        experiment: Experiment name. Uses the server's default experiment when omitted.
        run_name: Run display name. Generated when omitted.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to start the run. ``False`` returns a preview.

    Returns:
        dict: Preview of the run, or the created run (``run_id``, ``state``, ...).
    """
    for value, field in ((version, "version"), (experiment, "experiment"), (run_name, "run_name")):
        if value is not None:
            err = validate_display_name(value, field)
            if err is not None:
                return err.model_dump()
    pipeline_err = validate_display_name(pipeline, "pipeline")
    if pipeline_err is not None:
        return pipeline_err.model_dump()
    params_err = _validate_params(params)
    if params_err is not None:
        return params_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        # Resolve the pipeline to its object before running it. Passing the object, not
        # the name, keeps the SDK from ever treating the string as a local file path.
        pipeline_obj = client.get_pipeline(pipeline)
        if version is not None:
            client.get_pipeline_version(pipeline, version)
        if experiment is not None:
            client.get_experiment(experiment)
    except Exception as e:
        return pipelines_error(e, f"look up pipeline '{pipeline}'").model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will start a run of pipeline '{pipeline}' in namespace '{ns}'. "
                "Set confirmed=True to start it."
            ),
            config={
                "pipeline": pipeline_to_dict(pipeline_obj),
                "version": version or "latest",
                "params": params or {},
                "experiment": experiment or "default",
                "run_name": run_name,
                "namespace": ns,
            },
        ).model_dump()

    try:
        run = client.run(
            pipeline_obj, params=params, name=run_name, experiment=experiment, version=version
        )
    except Exception as e:
        return pipelines_error(e, f"start a run of pipeline '{pipeline}'").model_dump()

    data = run_to_dict(run)
    data["namespace"] = ns
    data["next_steps"] = [f"get_pipeline_run(run_id='{data.get('run_id')}') to follow it"]
    return ToolResponse(data=data).model_dump()


def stop_pipeline_run(
    run_id: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """[DESTRUCTIVE] Stop (terminate) a pipeline run (two-phase confirm).

    Call once with ``confirmed=False`` to preview, then again with ``confirmed=True``.
    Tasks still running are killed; finished tasks keep their outputs.

    Args:
        run_id: Run ID (a UUID).
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to stop the run. ``False`` returns a preview.

    Returns:
        dict: Preview, or a confirmation that termination was requested.
    """
    id_err = validate_run_id(run_id)
    if id_err is not None:
        return id_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        run = run_to_dict(client.get_run(run_id))
    except Exception as e:
        return pipelines_error(e, f"get pipeline run '{run_id}'").model_dump()

    if run.get("state") in TERMINAL_RUN_STATES:
        return ToolError(
            error=f"Run '{run_id}' already finished with state '{run.get('state')}'",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will stop run '{run.get('name')}' ({run_id}), now '{run.get('state')}'. "
                f"{_OWNERSHIP_NOTE} Set confirmed=True to stop it."
            ),
            config={"run": run, "namespace": ns, "action": "stop"},
        ).model_dump()

    try:
        client.kfp_client().terminate_run(run_id)
    except Exception as e:
        return pipelines_error(e, f"stop pipeline run '{run_id}'").model_dump()

    return ToolResponse(
        data={
            "run_id": run_id,
            "stopped": True,
            "message": "Termination requested; the run moves to 'canceling', then 'canceled'",
            "next_steps": [f"get_pipeline_run(run_id='{run_id}') to confirm it stopped"],
        }
    ).model_dump()


def retry_pipeline_run(
    run_id: str,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Retry a failed or stopped pipeline run (two-phase confirm).

    KFP re-runs the tasks that did not succeed; tasks that succeeded are not run again.

    Args:
        run_id: Run ID (a UUID) of a run in state ``failed`` or ``canceled``.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to retry the run. ``False`` returns a preview.

    Returns:
        dict: Preview, or a confirmation that the retry was requested.
    """
    id_err = validate_run_id(run_id)
    if id_err is not None:
        return id_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        run = run_to_dict(client.get_run(run_id))
    except Exception as e:
        return pipelines_error(e, f"get pipeline run '{run_id}'").model_dump()

    if run.get("state") not in _RETRYABLE_STATES:
        return ToolError(
            error=f"Run '{run_id}' is '{run.get('state')}'; only failed or canceled runs can be retried",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will retry run '{run.get('name')}' ({run_id}), now '{run.get('state')}'. "
                f"{_OWNERSHIP_NOTE} Set confirmed=True to retry it."
            ),
            config={"run": run, "namespace": ns, "action": "retry"},
        ).model_dump()

    try:
        # PipelinesClient has no retry method; kfp.Client's run API does. This is the
        # only call through a private attribute, pinned by sdk_contracts_test.py.
        client.kfp_client()._run_api.run_service_retry_run(run_id=run_id)
    except Exception as e:
        return pipelines_error(e, f"retry pipeline run '{run_id}'").model_dump()

    return ToolResponse(
        data={
            "run_id": run_id,
            "retried": True,
            "message": "Retry requested; failed tasks run again",
            "next_steps": [f"get_pipeline_run(run_id='{run_id}') to follow it"],
        }
    ).model_dump()
