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

"""Authoring tools for Kubeflow Pipelines: register agent-written pipeline code, delete pipelines.

``create_pipeline`` takes KFP v2 Python source, checks it, compiles it to pipeline
YAML in a child process and uploads that YAML unchanged with
``PipelinesClient.upload_pipeline``. ``create_pipeline_run`` then runs it by name.
See ``_compile.py`` for what is checked and why.
"""

import logging
import os
import tempfile
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.huggingface import check_hf_dataset_exists, validate_hf_reference
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.common.utils import get_pipelines_client
from kubeflow_mcp.pipelines.api._common import (
    _NOT_FOUND_RE,
    check_resource_namespace,
    pipelines_error,
    resolve_namespace,
    validate_confirmed,
    validate_display_name,
)
from kubeflow_mcp.pipelines.api._compile import (
    PipelineCodeSummary,
    check_pipeline_code,
    compile_pipeline_code,
    inspect_pipeline_spec,
)
from kubeflow_mcp.pipelines.api.experiments import validate_description
from kubeflow_mcp.pipelines.types import pipeline_to_dict

logger = logging.getLogger(__name__)

MAX_DATASET_URI_LENGTH = 2048
# Same switch run_custom_training uses to submit code the safety scan flagged.
UNSAFE_CODE_ENV = "KUBEFLOW_MCP_UNSAFE_SCRIPTS"


def _validate_dataset(dataset: Any) -> ToolError | None:
    if dataset is None:
        return None
    if not isinstance(dataset, str) or not dataset.strip():
        return ToolError(
            error="dataset must be a non-empty URI string", error_code=ErrorCode.VALIDATION_ERROR
        )
    if len(dataset) > MAX_DATASET_URI_LENGTH:
        return ToolError(
            error=f"dataset is too long ({len(dataset)} chars, max {MAX_DATASET_URI_LENGTH})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return validate_hf_reference(dataset, "dataset", dataset=True)


def _validate_inputs(
    name: str, version: str | None, description: str | None, dataset: str | None
) -> ToolError | None:
    err = validate_display_name(name, "name")
    if err is None and version is not None:
        err = validate_display_name(version, "version")
    return err or validate_description(description) or _validate_dataset(dataset)


def _find_pipeline(client: Any, name: str) -> Any | None:
    """Return the registered pipeline called *name*, or None when there is none."""
    try:
        return client.get_pipeline(name)
    except ValueError as e:
        if _NOT_FOUND_RE.match(str(e)):
            return None
        raise


def _unsafe_code_allowed() -> bool:
    return os.environ.get(UNSAFE_CODE_ENV, "").lower() in ("1", "true", "yes")


def _preview(
    name: str,
    ns: str,
    summary: PipelineCodeSummary,
    existing: Any | None,
    version: str | None,
    description: str | None,
    dataset: str | None,
) -> dict[str, Any]:
    if existing is not None:
        action = f"add a new version to the existing pipeline '{name}'"
    else:
        action = f"register a new pipeline '{name}'"
    message = (
        f"Will compile the pipeline code and {action} in namespace '{ns}'. Review the "
        "components, their images and the parameters, then set confirmed=True."
    )
    if summary.safety_warnings:
        message += (
            " The code safety scan flagged patterns (see safety_warnings); submission is "
            f"blocked unless the server sets {UNSAFE_CODE_ENV}=true."
        )
    return PreviewResponse(
        message=message,
        config={
            "name": name,
            "namespace": ns,
            "action": "new_version" if existing is not None else "create",
            "existing_pipeline": pipeline_to_dict(existing) if existing is not None else None,
            "version": version,
            "description": description,
            "dataset": dataset,
            **summary.to_dict(),
            "safety_warnings": summary.safety_warnings,
        },
    ).model_dump()


def _compile_and_upload(
    client: Any,
    name: str,
    pipeline_code: str,
    version: str | None,
    description: str | None,
) -> tuple[dict[str, Any] | None, Any, ToolError | None]:
    """Compile, inspect and upload. Returns (spec summary, uploaded version, error)."""
    with tempfile.TemporaryDirectory(
        prefix="kubeflow_mcp_pipeline_", ignore_cleanup_errors=True
    ) as workdir:
        compiled, err = compile_pipeline_code(pipeline_code, workdir)
        if err is not None:
            return None, None, err
        spec, err = inspect_pipeline_spec(compiled.documents)
        if err is not None:
            return None, None, err
        try:
            # The compiler's file is uploaded as is, so KFP's header and the
            # platform (Kubernetes) document stay intact.
            uploaded = client.upload_pipeline(
                compiled.path, name=name, version=version, description=description
            )
        except Exception as e:
            return None, None, pipelines_error(e, f"upload pipeline '{name}'")
    return spec, uploaded, None


def create_pipeline(
    name: str,
    pipeline_code: str,
    description: str | None = None,
    version: str | None = None,
    dataset: str | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Register a Kubeflow pipeline from KFP v2 Python code (two-phase confirm).

    Call once with ``confirmed=False`` (the default) to preview: the code is checked
    without running it and the preview lists its components, images and parameters.
    Then call again with ``confirmed=True`` to compile it to pipeline YAML and upload
    it. If a pipeline with this name already exists, the upload adds a new version.

    The code must contain ``from kfp import dsl`` style imports, ``@dsl.component``
    functions and exactly one ``@dsl.pipeline`` function. Imports of other packages
    go inside the components, which run in the cluster. See
    ``pipelines://guides/authoring``.

    Args:
        name: Pipeline display name to register (also used by ``create_pipeline_run``).
        pipeline_code: KFP v2 pipeline source code.
        description: Optional pipeline description.
        version: Optional version name. KFP generates one when omitted.
        dataset: Optional dataset URI the pipeline reads, checked like ``fine_tune``
            checks its dataset: ``hf://`` format in the preview, Hub existence on submit.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to compile and upload. ``False`` returns a preview.

    Returns:
        dict: Preview, or the registered pipeline (``pipeline_id``,
        ``pipeline_version_id``, ``parameters``, ``images``) and ``next_steps``.
    """
    confirm_err = validate_confirmed(confirmed)
    if confirm_err is not None:
        return confirm_err.model_dump()

    input_err = _validate_inputs(name, version, description, dataset)
    if input_err is not None:
        return input_err.model_dump()

    summary, code_err = check_pipeline_code(pipeline_code)
    if code_err is not None:
        return code_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        existing = _find_pipeline(client, name)
    except Exception as e:
        return pipelines_error(e, f"look up pipeline '{name}'").model_dump()

    if not confirmed:
        return _preview(name, ns, summary, existing, version, description, dataset)

    if summary.safety_warnings:
        if not _unsafe_code_allowed():
            return ToolError(
                error="Pipeline code blocked: unsafe patterns detected. "
                f"Set {UNSAFE_CODE_ENV}=true to override.",
                error_code=ErrorCode.VALIDATION_ERROR,
                details={"safety_warnings": summary.safety_warnings},
            ).model_dump()
        logger.warning(
            "Registering pipeline code with safety warnings (override enabled): %s",
            summary.safety_warnings,
        )

    if dataset is not None:
        dataset_err = check_hf_dataset_exists(dataset)
        if dataset_err is not None:
            return dataset_err.model_dump()

    spec, uploaded, err = _compile_and_upload(client, name, pipeline_code, version, description)
    if err is not None:
        return err.model_dump()

    # Defaults as written in the source: the compiled spec stores integers as floats.
    params = {p["name"]: p["default"] for p in summary.parameters if not p["required"]}
    data = {
        "name": name,
        "namespace": ns,
        "created_new_pipeline": existing is None,
        "pipeline_id": getattr(uploaded, "pipeline_id", None),
        "pipeline_version_id": getattr(uploaded, "pipeline_version_id", None),
        "version": getattr(uploaded, "display_name", None),
        **spec,
        "next_steps": [
            f"create_pipeline_run(pipeline='{name}', params={params!r}) to preview a run"
        ],
    }
    return ToolResponse(data=data).model_dump()


MAX_LISTED_VERSIONS = 200
_DELETE_NOTE = (
    "Runs of this pipeline are kept, but a run whose version is deleted can no longer "
    "be retried or re-run."
)


def _version_names(client: Any, name: str) -> tuple[list[str], bool]:
    """Return up to MAX_LISTED_VERSIONS version names and whether there are more."""
    response = client.list_pipeline_versions(name, page_size=MAX_LISTED_VERSIONS)
    versions = [
        getattr(v, "display_name", None) for v in getattr(response, "pipeline_versions", []) or []
    ]
    return [v for v in versions if v], bool(getattr(response, "next_page_token", None))


def delete_pipeline(
    name: str,
    version: str | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """[DESTRUCTIVE] Delete a registered pipeline, or one of its versions (two-phase confirm).

    Call once with ``confirmed=False`` to preview what will be deleted, then again with
    ``confirmed=True``. Without ``version`` the whole pipeline goes, with every version.
    Runs are kept. KFP pipelines carry no MCP ownership label, so this tool is limited
    to the ``platform-admin`` persona.

    Args:
        name: Pipeline display name, as shown by ``list_pipelines()``.
        version: Only delete this version. Deletes the whole pipeline when omitted.
        namespace: K8s namespace. Uses the configured default when omitted.
        confirmed: Set ``True`` to delete. ``False`` returns a preview.

    Returns:
        dict: Preview, or a confirmation of what was deleted.
    """
    confirm_err = validate_confirmed(confirmed)
    if confirm_err is not None:
        return confirm_err.model_dump()

    input_err = validate_display_name(name, "name")
    if input_err is None and version is not None:
        input_err = validate_display_name(version, "version")
    if input_err is not None:
        return input_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        pipeline = client.get_pipeline(name)
        owner_err = check_resource_namespace(pipeline, ns, "Pipeline")
        if owner_err is not None:
            return owner_err.model_dump()
        if version is not None:
            client.get_pipeline_version(name, version)
            versions, more = [version], False
        else:
            versions, more = _version_names(client, name)
    except Exception as e:
        return pipelines_error(e, f"look up pipeline '{name}'").model_dump()

    target = f"version '{version}' of pipeline '{name}'" if version else f"pipeline '{name}'"
    if not confirmed:
        count = f"{len(versions)}{'+' if more else ''}"
        scope = "" if version else f" and all its versions ({count})"
        return PreviewResponse(
            message=(
                f"Will delete {target}{scope} in namespace '{ns}'. {_DELETE_NOTE} "
                "Set confirmed=True to delete."
            ),
            config={
                "pipeline": pipeline_to_dict(pipeline),
                "versions_to_delete": versions,
                "versions_truncated": more,
                "namespace": ns,
                "action": "delete_version" if version else "delete_pipeline",
            },
        ).model_dump()

    try:
        # force=True deletes every version with the pipeline; the preview listed them.
        client.delete_pipeline(name, version=version, force=version is None)
    except Exception as e:
        return pipelines_error(e, f"delete {target}").model_dump()

    return ToolResponse(
        data={
            "name": name,
            "pipeline_id": getattr(pipeline, "pipeline_id", None),
            "deleted": "version" if version else "pipeline",
            "versions_deleted": versions,
            "message": _DELETE_NOTE,
            "next_steps": ["list_pipelines() to confirm the result"],
        }
    ).model_dump()
