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

"""Discovery tools for registered Kubeflow Pipelines."""

from typing import Any

from kubeflow_mcp.common.types import ToolResponse
from kubeflow_mcp.common.utils import get_pipelines_client
from kubeflow_mcp.pipelines.api._common import (
    page_fields,
    pipelines_error,
    resolve_namespace,
    validate_display_name,
    validate_limit,
)
from kubeflow_mcp.pipelines.types import pipeline_to_dict, pipeline_version_to_dict


def list_pipelines(
    limit: int = 50,
    page_token: str = "",
    namespace: str | None = None,
) -> dict[str, Any]:
    """List pipelines registered on the Kubeflow Pipelines server.

    Args:
        limit: Pipelines per page, 1 to 200. Defaults to 50.
        page_token: ``next_page_token`` from a previous call, to read the next page.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing ``pipelines`` (``pipeline_id``, ``name``,
        ``description``, ``namespace``, ``created_at``), ``next_page_token``,
        ``has_more`` and ``total``.
    """
    limit_err = validate_limit(limit)
    if limit_err is not None:
        return limit_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        response = get_pipelines_client(ns).list_pipelines(
            page_token=page_token or "", page_size=limit
        )
        pipelines = [pipeline_to_dict(p) for p in getattr(response, "pipelines", None) or []]
        return ToolResponse(
            data={"pipelines": pipelines, "namespace": ns, **page_fields(response)}
        ).model_dump()
    except Exception as e:
        return pipelines_error(e, "list pipelines").model_dump()


def get_pipeline(
    name: str,
    include_versions: bool = True,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Get a registered pipeline by name, with its most recent versions.

    Args:
        name: Pipeline display name, as shown by ``list_pipelines()``.
        include_versions: Also return up to 20 versions of the pipeline.
        namespace: K8s namespace. Uses the configured default when omitted.

    Returns:
        dict: Response containing the pipeline fields, plus ``versions`` and
        ``versions_has_more`` when ``include_versions`` is true.
    """
    name_err = validate_display_name(name, "name")
    if name_err is not None:
        return name_err.model_dump()

    ns, ns_err = resolve_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_pipelines_client(ns)
        data = pipeline_to_dict(client.get_pipeline(name))
        if include_versions:
            versions = client.list_pipeline_versions(name, page_size=20)
            data["versions"] = [
                pipeline_version_to_dict(v)
                for v in getattr(versions, "pipeline_versions", None) or []
            ]
            data["versions_has_more"] = bool(getattr(versions, "next_page_token", None))
        data["next_steps"] = [
            f"list_pipeline_runs(pipeline='{name}') to see its runs",
            f"create_pipeline_run(pipeline='{name}') to preview a new run",
        ]
        return ToolResponse(data=data).model_dump()
    except Exception as e:
        return pipelines_error(e, f"get pipeline '{name}'").model_dump()
