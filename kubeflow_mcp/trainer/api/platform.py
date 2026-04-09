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

"""Platform administration tools for CRD inspection, runtime management, and controller debugging.

These tools are primarily intended for the platform-admin persona but
inspect_crd/inspect_controller are also available to ml-engineer.
"""

import logging
from typing import Any

from kubeflow.trainer.constants import constants as trainer_constants

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found

logger = logging.getLogger(__name__)

_TRAINER_CRD_GROUP = "trainer.kubeflow.org"
_MAX_TAIL_LINES = 1000
_CONTROLLER_LABELS = [
    "app.kubernetes.io/name=trainer,app.kubernetes.io/component=manager",
    "app.kubernetes.io/name=trainer,app.kubernetes.io/part-of=kubeflow",
]
_LOG_TAIL_DEFAULT = 200


_DEFAULT_CONTROLLER_NAMESPACES = ["kubeflow", "kubeflow-system"]

_PATCH_ALLOWED_KEYS = frozenset({"spec", "metadata"})
_SPEC_ALLOWED_KEYS = frozenset({"template", "labels", "annotations"})


def _get_controller_namespace() -> str | None:
    """Get controller namespace from env/config override, or None for auto-scan."""
    import os

    ns = os.getenv("KUBEFLOW_MCP_CONTROLLER_NAMESPACE")
    if ns:
        return ns
    try:
        from kubeflow_mcp.core.config import load_config

        cfg = load_config()
        if cfg.trainer.controller_namespace:
            return cfg.trainer.controller_namespace
    except Exception:
        pass
    return None


def _find_controller_pod(namespace: str | None = None):
    """Find controller pod by label selectors.

    Resolution order:
    1. Explicit ``namespace`` arg (from tool call)
    2. KUBEFLOW_MCP_CONTROLLER_NAMESPACE env var / config file
    3. Scan default namespaces: kubeflow, kubeflow-system
    """
    core = mcp_utils.get_core_v1_api()
    configured_ns = namespace or _get_controller_namespace()
    namespaces = [configured_ns] if configured_ns else _DEFAULT_CONTROLLER_NAMESPACES

    for ns in namespaces:
        for label in _CONTROLLER_LABELS:
            try:
                pods = core.list_namespaced_pod(
                    namespace=ns,
                    label_selector=label,
                    _request_timeout=mcp_utils.K8S_TIMEOUT,
                )
                if pods.items:
                    return pods.items[0], ns, core
            except Exception:
                continue

    searched = configured_ns or ", ".join(_DEFAULT_CONTROLLER_NAMESPACES)
    return None, searched, core


def inspect_crd(name: str | None = None) -> dict[str, Any]:
    """List Trainer CRDs or get details for a specific one.

    Args:
        name: Full CRD name (e.g. ``trainjobs.trainer.kubeflow.org``).
            Omit to list all Trainer CRDs in the ``trainer.kubeflow.org`` group.

    Returns:
        dict: If name omitted: ``crds`` list with name, group, versions, scope.
            If name given: CRD metadata, versions, validation schema, conditions.
    """
    try:
        api = mcp_utils.get_apiextensions_api()

        if name is None:
            all_crds = api.list_custom_resource_definition(
                _request_timeout=mcp_utils.K8S_TIMEOUT,
            )
            trainer_crds = []
            for crd in all_crds.items:
                if crd.spec.group == _TRAINER_CRD_GROUP:
                    trainer_crds.append(
                        {
                            "name": crd.metadata.name,
                            "group": crd.spec.group,
                            "versions": [v.name for v in crd.spec.versions],
                            "scope": crd.spec.scope,
                            "served_versions": [v.name for v in crd.spec.versions if v.served],
                        }
                    )
            return ToolResponse(
                data={"crds": trainer_crds, "count": len(trainer_crds)}
            ).model_dump()

        crd = api.read_custom_resource_definition(
            name=name,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        versions = []
        for v in crd.spec.versions:
            version_info: dict[str, Any] = {
                "name": v.name,
                "served": v.served,
                "storage": v.storage,
            }
            if v.schema and v.schema.open_apiv3_schema:
                props = v.schema.open_apiv3_schema.properties or {}
                if "spec" in props and props["spec"].properties:
                    version_info["spec_fields"] = list(props["spec"].properties.keys())
            versions.append(version_info)

        return ToolResponse(
            data={
                "name": crd.metadata.name,
                "group": crd.spec.group,
                "scope": crd.spec.scope,
                "versions": versions,
                "conditions": [
                    {"type": c.type, "status": c.status} for c in (crd.status.conditions or [])
                ],
            }
        ).model_dump()

    except Exception as e:
        logger.warning("inspect_crd(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"CRD '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def inspect_controller(
    view: str = "logs",
    namespace: str | None = None,
    tail_lines: int = _LOG_TAIL_DEFAULT,
) -> dict[str, Any]:
    """Inspect the trainer-controller-manager pod.

    Args:
        view: ``"logs"`` for pod logs, ``"events"`` for K8s events.
        namespace: Controller namespace. Omit to auto-discover
            (scans kubeflow, kubeflow-system by default).
        tail_lines: Log lines to return (only for view="logs"). Default 200.

    Returns:
        dict: If view="logs": ``pod``, ``namespace``, ``logs``, ``tail_lines``.
            If view="events": ``pod``, ``namespace``, ``events`` list.
    """
    valid_views = ("logs", "events")
    if view not in valid_views:
        return ToolError(
            error=f"Invalid view '{view}'. Must be one of: {valid_views}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if tail_lines < 1:
        return ToolError(
            error=f"tail_lines must be >= 1, got {tail_lines}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    tail_lines = min(tail_lines, _MAX_TAIL_LINES)

    try:
        pod, searched_ns, core = _find_controller_pod(namespace)

        if not pod:
            return ToolError(
                error=f"No controller pod found in namespace '{searched_ns}'",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
                details={
                    "searched_namespace": searched_ns,
                    "hint": (
                        "Set KUBEFLOW_MCP_CONTROLLER_NAMESPACE env var or pass "
                        "namespace= to override"
                    ),
                },
            ).model_dump()

        pod_name = pod.metadata.name
        pod_ns = pod.metadata.namespace

        if view == "logs":
            logs = core.read_namespaced_pod_log(
                name=pod_name,
                namespace=pod_ns,
                tail_lines=tail_lines,
                _request_timeout=mcp_utils.K8S_TIMEOUT,
            )
            return ToolResponse(
                data={
                    "pod": pod_name,
                    "namespace": pod_ns,
                    "logs": logs,
                    "tail_lines": tail_lines,
                    "phase": pod.status.phase,
                }
            ).model_dump()

        events = core.list_namespaced_event(
            namespace=pod_ns,
            field_selector=f"involvedObject.name={pod_name}",
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )
        event_list = [
            {
                "type": ev.type,
                "reason": ev.reason,
                "message": ev.message,
                "count": ev.count,
                "first_seen": ev.first_timestamp.isoformat() if ev.first_timestamp else None,
                "last_seen": ev.last_timestamp.isoformat() if ev.last_timestamp else None,
            }
            for ev in events.items
        ]
        return ToolResponse(
            data={
                "pod": pod_name,
                "namespace": pod_ns,
                "events": event_list,
                "count": len(event_list),
            }
        ).model_dump()

    except Exception as e:
        logger.warning("inspect_controller(%s, %s) failed: %s", view, namespace, e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def patch_runtime(
    name: str,
    patch: dict[str, Any] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Strategic merge patch on a ClusterTrainingRuntime.

    Use to update images, add volumes, change defaults without full replacement.
    Top-level keys are restricted to ``spec`` and ``metadata``; deep structure
    is validated by the K8s API. Intended for platform-admin persona only.

    Args:
        name: ClusterTrainingRuntime name.
        patch: Strategic merge patch dict. Allowed top-level keys: spec, metadata.
        confirmed: Must be True to apply. False returns a preview.

    Returns:
        dict: Preview or applied patch result.
    """
    if not patch:
        return ToolError(
            error="patch parameter is required",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    bad_keys = set(patch.keys()) - _PATCH_ALLOWED_KEYS
    if bad_keys:
        return ToolError(
            error=f"Invalid top-level patch keys: {sorted(bad_keys)}. "
            f"Allowed: {sorted(_PATCH_ALLOWED_KEYS)}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if not confirmed:
        return ToolResponse(
            data={
                "action": "preview",
                "runtime": name,
                "patch": patch,
                "message": "Set confirmed=True to apply this patch.",
            }
        ).model_dump()

    try:
        api = mcp_utils.get_custom_objects_api()
        result = api.patch_cluster_custom_object(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            plural="clustertrainingruntimes",
            name=name,
            body=patch,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        return ToolResponse(
            data={
                "runtime": name,
                "patched": True,
                "message": f"ClusterTrainingRuntime '{name}' patched successfully",
                "resource_version": result.get("metadata", {}).get("resourceVersion"),
            }
        ).model_dump()

    except Exception as e:
        logger.warning("patch_runtime(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"ClusterTrainingRuntime '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def create_runtime(
    name: str,
    spec: dict[str, Any] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a new ClusterTrainingRuntime.

    Top-level spec keys are restricted to ``template``, ``labels``, and
    ``annotations``; deep structure is validated by the K8s API.
    Intended for platform-admin persona only.

    Args:
        name: Name for the new runtime.
        spec: Spec dict. Allowed top-level keys: template, labels, annotations.
        confirmed: Must be True to create. False returns a preview.

    Returns:
        dict: Preview or creation result.
    """
    if not spec:
        return ToolError(
            error="spec parameter is required",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    bad_keys = set(spec.keys()) - _SPEC_ALLOWED_KEYS
    if bad_keys:
        return ToolError(
            error=f"Invalid top-level spec keys: {sorted(bad_keys)}. "
            f"Allowed: {sorted(_SPEC_ALLOWED_KEYS)}",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    body = {
        "apiVersion": f"{trainer_constants.GROUP}/{trainer_constants.VERSION}",
        "kind": "ClusterTrainingRuntime",
        "metadata": {"name": name},
        "spec": spec,
    }

    if not confirmed:
        return ToolResponse(
            data={
                "action": "preview",
                "runtime": name,
                "body": body,
                "message": "Set confirmed=True to create this runtime.",
            }
        ).model_dump()

    try:
        api = mcp_utils.get_custom_objects_api()
        result = api.create_cluster_custom_object(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            plural="clustertrainingruntimes",
            body=body,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        return ToolResponse(
            data={
                "runtime": name,
                "created": True,
                "message": f"ClusterTrainingRuntime '{name}' created successfully",
                "resource_version": result.get("metadata", {}).get("resourceVersion"),
            }
        ).model_dump()

    except Exception as e:
        logger.warning("create_runtime(%s) failed: %s", name, e, exc_info=True)
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def delete_runtime(
    name: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete a ClusterTrainingRuntime.

    Lists dependent TrainJobs as a warning before deletion.

    Args:
        name: ClusterTrainingRuntime name to delete.
        confirmed: Must be True to delete. False returns a preview with dependents.

    Returns:
        dict: Preview with dependent jobs, or deletion result.
    """
    try:
        api = mcp_utils.get_custom_objects_api()

        dependent_jobs = []
        try:
            jobs = api.list_cluster_custom_object(
                group=trainer_constants.GROUP,
                version=trainer_constants.VERSION,
                plural=trainer_constants.TRAINJOB_PLURAL,
                _request_timeout=mcp_utils.K8S_TIMEOUT,
            )
            for job in jobs.get("items", []):
                runtime_ref = job.get("spec", {}).get("runtimeRef", {})
                if runtime_ref.get("name") == name:
                    dependent_jobs.append(
                        {
                            "name": job["metadata"]["name"],
                            "namespace": job["metadata"].get("namespace", "unknown"),
                        }
                    )
        except Exception:
            logger.debug("Could not list dependent jobs for runtime %s", name)

        if not confirmed:
            return ToolResponse(
                data={
                    "action": "preview",
                    "runtime": name,
                    "dependent_jobs": dependent_jobs,
                    "dependent_count": len(dependent_jobs),
                    "warning": (
                        f"{len(dependent_jobs)} TrainJob(s) reference this runtime. "
                        "They will fail if the runtime is deleted."
                        if dependent_jobs
                        else "No dependent TrainJobs found."
                    ),
                    "message": "Set confirmed=True to delete this runtime.",
                }
            ).model_dump()

        api.delete_cluster_custom_object(
            group=trainer_constants.GROUP,
            version=trainer_constants.VERSION,
            plural="clustertrainingruntimes",
            name=name,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )

        return ToolResponse(
            data={
                "runtime": name,
                "deleted": True,
                "message": f"ClusterTrainingRuntime '{name}' deleted successfully",
                "dependent_jobs_affected": len(dependent_jobs),
            }
        ).model_dump()

    except Exception as e:
        logger.warning("delete_runtime(%s) failed: %s", name, e, exc_info=True)
        if is_k8s_not_found(e):
            return ToolError(
                error=f"ClusterTrainingRuntime '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()
