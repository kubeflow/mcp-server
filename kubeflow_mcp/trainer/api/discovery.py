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

"""Discovery tools for training jobs and runtimes."""

import logging
import time
import uuid
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details, is_k8s_not_found
from kubeflow_mcp.common.utils import (
    get_core_v1_api,
    get_custom_objects_api,
    get_trainer_client,
    get_trainer_client_for_namespace,
    get_trainer_effective_namespace,
)
from kubeflow_mcp.core.security import check_namespace_allowed

logger = logging.getLogger(__name__)

_PACKAGES_POD_TIMEOUT = 60
_PACKAGES_POLL_INTERVAL = 3

# Legacy filter/docs used "Succeeded" for finished TrainJobs; API uses "Complete".
_JOB_STATUS_FILTER_ALIASES: dict[str, str] = {
    "Succeeded": "Complete",
}


def _trainjob_runtime_to_mcp(runtime: object | None) -> dict[str, str] | None:
    """Serialize Runtime object for MCP JSON response."""
    if runtime is None:
        return None
    name = getattr(runtime, "name", None)
    if isinstance(name, str) and name:
        return {"name": name}
    return None


MAX_LIST_LIMIT = 500


def list_training_jobs(
    runtime: str | None = None,
    status: str | None = None,
    namespace: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List training jobs.

    Args:
        runtime: Filter by ClusterTrainingRuntime name (e.g., ``torch-tune``).
        status: Filter by TrainJob status: ``Created``, ``Running``, ``Complete``,
            ``Failed``, ``Suspended``. ``Succeeded`` is accepted as an alias for ``Complete``.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        limit: Maximum jobs to return. Defaults to 50.

    Returns:
        dict: Response containing:

        - ``jobs`` (list): List of jobs with ``name``, ``status``, ``runtime``
        - ``total`` (int): Total matching jobs

    Example:
        >>> list_training_jobs(status="Running")
        {"data": {"jobs": [{"name": "fine-tune-abc", "status": "Running"}], "total": 1}}
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        if limit < 1:
            return ToolError(
                error=f"limit must be >= 1, got {limit}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        limit = min(limit, MAX_LIST_LIMIT)
        client = get_trainer_client_for_namespace(namespace)
        if runtime:
            rt_obj = client.get_runtime(name=runtime)
            jobs = client.list_jobs(runtime=rt_obj)
        else:
            jobs = client.list_jobs()

        job_list = []
        for job in jobs:
            jr = job.runtime if hasattr(job, "runtime") else None
            job_data = {
                "name": job.name,
                "status": job.status if hasattr(job, "status") else "Unknown",
                "runtime": _trainjob_runtime_to_mcp(jr),
            }
            job_list.append(job_data)

        if status:
            want = _JOB_STATUS_FILTER_ALIASES.get(status, status)
            job_list = [j for j in job_list if j.get("status") == want]

        return ToolResponse(data={"jobs": job_list[:limit], "total": len(job_list)}).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_training_job(name: str, namespace: str | None = None) -> dict[str, Any]:
    """Get details of a specific training job.

    Args:
        name: The TrainJob name.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.

    Returns:
        dict: Response containing ``name``, ``status``, ``runtime``.

    Raises:
        ToolError: If job not found (``RESOURCE_NOT_FOUND``).
    """
    ns_err = check_namespace_allowed(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        client = get_trainer_client_for_namespace(namespace)
        job = client.get_job(name=name)

        jr = job.runtime if hasattr(job, "runtime") else None
        status = job.status if hasattr(job, "status") else "Unknown"

        next_steps: list[str] = []
        if status == "Failed":
            next_steps = [
                f"get_training_events(name='{name}') — check for OOM/scheduling issues",
                f"get_training_logs(name='{name}') — check error output",
                "Read trainer://workflows/ops",
            ]
        elif status == "Running":
            next_steps = [f"get_training_logs(name='{name}') — check progress"]
        elif status == "Created":
            next_steps = [
                f"get_training_events(name='{name}') — check if scheduling or suspended",
            ]

        data: dict[str, Any] = {
            "name": job.name,
            "status": status,
            "runtime": _trainjob_runtime_to_mcp(jr),
        }
        if next_steps:
            data["next_steps"] = next_steps

        return ToolResponse(data=data).model_dump()

    except Exception as e:
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


def list_runtimes() -> dict[str, Any]:
    """List available ClusterTrainingRuntimes.

    Call this if ``fine_tune()`` fails with "runtime not found" to see
    what runtimes are installed in the cluster.

    Returns:
        dict: Response containing:

        - ``runtimes`` (list): Available runtimes with ``name``
        - ``total`` (int): Runtime count

    Example:
        >>> list_runtimes()
        {"data": {"runtimes": [{"name": "torch-tune"}, {"name": "torch-distributed"}], "total": 2}}
    """
    try:
        client = get_trainer_client()
        runtimes = client.list_runtimes()

        runtime_list = []
        for rt in runtimes:
            runtime_list.append(
                {
                    "name": rt.name if hasattr(rt, "name") else str(rt),
                }
            )

        return ToolResponse(
            data={"runtimes": runtime_list, "total": len(runtime_list)}
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def _get_runtime_image(name: str) -> str | None:
    """Extract container image from a ClusterTrainingRuntime CRD."""
    api = get_custom_objects_api()
    rt = api.get_cluster_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        plural="clustertrainingruntimes",
        name=name,
    )
    for rj in rt.get("spec", {}).get("template", {}).get("spec", {}).get("replicatedJobs", []):
        containers = (
            rj.get("template", {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for c in containers:
            img = c.get("image")
            if img:
                return img
    return None


def _fetch_packages_via_pod(runtime_name: str) -> dict[str, Any]:
    """Create a lightweight Pod to run ``pip list`` using the runtime's image.

    Creates a temporary Pod from the runtime's container image to list packages.
    Avoids creating a full TrainJob whose entrypoint would crash without a script.

    The pod includes emptyDir volumes for writable directories so it works
    on platforms with read-only root filesystems.
    """
    image = _get_runtime_image(runtime_name)
    if not image:
        return {
            "packages_error": f"Could not extract container image from runtime '{runtime_name}'"
        }

    ns = get_trainer_effective_namespace()
    ns_err = check_namespace_allowed(ns)
    if ns_err is not None:
        return {"packages_error": f"Namespace '{ns}' not allowed by policy"}

    pod_name = f"mcp-pip-list-{uuid.uuid4().hex[:8]}"
    core = get_core_v1_api()

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": ns,
            "labels": {"app.kubernetes.io/managed-by": "kubeflow-mcp", "purpose": "pip-list"},
        },
        "spec": {
            "restartPolicy": "Never",
            "volumes": [
                {"name": "tmp", "emptyDir": {}},
                {"name": "workspace", "emptyDir": {}},
                {"name": "dot-local", "emptyDir": {}},
                {"name": "dot-cache", "emptyDir": {}},
            ],
            "containers": [
                {
                    "name": "pip-list",
                    "image": image,
                    "command": [
                        "sh",
                        "-c",
                        "pip list --format=columns 2>/dev/null || pip list 2>/dev/null || echo 'pip not found'",
                    ],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "500m", "memory": "256Mi"},
                    },
                    "volumeMounts": [
                        {"name": "tmp", "mountPath": "/tmp"},
                        {"name": "workspace", "mountPath": "/workspace"},
                        {"name": "dot-local", "mountPath": "/.local"},
                        {"name": "dot-cache", "mountPath": "/.cache"},
                    ],
                }
            ],
        },
    }

    try:
        core.create_namespaced_pod(namespace=ns, body=pod_manifest)
    except Exception as e:
        return {"packages_error": f"Failed to create pip-list pod: {e}"}

    try:
        deadline = time.monotonic() + _PACKAGES_POD_TIMEOUT
        phase = None
        while time.monotonic() < deadline:
            pod = core.read_namespaced_pod(name=pod_name, namespace=ns)
            phase = pod.status.phase
            if phase in ("Succeeded", "Failed"):
                break
            time.sleep(_PACKAGES_POLL_INTERVAL)

        if phase not in ("Succeeded", "Failed"):
            return {
                "packages_error": f"Pod '{pod_name}' did not complete within {_PACKAGES_POD_TIMEOUT}s (phase: {phase})"
            }

        logs = core.read_namespaced_pod_log(name=pod_name, namespace=ns, tail_lines=500)

        if phase == "Failed":
            return {
                "packages_error": "pip-list pod failed",
                "logs": logs[-500:] if logs else "no logs",
            }

        packages = []
        for line in logs.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in ("Package", "---"):
                packages.append({"name": parts[0], "version": parts[1]})
        return {"packages": packages, "total": len(packages)}

    finally:
        try:
            core.delete_namespaced_pod(name=pod_name, namespace=ns, grace_period_seconds=0)
        except Exception:
            logger.warning("Failed to delete pip-list pod %s/%s", ns, pod_name)


def get_runtime(name: str, include_packages: bool = False) -> dict[str, Any]:
    """Get ClusterTrainingRuntime configuration.

    Args:
        name: Runtime name (e.g., ``torch-distributed``).
        include_packages: If True, fetches pip packages by creating a
            temporary Pod (slow, ~30-60s).  Default False.

    Returns:
        dict: Response containing runtime ``name``, configuration, and
        optionally ``packages`` list.

    Raises:
        ToolError: If runtime not found (``RESOURCE_NOT_FOUND``).
    """
    try:
        client = get_trainer_client()
        rt = client.get_runtime(name=name)

        rt_name = rt.name if hasattr(rt, "name") else name

        data: dict[str, Any] = {"name": rt_name}

        spec = getattr(rt, "spec", None)
        if spec is not None:
            ml_policy = getattr(spec, "ml_policy", None)
            data["framework"] = getattr(ml_policy, "torch", None) or getattr(ml_policy, "mpi", None)
            template = getattr(spec, "template", None)
            if template is not None:
                replicated_jobs = getattr(getattr(template, "spec", None), "replicated_jobs", None)
                if replicated_jobs:
                    data["replicated_jobs"] = [
                        {
                            "name": getattr(rj, "name", None),
                            "replicas": getattr(
                                getattr(getattr(rj, "template", None), "spec", None),
                                "completions",
                                None,
                            ),
                        }
                        for rj in replicated_jobs
                    ]

        if include_packages:
            data.update(_fetch_packages_via_pod(name))

        return ToolResponse(data=data).model_dump()

    except Exception as e:
        if is_k8s_not_found(e):
            return ToolError(
                error=f"Runtime '{name}' not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND,
            ).model_dump()
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()
