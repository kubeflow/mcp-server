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

"""Serialization helpers for Kubeflow Pipelines SDK types.

The SDK returns ``kfp_server_api`` model objects whose fields include datetimes and
nested models. These helpers read them with ``getattr`` so this module loads without
``kfp`` installed, and return plain, MCP-safe dicts.
"""

from typing import Any

# A run can carry many tasks; the agent needs their states, not an unbounded list.
MAX_TASKS = 50


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _state(value: Any) -> str | None:
    # The API reports states upper case (``SUCCEEDED``); the SDK's filters and
    # constants use lower case, so the tools speak lower case throughout.
    if value is None:
        return None
    return str(getattr(value, "value", value)).lower()


def _error_message(obj: Any) -> str | None:
    error = getattr(obj, "error", None)
    return getattr(error, "message", None) or None


def pipeline_to_dict(pipeline: Any) -> dict[str, Any]:
    """Convert a ``V2beta1Pipeline`` into a JSON-safe dict."""
    return {
        "pipeline_id": getattr(pipeline, "pipeline_id", None),
        "name": getattr(pipeline, "display_name", None),
        "description": getattr(pipeline, "description", None),
        "namespace": getattr(pipeline, "namespace", None),
        "created_at": _iso(getattr(pipeline, "created_at", None)),
    }


def pipeline_version_to_dict(version: Any) -> dict[str, Any]:
    """Convert a ``V2beta1PipelineVersion`` into a JSON-safe dict (spec left out)."""
    return {
        "pipeline_version_id": getattr(version, "pipeline_version_id", None),
        "name": getattr(version, "display_name", None),
        "description": getattr(version, "description", None),
        "created_at": _iso(getattr(version, "created_at", None)),
    }


def _task_to_dict(task: Any) -> dict[str, Any]:
    child_pods = [
        getattr(child, "pod_name", None) for child in getattr(task, "child_tasks", None) or []
    ]
    return {
        "task_id": getattr(task, "task_id", None),
        "name": getattr(task, "display_name", None),
        "state": _state(getattr(task, "state", None)),
        "pod_name": getattr(task, "pod_name", None),
        "child_pods": [pod for pod in child_pods if pod],
        "error": _error_message(task),
    }


def run_to_dict(run: Any, *, include_details: bool = False) -> dict[str, Any]:
    """Convert a ``V2beta1Run`` into a JSON-safe dict.

    The compiled pipeline spec is never included; it can be very large and the agent
    does not need it to follow a run. ``include_details`` adds the run parameters and
    up to ``MAX_TASKS`` task states.
    """
    reference = getattr(run, "pipeline_version_reference", None)
    data: dict[str, Any] = {
        "run_id": getattr(run, "run_id", None),
        "name": getattr(run, "display_name", None),
        "state": _state(getattr(run, "state", None)),
        "experiment_id": getattr(run, "experiment_id", None),
        "pipeline_id": getattr(reference, "pipeline_id", None),
        "pipeline_version_id": getattr(reference, "pipeline_version_id", None),
        "created_at": _iso(getattr(run, "created_at", None)),
        "finished_at": _iso(getattr(run, "finished_at", None)),
        "error": _error_message(run),
    }
    if include_details:
        runtime_config = getattr(run, "runtime_config", None)
        data["parameters"] = dict(getattr(runtime_config, "parameters", None) or {})
        tasks = getattr(getattr(run, "run_details", None), "task_details", None) or []
        data["tasks"] = [_task_to_dict(task) for task in tasks[:MAX_TASKS]]
        data["tasks_truncated"] = len(tasks) > MAX_TASKS
    return data


def run_node_ids(run: Any) -> list[tuple[str | None, str]]:
    """Return ``(task name, argo node id)`` for every node KFP recorded for a run's tasks.

    In KFP v2 / Argo 3.4+, the ``pod_name`` and ``child_tasks[].pod_name`` fields
    contain Argo Workflow Node IDs, not real Kubernetes Pod names. Use
    ``monitoring._resolve_run_pods()`` to map these to actual pods.
    """
    pods: list[tuple[str | None, str]] = []
    for task in getattr(getattr(run, "run_details", None), "task_details", None) or []:
        name = getattr(task, "display_name", None)
        pod = getattr(task, "pod_name", None)
        if pod:
            pods.append((name, pod))
        for child in getattr(task, "child_tasks", None) or []:
            child_pod = getattr(child, "pod_name", None)
            if child_pod:
                pods.append((name, child_pod))
    # The same node ID can be listed by a task and as a parent's child.
    return list(dict.fromkeys(pods))


# Backward-compatible alias.
run_pod_names = run_node_ids


__all__ = [
    "MAX_TASKS",
    "pipeline_to_dict",
    "pipeline_version_to_dict",
    "run_node_ids",
    "run_pod_names",
    "run_to_dict",
]
