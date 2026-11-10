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

"""Read-only pod diagnostics shared by discovery and wait responses."""

import logging
from typing import Any

from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client import V1ContainerStatus, V1Pod

from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_core_v1_api,
    get_trainer_effective_namespace,
)
from kubeflow_mcp.core.logging import _redact_text
from kubeflow_mcp.trainer.constants import MAX_DIAGNOSTIC_PODS

logger = logging.getLogger(__name__)


def _container_details(container: V1ContainerStatus) -> dict[str, Any]:
    details = {
        "name": container.name,
        "ready": container.ready,
        "restart_count": container.restart_count,
    }
    for state in ("waiting", "terminated", "running"):
        value = getattr(container.state, state, None)
        if value is not None:
            details["state"] = state
            details["reason"] = getattr(value, "reason", None)
            details["message"] = _redact_text(getattr(value, "message", None) or "")[:1000]
            details["exit_code"] = getattr(value, "exit_code", None)
            break
    previous = getattr(container.last_state, "terminated", None)
    if previous is not None:
        details["last_termination"] = {"reason": previous.reason, "exit_code": previous.exit_code}
    return details


def _pod_details(pod: V1Pod) -> dict[str, Any]:
    status = pod.status
    containers = [
        {**_container_details(container), "init": is_init}
        for is_init, statuses in (
            (True, status.init_container_statuses),
            (False, status.container_statuses),
        )
        for container in (statuses or [])
    ]
    conditions = [
        {
            "type": condition.type,
            "status": condition.status,
            "reason": condition.reason,
            "message": _redact_text(condition.message or "")[:1000],
        }
        for condition in (status.conditions or [])
        if condition.status != "True"
    ]
    return {
        "name": pod.metadata.name,
        "phase": status.phase,
        "node": pod.spec.node_name,
        "ready": any(c.type == "Ready" and c.status == "True" for c in (status.conditions or [])),
        "restart_count": sum(c["restart_count"] for c in containers),
        "containers": containers,
        "conditions": conditions,
    }


def _pod_blocking_reasons(pod: dict[str, Any]) -> list[str]:
    reasons = [
        f"{pod['name']}: {condition['reason']}: {condition['message']}"
        for condition in pod["conditions"]
        if condition["type"] == "PodScheduled"
    ]
    for container in pod["containers"]:
        if container.get("state") == "waiting" or container.get("exit_code"):
            reasons.append(
                f"{pod['name']}/{container['name']}: {container.get('reason')} "
                f"{container.get('message', '')}".strip()
            )
    if pod["phase"] == "Failed" and not reasons:
        reasons.append(f"{pod['name']}: pod Failed; use get_training_events for the cause")
    return reasons


def get_training_pod_diagnostics(name: str, namespace: str | None) -> dict[str, Any]:
    """Describe bounded job-associated pods; callers enforce name and namespace policy."""
    try:
        pods = (
            get_core_v1_api()
            .list_namespaced_pod(
                namespace=get_trainer_effective_namespace(namespace),
                label_selector=trainer_constants.POD_LABEL_SELECTOR.format(trainjob_name=name),
                _request_timeout=K8S_TIMEOUT,
            )
            .items
        )
        diagnostics = []
        for pod in pods:
            detail = _pod_details(pod)
            diagnostics.append((detail, _pod_blocking_reasons(detail)))
        # Keep blockers visible before applying the response cap; prefer active attempts on ties.
        diagnostics.sort(
            key=lambda item: (not item[1], item[0]["phase"] in ("Succeeded", "Failed"))
        )
        selected = diagnostics[:MAX_DIAGNOSTIC_PODS]
        reasons = [reason for _, blockers in selected for reason in blockers]
        omitted = sum(bool(blockers) for _, blockers in diagnostics[MAX_DIAGNOSTIC_PODS:])
        if omitted:
            reasons.append(
                f"{omitted} additional pods have blocking conditions; pod details are truncated"
            )
        return {
            "pods": [detail for detail, _ in selected],
            "pods_truncated": len(pods) > MAX_DIAGNOSTIC_PODS,
            "blocking_reasons": reasons,
        }
    except Exception:
        logger.debug("Pod diagnostics unavailable for %s", name, exc_info=True)
        return {
            "pod_diagnostics_error": "Could not read pod diagnostics; check namespace access and connectivity"
        }
