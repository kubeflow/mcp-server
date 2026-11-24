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

import json
import logging
from typing import Any

from kubeflow.trainer.constants import constants as trainer_constants
from kubernetes.client import V1ContainerStatus, V1Pod

from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    get_core_v1_api,
    get_trainer_effective_namespace,
    get_training_pod_step,
)
from kubeflow_mcp.core.logging import _redact_text
from kubeflow_mcp.trainer.constants import (
    MAX_DIAGNOSTIC_CHARS,
    MAX_DIAGNOSTIC_MESSAGE_CHARS,
    MAX_DIAGNOSTIC_PODS,
    TRANSIENT_WAITING_REASONS,
)

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
            details["message"] = _redact_text(getattr(value, "message", None) or "")[
                :MAX_DIAGNOSTIC_MESSAGE_CHARS
            ]
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
            "message": _redact_text(condition.message or "")[:MAX_DIAGNOSTIC_MESSAGE_CHARS],
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
        f"{pod['name']}: {condition['reason'] or condition['type']}"
        for condition in pod["conditions"]
        if condition["type"] == "PodScheduled"
    ]
    for container in pod["containers"]:
        waiting = container.get("state") == "waiting"
        if (waiting and container.get("reason") not in TRANSIENT_WAITING_REASONS) or container.get(
            "exit_code"
        ):
            exit_code = container.get("exit_code")
            reason = f"{pod['name']}/{container['name']}: {container.get('reason') or container['state']}"
            if exit_code:
                reason += f" (exit {exit_code})"
            reasons.append(reason)
    if pod["phase"] == "Failed" and not reasons:
        reasons.append(f"{pod['name']}: pod Failed; use get_training_events for the cause")
    return reasons


def _bounded_diagnostics(diagnostics: list[tuple[dict[str, Any], list[str]]]) -> dict[str, Any]:
    """Keep failure-first pod details and reasons within one serialized character budget."""
    data: dict[str, Any] = {"pods": [], "pods_truncated": False, "blocking_reasons": []}
    reasons_truncated = False
    reported: set[str] = set()
    # Leave room for the truncation summaries and the final boolean value.
    budget = MAX_DIAGNOSTIC_CHARS - 256
    for detail, blockers in diagnostics[:MAX_DIAGNOSTIC_PODS]:
        for reason in blockers:
            data["blocking_reasons"].append(reason)
            if len(json.dumps(data, ensure_ascii=False)) > budget:
                data["blocking_reasons"].pop()
                reasons_truncated = True
                break
            reported.add(detail["name"])
        data["pods"].append(detail)
        if len(json.dumps(data, ensure_ascii=False)) > budget:
            data["pods"].pop()
    included = {pod["name"] for pod in data["pods"]} | reported
    omitted = sum(
        bool(blockers) for detail, blockers in diagnostics if detail["name"] not in included
    )
    if omitted:
        data["blocking_reasons"].append(
            f"{omitted} additional pods have blocking conditions; pod details are truncated"
        )
    if reasons_truncated:
        data["blocking_reasons"].append(
            "Additional blocking reasons omitted; use get_training_events for details"
        )
    data["pods_truncated"] = len(data["pods"]) < len(diagnostics) or reasons_truncated
    return data


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
        seen_steps: set[str] = set()
        # Match the SDK's newest attempt per step; retries can leave failed pods behind.
        for pod in sorted(
            pods,
            key=lambda p: (
                p.metadata.creation_timestamp is not None,
                p.metadata.creation_timestamp,
            ),
            reverse=True,
        ):
            step = get_training_pod_step(pod) or pod.metadata.name
            if step in seen_steps:
                continue
            seen_steps.add(step)
            detail = _pod_details(pod)
            diagnostics.append((detail, _pod_blocking_reasons(detail)))
        # Keep blockers visible before applying the response cap; prefer active attempts on ties.
        diagnostics.sort(
            key=lambda item: (not item[1], item[0]["phase"] in ("Succeeded", "Failed"))
        )
        return _bounded_diagnostics(diagnostics)
    except Exception:
        logger.debug("Pod diagnostics unavailable for %s", name, exc_info=True)
        return {
            "pod_diagnostics_error": "Could not read pod diagnostics; check namespace access and connectivity"
        }
