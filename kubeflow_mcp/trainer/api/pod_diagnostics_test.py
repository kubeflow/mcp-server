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

"""Regression tests for job-scoped scheduling and container diagnostics."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client as k8s
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.trainer.api.pod_diagnostics import get_training_pod_diagnostics
from kubeflow_mcp.trainer.constants import (
    MAX_DIAGNOSTIC_CHARS,
    MAX_DIAGNOSTIC_MESSAGE_CHARS,
    MAX_DIAGNOSTIC_PODS,
)


def _pod(name="trainer-node-0", phase="Pending", **status):
    return k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(name=name),
        spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="node")]),
        status=k8s.V1PodStatus(phase=phase, **status),
    )


def test_pending_pod_explains_scheduling_and_scopes_lookup():
    pod = _pod(
        conditions=[
            k8s.V1PodCondition(
                type="PodScheduled",
                status="False",
                reason="Unschedulable",
                message="0/2 nodes available: Insufficient nvidia.com/gpu",
            )
        ]
    )
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [pod]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    args = api.list_namespaced_pod.call_args.kwargs
    assert args["namespace"] == "workloads"
    assert "trainer" in args["label_selector"]
    assert args["_request_timeout"] > 0
    assert result["pods"][0]["ready"] is False
    assert "Unschedulable" in result["blocking_reasons"][0]
    assert "Insufficient nvidia.com/gpu" in result["pods"][0]["conditions"][0]["message"]


def test_running_pod_surfaces_crashloop_and_initializer_termination():
    crashed = k8s.V1ContainerStatus(
        name="node",
        image="trainer",
        image_id="id",
        ready=False,
        restart_count=3,
        state=k8s.V1ContainerState(
            waiting=k8s.V1ContainerStateWaiting(
                reason="CrashLoopBackOff", message="back-off restarting failed container"
            )
        ),
        last_state=k8s.V1ContainerState(
            terminated=k8s.V1ContainerStateTerminated(exit_code=137, reason="OOMKilled")
        ),
    )
    init = k8s.V1ContainerStatus(
        name="dataset-initializer",
        image="initializer",
        image_id="id",
        ready=False,
        restart_count=0,
        state=k8s.V1ContainerState(
            terminated=k8s.V1ContainerStateTerminated(
                exit_code=1, reason="Error", message="HF_TOKEN=private-token download failed"
            )
        ),
    )
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [
        _pod(phase="Running", container_statuses=[crashed], init_container_statuses=[init])
    ]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    pod = result["pods"][0]
    assert pod["phase"] == "Running"
    assert pod["restart_count"] == 3
    assert pod["containers"][1]["last_termination"] == {"reason": "OOMKilled", "exit_code": 137}
    assert pod["containers"][0]["init"] is True
    assert "private-token" not in str(result)
    assert any("CrashLoopBackOff" in r for r in result["blocking_reasons"])
    assert any("exit 1" in r for r in result["blocking_reasons"])


def test_pod_diagnostics_bounded_and_errors_explicit():
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [
        _pod(name=f"old-{n}", phase="Succeeded") for n in range(MAX_DIAGNOSTIC_PODS)
    ] + [_pod()]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
        assert result["pods_truncated"] is True
        assert len(result["pods"]) == MAX_DIAGNOSTIC_PODS
        assert result["pods"][0]["phase"] == "Pending"
        api.list_namespaced_pod.side_effect = ApiException(status=403)
        denied = get_training_pod_diagnostics("trainer", "workloads")
    assert "pod_diagnostics_error" in denied
    assert "pods" not in denied


@pytest.mark.parametrize("failure", ["crashloop", "unschedulable", "failed"])
def test_blocked_pod_beyond_cap_is_prioritized(failure: str) -> None:
    blocked = _pod(name="blocked", phase="Running")
    if failure == "crashloop":
        blocked.status.container_statuses = [
            k8s.V1ContainerStatus(
                name="node",
                image="trainer",
                image_id="id",
                ready=False,
                restart_count=4,
                state=k8s.V1ContainerState(
                    waiting=k8s.V1ContainerStateWaiting(reason="CrashLoopBackOff")
                ),
            )
        ]
    elif failure == "unschedulable":
        blocked.status.phase = "Pending"
        blocked.status.conditions = [
            k8s.V1PodCondition(
                type="PodScheduled",
                status="False",
                reason="Unschedulable",
                message="Insufficient nvidia.com/gpu",
            )
        ]
    else:
        blocked.status.phase = "Failed"
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [
        _pod(name=f"healthy-{i}", phase="Running") for i in range(MAX_DIAGNOSTIC_PODS)
    ] + [blocked]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    assert result["pods_truncated"] is True
    assert len(result["pods"]) == MAX_DIAGNOSTIC_PODS
    assert result["pods"][0]["name"] == "blocked"
    assert result["blocking_reasons"]
    assert "blocked" in result["blocking_reasons"][0]


def test_omitted_blockers_are_summarized_without_expanding_response() -> None:
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [
        _pod(name=f"failed-{i}", phase="Failed") for i in range(MAX_DIAGNOSTIC_PODS + 2)
    ]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    assert len(result["pods"]) == MAX_DIAGNOSTIC_PODS
    assert len(result["blocking_reasons"]) == MAX_DIAGNOSTIC_PODS + 1
    assert result["blocking_reasons"][-1].startswith("2 additional pods have blocking conditions")


def test_wide_failed_jobs_respect_one_diagnostic_character_budget() -> None:
    container = k8s.V1ContainerStatus(
        name="node",
        image="trainer",
        image_id="id",
        ready=False,
        restart_count=2,
        state=k8s.V1ContainerState(
            waiting=k8s.V1ContainerStateWaiting(
                reason="CrashLoopBackOff",
                message="x" * (MAX_DIAGNOSTIC_MESSAGE_CHARS + 100),
            )
        ),
    )
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [
        _pod(name=f"blocked-{i}", container_statuses=[container] * 20)
        for i in range(MAX_DIAGNOSTIC_PODS)
    ]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    assert len(json.dumps(result, ensure_ascii=False)) <= MAX_DIAGNOSTIC_CHARS
    assert result["pods_truncated"] is True
    assert result["blocking_reasons"][0].startswith("blocked-0/node: CrashLoopBackOff")
    assert any("additional pods" in reason for reason in result["blocking_reasons"])


def test_individual_condition_and_container_messages_are_capped() -> None:
    from kubeflow_mcp.trainer.api.pod_diagnostics import _pod_details

    long_message = "x" * (MAX_DIAGNOSTIC_MESSAGE_CHARS + 100)
    container = k8s.V1ContainerStatus(
        name="node",
        image="trainer",
        image_id="id",
        ready=False,
        restart_count=0,
        state=k8s.V1ContainerState(waiting=k8s.V1ContainerStateWaiting(message=long_message)),
    )
    detail = _pod_details(
        _pod(
            container_statuses=[container],
            conditions=[
                k8s.V1PodCondition(type="PodScheduled", status="False", message=long_message),
            ],
        )
    )
    assert len(detail["containers"][0]["message"]) == MAX_DIAGNOSTIC_MESSAGE_CHARS
    assert len(detail["conditions"][0]["message"]) == MAX_DIAGNOSTIC_MESSAGE_CHARS


@pytest.mark.parametrize("reason", ["ContainerCreating", "PodInitializing"])
def test_starting_containers_remain_visible_without_blockers(reason: str) -> None:
    container = k8s.V1ContainerStatus(
        name="node",
        image="trainer",
        image_id="id",
        ready=False,
        restart_count=0,
        state=k8s.V1ContainerState(waiting=k8s.V1ContainerStateWaiting(reason=reason)),
    )
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [_pod(container_statuses=[container])]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    assert result["blocking_reasons"] == []
    assert result["pods"][0]["containers"][0]["reason"] == reason


@pytest.mark.parametrize("role", ["node", "launcher", "dataset-initializer"])
@pytest.mark.parametrize("phase", ["Running", "Failed"])
def test_only_the_latest_attempt_of_each_step_contributes_diagnostics(
    role: str, phase: str
) -> None:
    from kubeflow.trainer.constants import constants

    old = _pod(name="old-attempt", phase="Failed")
    current = _pod(name="current-attempt", phase=phase)
    for pod, age in ((old, 1), (current, 0)):
        pod.metadata.labels = {constants.JOBSET_RJOB_NAME_LABEL: role}
        if role in (constants.NODE, constants.LAUNCHER):
            pod.metadata.labels[constants.JOB_INDEX_LABEL] = "0"
        pod.metadata.creation_timestamp = datetime(2026, 11, 19, tzinfo=timezone.utc) - timedelta(
            minutes=age
        )
    api = MagicMock()
    api.list_namespaced_pod.return_value.items = [old, current, _pod(name="unlabeled")]
    with patch("kubeflow_mcp.trainer.api.pod_diagnostics.get_core_v1_api", return_value=api):
        result = get_training_pod_diagnostics("trainer", "workloads")
    assert {p["name"] for p in result["pods"]} == {"current-attempt", "unlabeled"}
    assert bool(result["blocking_reasons"]) == (phase == "Failed")
    assert "old-attempt" not in str(result)


def test_omitted_count_excludes_pods_whose_reasons_were_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubeflow_mcp.trainer.api import pod_diagnostics

    monkeypatch.setattr(pod_diagnostics, "MAX_DIAGNOSTIC_CHARS", 800)
    result = pod_diagnostics._bounded_diagnostics(
        [
            ({"name": "reported", "message": "x" * 1000}, ["reported: Failed"]),
            ({"name": "omitted", "message": "x" * 1000}, ["omitted: " + "x" * 1000]),
        ]
    )
    assert result["pods"] == []
    assert result["blocking_reasons"][0] == "reported: Failed"
    assert any(r.startswith("1 additional pods") for r in result["blocking_reasons"])
    assert len(json.dumps(result)) <= 800
