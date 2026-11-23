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

"""A fake Katib cluster for the optimizer unit tests.

The optimizer tools read Katib's custom resources and trial pods directly, so
the tests fake the two Kubernetes clients they use rather than the SDK. The
fake answers like the API server: unknown objects raise a 404 ``ApiException``
and label selectors filter lists, so a tool that reads the wrong resource fails
here the way it would against a cluster.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE

NAMESPACE = "kubeflow"

# Katib Trial condition for each state the tools report.
_STATE_CONDITIONS = {
    "Created": ["Created"],
    "Running": ["Created", "Running"],
    "Complete": ["Created", "Succeeded"],
    "Failed": ["Created", "Failed"],
    "Killed": ["Created", "Killed"],
    "EarlyStopped": ["Created", "EarlyStopped"],
    "MetricsUnavailable": ["Created", "MetricsUnavailable"],
}


def experiment_cr(
    name: str = "exp-1",
    *,
    algorithm: str = "random",
    parallel: int = 2,
    max_trials: int = 10,
    max_failed: int | None = None,
    managed: bool = True,
    resource_version: str = "7",
    conditions: list[dict[str, Any]] | None = None,
    status: dict[str, Any] | None = None,
    **spec: Any,
) -> dict[str, Any]:
    """An Experiment CR as the API server returns it (defaults applied)."""
    body: dict[str, Any] = {
        "objective": {"type": "maximize", "objectiveMetricName": "accuracy"},
        "algorithm": {"algorithmName": algorithm},
        "parallelTrialCount": parallel,
        "maxTrialCount": max_trials,
        "parameters": [
            {
                "name": "lr",
                "parameterType": "double",
                "feasibleSpace": {"min": "0.001", "max": "0.1", "distribution": "logUniform"},
            }
        ],
        "resumePolicy": "Never",
        **spec,
    }
    if max_failed is not None:
        body["maxFailedTrialCount"] = max_failed
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "resourceVersion": resource_version,
            "uid": f"uid-{name}",
            "creationTimestamp": "2026-11-01T00:00:00Z",
            "labels": {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE} if managed else {},
        },
        "spec": body,
        "status": {"conditions": conditions or [], **(status or {})},
    }


def trial_cr(
    name: str,
    state: str = "Running",
    *,
    experiment: str = "exp-1",
    created: str = "2026-11-01T00:00:01Z",
    kind: str = "TrainJob",
    params: dict[str, str] | None = None,
    metrics: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """A Trial CR whose conditions put it in ``state``."""
    conditions = [{"type": c, "status": "True"} for c in _STATE_CONDITIONS[state]]
    trial_status: dict[str, Any] = {"conditions": conditions}
    if metrics is not None:
        trial_status["observation"] = {"metrics": metrics}
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Trial",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "creationTimestamp": created,
            "labels": {"katib.kubeflow.org/experiment": experiment},
        },
        "spec": {
            "primaryContainerName": "node" if kind == "TrainJob" else "train",
            "parameterAssignments": [
                {"name": k, "value": v} for k, v in (params or {"lr": "0.01"}).items()
            ],
            "runSpec": {"kind": kind, "metadata": {"name": name}},
        },
        "status": trial_status,
    }


def metric(name: str = "accuracy", latest: str = "0.95") -> dict[str, str]:
    return {"name": name, "min": "0.5", "max": latest, "latest": latest}


def pod(name: str, labels: dict[str, str], containers: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels=labels),
        spec=SimpleNamespace(containers=[SimpleNamespace(name=c) for c in containers]),
    )


def event(
    name: str, kind: str, reason: str, when: str, *, type_: str = "Normal"
) -> SimpleNamespace:
    return SimpleNamespace(
        type=type_,
        involved_object=SimpleNamespace(kind=kind, name=name),
        reason=reason,
        message=f"{reason} for {name}",
        last_timestamp=when,
        event_time=None,
        first_timestamp=when,
    )


def _not_found(kind: str, name: str) -> ApiException:
    return ApiException(status=404, reason=f"{kind} {name} not found")


def _matches(labels: dict[str, str], selector: str | None) -> bool:
    if not selector:
        return True
    key, _, value = selector.partition("=")
    return labels.get(key) == value


@dataclass
class FakeCustomObjects:
    """``CustomObjectsApi`` over in-memory Katib resources."""

    experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)
    suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    patch_error: Exception | None = None

    def _store(self, plural: str) -> dict[str, dict[str, Any]]:
        if plural == "experiments":
            return self.experiments
        if plural == "suggestions":
            return self.suggestions
        return {t["metadata"]["name"]: t for t in self.trials}

    def get_namespaced_custom_object(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get", kw))
        found = self._store(kw["plural"]).get(kw["name"])
        if found is None:
            raise _not_found(kw["plural"], kw["name"])
        return found

    def list_namespaced_custom_object(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("list", kw))
        items = list(self._store(kw["plural"]).values())
        selector = kw.get("label_selector")
        return {"items": [i for i in items if _matches(i["metadata"].get("labels", {}), selector)]}

    def patch_namespaced_custom_object(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("patch", kw))
        if self.patch_error is not None:
            raise self.patch_error
        stored = self._store(kw["plural"]).get(kw["name"]) or {}
        expected = (kw["body"].get("metadata") or {}).get("resourceVersion")
        if expected and expected != stored.get("metadata", {}).get("resourceVersion"):
            raise ApiException(status=409, reason="Conflict")
        return {}

    def delete_namespaced_custom_object(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("delete", kw))
        stored = self._store(kw["plural"]).get(kw["name"])
        if stored is None:
            raise _not_found(kw["plural"], kw["name"])
        metadata = stored["metadata"]
        preconditions = (kw.get("body") or {}).get("preconditions") or {}
        if any(metadata.get(key) != value for key, value in preconditions.items()):
            raise ApiException(status=409, reason="Conflict")
        return {}

    def create_namespaced_custom_object(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("create", kw))
        if kw.get("dry_run"):
            return kw["body"]
        stored = {**kw["body"], "status": {"conditions": []}}
        self._store(kw["plural"])[stored["metadata"]["name"]] = stored
        return stored

    def writes(self) -> list[tuple[str, dict[str, Any]]]:
        return [c for c in self.calls if c[0] in {"patch", "delete", "create"}]


@dataclass
class FakeCoreV1:
    """``CoreV1Api`` over in-memory pods, logs and events."""

    pods: list[SimpleNamespace] = field(default_factory=list)
    logs: dict[tuple[str, str], str] = field(default_factory=dict)
    events: list[SimpleNamespace] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def list_namespaced_pod(self, **kw: Any) -> SimpleNamespace:
        self.calls.append(("list_pods", kw))
        selector = kw.get("label_selector")
        return SimpleNamespace(
            items=[p for p in self.pods if _matches(p.metadata.labels, selector)]
        )

    def read_namespaced_pod_log(self, **kw: Any) -> str:
        self.calls.append(("read_log", kw))
        return self.logs.get((kw["name"], kw["container"]), "")

    def list_namespaced_event(self, **kw: Any) -> SimpleNamespace:
        self.calls.append(("list_events", kw))
        return SimpleNamespace(items=list(self.events))


@dataclass
class FakeKatib:
    custom: FakeCustomObjects = field(default_factory=FakeCustomObjects)
    core: FakeCoreV1 = field(default_factory=FakeCoreV1)

    def add_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        self.custom.experiments[experiment["metadata"]["name"]] = experiment
        return experiment

    def add_trials(self, *trials: dict[str, Any]) -> None:
        self.custom.trials.extend(trials)


# Modules that import the client factories by name, as well as the module that
# defines them, so the fake is seen however a tool reaches the API.
_PATCH_SITES = (
    "kubeflow_mcp.common.utils",
    "kubeflow_mcp.optimizer.api.discovery",
    "kubeflow_mcp.optimizer.api.monitoring",
)


@pytest.fixture
def katib() -> Iterator[FakeKatib]:
    """A fake cluster with Katib installed and the namespace resolved to ``kubeflow``."""
    fake = FakeKatib()
    patches = [
        patch(f"{site}.{name}", target, create=False)
        for site in _PATCH_SITES
        for name, target in (
            ("get_custom_objects_api", lambda: fake.custom),
            ("get_optimizer_effective_namespace", lambda ns=None: ns or NAMESPACE),
        )
    ]
    patches.append(patch("kubeflow_mcp.common.utils.get_core_v1_api", lambda: fake.core))
    for p in patches:
        p.start()
    try:
        yield fake
    finally:
        for p in reversed(patches):
            p.stop()
