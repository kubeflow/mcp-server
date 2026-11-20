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

"""Direct reads of Katib's own resources, shared by the optimizer tools.

The SDK's ``OptimizerClient`` readers only understand experiments it could have
created itself. ``get_job()`` converts the algorithm through a registry holding
only ``random`` and ``grid``, so every other algorithm the creation tools accept
raises; and every trial read resolves a TrainJob named after the trial, so a
``batch/v1`` Job (or any other) trial template fails, and is even reported as a
missing experiment. Both were hit in cluster testing.

Everything here reads the Experiment and Trial custom resources and the trial
pods directly. Those describe every algorithm and workload kind, and the reads
are bounded by explicit timeouts and size limits, which the SDK readers lack.
"""

from typing import Any

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_LABEL,
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
    METRICS_COLLECTOR_CONTAINER,
    TRIAL_LABEL,
    TRIAL_PLURAL,
)

# Labels each workload kind's own controller puts on its pods, keyed by the
# trial's runSpec kind and named after the workload (Katib names the workload
# after the trial). Used when Katib's own trial label is absent, which happens
# when the namespace has no metrics-collector injection.
_WORKLOAD_POD_LABELS = {
    "Job": "batch.kubernetes.io/job-name",
    "TrainJob": "jobset.sigs.k8s.io/jobset-name",
}


def read_experiment(name: str, namespace: str) -> dict[str, Any]:
    """Fetch an Experiment CR. Raises whatever the Kubernetes client raises."""
    return mcp_utils.get_custom_objects_api().get_namespaced_custom_object(
        group=KATIB_API_GROUP,
        version=KATIB_API_VERSION,
        namespace=namespace,
        plural=EXPERIMENT_PLURAL,
        name=name,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def list_trials(name: str, namespace: str) -> list[dict[str, Any]]:
    """Every Trial CR of an experiment, oldest first.

    Ordered by creation time, the order Katib itself uses when it chooses
    trials to delete, with the name as a tie-breaker so the result is stable.
    """
    resp = mcp_utils.get_custom_objects_api().list_namespaced_custom_object(
        group=KATIB_API_GROUP,
        version=KATIB_API_VERSION,
        namespace=namespace,
        plural=TRIAL_PLURAL,
        label_selector=f"{EXPERIMENT_LABEL}={name}",
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )
    trials = resp.get("items") or []
    return sorted(
        trials,
        key=lambda t: (
            (t.get("metadata") or {}).get("creationTimestamp") or "",
            (t.get("metadata") or {}).get("name") or "",
        ),
    )


def find_trial_pods(trial: dict[str, Any], namespace: str) -> list[Any]:
    """Pods running a trial, found by Katib's trial label or the workload's own."""
    name = (trial.get("metadata") or {}).get("name")
    kind = ((trial.get("spec") or {}).get("runSpec") or {}).get("kind")
    selectors = [f"{TRIAL_LABEL}={name}"]
    if kind in _WORKLOAD_POD_LABELS:
        selectors.append(f"{_WORKLOAD_POD_LABELS[kind]}={name}")

    core = mcp_utils.get_core_v1_api()
    for selector in selectors:
        pods = core.list_namespaced_pod(
            namespace=namespace,
            label_selector=selector,
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        ).items
        if pods:
            return sorted(pods, key=lambda p: p.metadata.name)
    return []


def pick_log_source(trial: dict[str, Any], pods: list[Any]) -> tuple[Any, str] | None:
    """Choose the pod and container whose log shows the trial's output.

    Katib's StdOut metrics collector redirects the primary container's output to
    a file and echoes it from its sidecar, so the sidecar's log is the trial's
    output whenever it is present (this is also what the SDK reads). Without a
    sidecar, the trial's primary container is used, then the first container.
    """
    primary = (trial.get("spec") or {}).get("primaryContainerName")
    for preferred in (METRICS_COLLECTOR_CONTAINER, primary):
        for pod in pods:
            names = [c.name for c in (pod.spec.containers or [])]
            if preferred and preferred in names:
                return pod, preferred
    for pod in pods:
        if pod.spec.containers:
            return pod, pod.spec.containers[0].name
    return None


def read_pod_log(
    pod: Any, container: str, namespace: str, tail_lines: int, limit_bytes: int
) -> str:
    """Read the end of one container's log, bounded on the server side.

    ``tail_lines`` and ``limit_bytes`` are applied by the API server, so a large
    log is never transferred or held in memory in full.
    """
    return mcp_utils.get_core_v1_api().read_namespaced_pod_log(
        name=pod.metadata.name,
        namespace=namespace,
        container=container,
        tail_lines=tail_lines,
        limit_bytes=limit_bytes,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )


def list_experiment_pods(name: str, namespace: str) -> list[Any]:
    """Pods carrying the experiment's label: its trial pods and suggestion pod."""
    return (
        mcp_utils.get_core_v1_api()
        .list_namespaced_pod(
            namespace=namespace,
            label_selector=f"{EXPERIMENT_LABEL}={name}",
            _request_timeout=mcp_utils.K8S_TIMEOUT,
        )
        .items
    )


def list_events_for(names: set[str], namespace: str) -> list[Any]:
    """Namespace events about any of the named objects."""
    events = (
        mcp_utils.get_core_v1_api()
        .list_namespaced_event(namespace=namespace, _request_timeout=mcp_utils.K8S_TIMEOUT)
        .items
    )
    return [
        e for e in events if getattr(getattr(e, "involved_object", None), "name", None) in names
    ]
