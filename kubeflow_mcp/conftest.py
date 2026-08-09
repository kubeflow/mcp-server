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

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─── Sentinel names for side-effect dispatchers ───────────────────────────────

VALID_JOB_NAME = "train-gemma-abc"
NOT_FOUND_NAME = "not-found-job"
TIMEOUT_NAME = "timeout-job"


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_k8s_client():
    """Patch kubernetes.client and return the mock for assertion."""
    with patch("kubernetes.client") as mock_client:
        yield mock_client


def _make_mock_job(name: str, *, status: str = "Running") -> MagicMock:
    job = MagicMock()
    job.name = name
    job.status = status
    runtime = MagicMock()
    runtime.name = "torchtune-llama"
    job.runtime = runtime
    return job


def _configure_trainer_client_mock(client: MagicMock) -> MagicMock:
    """Wire name-keyed side effects onto a TrainerClient mock."""

    def get_job_side_effect(*, name: str, **kwargs: Any) -> MagicMock:
        if name == NOT_FOUND_NAME:
            from kubernetes.client.exceptions import ApiException

            raise ApiException(status=404, reason="Not Found")
        if name == TIMEOUT_NAME:
            raise TimeoutError("timed out")
        return _make_mock_job(name)

    def delete_job_side_effect(*, name: str, **kwargs: Any) -> None:
        if name == NOT_FOUND_NAME:
            from kubernetes.client.exceptions import ApiException

            raise ApiException(status=404, reason="Not Found")
        if name == TIMEOUT_NAME:
            raise TimeoutError("timed out")

    def list_jobs_side_effect(*args: Any, **kwargs: Any) -> list[MagicMock]:
        return [_make_mock_job(VALID_JOB_NAME)]

    def create_job_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        name = kwargs.get("name") or VALID_JOB_NAME
        if name == TIMEOUT_NAME:
            raise TimeoutError("timed out")
        return _make_mock_job(name, status="Created")

    client.get_job.side_effect = get_job_side_effect
    client.delete_job.side_effect = delete_job_side_effect
    client.list_jobs.side_effect = list_jobs_side_effect
    client.create_job.side_effect = create_job_side_effect
    client.get_job_logs.return_value = iter(["Epoch 1/10: loss=2.34"])
    client.get_job_events.return_value = [{"type": "Normal", "message": "Scheduled"}]
    return client


_TRAINER_CLIENT_PATCHES = (
    "kubeflow_mcp.common.utils.get_trainer_client",
    "kubeflow_mcp.common.utils.get_trainer_client_for_namespace",
    "kubeflow_mcp.trainer.api.discovery.get_trainer_client",
    "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
    "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace",
    "kubeflow_mcp.trainer.api.training.get_trainer_client",
    "kubeflow_mcp.trainer.api.training.get_trainer_client_for_namespace",
)


@pytest.fixture
def mock_trainer_client():
    """Patch TrainerClient accessors and return a configured MagicMock."""
    client = _configure_trainer_client_mock(MagicMock())
    patches = [patch(target, return_value=client) for target in _TRAINER_CLIENT_PATCHES]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield client


@pytest.fixture
def mock_k8s_apis():
    """Patch CoreV1Api, CustomObjectsApi, and ApiextensionsV1Api accessors."""
    core_mock = MagicMock()
    custom_mock = MagicMock()
    apiext_mock = MagicMock()

    node = MagicMock()
    node.metadata.labels = {"node.openshift.io/os_id": "rhcos"}
    core_mock.list_node.return_value = MagicMock(items=[node])

    pod = MagicMock()
    pod.metadata.name = "trainer-controller-manager-abc"
    pod.status.phase = "Running"
    core_mock.list_namespaced_pod.return_value = MagicMock(items=[pod])
    core_mock.read_namespaced_pod_log.return_value = "controller started"

    custom_mock.list_cluster_custom_object.return_value = {
        "items": [create_mock_runtime()],
    }
    custom_mock.get_cluster_custom_object.return_value = create_mock_runtime()
    custom_mock.list_namespaced_custom_object.return_value = {
        "items": [create_mock_trainjob()],
    }

    apiext_mock.list_custom_resource_definition.return_value = MagicMock(
        items=[
            MagicMock(
                metadata=MagicMock(name="trainjobs.trainer.kubeflow.org"),
                spec=MagicMock(
                    group="trainer.kubeflow.org",
                    versions=[MagicMock(name="v1", served=True, storage=True)],
                ),
                status=MagicMock(conditions=[]),
            )
        ]
    )

    with (
        patch(
            "kubeflow_mcp.common.utils.get_core_v1_api",
            return_value=core_mock,
        ),
        patch(
            "kubeflow_mcp.common.utils.get_custom_objects_api",
            return_value=custom_mock,
        ),
        patch(
            "kubeflow_mcp.common.utils.get_apiextensions_api",
            return_value=apiext_mock,
        ),
        patch(
            "kubeflow_mcp.common.utils.get_version_api",
            return_value=MagicMock(
                get_code=MagicMock(
                    return_value=MagicMock(git_version="v1.29.3"),
                )
            ),
        ),
    ):
        yield {
            "core_v1": core_mock,
            "custom": custom_mock,
            "apiextensions": apiext_mock,
        }


@pytest.fixture
def tmp_policy_file(tmp_path):
    """Write a temporary policy YAML and patch the lookup path."""
    import yaml

    from kubeflow_mcp.core.policy import _get_cached_policy

    policy_path = tmp_path / ".kf-mcp-policy.yaml"

    def _write(data: dict):
        policy_path.write_text(yaml.dump(data))
        _get_cached_policy.cache_clear()
        return policy_path

    with patch(
        "kubeflow_mcp.core.policy._get_policy_paths",
        return_value=[policy_path],
    ):
        yield _write

    _get_cached_policy.cache_clear()


# ─── Helpers ────────────────────────────────────────────────────────────────


def create_mock_trainjob(
    name: str = VALID_JOB_NAME,
    namespace: str = "default",
    status: str = "Created",
    *,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a realistic TrainJob API response dict for testing."""
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "kubeflow-mcp",
                **(labels or {}),
            },
            "creationTimestamp": "2026-07-12T10:00:00Z",
        },
        "spec": {
            "runtimeRef": {"name": "torchtune-llama"},
        },
        "status": {
            "conditions": [
                {
                    "type": status,
                    "status": "True",
                    "lastTransitionTime": "2026-07-12T10:01:00Z",
                }
            ],
        },
    }


def create_mock_runtime(
    name: str = "torchtune-llama",
    *,
    num_nodes: int = 1,
    image: str = "ghcr.io/kubeflow/trainer/torchtune:latest",
) -> dict[str, Any]:
    """Create a realistic ClusterTrainingRuntime API response dict for testing."""
    return {
        "metadata": {
            "name": name,
            "creationTimestamp": "2026-06-01T00:00:00Z",
        },
        "spec": {
            "template": {
                "spec": {
                    "numNodes": num_nodes,
                    "replicatedJobs": [
                        {
                            "name": "Node",
                            "template": {
                                "spec": {
                                    "template": {
                                        "spec": {
                                            "containers": [{"name": "trainer", "image": image}]
                                        }
                                    }
                                }
                            },
                        }
                    ],
                }
            }
        },
    }


def verify_tool_success(result: dict[str, Any]) -> dict[str, Any]:
    """Verify a tool returned a success response and return the data."""
    assert result.get("success") is True, f"Expected success, got: {result}"
    assert "error" not in result
    return result.get("data", result)


def verify_tool_error(result: dict[str, Any], *, error_code: str | None = None) -> dict[str, Any]:
    """Verify a tool returned an error response."""
    assert result.get("success") is False, f"Expected error, got: {result}"
    assert "error" in result
    if error_code:
        assert result.get("error_code") == error_code
    return result
