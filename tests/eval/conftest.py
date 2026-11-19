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

"""Shared fixtures for the Tier 1 eval judges."""

import copy
import importlib.util
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona
from tests.eval.helpers import EXPERIMENT, FAILED_RUN_ID

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# The pipelines tools call a Kubeflow Pipelines client, which talks to KFP's REST API
# rather than the Kubernetes API, so the Kubernetes fake cannot see their writes.
KFP_TOOL_MODULES = ("authoring", "discovery", "experiments", "monitoring", "runs")

# What a GET of one training runtime returns, so previews that read the runtime's
# node job (run_custom_training) see a real one instead of a mock.
RUNTIME: dict[str, Any] = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainingRuntime",
    "metadata": {
        "name": "torch-distributed",
        "namespace": "default",
        "labels": {"trainer.kubeflow.org/framework": "torch"},
    },
    "spec": {
        "mlPolicy": {"numNodes": 1},
        "template": {
            "spec": {
                "replicatedJobs": [
                    {
                        "name": "node",
                        "template": {
                            "metadata": {
                                "labels": {"trainer.kubeflow.org/trainjob-ancestor-step": "trainer"}
                            },
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {"name": "node", "image": "python:3.11-slim"}
                                        ]
                                    }
                                }
                            },
                        },
                    }
                ]
            }
        },
    },
}
RUNTIME_PLURALS = {"trainingruntimes", "clustertrainingruntimes"}

# What a GET of one Katib Experiment returns, so update_hpo_experiment can read the
# live spec and status it previews against instead of a mock.
KATIB_EXPERIMENT: dict[str, Any] = {
    "apiVersion": "kubeflow.org/v1beta1",
    "kind": "Experiment",
    "metadata": {
        "name": "hpo-a",
        "namespace": "default",
        "resourceVersion": "1",
        "labels": {"kubeflow-mcp/managed-by": "mcp"},
    },
    "spec": {"parallelTrialCount": 3, "maxTrialCount": 12},
    "status": {"trials": 0},
}


@dataclass
class FakeKubernetes:
    """Records every request and answers with a mock (a real runtime for runtime GETs)."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def call_api(
        self, resource_path: str, method: str, path_params: Any = None, async_req: bool = False
    ) -> Any:
        self.calls.append((method, resource_path))
        params = path_params or {}
        body: Any = MagicMock()
        if method == "GET" and params.get("plural") in RUNTIME_PLURALS and "name" in params:
            body = copy.deepcopy(RUNTIME)
        elif method == "GET" and params.get("plural") == "experiments" and "name" in params:
            body = copy.deepcopy(KATIB_EXPERIMENT)
        elif method == "GET" and params.get("plural") == "trials":
            body = {"items": []}
        if not async_req or isinstance(body, MagicMock):
            return body
        # The SDK calls with async_req=True and reads the body from thread.get().
        thread = MagicMock()
        thread.get.return_value = body
        return thread

    @property
    def writes(self) -> list[tuple[str, str]]:
        return [call for call in self.calls if call[0] in WRITE_METHODS]


@pytest.fixture
def k8s():
    """Fake the Kubernetes API at ApiClient.call_api.

    Both the Kubeflow SDK and the raw Kubernetes clients send every request
    through this method, so a write is seen whichever client a tool uses.
    """
    fake = FakeKubernetes()

    def fake_call_api(self, resource_path, method, path_params=None, *args, **kwargs):
        return fake.call_api(resource_path, method, path_params, kwargs.get("async_req", False))

    with (
        patch("kubernetes.config.load_kube_config"),
        patch("kubernetes.config.load_incluster_config"),
        patch("kubernetes.client.ApiClient.call_api", fake_call_api),
        # No GPUs exist behind the fake API; fine_tune refuses to preview without them.
        patch("kubeflow_mcp.trainer.api.training._check_gpu_available", return_value=None),
    ):
        mcp_utils.reset_clients()
        yield fake
    mcp_utils.reset_clients()


@dataclass
class FakeKfp:
    """A Kubeflow Pipelines client that records every method called on it."""

    client: MagicMock = field(default_factory=lambda: MagicMock(name="PipelinesClient"))

    @property
    def calls(self) -> list[str]:
        # "get_run().state" is an attribute of a returned object, not a request.
        return [name for name, _, _ in self.client.mock_calls if "()" not in name]

    @property
    def writes(self) -> list[str]:
        """Every call that is not a get or list, so an unlisted write method counts too."""
        return [name for name in self.calls if not _is_kfp_read(name.rsplit(".", 1)[-1])]


def _get_experiment(name: str, *args: Any, **kwargs: Any) -> SimpleNamespace:
    # Only EXPERIMENT exists, so create_experiment can preview a new one.
    if name != EXPERIMENT:
        raise ValueError(f"Experiment not found: {name}")
    return SimpleNamespace(experiment_id="e-1", display_name=EXPERIMENT, namespace="default")


def _is_kfp_read(method: str) -> bool:
    return method.startswith(("get_", "list_")) or "_service_get_" in method


@pytest.fixture(autouse=True)
def kfp():
    """Fake the Kubeflow Pipelines client, so no judge reaches a real KFP server."""
    fake = FakeKfp()
    client = fake.client
    client.get_pipeline.return_value = SimpleNamespace(
        pipeline_id="p-1", display_name="pipe-a", description=None, namespace="default"
    )
    client.list_pipeline_versions.return_value = SimpleNamespace(
        pipeline_versions=[SimpleNamespace(pipeline_version_id="v-1")]
    )
    client.get_run.side_effect = lambda run_id, *args, **kwargs: SimpleNamespace(
        run_id=run_id,
        display_name="run-a",
        state="FAILED" if run_id == FAILED_RUN_ID else "RUNNING",
        experiment_id="e-1",
        error=None,
    )
    client.get_experiment.side_effect = _get_experiment
    # Empty pages, so list tools return a response instead of a mock they cannot serialize.
    page = {"next_page_token": None, "total_size": 0}
    client.list_pipelines.return_value = SimpleNamespace(pipelines=[], **page)
    client.list_runs.return_value = SimpleNamespace(runs=[], **page)
    client.list_experiments.return_value = SimpleNamespace(experiments=[], **page)
    # stop and retry check the run's experiment namespace before they preview.
    experiments = client._backend.experiment_api
    experiments.experiment_service_get_experiment.return_value = SimpleNamespace(
        experiment_id="e-1", namespace="default"
    )
    with ExitStack() as stack:
        if importlib.util.find_spec("kubeflow_mcp.pipelines") is not None:
            for module in KFP_TOOL_MODULES:
                stack.enter_context(
                    patch(
                        f"kubeflow_mcp.pipelines.api.{module}.get_pipelines_client",
                        lambda *args, **kwargs: fake.client,
                    )
                )
        yield fake


@pytest.fixture(autouse=True)
def _restore_persona():
    """create_server() mutates the process-global persona; restore it."""
    previous = get_effective_persona()
    yield
    set_effective_persona(previous)
