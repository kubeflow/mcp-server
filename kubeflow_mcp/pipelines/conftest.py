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

"""Shared fixtures for the pipelines tests.

The tools are exercised against a mocked ``PipelinesClient`` and fake SDK objects
shaped like the ``kfp_server_api`` models, so no ``kfp`` install or cluster is needed.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

RUN_ID = "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"
CREATED = datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc)

_API_MODULES = ("discovery", "monitoring", "runs")


class ApiException(Exception):
    """Stand-in for ``kfp_server_api`` / ``kubernetes`` ``ApiException``."""

    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"({status}) {reason}")
        self.status = status


def make_pipeline(name: str = "churn-prediction", pipeline_id: str = "p-1") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_id=pipeline_id,
        display_name=name,
        description="Predicts churn",
        namespace="team-a",
        created_at=CREATED,
        pipeline_spec={"huge": "spec"},
    )


def make_version(name: str = "v1", version_id: str = "pv-1") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_version_id=version_id,
        display_name=name,
        description=None,
        created_at=CREATED,
        pipeline_spec={"huge": "spec"},
    )


def make_task(name: str, state: str = "SUCCEEDED", pod: str | None = None, children=()):
    return SimpleNamespace(
        task_id=f"t-{name}",
        display_name=name,
        state=state,
        pod_name=pod,
        child_tasks=[SimpleNamespace(task_id=f"c-{p}", pod_name=p) for p in children],
        error=None,
    )


def make_run(
    state: str = "RUNNING",
    run_id: str = RUN_ID,
    tasks: list | None = None,
    error: str | None = None,
    params: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        display_name="churn-run",
        state=state,
        experiment_id="e-1",
        pipeline_version_reference=SimpleNamespace(pipeline_id="p-1", pipeline_version_id="pv-1"),
        created_at=CREATED,
        finished_at=None,
        error=SimpleNamespace(message=error) if error else None,
        runtime_config=SimpleNamespace(parameters=params or {"lr": 0.01}),
        run_details=SimpleNamespace(task_details=tasks or []),
        pipeline_spec={"huge": "spec"},
    )


def make_page(field: str, items: list, token: str = "", total: int | None = None):
    return SimpleNamespace(**{field: items, "next_page_token": token, "total_size": total})


@pytest.fixture
def kfp_client(monkeypatch):
    """A mocked PipelinesClient wired into every tool module.

    ``kfp_client.namespaces`` records the namespace each client was built for, and the
    default namespace resolves to ``team-a``.
    """
    client = MagicMock(name="PipelinesClient")
    client.namespaces = []
    client.kfp_client.get_experiment.return_value = SimpleNamespace(
        experiment_id="e-1", namespace="team-a"
    )

    def factory(namespace):
        client.namespaces.append(namespace)
        return client

    for module in _API_MODULES:
        monkeypatch.setattr(f"kubeflow_mcp.pipelines.api.{module}.get_pipelines_client", factory)
    monkeypatch.setattr(
        "kubeflow_mcp.pipelines.api._common.get_pipelines_effective_namespace",
        lambda namespace=None: namespace or "team-a",
    )
    monkeypatch.setattr("kubeflow_mcp.core.policy.get_allowed_namespaces", lambda: None)
    return client
