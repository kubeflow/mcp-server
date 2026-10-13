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

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.core.policy import get_effective_persona, set_effective_persona

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Minimal valid arguments that get each mutating tool as far as its preview.
# A new mutating tool fails test_every_mutating_tool_has_gate_arguments until
# it is added here, so the gate check cannot silently skip it.
GATE_ARGS: dict[str, dict[str, Any]] = {
    "create_runtime": {"name": "rt-a", "spec": {"template": {}}},
    "delete_runtime": {"name": "rt-a"},
    "delete_training_job": {"name": "job-a"},
    "fine_tune": {"model": "hf://google/gemma-2b", "dataset": "hf://tatsu-lab/alpaca"},
    "patch_runtime": {"name": "rt-a", "patch": {"metadata": {"labels": {"a": "b"}}}},
    "run_container_training": {"image": "busybox:1.36"},
    "run_custom_training": {"script": "print('hi')"},
    "update_training_job": {"name": "job-a", "action": "suspend"},
}


@dataclass
class FakeKubernetes:
    """Records every request and answers with a mock."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def call_api(self, resource_path: str, method: str) -> Any:
        self.calls.append((method, resource_path))
        return MagicMock()

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

    def fake_call_api(self, resource_path, method, *args, **kwargs):
        return fake.call_api(resource_path, method)

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


@pytest.fixture(autouse=True)
def _restore_persona():
    """create_server() mutates the process-global persona; restore it."""
    previous = get_effective_persona()
    yield
    set_effective_persona(previous)


def payload(result: Any) -> dict[str, Any]:
    """Return the tool's response dict from an MCP CallToolResult."""
    body = result.structured_content or {}
    if set(body) == {"result"}:
        body = body["result"]
    return body
