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

"""Shared data and helpers for the Tier 1 eval judges."""

from typing import Any

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

# Valid arguments for each read-only tool that some persona hides, so the persona
# judges also try hidden tools that only read. A new one fails
# test_every_hidden_read_only_tool_has_arguments until it is added here.
READ_ARGS: dict[str, dict[str, Any]] = {
    "inspect_controller": {},
    "inspect_crd": {},
    "wait_for_training": {"name": "job-a", "timeout_seconds": 1, "polling_interval": 1},
}


def payload(result: Any) -> dict[str, Any]:
    """Return the tool's response dict from an MCP CallToolResult."""
    body = result.structured_content or {}
    if set(body) == {"result"}:
        body = body["result"]
    return body
