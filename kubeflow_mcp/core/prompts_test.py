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
"""Tests for MCP prompts."""

import asyncio

from fastmcp import FastMCP

from kubeflow_mcp.core.prompts import register_prompts


def _render(mcp: FastMCP, name: str, arguments: dict | None = None) -> str:
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(mcp.render_prompt(name, arguments or {}))
        return result.messages[0].content.text
    finally:
        loop.close()


class TestRegisterPrompts:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)
        loop = asyncio.new_event_loop()
        self.prompts = {p.name: p for p in loop.run_until_complete(self.mcp._list_prompts())}
        loop.close()

    def test_registers_all_prompts(self):
        expected = {
            "fine_tuning_workflow",
            "custom_training_workflow",
            "troubleshooting_guide",
            "resource_planning",
            "monitoring_workflow",
        }
        assert set(self.prompts.keys()) == expected

    def test_prompts_have_descriptions(self):
        for name, prompt in self.prompts.items():
            assert prompt.description, f"{name} missing description"


class TestFineTuningWorkflow:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)

    def test_default_placeholders(self):
        text = _render(self.mcp, "fine_tuning_workflow")
        assert "your-model" in text
        assert "your-dataset" in text

    def test_with_model_and_dataset(self):
        text = _render(self.mcp, "fine_tuning_workflow", {"model": "gemma-2b", "dataset": "alpaca"})
        assert "gemma-2b" in text
        assert "alpaca" in text

    def test_contains_workflow_steps(self):
        text = _render(self.mcp, "fine_tuning_workflow")
        assert "Step 1" in text
        assert "get_cluster_resources" in text
        assert "confirmed=False" in text
        assert "confirmed=True" in text


class TestCustomTrainingWorkflow:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)

    def test_script_type_default(self):
        text = _render(self.mcp, "custom_training_workflow")
        assert "Custom Script Training" in text
        assert "run_custom_training" in text

    def test_container_type(self):
        text = _render(self.mcp, "custom_training_workflow", {"training_type": "container"})
        assert "Container Training" in text
        assert "run_container_training" in text

    def test_script_type_includes_safety(self):
        text = _render(self.mcp, "custom_training_workflow", {"training_type": "script"})
        assert "safety" in text.lower()


class TestTroubleshootingGuide:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)

    def test_default_guide(self):
        text = _render(self.mcp, "troubleshooting_guide")
        assert "Diagnostic Workflow" in text
        assert "get_training_job" in text

    def test_oom_guide(self):
        text = _render(self.mcp, "troubleshooting_guide", {"error_type": "oom"})
        assert "OOMKilled" in text
        assert "batch_size" in text

    def test_pending_guide(self):
        text = _render(self.mcp, "troubleshooting_guide", {"error_type": "pending"})
        assert "Pending" in text
        assert "FailedScheduling" in text

    def test_image_guide(self):
        text = _render(self.mcp, "troubleshooting_guide", {"error_type": "image"})
        assert "Image Pull" in text

    def test_nccl_guide(self):
        text = _render(self.mcp, "troubleshooting_guide", {"error_type": "nccl"})
        assert "NCCL" in text

    def test_unknown_type_returns_general(self):
        text = _render(self.mcp, "troubleshooting_guide", {"error_type": "unknown_xyz"})
        assert "Common Issues" in text


class TestResourcePlanning:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)

    def test_default_model(self):
        text = _render(self.mcp, "resource_planning")
        assert "your model" in text

    def test_with_model(self):
        text = _render(self.mcp, "resource_planning", {"model": "llama-3-8b"})
        assert "llama-3-8b" in text

    def test_contains_reference_tables(self):
        text = _render(self.mcp, "resource_planning")
        assert "GPU Memory Reference" in text
        assert "Batch Size Guide" in text


class TestMonitoringWorkflow:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_prompts(self.mcp)

    def test_default_placeholder(self):
        text = _render(self.mcp, "monitoring_workflow")
        assert "<job-name>" in text

    def test_with_job_name(self):
        text = _render(self.mcp, "monitoring_workflow", {"job_name": "train-gemma-abc"})
        assert "train-gemma-abc" in text

    def test_contains_recovery_actions(self):
        text = _render(self.mcp, "monitoring_workflow")
        assert "delete_training_job" in text
        assert "suspend_training_job" in text
        assert "resume_training_job" in text
