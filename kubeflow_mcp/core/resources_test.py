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
"""Tests for MCP resources."""

import asyncio

from fastmcp import FastMCP

from kubeflow_mcp.core.resources import register_resources


def _read_resource(mcp: FastMCP, uri: str) -> str:
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(mcp.read_resource(uri))
        return str(result)
    finally:
        loop.close()


class TestRegisterResources:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_resources(self.mcp)

    def test_supported_models_registered(self):
        text = _read_resource(self.mcp, "trainer://models/supported")
        assert "Supported Models" in text

    def test_runtime_info_registered(self):
        text = _read_resource(self.mcp, "trainer://runtimes/info")
        assert "Training Runtimes" in text

    def test_quickstart_registered(self):
        text = _read_resource(self.mcp, "trainer://guides/quickstart")
        assert "Quick Start Guide" in text

    def test_troubleshooting_registered(self):
        text = _read_resource(self.mcp, "trainer://guides/troubleshooting")
        assert "Troubleshooting" in text


class TestSupportedModels:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_resources(self.mcp)
        self.text = _read_resource(self.mcp, "trainer://models/supported")

    def test_contains_small_models(self):
        assert "gemma-2b" in self.text
        assert "Llama-3.2-1B" in self.text

    def test_contains_medium_models(self):
        assert "Mistral-7B" in self.text
        assert "Llama-3.1-8B" in self.text

    def test_contains_large_models(self):
        assert "Llama-3.1-70B" in self.text

    def test_contains_dataset_info(self):
        assert "tatsu-lab/alpaca" in self.text
        assert "databricks/dolly-15k" in self.text

    def test_contains_id_format_guide(self):
        assert "estimate_resources()" in self.text
        assert "fine_tune()" in self.text
        assert "hf://" in self.text


class TestRuntimeInfo:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_resources(self.mcp)
        self.text = _read_resource(self.mcp, "trainer://runtimes/info")

    def test_contains_torch_tune(self):
        assert "torch-tune" in self.text
        assert "fine_tune()" in self.text

    def test_contains_torch_distributed(self):
        assert "torch-distributed" in self.text
        assert "run_custom_training()" in self.text

    def test_contains_runtime_patches(self):
        assert "node_selector" in self.text
        assert "tolerations" in self.text

    def test_contains_list_runtimes_example(self):
        assert "list_runtimes()" in self.text


class TestQuickstartGuide:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_resources(self.mcp)
        self.text = _read_resource(self.mcp, "trainer://guides/quickstart")

    def test_has_numbered_steps(self):
        assert "## 1." in self.text
        assert "## 2." in self.text
        assert "## 3." in self.text
        assert "## 4." in self.text

    def test_contains_preview_pattern(self):
        assert "confirmed=False" in self.text
        assert "confirmed=True" in self.text

    def test_contains_common_issues_table(self):
        assert "OOMKilled" in self.text
        assert "batch_size" in self.text


class TestTroubleshootingQuickRef:
    def setup_method(self):
        self.mcp = FastMCP("test")
        register_resources(self.mcp)
        self.text = _read_resource(self.mcp, "trainer://guides/troubleshooting")

    def test_contains_diagnostic_commands(self):
        assert "get_training_job" in self.text
        assert "get_training_events" in self.text
        assert "get_training_logs" in self.text

    def test_contains_status_table(self):
        assert "Created" in self.text
        assert "Running" in self.text
        assert "Failed" in self.text
        assert "Complete" in self.text

    def test_contains_error_sections(self):
        assert "OOMKilled" in self.text
        assert "FailedScheduling" in self.text
        assert "ImagePullBackOff" in self.text
        assert "NCCL Timeout" in self.text

    def test_contains_recovery_commands(self):
        assert "delete_training_job" in self.text
        assert "suspend_training_job" in self.text
        assert "resume_training_job" in self.text
