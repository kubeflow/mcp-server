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
import importlib

from fastmcp import FastMCP

from kubeflow_mcp.core.resources import register_resources


def _read_resource(mcp: FastMCP, uri: str) -> str:
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(mcp.read_resource(uri))
        contents = getattr(result, "contents", None)
        if contents:
            return getattr(contents[0], "content", str(result))
        return str(result)
    finally:
        loop.close()


def _make_mcp_with_trainer() -> FastMCP:
    mcp = FastMCP("test")
    trainer = importlib.import_module("kubeflow_mcp.trainer")
    register_resources(mcp, loaded_modules={"trainer": trainer})
    return mcp


class TestRegisterResources:
    def setup_method(self):
        self.mcp = _make_mcp_with_trainer()

    def test_training_patterns_registered(self):
        text = _read_resource(self.mcp, "trainer://guides/training-patterns")
        assert len(text) > 0

    def test_platform_fixes_registered(self):
        text = _read_resource(self.mcp, "trainer://guides/platform-fixes")
        assert len(text) > 0

    def test_troubleshooting_registered(self):
        text = _read_resource(self.mcp, "trainer://guides/troubleshooting")
        assert len(text) > 0

    def test_empty_modules_registers_nothing(self):
        mcp = FastMCP("test")
        register_resources(mcp, loaded_modules={})
        loop = asyncio.new_event_loop()
        try:
            resources = loop.run_until_complete(mcp._list_resources())
            assert len(resources) == 0
        finally:
            loop.close()


class TestTrainingPatterns:
    def setup_method(self):
        self.mcp = _make_mcp_with_trainer()
        self.text = _read_resource(self.mcp, "trainer://guides/training-patterns")

    def test_contains_fine_tune_example(self):
        assert "run_custom_training" in self.text or "SFTTrainer" in self.text

    def test_contains_custom_training(self):
        assert "run_custom_training" in self.text


class TestTroubleshootingGuide:
    def setup_method(self):
        self.mcp = _make_mcp_with_trainer()
        self.text = _read_resource(self.mcp, "trainer://guides/troubleshooting")

    def test_contains_diagnostic_commands(self):
        assert "get_training_job" in self.text
        assert "get_training_events" in self.text
        assert "get_training_logs" in self.text

    def test_contains_status_values(self):
        assert "Failed" in self.text
        assert "Created" in self.text


class TestPlatformFixes:
    def setup_method(self):
        self.mcp = _make_mcp_with_trainer()
        self.text = _read_resource(self.mcp, "trainer://guides/platform-fixes")

    def test_contains_volume_guidance(self):
        assert "emptyDir" in self.text or "volume" in self.text.lower()
