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

"""Unit tests for core/resources.py - MCP resource registration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from kubeflow_mcp.core.resources import register_resources


def _fake_mcp() -> tuple[MagicMock, dict]:
    """Return a mock FastMCP that records each registered handler by URI."""
    handlers: dict = {}

    def resource(uri):
        def decorator(fn):
            handlers[uri] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.resource.side_effect = resource
    return mcp, handlers


def _fake_module(tmp_path, resources: dict) -> SimpleNamespace:
    return SimpleNamespace(__file__=str(tmp_path / "__init__.py"), CLIENT_RESOURCES=resources)


def test_registered_resource_returns_file_content(tmp_path):
    (tmp_path / "guide.md").write_text("# Guide\nsteps", encoding="utf-8")
    module = _fake_module(tmp_path, {"test://guides/my-guide": ("guide.md", "A guide")})
    mcp, handlers = _fake_mcp()

    assert register_resources(mcp, {"test": module}) is True

    handler = handlers["test://guides/my-guide"]
    assert handler() == "# Guide\nsteps"
    assert handler.__doc__ == "A guide"
    assert handler.__name__ == "my_guide"


def test_missing_resource_file_returns_false(tmp_path):
    module = _fake_module(tmp_path, {"test://guides/missing": ("missing.md", "Missing")})
    mcp, handlers = _fake_mcp()

    assert register_resources(mcp, {"test": module}) is False
    assert handlers == {}


def test_module_without_resources_is_skipped(tmp_path):
    mcp, handlers = _fake_mcp()

    assert register_resources(mcp, {"stub": SimpleNamespace()}) is True
    assert handlers == {}
