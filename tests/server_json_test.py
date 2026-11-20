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

"""MCP Registry metadata checks for ``server.json``.

``mcp-publisher publish`` validates ``server.json`` against the registry schema
and rejects the whole submission on the first violation, so these checks catch
problems in CI instead of at publish time. The limits mirror
https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json and
are hard-coded to keep the test offline.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from kubeflow_mcp import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent

# ServerDetail constraints from the registry schema.
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$")
MAX_TEXT_LENGTH = 100


@pytest.fixture(scope="module")
def server_json() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))


def test_required_fields_present(server_json: dict[str, Any]) -> None:
    for field in ("name", "description", "version"):
        assert server_json.get(field), f"server.json is missing required field {field!r}"


def test_name_matches_registry_pattern(server_json: dict[str, Any]) -> None:
    assert NAME_PATTERN.match(server_json["name"])


@pytest.mark.parametrize("field", ["description", "title"])
def test_text_fields_within_registry_limit(server_json: dict[str, Any], field: str) -> None:
    value = server_json[field]
    assert 1 <= len(value) <= MAX_TEXT_LENGTH, (
        f"server.json {field} is {len(value)} characters; the MCP Registry allows "
        f"at most {MAX_TEXT_LENGTH}"
    )


def test_versions_match_package_version(server_json: dict[str, Any]) -> None:
    assert server_json["version"] == __version__
    for package in server_json["packages"]:
        assert package["version"] == __version__, package["identifier"]


def test_readme_declares_registry_name(server_json: dict[str, Any]) -> None:
    # The registry proves PyPI ownership by finding this marker in the package
    # long description, which is README.md.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {server_json['name']}" in readme
