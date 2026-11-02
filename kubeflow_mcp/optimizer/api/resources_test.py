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

"""Optimizer MCP resources: registration and factual consistency.

The guides are read by agents, so wrong content is worse than no content —
these tests pin the claims that would silently rot as the tools change.
"""

from pathlib import Path

import pytest

import kubeflow_mcp.optimizer as optimizer_module
from kubeflow_mcp.optimizer import CLIENT_RESOURCES, TOOLS
from kubeflow_mcp.optimizer.api.optimization import ALGORITHMS

RESOURCE_DIR = Path(optimizer_module.__file__).parent


def _text(filename: str) -> str:
    return (RESOURCE_DIR / filename).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def hpo_patterns() -> str:
    return _text("resources/hpo-patterns.md")


@pytest.fixture(scope="module")
def troubleshooting() -> str:
    return _text("resources/troubleshooting.md")


# ─── registration ──────────────────────────────────────────────────────────


def test_every_declared_resource_file_exists():
    """register_resources() silently skips missing files, so assert here."""
    for uri, (filename, _desc) in CLIENT_RESOURCES.items():
        assert (RESOURCE_DIR / filename).is_file(), f"{uri} -> {filename} missing"


def test_resources_register_on_the_server():
    from kubeflow_mcp.core.resources import register_resources

    mcp = _RecordingMCP()
    complete = register_resources(mcp, {"optimizer": optimizer_module})
    assert complete is True
    assert set(mcp.registered) == set(CLIENT_RESOURCES)


def test_resource_uris_are_namespaced_and_described():
    for uri, (_filename, desc) in CLIENT_RESOURCES.items():
        assert uri.startswith("optimizer://"), uri
        assert desc.strip(), f"{uri} has no description"


class _RecordingMCP:
    """Minimal stand-in capturing what register_resources() registers."""

    def __init__(self) -> None:
        self.registered: list[str] = []

    def resource(self, uri: str):
        self.registered.append(uri)

        def decorator(fn):
            # Handler must return the file's content without touching disk again.
            assert fn().strip(), f"{uri} resolved to empty content"
            return fn

        return decorator


# ─── factual consistency with the implementation ───────────────────────────


def test_every_algorithm_is_documented(hpo_patterns):
    """The guide's algorithm table must cover exactly what the tool accepts."""
    for algorithm in ALGORITHMS:
        assert f'"{algorithm}"' in hpo_patterns, f"{algorithm} missing from hpo-patterns.md"


def test_guide_does_not_claim_algorithms_need_the_raw_spec(hpo_patterns):
    """Regression: the guide used to say tpe/cmaes/hyperband were unreachable
    from create_hpo_experiment. Building the CR directly made that false."""
    table = hpo_patterns.split("## Algorithm selection", 1)[1].split("##", 1)[0]
    assert "create_experiment_from_spec" not in table


def test_guides_only_reference_real_tools(hpo_patterns, troubleshooting):
    """Catch tool names that were renamed or never existed."""
    import re

    known = {t.__name__ for t in TOOLS} | {
        "fine_tune",
        "run_custom_training",
        "list_runtimes",
    }
    for doc, name in ((hpo_patterns, "hpo-patterns"), (troubleshooting, "troubleshooting")):
        referenced = set(re.findall(r"\b([a-z_]+)\(", doc))
        unknown = {r for r in referenced if r.endswith(("_experiment", "_trials", "_trial"))}
        assert unknown <= known, f"{name}.md references unknown tools: {unknown - known}"


def test_troubleshooting_documents_the_complete_vs_succeeded_trap(troubleshooting):
    """Trial status comes from the TrainJob, so "Succeeded" never matches."""
    assert 'status="Complete"' in troubleshooting
    assert "Succeeded" in troubleshooting


def test_troubleshooting_covers_the_not_ready_controller(troubleshooting):
    """The failure mode a broken RBAC install actually produces."""
    assert "not Ready" in troubleshooting
    assert "RBAC" in troubleshooting


def test_documented_limits_match_the_code(hpo_patterns):
    from kubeflow_mcp.optimizer.api._common import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT
    from kubeflow_mcp.optimizer.api.optimization import (
        MAX_PARALLEL_TRIAL_LIMIT,
        MAX_TRIAL_COUNT_LIMIT,
    )

    assert f"**{MAX_TRIAL_COUNT_LIMIT}**" in hpo_patterns
    assert f"**{MAX_PARALLEL_TRIAL_LIMIT}**" in hpo_patterns
    assert f"**{DEFAULT_LIST_LIMIT}**" in hpo_patterns
    assert f"**{MAX_LIST_LIMIT}**" in hpo_patterns
