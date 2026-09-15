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

"""Wheel content test.

Builds the wheel in-process with hatchling and asserts that the top-level
``skills/`` directory (Agent Skills, SEP-2640) is shipped under
``kubeflow_mcp/skills/`` so PyPI and container installations can discover
it. Guards the ``force-include`` rule in ``pyproject.toml``.
"""

import os
import zipfile
from pathlib import Path

import pytest

hatchling_build = pytest.importorskip("hatchling.build")

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
WHEEL_SKILLS_PREFIX = "kubeflow_mcp/skills/"


def _source_skill_files() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.rglob("*") if p.is_file())


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("wheel")
    # hatchling resolves the project from the current working directory.
    previous_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        wheel_name = hatchling_build.build_wheel(str(out_dir))
    finally:
        os.chdir(previous_cwd)
    return out_dir / wheel_name


@pytest.fixture(scope="module")
def wheel_entries(wheel_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel_path) as wheel:
        return {name: wheel.read(name) for name in wheel.namelist()}


def test_source_tree_has_skill_files():
    files = _source_skill_files()
    assert files, f"no skill files found under {SKILLS_DIR}"
    assert SKILLS_DIR / "kubeflow-training" / "SKILL.md" in files


def test_wheel_ships_every_skill_file(wheel_entries: dict[str, bytes]):
    for source in _source_skill_files():
        relative = source.relative_to(SKILLS_DIR).as_posix()
        wheel_name = WHEEL_SKILLS_PREFIX + relative
        assert wheel_name in wheel_entries, f"{wheel_name} missing from wheel"
        assert wheel_entries[wheel_name] == source.read_bytes(), (
            f"{wheel_name} content differs from {source}"
        )


def test_wheel_does_not_leak_skills_at_top_level(wheel_entries: dict[str, bytes]):
    leaked = [name for name in wheel_entries if name.startswith("skills/")]
    assert not leaked, f"skills must live under {WHEEL_SKILLS_PREFIX}, found: {leaked}"
