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

"""Built-artifact content tests for the Agent Skills package.

The skill files live at the repo top level (``skills/``), outside the
``kubeflow_mcp`` package, so nothing but the Hatch ``force-include`` rule
puts them in the wheel. These tests inspect the built artifact rather than
the source checkout:

- the wheel is built in-process with hatchling and must contain
  ``SKILL.md`` and ``mcp.json`` under ``kubeflow_mcp/skills/``;
- the container image installs that same wheel from the files the
  Dockerfile copies into the build stage, so every ``force-include``
  source must be copied before the project is installed.

hatchling is a hard requirement here (dev dependency group): skipping
would let a packaging regression pass CI.
"""

import os
import re
import zipfile
from pathlib import Path

import pytest
from hatchling import build as hatchling_build

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
WHEEL_SKILLS_PREFIX = "kubeflow_mcp/skills/"
REQUIRED_SKILL_FILES = ("kubeflow-training/SKILL.md", "kubeflow-training/mcp.json")


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


@pytest.mark.parametrize("relative", REQUIRED_SKILL_FILES)
def test_wheel_contains_required_skill_file(wheel_entries: dict[str, bytes], relative: str):
    wheel_name = WHEEL_SKILLS_PREFIX + relative
    assert wheel_name in wheel_entries, f"{wheel_name} missing from built wheel"
    assert wheel_entries[wheel_name], f"{wheel_name} is empty in built wheel"
    assert wheel_entries[wheel_name] == (SKILLS_DIR / relative).read_bytes()


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


def _force_include_sources() -> list[str]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"])


def test_force_include_maps_skills_into_package():
    assert "skills" in _force_include_sources()


def test_container_build_stage_copies_force_included_sources():
    """The image installs the wheel built from what the Dockerfile copies in."""
    lines = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    stage_starts = [i for i, line in enumerate(lines) if line.startswith("FROM ")]
    assert len(stage_starts) >= 2, "expected a multi-stage Dockerfile (builder + runtime)"
    builder = [line.strip() for line in lines[stage_starts[0] : stage_starts[1]]]

    install_steps = [
        i
        for i, line in enumerate(builder)
        if line.startswith("RUN uv sync") and "--no-install-project" not in line
    ]
    assert install_steps, "builder stage no longer installs the project with uv sync"
    before_install = builder[: install_steps[0]]

    for source in _force_include_sources():
        pattern = re.compile(rf"^COPY\s+{re.escape(source)}\s+\./{re.escape(source)}$")
        assert any(pattern.match(line) for line in before_install), (
            f"Dockerfile must COPY {source} into the builder stage before installing "
            "the project, otherwise the image ships without it"
        )
