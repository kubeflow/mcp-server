#!/usr/bin/env python3
# Copyright The Kubeflow Authors.
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

"""Validate that pyproject.toml and kubeflow_mcp/__init__.py declare the same version.

This is the single-source version-sync check used by both CI (release.yaml)
and local development (make check-version).  Keep all version-extraction
logic here so it never drifts between the workflow and the docs.

Usage:
    # Local check — prints versions and exits 0/1:
    python scripts/check_version.py

    # CI — also writes GITHUB_OUTPUT variables:
    python scripts/check_version.py --github-output

    # Build-job verification — asserts an expected version:
    python scripts/check_version.py --expected-version 0.2.0
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_INIT = _ROOT / "kubeflow_mcp" / "__init__.py"

# PEP 440 pre-release segments: .devN, aN, bN, rcN  (also handles non-standard '-dev')
_PRERELEASE_RE = re.compile(r"(\.dev\d*|dev\d*|a\d+|b\d+|rc\d+)", re.IGNORECASE)


def _read_pyproject_version() -> tuple[str, str]:
    """Return (package_name, version) from pyproject.toml."""
    with _PYPROJECT.open("rb") as f:
        project = tomllib.load(f)["project"]
    return project["name"], project["version"]


def _read_code_version() -> str:
    """Return __version__ from kubeflow_mcp/__init__.py using AST parsing."""
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise SystemExit("__version__ not found in kubeflow_mcp/__init__.py")


def is_prerelease(version: str) -> bool:
    """Check whether *version* contains a PEP 440 pre-release segment."""
    return bool(_PRERELEASE_RE.search(version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Assert the codebase version matches this value (used in CI build verification).",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write workflow output variables to $GITHUB_OUTPUT.",
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help="GitHub release tag to validate against the project version.",
    )
    args = parser.parse_args()

    package_name, project_version = _read_pyproject_version()
    code_version = _read_code_version()

    print(f"Package name:     {package_name}")
    print(f"pyproject.toml:   {project_version}")
    print(f"__init__.py:      {code_version}")

    if project_version != code_version:
        print(
            "ERROR: Version mismatch between pyproject.toml and kubeflow_mcp/__init__.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.expected_version and args.expected_version != project_version:
        print(
            f"ERROR: Expected version {args.expected_version!r} but found {project_version!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.release_tag and args.release_tag != project_version:
        print(
            f"ERROR: Release tag {args.release_tag!r} does not match project version {project_version!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    prerelease = is_prerelease(project_version)
    print(f"Pre-release:      {prerelease}")
    print("Version OK ✓")

    if args.github_output:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if not output_file:
            print("WARNING: $GITHUB_OUTPUT not set, skipping output", file=sys.stderr)
            return
        with Path(output_file).open("a", encoding="utf-8") as fh:
            for key, value in {
                "artifact-name": f"dist-{project_version}",
                "is-prerelease": str(prerelease).lower(),
                "package-name": package_name,
                "version": project_version,
            }.items():
                print(f"{key}={value}", file=fh)


if __name__ == "__main__":
    main()
