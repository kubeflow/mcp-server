# Releasing kubeflow-mcp

This repository publishes the `kubeflow-mcp` Python package to PyPI using
GitHub Actions and PyPI trusted publishing. PyPI API tokens are not stored in
GitHub secrets.

## Versioning

Kubeflow MCP version format follows Python's [PEP 440](https://peps.python.org/pep-0440/).
Versions are in the format of `X.Y.Z`, where `X` is the major version, `Y` is
the minor version, and `Z` is the patch version.

Pre-releases use the format `X.Y.ZrcN` where `N` is the release candidate
number (following [Kubeflow SDK conventions](https://github.com/kubeflow/sdk/blob/main/RELEASE.md)).

Update both version sources in the same pull request:

- `pyproject.toml`: `[project].version`
- `kubeflow_mcp/__init__.py`: `__version__`

The workflow fails the build if those values differ.

Pre-release strings such as `dev` or `rc` are allowed for Test PyPI dry runs,
but they are blocked from production PyPI publish by default.

If a GitHub Release is published for a prerelease version, the workflow keeps
the GitHub Release marked as prerelease and skips production PyPI publish.

## Release Paths

The workflow has two supported paths:

1. Manual Test PyPI dry run.

   Run the `Release` workflow with `workflow_dispatch` on the ref you want to
   test. You must enter the expected version string — the workflow validates
   it matches both `pyproject.toml` and `kubeflow_mcp/__init__.py`.

2. Published GitHub Release.

   Publish a GitHub Release with the final release tag. The workflow builds the
   tagged source, publishes a stable build to PyPI, and attaches the
   distribution artifacts to the GitHub Release.

## Pre-production Checklist

Use this flow for RC or dev validation before a production release.

1. Bump the version in both files.

   Keep `pyproject.toml` and `kubeflow_mcp/__init__.py` identical.

2. Run the local checks.

   Use the commands in the next section to verify version sync, unit tests, and
   package build health.

3. Open GitHub Actions -> `Release` -> `Run workflow`.

   Select the branch or tag that contains the version you want to test.
   Enter the expected version string when prompted.

4. Let the workflow publish to Test PyPI.

   The workflow uploads the build artifact to Test PyPI through OIDC trusted
   publishing. No production PyPI publish happens on this path.

5. Verify the package from Test PyPI.

   ```sh
   pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     kubeflow-mcp==X.Y.ZrcN
   kubeflow-mcp --version
   ```

## Production Checklist

Use this flow only for a final `X.Y.Z` release.

1. Bump the version in both files to the same stable version.

   Production releases should not contain `dev` or `rc` in the version string.

2. Run the local checks.

   Confirm the version matches in both files and the package builds cleanly.

3. Publish a GitHub Release with the exact version tag.

   The workflow runs on `release: published`, builds the tagged source, and
   publishes to PyPI using trusted publishing.

4. Approve the `release` environment gate if your repository requires it.

   This is the production OIDC trusted publishing path for PyPI.

5. Verify the release.

   ```sh
   pip install kubeflow-mcp==X.Y.Z
   kubeflow-mcp --version
   ```

## Local Testing

Run these checks from the repository root before publishing.

1. Confirm the version is synchronized.

   ```sh
   python scripts/check_version.py
   ```

2. Run lint, formatting, and unit tests.

   ```sh
   make verify
   make test-python
   ```

3. Build and validate the distribution artifacts.

   ```sh
   rm -rf dist/
   uv build
   uvx twine check dist/*
   ```

4. Inspect the built wheel in a clean virtual environment.

   ```sh
   python -m venv /tmp/kubeflow-mcp-release-test
   /tmp/kubeflow-mcp-release-test/bin/python -m pip install --upgrade pip
   /tmp/kubeflow-mcp-release-test/bin/python -m pip install dist/*.whl
   /tmp/kubeflow-mcp-release-test/bin/kubeflow-mcp --version
   /tmp/kubeflow-mcp-release-test/bin/python -c "import kubeflow_mcp; print(kubeflow_mcp.__version__)"
   ```

5. Clean up the temporary environment.

   ```sh
   rm -rf /tmp/kubeflow-mcp-release-test
   ```

Trusted publishing cannot be fully tested locally because it depends on GitHub
Actions OIDC tokens and the configured GitHub environments. Use the Test PyPI
dry run for end-to-end publish validation.

## Trusted Publisher Setup

Configure trusted publishing for both PyPI and Test PyPI. The workflow uses
`id-token: write` permissions and does not read a PyPI token from secrets.

For PyPI, add a publisher for the `kubeflow-mcp` project with these values:

| Field | Value |
| --- | --- |
| Owner | `kubeflow` |
| Repository name | `mcp-server` |
| Workflow name | `release.yaml` |
| Environment name | `release` |

For Test PyPI, add a publisher for the `kubeflow-mcp` project with these values:

| Field | Value |
| --- | --- |
| Owner | `kubeflow` |
| Repository name | `mcp-server` |
| Workflow name | `release.yaml` |
| Environment name | `test-pypi` |

Create matching GitHub environments named `release` and `test-pypi`. Configure
required reviewers on those environments to require manual approval before
publishing.
