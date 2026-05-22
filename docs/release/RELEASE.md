# Releasing kubeflow-mcp

This repository publishes the `kubeflow-mcp` Python package to PyPI using
GitHub Actions and PyPI trusted publishing. PyPI API tokens are not stored in
GitHub secrets.

## Versioning

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
   test. The workflow builds the package and publishes it to Test PyPI only.

2. Published GitHub Release.

   Publish a GitHub Release with the final release tag. The workflow builds the
   tagged source, publishes a stable build to PyPI, and creates or updates the
   GitHub Release asset set.

## Pre-production Checklist

Use this flow for RC or dev validation before a production release.

1. Bump the version in both files.

   Keep `pyproject.toml` and `kubeflow_mcp/__init__.py` identical.

2. Run the local checks.

   Use the commands in the next section to verify version sync, unit tests, and
   package build health.

3. Open GitHub Actions -> `Release` -> `Run workflow`.

   Select the branch or tag that contains the version you want to test.

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
   python - <<'PY'
   import ast
   import tomllib

   with open("pyproject.toml", "rb") as f:
       project_version = tomllib.load(f)["project"]["version"]

   module = ast.parse(open("kubeflow_mcp/__init__.py", encoding="utf-8").read())
   code_version = None
   for node in module.body:
       if (
           isinstance(node, ast.Assign)
           and len(node.targets) == 1
           and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id == "__version__"
           and isinstance(node.value, ast.Constant)
           and isinstance(node.value.value, str)
       ):
           code_version = node.value.value
           break

   if code_version is None:
       raise SystemExit("__version__ not found in kubeflow_mcp/__init__.py")

   print(f"pyproject.toml version: {project_version}")
   print(f"kubeflow_mcp version:   {code_version}")

   if project_version != code_version:
       raise SystemExit("version mismatch")
   PY
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
