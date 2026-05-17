# Releasing kubeflow-mcp

This repository publishes the `kubeflow-mcp` Python package to PyPI using
GitHub Actions and PyPI trusted publishing. PyPI API tokens are not stored in
GitHub secrets.

## Versioning

`kubeflow-mcp` follows Python's [PEP 440](https://peps.python.org/pep-0440/)
release version format:

- Final releases use `X.Y.Z`, for example `0.1.0`.
- Release candidates use `X.Y.ZrcN`, for example `0.1.0rc1`.

The version must be updated in both files in the same pull request:

- `pyproject.toml`: `[project] version`
- `kubeflow_mcp/__init__.py`: `__version__`

Development versions such as `0.1.0-dev` are not publishable by the release
workflow.

## Release Branches and Tags

Release tags match the package version exactly:

- `0.1.0`
- `0.1.0rc1`

Do not prefix release tags with `v`.

Release branches use `release-X.Y`, where `X.Y` is the release minor version.
For example, `0.1.0` and `0.1.1` are released from `release-0.1`.

Do not bump release versions directly on `main`. Create a normal PR branch,
merge it to `main`, and let the release workflow create or update the matching
`release-X.Y` branch.

## Changelog

This repository does not currently keep a committed `CHANGELOG.md` or
`CHANGELOG/` directory on `main`. The release workflow creates GitHub Releases
with generated release notes. If a changelog file or directory is added later,
update it in the same PR as the version bump and update this guide with the new
convention.

## Automated Release Workflow

The `Release` GitHub Actions workflow supports three paths:

1. Version PR merged to `main`.

   The workflow detects changes to version files, creates or updates
   `release-X.Y`, builds from that release branch, creates the `X.Y.Z` tag,
   publishes to PyPI after approval, and then creates the GitHub Release.

2. GitHub Release published manually.

   The workflow builds the tag and publishes to PyPI after approval. This path
   is kept for compatibility, but the preferred flow is the version PR path.

3. Manual Test PyPI dry run.

   The workflow builds the selected branch or tag and publishes to Test PyPI
   when `test_pypi` is enabled.

## Release Process

1. Create a release preparation branch from `main`.

   ```sh
   git checkout main
   git pull
   git checkout -b chore/release-X.Y.Z
   ```

2. Update the version.

   Change both `pyproject.toml` and `kubeflow_mcp/__init__.py` to the same
   release version.

3. Run the local release checks.

   Follow the local testing section below before opening the PR.

4. Open a PR to `main` and get it reviewed.

   The PR should contain the version bump and any release note or changelog
   updates that apply to the release.

5. Merge the PR to `main`.

   The `Release` workflow starts automatically. It creates or updates the
   matching `release-X.Y` branch from the merged commit.

6. Approve the `release` environment gate for PyPI publishing.

   After approval, the workflow publishes the package to PyPI using trusted
   publishing.

7. Approve the `release` environment gate for GitHub Release creation if the
   environment requires a second approval.

   The workflow creates a GitHub Release for the package version and attaches
   the built artifacts.

8. Verify the release.

   ```sh
   pip install kubeflow-mcp==X.Y.Z
   kubeflow-mcp --version
   ```

## Local Testing

Run these checks from the repository root before creating the version bump PR.
They mirror the build job in `.github/workflows/release.yaml`.

1. Confirm the release version is valid and synchronized.

   ```sh
   python - <<'PY'
   import ast
   import re
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

   if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?", project_version):
       raise SystemExit("version must be X.Y.Z or X.Y.ZrcN")
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

4. Install the built wheel into a clean virtual environment.

   ```sh
   python -m venv /tmp/kubeflow-mcp-release-test
   /tmp/kubeflow-mcp-release-test/bin/python -m pip install --upgrade pip
   /tmp/kubeflow-mcp-release-test/bin/python -m pip install dist/*.whl
   /tmp/kubeflow-mcp-release-test/bin/kubeflow-mcp --version
   /tmp/kubeflow-mcp-release-test/bin/python -c "import kubeflow_mcp; print(kubeflow_mcp.__version__)"
   ```

5. Inspect package metadata and dependency consistency.

   ```sh
   /tmp/kubeflow-mcp-release-test/bin/python -m pip show kubeflow-mcp
   /tmp/kubeflow-mcp-release-test/bin/python -m pip check
   ```

6. Clean up the temporary environment.

   ```sh
   rm -rf /tmp/kubeflow-mcp-release-test
   ```

Trusted publishing cannot be fully tested locally because it depends on GitHub
Actions OIDC tokens and the configured GitHub environment. Use the Test PyPI dry
run for end-to-end publish validation.

## Test PyPI Dry Run

Use Test PyPI before publishing when validating a release candidate or checking
the publishing path.

1. Go to GitHub Actions -> `Release` -> `Run workflow`.
2. Select the branch or tag that contains the version to test.
3. Set `test_pypi` to `true`.
4. Approve the `test-pypi` environment gate.
5. Verify the package from Test PyPI.

   ```sh
   pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     kubeflow-mcp==X.Y.ZrcN
   kubeflow-mcp --version
   ```

Each Test PyPI upload must use a version that has not already been uploaded to
Test PyPI.

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
