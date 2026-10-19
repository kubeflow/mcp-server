# Releasing kubeflow-mcp

The release workflow publishes the Python package to PyPI, the container image to GHCR, and a GitHub Release. A release starts when a version bump is merged to `main` or a `release-X.Y` branch.

## Prerequisites

- Write access to the repository.
- Docker or Podman for changelog generation. Docker is the default; use `CONTAINER_RUNTIME=podman` when Podman is installed.
- (Optional, recommended) A GitHub token exported as `GITHUB_TOKEN` to enrich changelog entries with pull-request and author metadata.

When no GitHub token is available, or for local dry-runs from commits not yet pushed to GitHub, add `OFFLINE=1` to skip GitHub metadata enrichment.

## Version and branch rules

- Stable releases use `X.Y.Z`; release candidates use `X.Y.ZrcN`.
- The version is defined in `kubeflow_mcp/__init__.py`; Hatch reads it dynamically.
- Release branches use the `release-X.Y` format.
- Release the current minor version from `main`; release later patches from the matching `release-X.Y` branch.
- Stable tags use the bare version, for example `0.1.2`.

## Create a release PR

If you have a GitHub token, export it before creating the release PR to enrich the changelog:

```bash
export GITHUB_TOKEN
make release VERSION=X.Y.Z
```

If generating the release without a token or offline, pass `OFFLINE=1`:

```bash
make release VERSION=X.Y.Z OFFLINE=1
```

This updates the package version, `server.json`, and the changelog. Review the changes, then open a signed PR against `main` or the appropriate release branch.

Use the same command with an `rcN` version for a release candidate; release candidates skip stable changelog generation.

To generate or preview only the changelog:

```bash
make changelog VERSION=X.Y.Z
make changelog VERSION=X.Y.Z DRY_RUN=1
make changelog VERSION=X.Y.Z DRY_RUN=1 CONTAINER_RUNTIME=podman OFFLINE=1
```

## Publish the release

After the version PR merges, GitHub Actions:

1. Creates or updates the release branch.
2. Verifies, tests, and packages the project.
3. Creates the version tag.
4. Publishes the package to PyPI and the image to GHCR.
5. Creates the GitHub Release.

PyPI and GitHub Release publication require approval in the release environment.

Verify the published package and CLI:

```bash
pip install kubeflow-mcp==X.Y.Z
kubeflow-mcp --version
```

Also confirm that the package, container image, and GitHub Release have the expected version.

## Re-run or recover a release

If a release workflow fails or requires manual recovery:

### Failures before publication (build, test, or tag creation)

If the workflow fails before publishing to PyPI:

1. Navigate to **Actions** → **Release** workflow.
2. Click **Run workflow**.
3. Select `main` (for minor releases) or the target `release-X.Y` branch (for patch releases).
4. Click **Run workflow** to start the release.

Alternatively, you can use **Re-run all jobs** on the failed run.

### Failures after partial publication (e.g. PyPI succeeded, but Docker or GitHub Release failed)

If PyPI publication has already succeeded, do **not** dispatch a fresh workflow run, because PyPI rejects re-uploading an existing distribution version:

1. Open the failed release workflow run in **Actions**.
2. Click **Re-run failed jobs** (instead of re-running all jobs or triggering a new workflow dispatch).
3. This retries only the incomplete jobs (such as Docker publish or GitHub Release) without attempting to re-upload to PyPI.

### Requirements for manual dispatch

- Manual dispatch must be started from `main` or the relevant `release-X.Y` branch. The workflow explicitly rejects runs dispatched from any other branch.
- The selected branch must already contain a valid release version (`X.Y.Z` or `X.Y.ZrcN`) in `kubeflow_mcp/__init__.py`.
