# Releasing kubeflow-mcp

The release workflow publishes the Python package to PyPI, the container image to GHCR, and a GitHub Release. A release starts when a version bump is merged to `main` or a `release-X.Y` branch.

## Prerequisites

- Write access to the repository.
- Docker or Podman for changelog generation. Docker is the default; use `CONTAINER_RUNTIME=podman` when Podman is installed.
- A GitHub token exported as `GITHUB_TOKEN` for pull-request and author metadata.

For local dry-runs from commits that are not pushed to GitHub, add `OFFLINE=1` to skip GitHub metadata enrichment.

## Version and branch rules

- Stable releases use `X.Y.Z`; release candidates use `X.Y.ZrcN`.
- The version is defined in `kubeflow_mcp/__init__.py`; Hatch reads it dynamically.
- Release branches use the `release-X.Y` format.
- Release the current minor version from `main`; release later patches from the matching `release-X.Y` branch.
- Stable tags use the bare version, for example `0.1.2`.

## Create a release PR

Export the token and run:

```bash
export GITHUB_TOKEN
make release VERSION=X.Y.Z
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
