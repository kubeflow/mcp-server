# Contributing to Kubeflow MCP Server

Thank you for your interest in contributing! Checkout the general Kubeflow contributing guidelines [here](https://www.kubeflow.org/docs/about/contributing/).

We encourage the judicious use of AI/LLM tools; please refer to the [Kubeflow AI Policy](https://www.kubeflow.org/docs/about/ai_policy/) for more information.

## Requirements
- [Supported Python version](./pyproject.toml)
- [pre-commit](https://pre-commit.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mcp-server.git
   cd mcp-server
   ```

3. Install the [uv](https://docs.astral.sh/uv/) CLI if you do not already have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (`make install-dev` can also auto-install uv via the `make uv` target.)

4. Set up development environment:
   ```bash
   make install-dev
   ```

5. Create a branch:
   ```bash
   git checkout -b feat/your-feature
   ```

## Development

The Kubeflow MCP Server project includes a `Makefile` with several helpful commands to streamline your development workflow.

### Coding Style

Before creating git commits, ensure you have installed pre-commit hooks:

```bash
uv run pre-commit install
```

The pre-commit hooks ensure code quality and consistency (linting and formatting with `ruff`). They are also executed in CI.

To run verification checks locally (matches CI: lockfile check + `pre-commit run --all-files`):

```bash
make verify
```

## Testing

The project includes unit tests to ensure code quality and functionality.

### Unit Testing
To run unit tests locally, use the following `make` command:

```bash
make test-python
```

### Conformance Testing

We validate that the server correctly implements the MCP protocol
(`initialize`, `tools/list`, `tools/call`, etc.) using the official
[MCP conformance framework](https://github.com/modelcontextprotocol/conformance).
This runs in CI on every PR (`.github/workflows/conformance.yaml`) and needs
**no Kubernetes cluster and no LLM** — the server starts against an unreachable
cluster and its tools degrade gracefully.

To run the same suite locally (requires Node.js 20+):

```bash
make conformance
```

This starts `kubeflow-mcp serve --transport http` in the background, waits for
it to answer `initialize`, then runs the conformance `active` suite against
`http://localhost:8000/mcp`.

**Baseline file.** `tests/conformance/expected-failures.yaml` lists scenarios
that are allowed to fail — these are known spec gaps for this server (mostly
scenarios that assume the reference server's test fixtures, plus a few
unadvertised capabilities). Every scenario *not* listed must pass, and CI also
fails if a listed scenario unexpectedly starts passing. When you add protocol
features, remove the now-passing scenarios from the baseline; if you
intentionally change behavior, update the list with a one-line reason.

**DNS rebinding protection.** The HTTP/SSE transports validate the `Host` and
`Origin` headers (`kubeflow_mcp/core/transport_security.py`) so a locally-bound
server cannot be reached by a malicious web page via DNS rebinding. Protection
is on by default and allows loopback hosts; override the allowlists with
`KUBEFLOW_MCP_ALLOWED_HOSTS` / `KUBEFLOW_MCP_ALLOWED_ORIGINS` (comma-separated,
`:*` port wildcard supported) when serving on a non-loopback address, or set
`KUBEFLOW_MCP_DNS_REBINDING_PROTECTION=false` to disable it (not recommended).

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:** feat, fix, revert, chore

**Examples:**
- `feat(trainer): add create_training_job tool`
- `fix(core): handle timeout in k8s client`
- `docs: update README with usage examples`

## Pull Request Process

1. Update tests for your changes
2. Ensure all checks pass (`make verify` and `make test-python`)
3. Update documentation if needed
4. Request review from maintainers

## Areas Open for Contribution

- **OptimizerClient tools** - Hyperparameter optimization integration
- **ModelRegistryClient tools** - Model registry integration
- **Documentation** - Examples and tutorials
- **Testing** - Increase test coverage

## Code of Conduct

This project follows the [Kubeflow Code of Conduct](https://github.com/kubeflow/community/blob/master/CODE_OF_CONDUCT.md).

## Questions?

Open an issue or reach out to maintainers on the CNCF Slack `#kubeflow-ml-experience` channel.
