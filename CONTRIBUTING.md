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

3. Set up development environment:
   ```bash
   make install-dev
   ```

4. Create a branch:
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

To run verification checks locally:

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

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:** feat, fix, revert, chore, docs, proposal

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
