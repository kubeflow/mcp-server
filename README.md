# Kubeflow MCP Server

AI-powered interface for Kubeflow Training via [Model Context Protocol](https://modelcontextprotocol.io/).

Proposal: https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server

> ⚠️ **Note:** This project is in early development. We currently accept PRs only after prior discussion on Slack — join `#kubeflow-ml-experience` on the [CNCF Slack](https://www.kubeflow.org/docs/about/community/). For more discussion, join on bi-weekly ML Experience WG call on Wednesdays.

## Overview

This MCP server enables LLM agents (Claude, Cursor, etc.) to interact with Kubeflow Training through natural language. It wraps the [Kubeflow SDK](https://github.com/kubeflow/sdk) with MCP tools for fine-tuning, training job management, and monitoring.

## Compatibility

| MCP Server | Kubeflow SDK | Python      | Kubernetes |
|------------|-------------|-------------|------------|
| 0.1.x      | ≥ 0.4.0     | 3.10 – 3.12 | ≥ 1.27     |

## Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | 🚧 In Progress |
| TrainerClient Tools | 🚧 In Progress |
| OptimizerClient Tools | ⬜ Planned (Contributors Welcome) |
| ModelRegistryClient Tools | ⬜ Planned (Contributors Welcome) |
| PipelinesClient Tools | ⬜ Planned (Contributors Welcome) |
| SparkClient Tools | ⬜ Planned (Contributors Welcome) |
| FeastClient Tools | ⬜ Planned (Contributors Welcome) |

## Quick Start

```bash
# Install (trainer + optimizer included by default)
pip install kubeflow-mcp

# Install with hub or spark extras
pip install kubeflow-mcp[hub]
pip install kubeflow-mcp[spark]

# Run
kubeflow-mcp serve --clients trainer
```

## Development

The project uses `uv` and a `Makefile` to manage the development environment.

```bash
# Setup development environment
make install-dev

# Run verification (lint, format)
make verify

# Run unit tests
make test-python
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - See [LICENSE](LICENSE)
