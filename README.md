# Kubeflow MCP Server

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org) [![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)

Proposal: [KEP-936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server) · [ROADMAP](ROADMAP.md) · [SECURITY](SECURITY.md) · [CONTRIBUTING](CONTRIBUTING.md)

## Overview

The Kubeflow MCP Server exposes Kubeflow Training operations as [Model Context Protocol](https://modelcontextprotocol.io/) tools, enabling AI agents (Claude, Cursor, Claude Code, or any custom agents etc.) to plan, submit, monitor, and manage training jobs through natural language — without users needing to learn Kubernetes or the Kubeflow SDK directly.

### Benefits

- **Agent-Native**: Tools auto-discovered via MCP — no manual API wiring
- **Guided Workflow**: Phase ordering with next-step hints (Plan → Discover → Train → Monitor)
- **Preview-Before-Submit**: Every mutating operation requires explicit confirmation
- **Security-First**: Persona gating, namespace enforcement, input validation, bearer/JWT auth
- **Multi-Platform**: Auto-detects OpenShift, EKS, GKE with platform-specific guidance
- **Token-Efficient**: Progressive/semantic modes compress 23 tools into 2-3 meta-tools
- **Extensible**: Plugin architecture for additional Kubeflow clients (TODO: optimizer, hub)

## Get Started

### Install from source

```bash
git clone https://github.com/kubeflow/mcp-server.git
cd mcp-server
pip install .
```

### Run the server

```bash
kubeflow-mcp serve
```

> Once published to PyPI, install with `pip install kubeflow-mcp`.

### Example: Fine-tune a model via AI agent

Once connected, your AI agent can run a complete training workflow through natural language:

```
User: "Fine-tune gemma-2b on the alpaca dataset"

Agent calls: check_compatibility()        → ✅ K8s 1.29, Trainer CRD installed
Agent calls: get_cluster_resources()      → 4x A100 GPUs available
Agent calls: estimate_resources("google/gemma-2b") → needs ~16GB GPU, 1x A100
Agent calls: list_runtimes()              → torchtune-llama, torchtune-gemma, ...
Agent calls: fine_tune(                   → preview config (confirmed=False)
    model="hf://google/gemma-2b",
    dataset="hf://tatsu-lab/alpaca",
    runtime="torchtune-gemma-2b"
)
Agent calls: fine_tune(..., confirmed=True) → TrainJob "train-gemma-abc" created
Agent calls: get_training_logs("train-gemma-abc") → training progress...
```

Every mutating tool requires `confirmed=True` — agents always preview before submitting.

### MCP Client Config


<details>
<summary>Cursor</summary>

Add to `.cursor/mcp.json` (or use the `.mcp.json` at the repo root for local dev):

```json
{
  "mcpServers": {
    "kubeflow": {
      "command": "uv",
      "args": ["run", "kubeflow-mcp", "serve"]
    }
  }
}
```

</details>

<details>
<summary>Claude Code</summary>

```bash
claude mcp add kubeflow -- kubeflow-mcp serve
```

</details>

## Tools

23 tools organized by workflow phase:

| Phase | Tools | Description |
|-------|-------|-------------|
| Planning | `pre_flight`, `check_compatibility`, `get_cluster_resources`, `estimate_resources` | Environment validation and resource estimation |
| Discovery | `list_training_jobs`, `get_training_job`, `list_runtimes`, `get_runtime` | Browse jobs and available runtimes |
| Training | `fine_tune`, `run_custom_training`, `run_container_training` | Submit LoRA/QLoRA fine-tuning, custom scripts, or container jobs |
| Monitoring | `get_training_logs`, `get_training_events`, `wait_for_training` | Track progress, debug failures |
| Lifecycle | `delete_training_job`, `update_training_job` | Manage existing jobs (ownership-guarded) |
| Platform | `inspect_crd`, `inspect_controller`, `patch_runtime`, `create_runtime`, `delete_runtime` | Cluster inspection and runtime management |
| Health | `health_check`, `get_server_logs` | Server diagnostics |


### Requirements

| MCP Server | Kubeflow Trainer | Kubeflow SDK | Python      | Kubernetes |
|------------|------------------|--------------|-------------|------------|
| 0.1.x      | >= 2.2.0         | >= 0.4.0     | 3.10 - 3.12 | >= 1.27    |

## CLI Reference

### `kubeflow-mcp serve`

```bash
kubeflow-mcp serve \
  --clients trainer \             # modules: trainer, optimizer (stub), hub (stub)
  --persona ml-engineer \         # readonly | data-scientist | ml-engineer | platform-admin
  --mode full \                   # full | progressive | semantic
  --instruction-tier full \       # full | compact | minimal
  --transport stdio \             # stdio | http | sse
  --auth-token SECRET \           # bearer token for HTTP auth (dev/staging)
  --otel-endpoint URL \           # OTLP HTTP endpoint (optional tracing)
  --log-level INFO \              # DEBUG | INFO | WARNING | ERROR
  --log-format console \          # console | json (auto-detected if omitted)
  --no-banner                     # suppress startup banner
```

`--mode progressive` exposes 3 meta-tools (~85 tokens) for hierarchical discovery. `--mode semantic` exposes 2 meta-tools (~69 tokens) using embedding search. Both reduce token consumption significantly for agent workflows.

<details>
<summary> HTTP Authentication</summary>

When using `--transport http`, configure auth to secure the endpoint:

```bash
# Simple API key (dev/staging)
kubeflow-mcp serve --transport http --auth-token my-secret-token

# Or via env var
export KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token
kubeflow-mcp serve --transport http

# JWT verification (production)
export KUBEFLOW_MCP_JWKS_URI=https://auth.example.com/.well-known/jwks.json
export KUBEFLOW_MCP_JWT_ISSUER=https://auth.example.com
export KUBEFLOW_MCP_JWT_AUDIENCE=kubeflow-mcp
kubeflow-mcp serve --transport http
```

Without auth configured, the server logs a warning that the HTTP endpoint is open.

</details>

<details>
<summary>Agent Subcommand</summary>

```bash
kubeflow-mcp agent \
  --backend ollama \              # ollama (default; more backends planned)
  --model qwen3:8b \              # model name for the backend
  --mode full \                   # full | progressive | semantic
  --thinking                      # enable thinking output (supported models)
```

</details>

## Observability

OpenTelemetry tracing is optional and can be enabled without changing tool code.

- Install optional dependencies: `pip install ".[otel]"`
- Enable tracing with CLI flag or env var:

```bash
kubeflow-mcp serve --otel-endpoint http://localhost:4318/v1/traces
# or
export KUBEFLOW_MCP_OTEL_ENDPOINT=http://localhost:4318/v1/traces
kubeflow-mcp serve
```

Each tool invocation emits a span with attributes:
`tool.name`, `tool.success`, `tool.duration_ms`, `kubeflow.persona`, and `correlation_id`.

## Development

```bash
make install-dev                  # setup environment
make verify                       # lint + format check
make test-python                  # run tests
make inspector                    # launch MCP Inspector (stdio)
make inspector TRANSPORT=http     # Inspector + Streamable HTTP (start server separately)
make inspector TRANSPORT=sse      # Inspector + SSE (start server separately)
```

## Community

- **Slack**: Join [#kubeflow-ml-experience](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) on CNCF Slack
- **Meetings**: Attend the [Kubeflow SDK and ML Experience](https://bit.ly/kf-ml-experience) bi-weekly call
- **GitHub**: Issues and contributions at [kubeflow/mcp-server](https://github.com/kubeflow/mcp-server)

## Documentation


- **[CONTRIBUTING](CONTRIBUTING.md)**: Development workflow and PR guidelines
- **[KEP-936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server)**: Design proposal

## License

Apache License 2.0 — see [LICENSE](LICENSE).
