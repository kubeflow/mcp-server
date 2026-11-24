# Kubeflow MCP Server

<!-- mcp-name: io.github.kubeflow/mcp-server -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org) [![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) [![Coverage Status](https://coveralls.io/repos/github/kubeflow/mcp-server/badge.svg?branch=main)](https://coveralls.io/github/kubeflow/mcp-server?branch=main) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/mcp-server)

[Quickstart](docs/quickstart.md) · [Architecture](ARCHITECTURE.md) · [Blog](https://blog.kubeflow.org/introducing-kubeflow-mcp/) · [KEP-936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server) · [Roadmap](ROADMAP.md) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## Overview

The Kubeflow MCP Server exposes Kubeflow operations as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tools. AI agents (Claude, Cursor, Claude Code, or custom agents) can plan, submit, monitor, and manage ML workloads in natural language — without requiring users to write Kubernetes YAML or Kubeflow SDK boilerplate.

### Why Kubeflow MCP Server?

- **Agent-Native**: Tools are auto-discovered via MCP with guided workflow phases (`Plan → Discover → Run → Monitor`)
- **Preview-Before-Submit**: Every mutating operation previews its action first and requires explicit `confirmed=True`
- **Security-First**: Persona gating (`readonly` by default), namespace allowlists, input validation, and Bearer/JWT HTTP auth
- **Multi-Platform**: Auto-detects OpenShift, EKS, and GKE with platform-specific guidance
- **Token-Efficient**: Progressive and semantic tool modes compress tools into 2–3 meta-tools
- **Modular Clients**: Load only what you need (`trainer`, `spark`, `pipelines`, `optimizer`; planned: `hub`)

## Demo

[![Kubeflow MCP Server](https://img.youtube.com/vi/cZ2BP5hQjc8/0.jpg)](https://youtu.be/cZ2BP5hQjc8)

## Quick Start

> New to Kubeflow MCP? Follow the **[Quickstart & Troubleshooting Guide](docs/quickstart.md)** for step-by-step cluster verification, RBAC checks, and common fixes.

### Prerequisites

- **Python** 3.10, 3.11, or 3.12
- **Kubernetes cluster** with `kubectl` and a configured `kubeconfig` (see [RBAC Configuration](ARCHITECTURE.md#rbac-configuration))
- **Kubeflow component(s)** installed for the clients you enable: Kubeflow Trainer, Spark Operator, and/or Kubeflow Pipelines

### Install & Run

```bash
# Default (Trainer client)
pip install kubeflow-mcp

# Optional client extras
pip install "kubeflow-mcp[pipelines]"   # Kubeflow Pipelines (kfp)
pip install "kubeflow-mcp[spark]"       # Kubeflow Spark

# Start the MCP server (stdio transport by default)
kubeflow-mcp serve
```

### Connect an MCP Client

<details>
<summary><b>Cursor</b> (stdio)</summary>

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "kubeflow": {
      "command": "kubeflow-mcp",
      "args": ["serve"]
    }
  }
}
```

</details>

<details>
<summary><b>Claude Code</b> (stdio)</summary>

```bash
claude mcp add kubeflow -- kubeflow-mcp serve
```

</details>

<details>
<summary><b>Docker & HTTP Transport</b> (remote / in-cluster)</summary>

Pre-built multi-arch images are published to GHCR on every release. Replace `<your-secret-token>` with a unique secret before running:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -v ~/.kube/config:/kubeconfig:ro \
  -e KUBECONFIG=/kubeconfig \
  -e KUBEFLOW_MCP_AUTH_TOKEN="<your-secret-token>" \
  ghcr.io/kubeflow/mcp-server:latest
```

Connect your MCP client to `http://localhost:8000/mcp`:

```json
{
  "mcpServers": {
    "kubeflow": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer <your-secret-token>" }
    }
  }
}
```

For local clusters (e.g., Kind/Minikube) whose kubeconfig points to `127.0.0.1`, use `--network host` (Linux) or a kubeconfig targeting `host.docker.internal` so the container can reach the host API server. For in-cluster Kubernetes deployments, omit the kubeconfig mount (the pod uses its ServiceAccount credentials) and set `KUBEFLOW_MCP_ALLOWED_HOSTS` when exposing behind a Service or Ingress. Unauthenticated container probes are exposed at `GET /health` (liveness) and `GET /ready` (readiness).

</details>

### Example Workflow

Start the server with `data-scientist` (the default `readonly` persona blocks job submission):

```bash
kubeflow-mcp serve --persona data-scientist
```

```text
User: "Fine-tune gemma-2b on the alpaca dataset"

Agent → check_compatibility()                      # K8s + Trainer CRD verified
Agent → get_cluster_resources()                    # 4x A100 GPUs available
Agent → estimate_resources("google/gemma-2b")      # Needs ~16GB GPU, 1x A100
Agent → list_runtimes()                            # torchtune-gemma-2b found
Agent → fine_tune(..., confirmed=False)            # Returns preview for approval
Agent → fine_tune(..., confirmed=True)             # TrainJob "train-gemma-abc" created
Agent → get_training_logs("train-gemma-abc")       # Streams training progress
```

## Available Tools

### Trainer (`--clients trainer`, default — 23 tools)

| Phase | Tools | Description |
|-------|-------|-------------|
| **Planning** | `pre_flight`, `check_compatibility`, `get_cluster_resources`, `estimate_resources` | Environment checks and GPU/memory estimation |
| **Discovery** | `list_training_jobs`, `get_training_job`, `list_runtimes`, `get_runtime` | Browse TrainJobs and ClusterTrainingRuntimes |
| **Training** | `fine_tune`, `run_custom_training`, `run_container_training` | Submit LoRA/QLoRA fine-tuning, Python scripts, or container jobs |
| **Monitoring** | `get_training_logs`, `get_training_events`, `wait_for_training` | Track job status, logs, and Kubernetes events |
| **Lifecycle** | `update_training_job`, `delete_training_job` | Suspend, resume, or delete MCP-managed jobs |
| **Platform** | `inspect_crd`, `inspect_controller`, `create_runtime`, `patch_runtime`, `delete_runtime` | Cluster inspection and runtime administration |
| **Health** | `health_check`, `get_server_logs` | Server diagnostics |

### Pipelines (`--clients pipelines` — 14 tools)

Requires `pip install "kubeflow-mcp[pipelines]"` and Kubeflow Pipelines on the cluster.

| Phase | Tools | Description |
|-------|-------|-------------|
| **Discovery** | `list_pipelines`, `get_pipeline`, `list_experiments`, `get_experiment` | Browse pipelines, versions, and experiments |
| **Authoring** | `create_pipeline`, `delete_pipeline` | Compile and register KFP v2 pipelines from Python, or delete them |
| **Monitoring** | `list_pipeline_runs`, `get_pipeline_run`, `get_pipeline_logs` | Follow runs and read task pod logs |
| **Runs** | `create_pipeline_run`, `stop_pipeline_run`, `retry_pipeline_run`, `create_experiment`, `delete_experiment` | Launch, stop, or retry runs and manage experiments |

### Spark (`--clients spark` — 5 tools)

Requires `kubeflow[spark]` extra.

| Phase | Tools | Description |
|-------|-------|-------------|
| **Discovery** | `list_spark_sessions`, `get_spark_session` | Browse SparkConnect sessions |
| **Sessions** | `create_spark_session`, `delete_spark_session` | Provision and delete MCP-managed SparkConnect sessions |
| **Monitoring** | `get_spark_session_logs` | Read Spark driver logs |

### Optimizer (`--clients optimizer` — 17 tools)

Requires Katib v0.19 on the cluster, and the `katib.kubeflow.org/metrics-collector-injection=enabled` label on the experiment namespace.

| Phase | Tools | Description |
|-------|-------|-------------|
| **Planning** | `katib_pre_flight` | Check the Experiment CRD, controller readiness, and admission webhooks |
| **Discovery** | `list_hpo_experiments`, `get_hpo_experiment`, `get_hpo_experiment_status`, `get_trial`, `get_successful_trials`, `list_suggestions` | Browse Katib experiments, trials, and suggestions |
| **Optimization** | `create_hpo_experiment`, `create_hpo_experiment_from_spec` | Create experiments from flat parameters or a full `V1beta1Experiment` manifest |
| **Monitoring** | `get_hpo_experiment_trials`, `get_best_trial`, `get_suggestion`, `get_hpo_experiment_trial_logs`, `get_hpo_experiment_events` | Follow trials and read the best result, trial logs, and events |
| **Lifecycle** | `wait_for_hpo_experiment`, `update_hpo_experiment`, `delete_hpo_experiment` | Wait for completion, change the trial budget or parallelism, or delete MCP-managed experiments |

Katib has no pause: lowering `parallel_trial_count` makes Katib delete trials, newest first and whatever their state, so `update_hpo_experiment` is destructive and its preview lists those trials. See `optimizer://guides/hpo-patterns`.

## Personas & Configuration

`--persona` controls which tools are exposed. Each level includes all tools from the level above it:

| Persona | Scope |
|---------|-------|
| `readonly` *(default)* | Planning, browsing resources/jobs, reading logs and events |
| `data-scientist` | Submitting training/pipeline runs, deleting MCP-created jobs/sessions |
| `ml-engineer` | Container training, job lifecycle controls, pipeline authoring, CRD/controller inspection |
| `platform-admin` | Full access, including runtime and pipeline/experiment administration |

Custom personas can also be defined in `~/.kf-mcp-policy.yaml` (see [Architecture](ARCHITECTURE.md#security-model)).

### CLI Reference (`kubeflow-mcp serve`)

| Flag | Values / Default | Description |
|------|------------------|-------------|
| `--clients` | `trainer`, `spark`, `pipelines`, `optimizer` *(default: `trainer`)* | Comma-separated client modules to enable |
| `--persona` | `readonly` \| `data-scientist` \| `ml-engineer` \| `platform-admin` | Tool access persona |
| `--mode` | `full` \| `progressive` \| `semantic` | `progressive` (3 meta-tools) and `semantic` (2 meta-tools) reduce token usage |
| `--instruction-tier` | `full` \| `compact` \| `minimal` | Agent instruction verbosity |
| `--transport` | `stdio` \| `http` *(default: `stdio`)* | Transport protocol (`sse` is deprecated) |
| `--auth-token` | `<token>` | Bearer token for HTTP authentication |
| `--otel-endpoint` | `<url>` | Optional OTLP HTTP endpoint for OpenTelemetry tracing |
| `--log-level` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | Server log verbosity |
| `--log-format` | `console` \| `json` | Auto-detected if omitted |
| `--no-banner` | flag | Suppress the startup banner |

<details>
<summary><b>Environment Variables & HTTP Authentication</b></summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | Transport protocol (`http`, `stdio`) |
| `KUBEFLOW_MCP_CLIENTS` | `trainer` | Comma-separated client modules |
| `KUBEFLOW_MCP_PERSONA` | `readonly` | Active persona name |
| `KUBEFLOW_MCP_INSTRUCTION_TIER` | `full` | Instruction verbosity (`full`, `compact`, `minimal`) |
| `KUBEFLOW_MCP_AUTH_TOKEN` | _(none)_ | Bearer token for HTTP auth (dev/staging) |
| `KUBEFLOW_MCP_JWKS_URI` | _(none)_ | JWKS endpoint for JWT verification (production) |
| `KUBEFLOW_MCP_JWT_ISSUER` | _(none)_ | Expected JWT issuer (required with `KUBEFLOW_MCP_JWKS_URI`) |
| `KUBEFLOW_MCP_JWT_AUDIENCE` | _(none)_ | Expected JWT audience (required with `KUBEFLOW_MCP_JWKS_URI`) |
| `KUBEFLOW_MCP_ALLOWED_HOSTS` | _(loopback)_ | Allowed `Host` headers for DNS rebinding protection |
| `KUBEFLOW_MCP_ALLOWED_ORIGINS` | _(loopback)_ | Allowed `Origin` headers |
| `KUBEFLOW_MCP_DNS_REBINDING_PROTECTION` | `true` | Set `false` to disable Host/Origin checks (not recommended) |
| `KUBEFLOW_MCP_PIPELINES_URL` | _(auto)_ | Override Kubeflow Pipelines endpoint URL |
| `KUBEFLOW_MCP_PIPELINES_TOKEN` | _(auto)_ | Override Kubeflow Pipelines auth token |
| `KUBEFLOW_MCP_PIPELINES_NAMESPACE` | _(auto)_ | Override Kubeflow Pipelines namespace |
| `LOG_FORMAT` / `LOG_LEVEL` | `auto` / `INFO` | Logging format and level |

`KUBEFLOW_MCP_JWT_ISSUER` and `KUBEFLOW_MCP_JWT_AUDIENCE` are required with `KUBEFLOW_MCP_JWKS_URI`; the server refuses to start without them.
Without `--auth-token` or JWT settings on `--transport http`, the server logs a warning that the endpoint is open.

</details>

## Observability

Optional OpenTelemetry tracing can be enabled with `--otel-endpoint http://localhost:4318` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) after installing `uv sync --group otel`. See [examples/observability/README.md](examples/observability/README.md) for collector setup, span attributes, and log correlation.

## Compatibility

| MCP Server | Kubeflow Trainer | Kubeflow SDK | Python | Kubernetes |
|------------|------------------|--------------|--------|------------|
| `0.2.x` *(upcoming)* | `>= 2.3.0, < 2.4.0` | `>= 0.5.0, < 0.6.0` | `3.10 – 3.12` | `>= 1.27` |
| `0.1.x` | `== 2.2.1` | `== 0.4.1` | `3.10 – 3.12` | `>= 1.27` |

## Development

```bash
git clone https://github.com/kubeflow/mcp-server.git
cd mcp-server
make install-dev                  # setup environment
make verify                       # lint + format check
make test-python                  # run unit tests
make inspector                    # launch MCP Inspector (stdio; or TRANSPORT=http)
```

## Community & Documentation

- **Slack**: [#kubeflow-wg-agents](https://app.slack.com/client/T08PSQ7BQ/C0BQQRSPK6Z) and [#kubeflow-mcp-server](https://app.slack.com/client/T08PSQ7BQ/C0BSRBPNDHP) on [CNCF Slack](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)
- **Meetings**: [Kubeflow WG Agents Meeting Notes](https://bit.ly/kubeflow-mcp-wg-notes)
- **Blog**: [Introducing Kubeflow MCP: An Agent Interface for Cloud Native AI at Scale](https://blog.kubeflow.org/introducing-kubeflow-mcp/)
- **Docs**: [Quickstart](docs/quickstart.md) · [Architecture](ARCHITECTURE.md) · [Client Conventions](docs/CONVENTIONS.md) · [Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Security](SECURITY.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contributors

<a href="https://github.com/kubeflow/mcp-server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kubeflow/mcp-server" />
</a>
