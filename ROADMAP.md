# Kubeflow MCP Server ROADMAP

This roadmap tracks the phased delivery plan for the Kubeflow MCP Server, as proposed in
[KEP-936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server).
It covers the MCP server runtime only; higher-level toolkits and marketplaces
(for example, a `kubeflow/ai-toolkit` repo) are considered under future scope.

## 2026

### Core MCP Server (Trainer, Alpha/Beta)

- **Core Trainer MCP server (Phase 1)**
  - Ship 23 trainer tools (planning, training, discovery, monitoring, lifecycle, health)
  - Dynamic client loading (`--clients trainer,optimizer,hub`)
  - Tool modes: `full`, `progressive`, `semantic`
  - Persona & policy system (`readonly`, `data-scientist`, `ml-engineer`, `platform-admin`)
  - Two-phase confirmation for all mutating tools (`confirmed=False/True`)
  - Structured audit logging and failure hints

- **Production hardening (Phase 2)**
  - HTTP auth: bearer token + JWT verification
  - Identity -> `TrainerClient` wiring (per-user kube context), aligned with [kubeflow/sdk#281](https://github.com/kubeflow/sdk/issues/281) (unified auth mechanism)
  - Anthropic "native tool" adapter to expose these tools via the Messages API
    (using `tool_search_tool` and `defer_loading`)
  - Reliability controls (rate limiting, circuit breaking, timeouts)
  - CI: `kind` smoke tests, coverage gates, packaging and PyPI release, Dockerfile

- **Observability integrations (Phase 2)**
  - OpenTelemetry traces/spans per MCP tool call via [FastMCP native instrumentation](https://gofastmcp.com/servers/telemetry)
  - Prometheus `/metrics` endpoint — latency histograms and success/error counters per tool
  - Structured JSON audit logs in JSONL format for HTTP mode
  - Experiment tracking integration with MLflow (aligned with [kubeflow/sdk#63](https://github.com/kubeflow/sdk/issues/63))

- **Test and benchmark suite (Phase 2)**
  - Unit tests for all trainer tools and core modules (auth, config, policy, resilience, security, logging)
  - Integration and workflow tests (end-to-end tool chains: plan -> train -> monitor)
  - Performance benchmarks: latency (P50/P95/P99 per tool), token usage per mode, CPU/memory profiling
  - HTML benchmark dashboard (`make benchmark`)

### Enterprise & In-Cluster (Phase 3)

- **In-cluster deployment**
  - Helm chart / Kustomize overlays
  - ServiceAccount + impersonation for StreamableHTTP
  - OAuth2.1 / OIDC gateway deployment pattern

- **Multi-tenancy & multi-MCP**
  - Per-user namespace scoping (Istio headers)
  - ResourceQuota pre-flight checks
  - Coordination patterns with other MCP servers ([kubernetes-mcp-server](https://github.com/containers/kubernetes-mcp-server), [hf-mcp-server](https://github.com/huggingface/hf-mcp-server), Spark MCP, etc.)
  - Explore [AGNTCY Identity](https://github.com/agntcy/identity) for cryptographic signatures on tool calls

- **Gateway deployment pattern with agentgateway**
  - Use [agentgateway](https://github.com/agentgateway/agentgateway) (Linux Foundation, Rust/Go) as the recommended gateway layer for in-cluster/production deployments
  - agentgateway provides OIDC/JWT auth, CEL-based RBAC, rate limiting, OTel observability, and tool federation — eliminating the need to build those at the MCP server layer
  - Recommended topology: `AI Agent → agentgateway (auth, RBAC, OTel, rate-limit) → kubeflow-mcp (stdio or plain HTTP) → TrainerClient → K8s API`
  - Document Helm manifests, K8s Gateway API integration, and multi-MCP server federation patterns

- **Agent UX**
  - MCP Elicitation support via `ctx.elicit()` to replace `confirmed=False/True` pattern
    (blocked on client adoption — Claude Desktop, Cursor, VS Code don't fully support it yet; `confirmed` stays as fallback)
  - Convenience flows: auto-validate -> submit, structured error taxonomy

### Advanced Training (Phase 4)

- **Training lifecycle**
  - Checkpoints: `list_checkpoints()`, `restore_checkpoint()` (aligned with [KEP-2777](https://github.com/kubeflow/trainer/issues/2777) transparent GPU checkpointing with CRIU)
  - Progress and metrics: `get_training_progress()`, `get_training_metrics()` (aligned with [KEP-2779](https://github.com/kubeflow/trainer/tree/master/docs/proposals/2779-trainjob-progress))
  - Dynamic scaling: `scale_training_job()` for running jobs
  - Workspace snapshot: expose `snapshot_workspace()` once upstream support lands ([kubeflow/sdk#48](https://github.com/kubeflow/sdk/issues/48))

- **GPU & multi-cluster**
  - GPU visibility for active TrainJobs (aligned with [kubeflow/sdk#159](https://github.com/kubeflow/sdk/issues/159))
  - Multi-cluster support and selection patterns (aligned with [kubeflow/sdk#23](https://github.com/kubeflow/sdk/issues/23))

- **Dynamic LLM Trainer backends**
  - Track [KEP-2839](https://github.com/kubeflow/trainer/issues/2839) for TRL, Unsloth, and other LLM trainer frameworks
  - Keep `list_runtimes()`, `get_runtime()`, and `fine_tune()` in sync as new runtimes land upstream

### Additional Client Modules (Phase 5)

- **Optimizer (Katib) module**
  - `create_experiment`, `get_experiment`, `list_experiments`, `get_optimal_trial`, `delete_experiment`
  - `list_algorithms`, `suggest_config`, `compare_trials`

- **Hub / Model Registry module**
  - `register_model`, `list_models`, `get_model`, `list_model_versions`, `get_model_version`
  - `get_model_artifact`, `update_model`, `promote_model`, `compare_models`, `get_model_lineage`, `export_model`

- **Local and self-hosted agent backends**
  - Ollama agent: LlamaIndex `FunctionAgent` wired to the MCP server with Ollama LLM backend
  - `progressive` and `semantic` meta-tool modes specifically designed for smaller local models
    that cannot fit 20+ tool schemas in context (see [issue #3](https://github.com/kubeflow/mcp-server/issues/3))
  - vLLM and other self-hosted model backends
  - Claude (Anthropic Messages API), OpenAI (Responses API), LangGraph backends

- **Tool scalability**
  - Documented integration with [mcp-optimizer](https://github.com/StacklokLabs/mcp-optimizer) middleware
  - Anthropic `defer_loading` + `tool_search_tool` adapter for native tool scaling

### Future Modules (Phase 6+)

- **Pipelines MCP client**
  - Pipeline definition, compilation, submission, run management
  - Aligned with [kubeflow/sdk#125](https://github.com/kubeflow/sdk/issues/125)

- **Spark MCP client**
  - Spark job submission, monitoring, diagnostics
  - Aligned with [kubeflow/sdk#107](https://github.com/kubeflow/sdk/issues/107) and [kubeflow/sdk#238](https://github.com/kubeflow/sdk/issues/238) (AI-Assisted SparkClient)

- **Feast MCP client**
  - Feature retrieval and feature store operations
  - Aligned with [kubeflow/sdk#239](https://github.com/kubeflow/sdk/issues/239)

- **Notebooks & UI**
  - Notebook server management tools
  - Web dashboard, VS Code extension, Slack/ChatOps integrations built on top of the MCP server

## Graduation Criteria

- **Alpha**
  - Trainer MCP server with 23 tools, dynamic modes, personas, policy, and `SECURITY.md`
  - Basic CI and a passing automated test suite for trainer workflows (unit + integration)

- **Beta**
  - Production-ready HTTP mode (auth, identity, observability)
  - OpenTelemetry and Prometheus integrated and documented
  - Benchmark suite establishing baseline latency and token usage per mode
  - In-cluster / gateway deployment patterns and `kind`-based smoke tests
  - End-to-end tests with a real Kubeflow cluster and at least one MCP client (Claude Code, Cursor, etc.)

- **Stable**
  - Multi-client validation (Trainer + Optimizer + Hub at minimum)
  - Local agent backend (Ollama) validated end-to-end
  - Documented support matrix for Kubeflow SDK, Trainer, and Kubernetes versions
  - Full user and operator documentation (including docs site) and a well-defined deprecation policy

See also the [Kubeflow SDK ROADMAP](https://github.com/kubeflow/sdk/blob/main/ROADMAP.md)
for complementary SDK work that this MCP server depends on.
