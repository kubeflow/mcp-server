# Kubeflow MCP Server ROADMAP

This roadmap tracks the phased delivery plan for the Kubeflow MCP Server, as proposed in
[KEP‑936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server).
It covers the MCP server runtime only; higher-level toolkits and marketplaces
(for example, a `kubeflow/ai-toolkit` repo) is considered under future scope.

## 2026

### Core MCP Server (Trainer, Alpha/Beta)

- **Core Trainer MCP server (Phase 1)**
  - Ship 23 trainer tools (planning, training, discovery, monitoring, lifecycle, health)
  - Dynamic client loading (`--clients trainer,optimizer,hub`)
  - Tool modes: `full`, `progressive`, `semantic`
  - Persona & policy system (`readonly`, `data‑scientist`, `ml‑engineer`, `platform‑admin`)
  - Two‑phase confirmation for all mutating tools (`confirmed=False/True`)
  - Structured audit logging and failure hints

- **Production hardening (Phase 2)**
  - HTTP auth: bearer token + JWT verification
  - Identity → `TrainerClient` wiring (per‑user kube context)
  - Anthropic “native tool” adapter to expose these tools via the Messages API
    (using `tool_search_tool` and `defer_loading`)
  - Reliability controls (rate limiting, circuit breaking, timeouts)
  - Observability: OpenTelemetry integration and Prometheus `/metrics`
  - CI: `kind` smoke tests, coverage gates, packaging and PyPI release, Dockerfile

### Enterprise & In‑Cluster (Phase 3)

- **In‑cluster deployment**
  - Helm chart / Kustomize overlays
  - ServiceAccount + impersonation for StreamableHTTP
  - OAuth2.1 / OIDC gateway deployment pattern

- **Multi‑tenancy & multi‑MCP**
  - Per‑user namespace scoping (Istio headers)
  - ResourceQuota pre‑flight checks
  - Coordination patterns with other MCP servers (Kubernetes, HF, Spark, etc.)

- **Agent UX**
  - First‑class support for elicitation (once widely available in MCP clients)
  - Convenience flows: auto‑validate → submit, structured error taxonomy

### Advanced Training (Phase 4)

- **Training lifecycle**
  - Checkpoints: `list_checkpoints()`, `restore_checkpoint()`
  - Progress and metrics: `get_training_progress()`, `get_training_metrics()`
  - Dynamic scaling: `scale_training_job()` for running jobs

- **GPU & multi‑cluster**
  - GPU visibility for active TrainJobs
  - Multi‑cluster support and selection patterns

### Additional Client Modules (Phase 5)

- **Optimizer (Katib) module**
  - `create_experiment`, `get_experiment`, `list_experiments`, `get_optimal_trial`, `delete_experiment`
  - `list_algorithms`, `suggest_config`, `compare_trials`

- **Hub / Model Registry module**
  - `register_model`, `list_models`, `get_model`, `list_model_versions`, `get_model_version`
  - `get_model_artifact`, `update_model`, `promote_model`, `compare_models`, `get_model_lineage`, `export_model`

- **Tool scalability & agent backends**
  - Documented integration with `mcp-optimizer` and other middleware
  - Agent backends: Claude, OpenAI, LlamaIndex/LangGraph powered agents using the same tool surface

### Future Modules (Phase 6+)

- **Pipelines MCP client**
  - Pipeline definition, compilation, submission, run management

- **Spark MCP client**
  - Spark job submission, monitoring, diagnostics (aligned with Spark MCP / KFP integrations)

- **Feast MCP client**
  - Feature retrieval and feature store operations

- **Notebooks & UI**
  - Notebook server management tools
  - Web dashboard, VS Code extension, Slack/ChatOps integrations built on top of the MCP server

## Graduation Criteria

- **Alpha**
  - Trainer MCP server with 23 tools, dynamic modes, personas, policy, and `SECURITY.md`
  - Basic CI and a passing automated test suite for trainer workflows (unit + integration)

- **Beta**
  - Production‑ready HTTP mode (auth, identity, observability)
  - In‑cluster / gateway deployment patterns and `kind`‑based smoke tests
  - End‑to‑end tests with a real Kubeflow cluster and at least one MCP client (Claude Code, Cursor, etc.)

- **Stable**
  - Multi‑client validation (Trainer + Optimizer + Hub at minimum)
  - Documented support matrix for Kubeflow SDK, Trainer, and Kubernetes versions
  - Full user and operator documentation (including docs site) and a well‑defined deprecation policy

See also the [Kubeflow SDK ROADMAP](https://github.com/kubeflow/sdk/blob/main/ROADMAP.md)
for complementary SDK work that this MCP server depends on.
