# Architecture Overview

This document describes the Kubeflow MCP Server architecture in two parts: the **current state**
of the `main` branch, and the **planned target architecture** shown in the diagram below.

For the phased delivery plan, see [ROADMAP.md](ROADMAP.md).
For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Current Architecture

### What ships today

```
IDE (Cursor · Claude Code · VSCode)
Orchestrator Agent
         │
         │  stdio  │  Streamable HTTP  │  SSE (legacy)
         ▼
┌────────────────────────────────────────────-─┐
│  kubeflow-mcp serve  (FastMCP)               │
│                                              │
│  Auth          Bearer token · JWT (JWKS)     │
│  Policy        Persona-based tool filter     │
│  Tool modes    full · progressive · semantic │
│  Resilience    Rate limiter · Circuit breaker│
│  Health        /health  /ready  /metrics     │
│  Security      Input validation · masking    │
└──────────────────────┬──────────────────────-┘
                       │  Kubeflow SDK
                       ▼
               Trainer (23 tools)
               Optimizer stub · Hub stub
                       │
                       ▼
               Kubernetes / Kubeflow Trainer v2
               │
               Infrastructure
               Local · Kind · OpenShift · EKS · GKE
```

### Components

| Component | Module | Status |
|-----------|--------|--------|
| MCP server | `core/server.py` | ✅ Available |
| Auth (bearer / JWT) | `core/auth.py` | ✅ Available |
| Policy / personas | `core/policy.py` | ✅ Available |
| Resilience (rate limit, circuit breaker) | `core/resilience.py` | ✅ Available |
| Health endpoints | `core/health.py` | ✅ Available |
| Input validation | `core/security.py` | ✅ Available |
| HTTP edge | `core/http_edge.py` | ✅ Available |
| Trainer tools (23) | `trainer/api/` | ✅ Available |
| Optimizer | `optimizer/` | 🔲 Stub only |
| Hub / Model Registry | `hub/` | 🔲 Stub only |
| OpenTelemetry tracing | — | 🔄 In review ([#21](https://github.com/kubeflow/mcp-server/pull/21)) |
| Docker image + CI | — | 🔄 In review ([#25](https://github.com/kubeflow/mcp-server/pull/25)) |
| PyPI release workflow | — | 🔄 In review ([#28](https://github.com/kubeflow/mcp-server/pull/28)) |
| CLI Agent Runtime | — | 🔲 Planned — Phase 3 ([#15](https://github.com/kubeflow/mcp-server/issues/15)) |
| Gateway layer support | — | 🔲 Planned — Phase 4 |
| Spark Connect | — | 🔲 Planned — Phase 6 ([#5](https://github.com/kubeflow/mcp-server/issues/5)) |

---

## Target Architecture

The diagram below shows the full target architecture (Phases 1–6). Sections not yet available in
`main` are marked **Planned**.

![Kubeflow MCP Server — Target Architecture](docs/assets/architecture.svg)

### Planned additions

#### CLI Agent Runtime — Phase 3 ([#15](https://github.com/kubeflow/mcp-server/issues/15))

A new `kubeflow-mcp agent` command with a common `AgentProvider` protocol and a per-session
LLM router supporting multiple backends.

```
kubeflow-mcp agent --provider litellm | ollama
         │
LiteLLM / Ollama Agent  →  Per-session LLM Router
         │
         ├──► Ollama       (local inference · thinking mode)
         ├──► vLLM         (self-hosted GPU serving)
         ├──► Cloud APIs   (OpenAI · Anthropic · Gemini · …)
         └──► auto-fallback
         │
         │  MCP tool calls (stdio)
         ▼
kubeflow-mcp serve
```

#### Observability — Phase 2 (in review)

OTel tracing is being added in [#21](https://github.com/kubeflow/mcp-server/pull/21) using the
[MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/).

```
kubeflow-mcp serve ──►                  ┌──► Jaeger · Prometheus · Grafana
CLI Agent Runtime  ──► OTel Collector ──┤
Kubernetes cluster ──►                  ├──► MLflow  (optional — if MLFLOW_TRACKING_URI set)
                                        └──► Langfuse (LLM traces + token cost)
```

Enable with `--otel-endpoint <url>` or `KUBEFLOW_MCP_OTEL_ENDPOINT`.

#### Gateway Layer — Phase 4

`kubeflow-mcp serve` is single-tenant today — one token, one persona, one cluster. Phase 4 absorbs
five gateway capabilities natively, making the external gateway stack **opt-in** for most teams.

The design boundary:

- **Absorb natively** — identity (OIDC/OAuth 2.1), K8s RBAC (`SubjectAccessReview`), per-user rate
  limiting, MCP Server Card, and A2A delegation. These are single-server concerns that shouldn't
  require a sidecar proxy.
- **Stay external permanently** — LiteLLM Proxy (org-level LLM budget enforcement) and
  agentgateway (multi-server MCP federation is by definition a routing-layer concern).

Target topology for a standard single-team deployment after Phase 4:

```
Consumers
    │  Streamable HTTP (OIDC-authenticated)
    ▼
kubeflow-mcp serve
  ├─ OIDC / OAuth 2.1 native auth
  ├─ K8s RBAC per caller (SubjectAccessReview)
  ├─ Per-user rate limiting
  ├─ MCP Server Card (/.well-known/mcp.json)
  └─ A2A endpoint (/a2a)
```

The full gateway stack (agentgateway + LiteLLM Proxy) remains available for orgs that need
multi-server federation or org-wide LLM cost control. See [ROADMAP.md — Phase 4](ROADMAP.md#phase-4--enterprise--in-cluster-to-do) for the complete delivery list.

#### Eval Pipeline — Phase 2 ([#10](https://github.com/kubeflow/mcp-server/issues/10))

Not part of the serving path. Three tiers to balance speed, cost, and signal quality:

**Tier 1 — Every PR** (fast · free · deterministic)
```
GitHub Actions
    │
Rule-based safety judges
  · confirm gate never submits with confirmed=False
  · pre_flight runs before fine_tune
  · destructive tools blocked for readonly / data-scientist personas
  · MCP protocol conformance (tool schema, response types)
```

**Tier 2 — Nightly on main** (LLM judge · costs money)
```
Scheduled GitHub Actions
    │
Golden Dataset → LLM-as-judge + DeepEval
    │
Score Report → GHA artifact          (always)
            → MLflow                 (optional — if MLFLOW_TRACKING_URI is set)
            ← compare eval/baseline.json → fail run on regression
```

**Tier 3 — On-demand** (release candidates / major feature PRs)
```
Triggered manually by maintainer
    │
Full eval run → GHA artifact + PR comment (score delta)
             → update eval/baseline.json intentionally
```

> LLM judges run **nightly / on-demand only** — never per-PR. Per-PR LLM judging causes
> non-deterministic CI failures, unbounded API costs, and slow feedback loops.

---

## Security

- **stdio** — OS-level trust boundary; no network exposure, no auth.
- **HTTP / SSE** — Bearer API key (`KUBEFLOW_MCP_AUTH_TOKEN`) or JWT (`KUBEFLOW_MCP_JWKS_URI`).
- **Tool authorization** — Persona-based filtering; four built-in roles (`readonly`, `data-scientist`, `ml-engineer`, `platform-admin`).
- **Confirm gate** — All mutating tools require `confirmed=True`; agents preview before executing.
- **Input validation** — Kubernetes name/namespace checks, Python script safety, sensitive-data masking (`core/security.py`).

See [SECURITY.md](SECURITY.md) for the full threat model, RBAC guidance, and responsible disclosure policy.

---

## Glossary

| Term | Definition |
|------|------------|
| **MCP** | Model Context Protocol — open standard for exposing tools and resources to AI agents |
| **FastMCP** | Python MCP server framework used as the serving layer |
| **TrainJob** | Kubeflow Trainer v2 CRD representing a distributed training job |
| **ClusterTrainingRuntime** | Cluster-scoped CRD defining the runtime environment (images, backend, defaults) |
| **`full` mode** | All tools exposed directly; highest token cost, best for capable cloud models |
| **`progressive` mode** | 3 meta-tools (`list_tools` → `describe_tools` → `execute_tool`); lower initial token cost |
| **`semantic` mode** | 2 meta-tools (`find_tools` → `execute_tool`) using NL embeddings with keyword fallback |
| **confirm gate** | Mutating tools return a preview with `confirmed=False`; execute only with `confirmed=True` |
| **persona** | Server-side role filter restricting which tools are visible to a caller |
| **agentgateway** | Planned gateway providing RBAC, OTel, A2A, and MCP federation (Phase 4) |
| **KEP-936** | Kubeflow Enhancement Proposal defining this MCP server |
