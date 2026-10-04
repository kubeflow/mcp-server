# Kubeflow MCP Server — Quickstart & Troubleshooting

A task-focused guide: install the server, connect a supported MCP client, verify connectivity, and
diagnose common failures. For full CLI flags and environment variables see the [README](../README.md);
for the security model and RBAC YAML see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python** 3.10 – 3.12 | Check the [compatibility matrix](../README.md#requirements) for exact SDK versions |
| **Kubernetes** ≥ 1.27 | `kubectl` configured and pointing at your cluster (`kubectl cluster-info`) |
| **Kubeflow Trainer** ≥ 2.3.0 | CRD must be installed; verify with `kubectl get crd trainjobs.trainer.kubeflow.org` |
| **MCP client** | Cursor, Claude Code, VS Code, or any client that speaks MCP |

> **Note:** The server itself does not need GPU resources — only the training jobs it submits do.

---

## Install

### Option A — pip (recommended for local dev)

```bash
pip install kubeflow-mcp
kubeflow-mcp serve          # starts with stdio transport by default
```

### Option B — Docker (recommended for shared / in-cluster deployments)

Pre-built multi-arch images are published to GHCR on every release:

```bash
docker run --rm -p 8000:8000 \
  -e KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token \
  -v ~/.kube:/root/.kube:ro \
  ghcr.io/kubeflow/mcp-server:latest
```

The server listens at `http://localhost:8000/mcp`.
For in-cluster deployment with Kubernetes manifests see [`examples/kubernetes/`](../examples/kubernetes/).

---

## Connect Your MCP Client

Choose the transport that fits your setup:

| Transport | When to use | How it works |
|-----------|-------------|--------------|
| `stdio` | Local IDE plugins (Cursor, Claude Code) | Client spawns the server as a child process; communicates over stdin/stdout |
| `http` | Remote server or Docker container | Client connects over HTTP; requires auth configuration |
| `sse` | Legacy remote clients | Deprecated in favour of Streamable HTTP; use `http` for new setups |

### Cursor

Add to `.cursor/mcp.json` (or use the repo-root [`.mcp.json`](../.mcp.json) for local dev):

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

Transport: `stdio` (Cursor spawns the server process; no port or auth required).

### Claude Code

```bash
claude mcp add kubeflow -- kubeflow-mcp serve
```

Transport: `stdio`. Claude Code manages the server lifecycle.

### VS Code (or any HTTP client)

Start the server separately with HTTP transport, then point your client at it:

```bash
export KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token
kubeflow-mcp serve --transport http
```

Client config:

```json
{
  "mcpServers": {
    "kubeflow": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer my-secret-token" }
    }
  }
}
```

---

## Verify Connectivity

### 1. Ask your agent

Once the client is connected, ask:

> *"Check if my cluster is compatible with Kubeflow Training."*

The agent calls `pre_flight()` then `check_compatibility()`. A healthy response looks like:

```
✅ Kubernetes 1.29 reachable
✅ Kubeflow Trainer CRD installed (v2.3.0)
✅ SDK version compatible
```

If the agent returns an error, jump to [Troubleshooting](#troubleshooting) below.

### 2. HTTP health endpoints (http/sse transport only)

These endpoints do **not** require authentication and are safe to use from load-balancer probes:

```bash
curl http://localhost:8000/health   # liveness  — server process is up
curl http://localhost:8000/ready    # readiness — clients loaded, resources ready
```

`/ready` returns `200` only when both `clients_ready` and `resources_ready` are `true`. It does
**not** contact Kubernetes; a 503 means a packaged resource file is missing — check server logs, not
cluster health.

### 3. MCP Inspector (no LLM required)

Use the MCP Inspector to verify tools are registered independently of your AI client:

```bash
make inspector                      # stdio
make inspector TRANSPORT=http       # HTTP (start the server separately first)
```

---

## Configuration

Settings are resolved in this order (highest priority first):

```
CLI flag  →  Environment variable  →  ~/.kubeflow-mcp.yaml  →  built-in default
```

**Example config file** (`~/.kubeflow-mcp.yaml`):

```yaml
server:
  clients: [trainer]
  persona: data-scientist   # readonly | data-scientist | ml-engineer | platform-admin
  transport: stdio

auth:
  auth_token: my-secret-token   # dev/staging bearer token

logging:
  level: INFO
  format: console              # console | json (auto-detected if omitted)
```

Config file locations searched in order:

1. `./.kubeflow-mcp.yaml` (project-local)
2. `~/.kubeflow-mcp.yaml`
3. `~/.kubeflow-mcp.yml`
4. `~/.config/kubeflow-mcp/config.yaml`

For the full environment variable reference see the [README env vars table](../README.md#run-with-docker).

### Persona selection

The default persona is `readonly` — it exposes only read/inspect tools and hides all write
operations. Expand access as needed:

| Persona | Adds |
|---------|------|
| `readonly` | `list_*`, `get_*`, planning, monitoring (default) |
| `data-scientist` | + `fine_tune`, `run_*`, delete own MCP-created resources |
| `ml-engineer` | + `update_*`, platform inspect, advanced submit |
| `platform-admin` | all tools |

Set via `--persona ml-engineer`, `KUBEFLOW_MCP_PERSONA=ml-engineer`, or the config file.

---

## Authentication

### stdio transport

No authentication configuration needed. The server inherits your OS-level identity and reads your
local `~/.kube/config` directly.

### HTTP transport — development (bearer token)

```bash
# CLI flag
kubeflow-mcp serve --transport http --auth-token my-secret-token

# or environment variable
export KUBEFLOW_MCP_AUTH_TOKEN=my-secret-token
kubeflow-mcp serve --transport http
```

The server logs a warning if HTTP transport starts without any auth configured.

### HTTP transport — production (JWT / OIDC)

```bash
export KUBEFLOW_MCP_JWKS_URI=https://auth.example.com/.well-known/jwks.json
export KUBEFLOW_MCP_JWT_ISSUER=https://auth.example.com
export KUBEFLOW_MCP_JWT_AUDIENCE=kubeflow-mcp
kubeflow-mcp serve --transport http
```

For a defence-in-depth setup, place the server behind an authenticating reverse proxy with TLS in
addition to the built-in token/JWT check. See
[ARCHITECTURE.md — HTTP Transport](../ARCHITECTURE.md#2-http-transport--authentication) for the
full threat model.

### DNS rebinding protection

When using HTTP/SSE transport, the server validates `Host` and `Origin` headers to prevent DNS
rebinding attacks. By default only loopback addresses (`localhost`, `127.0.0.1`, `[::1]`) are
allowed.

If you expose the server through a Kubernetes Service, Ingress, or any non-loopback address, add the
hostname to the allowlist or requests will be rejected with **HTTP 421**:

```bash
export KUBEFLOW_MCP_ALLOWED_HOSTS="kubeflow-mcp.kubeflow.svc:*,mcp.example.com"
# For browser-based clients also set origins:
export KUBEFLOW_MCP_ALLOWED_ORIGINS="https://mcp.example.com"
```

The `:*` wildcard matches any port for that host. `kubectl port-forward` is unaffected because it
sends a loopback `Host` header.

---

## Kubernetes Access

### How the server finds your cluster

| Context | Mechanism |
|---------|-----------|
| Local workstation | `~/.kube/config` (standard `kubectl` config) |
| In-cluster pod | ServiceAccount token at `/var/run/secrets/kubernetes.io/serviceaccount/` |

The server does not manage kubeconfig files. Use `kubectl config use-context` to switch clusters
before starting the server locally.

### Minimum RBAC

The server needs read access to nodes, namespaces, and the Trainer CRD to function, plus TrainJob
lifecycle permissions for write personas. The full `ClusterRole` YAML is in
[ARCHITECTURE.md — RBAC Configuration](../ARCHITECTURE.md#rbac-configuration).

For a quick connectivity check, the `health_check` tool probes `list_namespace` — if Kubernetes is
unreachable it returns `"status": "degraded"`.

### Namespace restrictions (policy file)

Restrict which namespaces tools may target by creating `~/.kf-mcp-policy.yaml`:

```yaml
policy:
  namespaces:
    - team-a
    - team-b
```

> **Warning:** A policy file that exists but cannot be parsed (e.g. YAML syntax error) causes the
> server to refuse to start rather than silently drop restrictions.

---

## Troubleshooting

### Diagnostic commands

```bash
# Enable verbose server logs
LOG_LEVEL=DEBUG kubeflow-mcp serve

# Check K8s connectivity from inside an agent session
# Ask: "Run health_check" → returns {"status": "healthy|degraded", "kubernetes": true|false}

# HTTP health endpoints (no auth required)
curl http://localhost:8000/health
curl http://localhost:8000/ready

# MCP Inspector — test tools without an LLM
make inspector
```

---

### Server won't start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: kubeflow_mcp` | Package not installed | `pip install kubeflow-mcp` |
| Unsupported Python version error | Wrong Python version | Use Python 3.10 – 3.12 |
| Server exits immediately on stdio | The MCP client is not reading from the process stdin | Start with `--transport http` first to isolate config errors |
| `Policy file … could not be loaded` | Syntax error in `~/.kf-mcp-policy.yaml` | Fix YAML syntax; a bad policy file is fatal by design |
| `PyYAML not installed, skipping config file` | Optional YAML dep missing | `pip install pyyaml` |

---

### Client can't connect

| Symptom | Cause | Fix |
|---------|-------|-----|
| `HTTP 421 Misdirected Request` | DNS rebinding protection blocked the `Host` header | Add the hostname to `KUBEFLOW_MCP_ALLOWED_HOSTS` (see [Authentication](#authentication)) |
| `HTTP 403` from a browser-based client | `Origin` header not in allowlist | Add to `KUBEFLOW_MCP_ALLOWED_ORIGINS`; host and origin allowlists are independent |
| `HTTP 401 Unauthorized` | Wrong bearer token | Verify `KUBEFLOW_MCP_AUTH_TOKEN` matches the token sent by the client |
| Connection refused on port 8000 | Server started with `stdio` not `http` | Restart with `--transport http` |
| Agent says "no tools available" | Client config points to wrong URL | Check client URL; test with `curl http://localhost:8000/health` |
| Pod stuck in `CreateContainerConfigError` | `kubeflow-mcp-auth` Secret not created | Create the Secret before applying manifests (see [`examples/kubernetes/`](../examples/kubernetes/)) |

---

### Tools report errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `"kubernetes": false` in `health_check` | Server cannot reach the K8s API | Check `kubectl cluster-info`; verify kubeconfig or ServiceAccount permissions |
| `Trainer CRD not found` | Kubeflow Trainer not installed or wrong version | Install Trainer ≥ 2.3.0; check the [compatibility matrix](../README.md#requirements) |
| `"status": "degraded"` from `/ready` | A packaged Markdown resource file is missing | Check server logs; reinstall the package |
| Agent has no tool for submitting a job | Persona is `readonly` | Set `KUBEFLOW_MCP_PERSONA=data-scientist` or higher |
| `"was not created by MCP"` when deleting | Job was not submitted through MCP tools | Use `platform-admin` persona, or re-create the job via MCP tools |
| Empty namespace or 403 listing runtimes | In-cluster pod running in the wrong namespace | Deploy in the same namespace as your TrainJobs (see [`examples/kubernetes/`](../examples/kubernetes/)) |
| `Trainer control-plane version … (403)` | Missing Role for the `kubeflow-trainer-public` ConfigMap | Verify the `kubeflow-mcp-trainer-version` Role and Binding exist |

---

### Training job failures

| Error / Event | Cause | Fix |
|--------------|-------|-----|
| `OOMKilled` | GPU or CPU memory exceeded | Reduce `batch_size`; enable QLoRA (`quantize_base=True`); use gradient checkpointing |
| `FailedScheduling` | No node matches resource request | Check `get_cluster_resources()`; reduce `gpu_per_node`; add `tolerations` |
| `ErrImagePull / ImagePullBackOff` | Image not found or auth failed | Verify image name; add `image_pull_secrets` parameter |
| `NCCL timeout` | Multi-node communication failure | Pass `env={"NCCL_TIMEOUT": "1800"}`; try gloo backend; check cluster network |
| `403 Forbidden` (HuggingFace) | Gated model, no token | Accept the model licence on HuggingFace; pass `hf_token` parameter |
| `Read-only filesystem` | Platform enforces read-only root FS (e.g. OpenShift) | Add emptyDir volumes for `/.local`, `/.cache`, `/tmp` |
| `ProcessGroupNCCL … no GPUs` | torchtune on CPU-only cluster | Use `run_custom_training()` with gloo backend |
| Script syntax error | Invalid Python in `run_custom_training` | Script body is wrapped into a function — no top-level indentation, no `if __name__` guards |

For GPU memory sizing, batch size guidance, and additional scheduling parameters see
[`kubeflow_mcp/trainer/resources/troubleshooting.md`](../kubeflow_mcp/trainer/resources/troubleshooting.md).

---

## Next Steps

- [README](../README.md) — full CLI reference, env var table, tool list, observability setup
- [ARCHITECTURE.md](../ARCHITECTURE.md) — security model, trust boundaries, RBAC YAML, hardening checklist
- [examples/kubernetes/](../examples/kubernetes/) — complete in-cluster deployment with manifests
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development workflow, running tests, adding a new client module
- [Kubeflow Slack](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) — `#kubeflow-ml-experience` for questions
