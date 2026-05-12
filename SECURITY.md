# Security Policy

## Project Status

This project is in **early development** and has not yet reached a stable release (1.0). The API and features may change between versions.

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 0.x     | ✅ Yes    | Pre-release, actively developed |

Once the project reaches 1.0, a formal support policy for stable releases will be established.

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

### GitHub Security Advisories (Preferred)

1. Go to the [Security tab](https://github.com/kubeflow/mcp-server/security)
2. Click "Report a vulnerability"
3. Fill out the form with details

### Email

- **Email:** [kubeflow-discuss@googlegroups.com](mailto:kubeflow-discuss@googlegroups.com)
- **Subject:** `[SECURITY] kubeflow/mcp-server — <brief description>`

### What to Include

- Type of vulnerability (e.g., injection, authentication bypass)
- Location of the affected code (file path, line number)
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity (critical: ASAP, high: 2 weeks, medium: 1 month)

## Security Model

The MCP server is a **translation layer** between AI agents and the Kubeflow SDK. For HTTP transport, it supports bearer token and JWT/OIDC authentication (see `core/auth.py`). For Kubernetes API access, it inherits the identity of the process running it — per-user identity mapping is planned (ROADMAP Step 3).

### Trust Boundaries

```
AI Agent (Claude, Cursor, etc.)
    │
    ▼
MCP Server process  ← YOU ARE HERE (kubeflow-mcp)
    │
    ▼
Kubeflow SDK (TrainerClient)
    │
    ▼
Kubernetes API Server (RBAC enforced)
```

### What the MCP Server Controls

- **Persona-based tool filtering** — restricts which tools are visible to the AI agent (default: `--persona readonly`, which hides all write tools)
- **Policy file** — `~/.kf-mcp-policy.yaml` can further restrict tools and namespaces
- **Two-phase confirmation** — write tools require `confirmed=True` (preview first, submit after)
- **Input validation** — K8s name format, CPU/memory format, resource limits, training parameter bounds (batch_size, epochs, nodes, GPU count, LoRA rank, script size, package count)
- **Namespace restrictions** — policy enforcement on both lifecycle and training tools (training tools use per-call `TrainerClient` with `KubernetesBackendConfig(namespace=...)`)
- **Error sanitization** — stack traces are only included in error responses when the logger is at DEBUG level; production responses contain error messages only
- **External API hardening** — HuggingFace Hub calls use a 10s timeout and model ID format validation to prevent SSRF
- **Thread safety** — `RateLimiter` uses `threading.Lock`; policy cache uses `functools.lru_cache` (GIL-safe)
- **Audit logging** — every tool call is logged with masked parameters, duration, and correlation ID

### What the MCP Server Does NOT Control

- **Kubernetes RBAC** — the server operates under the caller's permissions; it cannot grant access beyond what the Kubeconfig allows
- **Network security** — TLS, ingress, and API gateway configuration are infrastructure concerns
- **Secret management** — the server does not store credentials; it reads Kubeconfig from the environment

## Known Security Considerations

### 1. `run_custom_training` — Host Code Execution

**Severity: High** | **Status: By design, documented**

The `run_custom_training` tool accepts a Python script as a string. The script is embedded inside a `train_func()` closure that is serialized by `cloudpickle` and executed **inside the Kubernetes training pod**, not on the MCP host.

The `exec(compile(script, ...))` call is **deferred** — it runs inside `train_func()` at pod runtime, not at submission time on the host. However, the closure is constructed on the host and serialized, so malformed scripts could theoretically cause issues during serialization.

**Mitigations in place:**
- `exec()` is called with `{"__builtins__": __builtins__}` — the **full** Python builtins are available (no language-level sandboxing); no additional globals are injected
- An AST-based safety check (`is_safe_python_code`) walks the parsed AST and flags: dangerous calls (`eval`, `exec`, `compile`, `__import__`), dangerous module calls (`os.system`, `os.popen`, `subprocess.*`, `shutil.rmtree`), dangerous imports (`ctypes`, `socket`), and dunder attribute access (`__builtins__`, `__subclasses__`, `__globals__`, `__code__`)
- The safety check runs on **both** preview and confirmed paths; `safety_warnings` are included in the preview response
- On the confirmed (submit) path, unsafe scripts are **blocked by default** with a `VALIDATION_ERROR`. Set `KUBEFLOW_MCP_UNSAFE_SCRIPTS=true` to override (logged with a warning)

**Mitigations NOT in place:**
- The AST check is bypassable via indirection (e.g. `getattr`, string concatenation, `importlib`)
- There is no sandbox, seccomp profile, or process isolation around the `exec()` call inside the pod
- The script executes with full privileges of the training pod's ServiceAccount

**Recommendation:** In production, restrict `run_custom_training` to trusted users via persona policy (`--persona data-scientist` or higher). Consider running the MCP server process with minimal filesystem and network permissions.

### 2. HTTP Transport — Authentication

**Severity: Medium** | **Status: Mitigated (API key + JWT implemented)**

`kubeflow-mcp serve --transport http` exposes the MCP server over StreamableHTTP. Authentication is available via `--auth-token` (API key) or `KUBEFLOW_MCP_JWKS_URI` (JWT/OIDC). The server logs a warning if HTTP transport runs without any auth configured.

**Recommendation:** Always configure `--auth-token` or JWT for HTTP transport. Use stdio transport for local development. For production, place behind an authenticating reverse proxy with TLS for defense-in-depth.

### 3. Identity — Per-User Mapping

**Severity: Medium** | **Status: Open, tracked in [ROADMAP.md](ROADMAP.md) Step 3**

`core/auth.py` implements HTTP-layer authentication (API key via `--auth-token`, JWT/OIDC via `KUBEFLOW_MCP_JWKS_URI`). However, authenticated identities are not yet mapped to distinct Kubernetes identities. All tool calls run under whatever Kubeconfig the process was started with.

**Impact:** In multi-user deployments, all users share the same Kubernetes identity. There is no per-user RBAC enforcement at the MCP layer (the auth layer verifies *who* is calling, but cannot scope *what* they can do beyond persona-level filtering).

**Recommendation:** For single-user local development (stdio), this is acceptable — the user's own Kubeconfig is used. For multi-user HTTP deployments, configure `--auth-token` or JWT and enforce identity at the ingress/gateway layer until per-user Kubernetes impersonation is wired (Step 3).

## Recommended Deployment Practices

```yaml
# 1. Run in isolated namespace
apiVersion: v1
kind: Namespace
metadata:
  name: kubeflow-mcp
  labels:
    pod-security.kubernetes.io/enforce: restricted

---
# 2. Use NetworkPolicy to restrict access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-server-ingress
  namespace: kubeflow-mcp
spec:
  podSelector:
    matchLabels:
      app: kubeflow-mcp
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              trusted: "true"
      ports:
        - port: 8000
```

## RBAC Configuration

### Minimum ClusterRole for MCP Server

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubeflow-mcp-server
rules:
  # Planning: get_cluster_resources
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["list"]

  # Discovery + Monitoring: list/get jobs, logs, events
  - apiGroups: ["trainer.kubeflow.org"]
    resources: ["trainjobs"]
    verbs: ["list", "get", "create", "delete", "patch"]
  - apiGroups: [""]
    resources: ["pods", "pods/log", "events"]
    verbs: ["list", "get"]

  # Discovery: list/get runtimes
  - apiGroups: ["trainer.kubeflow.org"]
    resources: ["clustertrainingruntimes"]
    verbs: ["list", "get"]
```

### Read-Only ClusterRole

For `--persona readonly` or monitoring-only deployments:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubeflow-mcp-readonly
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["list"]
  - apiGroups: ["trainer.kubeflow.org"]
    resources: ["trainjobs", "clustertrainingruntimes"]
    verbs: ["list", "get"]
  - apiGroups: [""]
    resources: ["pods", "pods/log", "events"]
    verbs: ["list", "get"]
```

### ServiceAccount with Minimal Permissions

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kubeflow-mcp
  namespace: kubeflow-mcp
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kubeflow-mcp-binding
  namespace: kubeflow-mcp
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-mcp-server
subjects:
  - kind: ServiceAccount
    name: kubeflow-mcp
    namespace: kubeflow-mcp
```

### Namespace-Scoped (recommended for multi-tenant)

Replace `ClusterRole` / `ClusterRoleBinding` with `Role` / `RoleBinding` in specific namespaces, and configure `~/.kf-mcp-policy.yaml`:

```yaml
policy:
  namespaces:
    - team-a
    - team-b
```

## Dependency Security

- **Kubeflow SDK** (`kubeflow>=0.4.0`) — pinned range, tracks upstream releases
- **FastMCP** (`fastmcp>=2.0.0`) — MCP protocol implementation
- **Kubernetes client** (`kubernetes`) — transitive via SDK
- **No credentials stored** — the server reads Kubeconfig from the environment; HuggingFace tokens are passed per-request via `hf_token` parameter

## Hardening Checklist

For production deployments:

- [ ] Default persona is `readonly` — explicitly set `--persona ml-engineer` or `--persona data-scientist` only for users who need write access
- [ ] Configure `~/.kf-mcp-policy.yaml` with `policy.namespaces` and `read_only: true` if appropriate
- [ ] For HTTP transport: set `--auth-token` (dev) or `KUBEFLOW_MCP_JWKS_URI` (production JWT) — the server warns if HTTP runs without auth
- [ ] Use stdio transport for local dev; place HTTP behind an authenticated reverse proxy for additional defense-in-depth
- [ ] Bind the MCP server ServiceAccount to the minimum ClusterRole above
- [ ] Do not grant `run_custom_training` access to untrusted users
- [ ] Keep log level at INFO or above in production (DEBUG exposes stack traces in error responses)
- [ ] Ensure every new tool has a `CLIENT_TOOL_ANNOTATIONS` entry — `read_only` mode is fail-closed: tools without an explicit `readOnlyHint: True` annotation are treated as write tools and excluded
- [ ] Review audit logs (`tool_call` events with `"audit": true`) for unexpected tool usage
- [ ] Pin `kubeflow-mcp` to a specific version in production

## Resilience

All tool calls pass through a rate limiter and per-tool circuit breakers:

- **Rate limiter** (token bucket): prevents runaway agent loops. Configurable via `KUBEFLOW_MCP_RATE_LIMIT` (default 10 req/s) and `KUBEFLOW_MCP_RATE_CAPACITY` (default 20 burst).
- **Circuit breaker** (per-tool): trips on repeated K8s/SDK infrastructure errors (not validation errors). Auto-recovers after `KUBEFLOW_MCP_CB_RECOVERY_TIMEOUT` (default 30s). Prevents cascading failures when the K8s API is degraded.

## Security Roadmap

See [ROADMAP.md](ROADMAP.md) for planned security enhancements.
