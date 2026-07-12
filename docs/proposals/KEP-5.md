---
title: "SparkConnect (Spark) Client Support for kubeflow-mcp-server"
authors:
  - "@vedhakoushik"
status: provisional
creation-date: 2026-07-11
---

# KEP-5: SparkConnect (Spark) Client Support for kubeflow-mcp-server

This KEP proposes implementing the **Spark client module** for
`kubeflow-mcp-server`, exposing the SparkConnect session lifecycle as 6 MCP
tools across 4 categories. This implements the "Spark (Planned: Phase 6)" node
already identified in the architecture (module: `kubeflow_mcp.spark`), tracked
by [#5](https://github.com/kubeflow/mcp-server/issues/5) — the number this KEP
takes its name from, following the convention set by
[KEP-34](https://github.com/kubeflow/mcp-server/pull/48) (issue
[#34](https://github.com/kubeflow/mcp-server/issues/34)).

A reference implementation is available in
[#51](https://github.com/kubeflow/mcp-server/pull/51).

## Summary

This KEP proposes a **Spark client module** for `kubeflow-mcp-server`, exposing the
**SparkConnect session lifecycle** as **6 MCP tools** across 4 categories (Planning, Discovery,
Sessions, Monitoring). It implements the "Spark (Planned: Phase 6)" node identified in the
architecture (issue [#5](https://github.com/kubeflow/mcp-server/issues/5)) by wrapping the
already-released `kubeflow.spark.SparkClient` — the SparkConnect surface of the SDK's Spark
work — and mirrors the existing `trainer` client module.

Spark rounds out the data path: interactive Spark (SparkConnect) sessions are a common
precursor to training, so exposing them keeps the `prepare data -> train` loop inside the MCP
interface.

## Motivation

### Problem Statement

AI IDEs and orchestrator agents currently have no structured way to:

- Provision an interactive Spark (SparkConnect) session on Kubernetes from a natural-language
  request
- Inspect a session's state, driver pod, and connect info, or poll it to readiness
- Retrieve driver-pod logs to debug a failing session
- Tear a session down when finished

Today this requires leaving the MCP interface and using `kubectl` / the SDK directly.

### Architecture Context

```python
CLIENT_MODULES = {
    "trainer": "kubeflow_mcp.trainer",     # implemented
    "optimizer": "kubeflow_mcp.optimizer", # stub
    "hub": "kubeflow_mcp.hub",             # stub
    "spark": "kubeflow_mcp.spark",         # this KEP (6 tools)
}
```

ROADMAP Phase 6 lists **Spark** ([#5](https://github.com/kubeflow/mcp-server/issues/5),
[kubeflow/sdk#107](https://github.com/kubeflow/sdk/issues/107)). The module is opt-in via
`--clients trainer,spark`.

### SDK Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| `kubeflow` (unified SDK) | >= 0.4.0 | Provides `kubeflow.spark.SparkClient` via the `[spark]` extra (landed in SDK 0.4 — [kubeflow/sdk#225](https://github.com/kubeflow/sdk/pull/225)) |
| `kubeflow[spark]` / `pyspark` | pyspark major.minor must match cluster Spark (default 4.0.1) | Spark Connect gRPC client (port 15002) |
| Spark Operator (SparkConnect CRD) | `sparkoperator.k8s.io/v1alpha1` (kind `SparkConnect`) | Provisions SparkConnect servers |
| Kubeflow Trainer | >= v2.2.0 | Unchanged; Spark is independent but shares the SDK 0.4.0 baseline |
| Kubernetes | >= 1.27 | Matches existing MCP server requirement |
| Python | >= 3.10 | Matches existing MCP server requirement |

> **Note:** on the released 0.4.0/0.4.1 baseline, `SparkConnectInfo`'s driver-pod field is named
> `pod_name`; it was renamed to `driver_pod_name` only on unreleased SDK `main`. See
> "Serialize the released field name" under Key Design Decisions.

## Goals

- Guard `delete_spark_session` so non-`platform-admin` personas can only delete sessions created
  through MCP, mirroring the trainer's ownership check (see Key Design Decisions).
- Implement **6 MCP tools** across Planning, Discovery, Sessions, and Monitoring for the
  SparkConnect session lifecycle, including a `spark_pre_flight` readiness check (CRD, controller
  health, SDK importability) mirroring the trainer's `pre_flight()` and KEP-34's
  `katib_pre_flight`.
- Load the module (and its tool metadata) **without** requiring the optional `kubeflow[spark]`
  extra — all SDK imports are lazy.
- Apply the server's existing conventions: persona filtering, confirm-gate for mutations,
  destructive-tool policy, tool phases, and agent-facing resource guides.
- Update `SDK_COMPATIBILITY`, `TOOL_PHASES`, `TOOL_NEXT_HINTS`, and `PERSONAS` for the new tools.

## Non-Goals

- **Spark batch jobs (SparkJob / `SparkApplication`)** — depend on
  [kubeflow/sdk#521](https://github.com/kubeflow/sdk/pull/521), which is not yet in a released
  `kubeflow` version. A follow-up would add a parallel submit/list/get/logs/delete tool set
  under the same module.
- **Proxying Spark RPCs** through the MCP server — the server manages session *lifecycle*; the
  data plane attaches with PySpark using the returned connect info.
- **Attaching to a pre-existing external Spark Connect server** (`connect(base_url=…)` returns a
  live `SparkSession`, which is not serializable over MCP).
- **`follow=True` streaming logs** — a stateless MCP tool returns a snapshot, truncated
  client-side to the last `tail_lines` (see Risks for the retrieval-bound caveat).

## Proposal

### Module Structure

```
kubeflow_mcp/spark/
├── __init__.py        # MODULE_INFO, TOOLS, descriptions/annotations, resources, instruction sections
├── api/
│   ├── planning.py    # spark_pre_flight
│   ├── discovery.py   # list_spark_sessions, get_spark_session
│   ├── sessions.py    # create_spark_session, delete_spark_session
│   └── monitoring.py  # get_spark_session_logs
├── types/__init__.py  # SparkConnectInfo -> JSON-safe dict
└── resources/         # session-patterns.md, troubleshooting.md
```

### Key Design Decisions

- **Wrap only public, JSON-serializable methods.** `SparkConnectInfo` (a dataclass with an enum
  + datetime) is normalized to a plain dict. `connect()` returns a live pyspark `SparkSession`,
  which is handled specially (below).
- **Serialize the released field name, not `main`.** The released `kubeflow[spark]` baseline
  this KEP targets (0.4.0, 0.4.1) names the driver-pod field `SparkConnectInfo.pod_name`; the
  field was renamed to `driver_pod_name` only on unreleased SDK `main`. The serialization helper
  reads `pod_name` first and falls back to `driver_pod_name` (first non-`None` wins), so session
  info stays correct against the released baseline today and keeps working if/when the rename
  ships in a future release.
- **Lazy SDK imports.** `kubeflow.spark` / `pyspark` are imported inside the client factory and
  the create path only, so `import kubeflow_mcp.spark` works without the `[spark]` extra; a
  missing extra surfaces a friendly `SDK_ERROR`.
- **`create_spark_session` semantics.** The SDK's only public creation path is `connect()`
  (create mode), which provisions the `SparkConnect` CR, waits for readiness, and opens a driver
  connection. The tool provisions the session, **releases the transient `SparkSession`**, and
  returns session metadata + connect info; the data plane attaches separately. If `connect()`
  fails *after* the CR is created (e.g. the MCP host can't reach the driver), the tool reports
  the provisioned session with a warning rather than losing it.
  **Known limitation:** when the MCP server runs outside the cluster, `connect()` opens a
  `kubectl port-forward` subprocess to reach the driver; `SparkSession.stop()` does not
  terminate it, so repeated calls in an out-of-cluster deployment can leak subprocesses and
  listening ports (see Risks).
- **Ownership guard on delete.** `delete_spark_session` mirrors the trainer's `is_mcp_managed()`
  pattern (`kubeflow_mcp/common/utils.py`, `trainer/api/lifecycle.py`): `create_spark_session`
  labels the session with `MCP_MANAGED_LABEL` via the SDK's `Labels` option, and for
  non-`platform-admin` personas, `delete_spark_session` checks that label before deleting and
  rejects the request with a clear error if the session wasn't created through MCP. Without this,
  any persona with delete access could remove *any* SparkConnect session in an allowed namespace,
  not only MCP-created ones.
- **Namespace safety.** A `None` namespace resolves the effective namespace via the same resolver
  `check_namespace_allowed` uses, so the policy-checked namespace always matches the one the
  client operates in.
- **Mirror the `trainer` module** layout, metadata shape, and server registration.

### MCP Tools

#### Planning (phase `spark_planning`)

| Tool | Description | SDK Method |
|------|-------------|------------|
| `spark_pre_flight` | Verify the SparkConnect CRD (`sparkoperator.k8s.io/v1alpha1`) is installed, the Spark Operator controller is healthy, and `kubeflow[spark]` is importable | `ApiextensionsV1Api` + `CoreV1Api` + import check |

> **Scope note**: mirrors the trainer's `pre_flight()` and KEP-34's
> `katib_pre_flight` — a compound readiness check run first, before any other
> Spark tool. Unlike the trainer's 4-sub-check `pre_flight()`, this is
> narrower by design: it validates Spark-specific readiness only (CRD,
> controller, SDK availability), not cluster resource sizing (Spark sessions
> are user-sized via `num_executors`/`executor_resources`, not estimated from
> a model like `fine_tune()`).

#### Discovery (phase `spark_discovery`)

| Tool | Description | SDK Method |
|------|-------------|------------|
| `list_spark_sessions` | List SparkConnect sessions; filter by state / namespace | `list_sessions()` |
| `get_spark_session` | Get a session's state, driver pod, service, and connect info | `get_session()` |

#### Sessions (phase `spark_sessions`)

| Tool | Description | SDK Method |
|------|-------------|------------|
| `create_spark_session` | Create a session (two-phase confirm); returns connect info | `connect()` (create mode) |
| `delete_spark_session` | [DESTRUCTIVE] Delete a session (two-phase confirm; ownership-guarded for non-admin personas) | `delete_session()` |

#### Monitoring (phase `spark_monitoring`)

| Tool | Description | SDK Method |
|------|-------------|------------|
| `get_spark_session_logs` | Driver-pod logs, truncated to the last `tail_lines` after retrieval | `get_session_logs()` |

### Persona Coverage

| Persona | Spark tools |
|---------|-------------|
| `readonly` | `spark_pre_flight`, `list_spark_sessions`, `get_spark_session`, `get_spark_session_logs` |
| `data-scientist` (inherits readonly) | `create_spark_session`, `delete_spark_session` |
| `ml-engineer` / `platform-admin` | inherit the above / unrestricted |

`delete_spark_session` is added to `DESTRUCTIVE_TOOLS` and, like `delete_training_job`, is
ownership-guarded: non-`platform-admin` personas may only delete sessions created through MCP
(see "Ownership guard on delete" under Key Design Decisions).

### Tool Phase Categories

Four phases are added to `TOOL_PHASES`: `spark_planning`, `spark_discovery`, `spark_sessions`,
`spark_monitoring`. The module's `PHASE_TO_SECTION` maps them onto the server's existing
instruction slots (`planning` / `monitoring` / `training`) so guidance surfaces for the right
personas.

### Instruction Sections

The module contributes Spark guidance under the `planning` section (`spark_pre_flight` — run
first), the `monitoring` section (session discovery + log inspection), and the `training` section
(create → attach → delete lifecycle).

### SDK Compatibility Update

#### SDK Method → MCP Tool Mapping

| SDK Method (`SparkClient`) | MCP Tool | Notes |
|----------------------------|----------|-------|
| `list_sessions()` | `list_spark_sessions` | |
| `get_session()` | `get_spark_session` | |
| `get_session_logs()` | `get_spark_session_logs` | client-side truncation only — the SDK call retrieves the full log before we truncate to `tail_lines` (see Risks); `follow=True` not exposed |
| `connect()` (create mode) | `create_spark_session` | releases the transient `SparkSession`, returns metadata; labels the session for ownership tracking |
| `delete_session()` | `delete_spark_session` | destructive, confirm-gate, ownership-guarded |

#### Tools Beyond SDK Abstraction

- **`spark_pre_flight`**: `SparkClient` has no readiness-check method. MCP exposes this using
  `ApiextensionsV1Api` (CRD existence), `CoreV1Api` (Spark Operator controller pod health), and a
  Python import check (`kubeflow[spark]` availability) — the same direct-API pattern the trainer
  uses in `check_compatibility()`/`pre_flight()` and KEP-34 uses for `katib_pre_flight`.

#### SDK Compatibility Snippet

```python
"spark": {
    "status": "implemented",
    "sdk_client": "kubeflow.spark.SparkClient",
    "covered_methods": [
        "connect",  # create mode, via create_spark_session
        "list_sessions",
        "get_session",
        "delete_session",
        "get_session_logs",
    ],
    "uncovered_methods": [
        "connect(base_url=...)",          # attaching to an existing server returns a live,
                                          # non-serializable pyspark SparkSession
        "get_session_logs(follow=True)",  # streaming not exposed
    ],
    "k8s_api_operations": [
        "spark_pre_flight (ApiextensionsV1Api CRD check + CoreV1Api controller health)",
    ],
},
```

### Cross-Client Integration

Spark (data preparation) and Trainer (training) together cover a common ML loop within the MCP
interface. Both share the SDK 0.4.0 baseline and the same server conventions (personas,
confirm-gate, phases, instruction composition), so an agent can move between them without a
context switch.

#### TOOL_NEXT_HINTS

Concrete `_meta.next` hints for clients that don't consume server instructions or resources
(e.g. Ollama, custom agents), following the pattern already used for trainer tools:

| Tool | Hint |
|------|------|
| `spark_pre_flight` | If checks pass, call `create_spark_session()` to provision a session |
| `list_spark_sessions` | Call `get_spark_session(name)` for details on a specific session |
| `get_spark_session` | Use `get_spark_session_logs(name)` for driver output, or `delete_spark_session(name)` to tear down |
| `get_spark_session_logs` | If errors found, check the session state with `get_spark_session(name)` |
| `create_spark_session` | Use connect info to attach PySpark, then `pre_flight()` to set up training |
| `delete_spark_session` | Confirm removal with `list_spark_sessions()` |

`create_spark_session`'s hint is the cross-client link into the trainer module, surfacing the
`prepare data -> train` handoff described above. The reverse direction (a training-side hint back
toward `create_spark_session` for further data prep) would mean editing the trainer module's
existing `TOOL_NEXT_HINTS`, which is out of scope for this Spark-focused KEP — noted here as a
possible small follow-up rather than committed to.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `kubeflow[spark]` extra not installed on the server host | All SDK imports are lazy; a missing extra returns a friendly `SDK_ERROR` telling the user to install `kubeflow[spark]` |
| MCP host cannot reach the driver during `create` | Session is still provisioned; the tool returns metadata + a warning and the agent polls `get_spark_session` |
| pyspark ↔ cluster Spark version mismatch | Documented in the compatibility table and `spark://guides/troubleshooting` |
| XXL PR scope | Scoped to SparkConnect only; SparkJob deferred to a follow-up gated on kubeflow/sdk#521 |
| SDK API stability | Pinned to the released 0.4.0 baseline; covered/uncovered methods enumerated in `SDK_COMPATIBILITY` |
| `delete_spark_session` lacks an ownership check | Mirror the trainer's `is_mcp_managed()` pattern: label MCP-created sessions and reject deletion of unowned sessions for non-`platform-admin` personas |
| `get_session_logs()` retrieves the entire log before truncation | SDK 0.4.x calls `read_namespaced_pod_log` without a server-side tail limit, so today's `tail_lines` truncation is client-side only (no network/memory bound). Mitigation: bypass the SDK wrapper and call `CoreV1Api.read_namespaced_pod_log(tail_lines=...)` directly for a true server-side bound (trainer's `discovery.py` already uses `CoreV1Api` directly for a comparable case), or request SDK support for a `tail_lines` passthrough |
| Out-of-cluster deployment leaks `kubectl port-forward` subprocesses on repeated `create_spark_session` calls | `SparkSession.stop()` does not terminate the backend's port-forward process. Mitigation: recommend in-cluster deployment of the MCP server as the supported mode for `create_spark_session`; track an SDK enhancement for explicit port-forward handle/cleanup exposure |

## Testing Plan

### Unit Tests

- **Location**: `tests/unit/spark/`. Covers tool metadata/annotation consistency, persona +
  phase + destructive-policy wiring, `SparkConnectInfo` serialization (including the
  `pod_name`/`driver_pod_name` dual-field fallback), `spark_pre_flight` CRD/controller detection,
  and mocked-`SparkClient` behavior for all six tools (state filter, not-found →
  `RESOURCE_NOT_FOUND`, log tailing/truncation, confirm-gate previews, namespace resolution,
  ownership-guard accept/reject paths on `delete_spark_session` for non-admin personas). The SDK
  is mocked, so tests run **without** the `kubeflow[spark]` extra.
- **Server load**: `create_server(clients=["trainer", "spark"])` registers all six tools in both
  `full` and `progressive` modes.

### Integration Tests (deferred)

Not part of this KEP's initial implementation, but planned as a follow-up — mirroring the
trainer's and KEP-34's integration test plans, which run against a live cluster rather than a
mocked SDK:

- Kind cluster with the Spark Operator installed: `spark_pre_flight`, `create_spark_session`,
  poll to `Ready` via `get_spark_session`, `get_spark_session_logs`, `delete_spark_session`
- Ownership guard: a `data-scientist`-persona deletion of a non-MCP-created session is rejected;
  an MCP-created session can be deleted
- Cross-client: `create_spark_session` followed by the trainer's `pre_flight()`, demonstrating
  the `prepare data -> train` handoff

## References

- [#5 — Spark client (ROADMAP Phase 6)](https://github.com/kubeflow/mcp-server/issues/5)
- [#51 — Reference implementation](https://github.com/kubeflow/mcp-server/pull/51)
- [KEP-936: Kubeflow MCP Server](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server)
- [kubeflow/sdk#107 — Integrating the Kubeflow Spark Application with the SDK](https://github.com/kubeflow/sdk/issues/107)
- [kubeflow/sdk#225 — SparkClient API for SparkConnect session management](https://github.com/kubeflow/sdk/pull/225)
- [kubeflow/sdk#521 — Spark batch job (SparkJob) APIs (follow-up)](https://github.com/kubeflow/sdk/pull/521)
- [SparkConnect CRD — `sparkoperator.k8s.io`](https://github.com/kubeflow/sdk/blob/main/hack/crds/sparkoperator.k8s.io_sparkconnects.yaml)
- [KEP-34: Katib (Optimizer) Client — sibling proposal (#48)](https://github.com/kubeflow/mcp-server/pull/48)
