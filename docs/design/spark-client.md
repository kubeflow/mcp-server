# Design: Spark Client Module (SparkConnect)

| | |
|---|---|
| **Status** | Proposed |
| **Issue** | [#5](https://github.com/kubeflow/mcp-server/issues/5) (ROADMAP Phase 6) |
| **SDK design reference** | [KEP-107 — Integrating the Kubeflow Spark Application with the Kubeflow SDK](https://github.com/kubeflow/sdk/issues/107) |
| **Related SDK work** | [kubeflow/sdk#521 — Spark batch job (SparkJob) submission & lifecycle APIs](https://github.com/kubeflow/sdk/pull/521) (in progress) |
| **Depends on** | `kubeflow[spark]` ≥ 0.4.0 (already released; SparkConnect surface of KEP-107) |

## Summary

This adds a `spark` client module to the Kubeflow MCP Server that exposes the
**SparkConnect session lifecycle** as MCP tools, so AI agents can create,
inspect, monitor, and tear down interactive Spark sessions on Kubernetes through
natural language. It wraps the already-released `kubeflow.spark.SparkClient` and
mirrors the existing `trainer` client module's architecture (`api/`, `types/`,
`resources/`, persona/phase wiring).

## Goals

- Expose the **public, JSON-serializable** `SparkClient` surface as MCP tools.
- Match the server's existing conventions: persona filtering, confirm-gate for
  mutations, destructive-tool policy, tool phases, agent-facing resource guides.
- Load without the optional `kubeflow[spark]` extra (lazy SDK imports), so the
  server and its tool metadata are unaffected when Spark isn't installed.

## Non-goals (explicitly out of scope for this PR)

- **Spark batch jobs (`SparkApplication` / SparkJob).** The SDK's batch-job APIs
  are still in progress in [kubeflow/sdk#521](https://github.com/kubeflow/sdk/pull/521)
  and are **not** in a released `kubeflow` version. SparkJob tools are a
  **follow-up PR gated on that SDK work landing** (see [Scope & phasing](#scope--phasing)).
- **Proxying Spark RPCs through the MCP server.** The server manages the session
  *lifecycle*; the data plane (a notebook/job/tool) attaches with PySpark using
  the returned connect info.
- Attaching to a pre-existing external Spark Connect server (`connect(base_url=…)`
  returns a live, non-serializable `SparkSession`) and `follow=True` log streaming.

## Background: SparkConnect vs SparkJob

KEP-107 covers two complementary surfaces on `SparkClient`:

1. **SparkConnect sessions** — interactive, long-lived Spark Connect servers
   (`SparkConnect` CRD). **Released today** via `connect()`, `list_sessions()`,
   `get_session()`, `delete_session()`, `get_session_logs()`.
2. **SparkJob / batch** — submit-and-run `SparkApplication` batch jobs. **Not yet
   released** — in progress in kubeflow/sdk#521.

This PR implements (1). (2) is a clean follow-up once the SDK exposes it.

## Tool surface

Five tools, grouped by phase and mapped 1:1 to public SDK methods:

| Tool | Phase | Kind | SDK method |
|---|---|---|---|
| `list_spark_sessions` | `spark_discovery` | read-only | `SparkClient.list_sessions()` |
| `get_spark_session` | `spark_discovery` | read-only | `SparkClient.get_session()` |
| `get_spark_session_logs` | `spark_monitoring` | read-only | `SparkClient.get_session_logs()` |
| `create_spark_session` | `spark_sessions` | write (confirm-gate) | `SparkClient.connect()` (create mode) |
| `delete_spark_session` | `spark_sessions` | write, destructive (confirm-gate) | `SparkClient.delete_session()` |

- **Personas:** `readonly` gets the three read tools; `data-scientist`+ additionally
  get `create`/`delete`. `delete_spark_session` is in `DESTRUCTIVE_TOOLS`.
- **Confirm-gate:** both write tools preview with `confirmed=False`, execute with
  `confirmed=True` (consistent with the trainer tools).
- **Resources:** two agent-facing guides — `spark://guides/session-patterns` and
  `spark://guides/troubleshooting`.

## Design decisions

1. **Wrap only public, JSON-serializable methods.** `SparkConnectInfo` (dataclass
   with an enum + datetime) is normalized to a plain dict via a small `types`
   helper. `connect()` returns a live pyspark `SparkSession`, which is not
   serializable over MCP — see below.
2. **Lazy SDK imports.** `kubeflow.spark`/`pyspark` are imported inside the client
   factory and the create path only, so `import kubeflow_mcp.spark` (and the
   module's metadata) works without the `kubeflow[spark]` extra; a missing extra
   surfaces a friendly `SDK_ERROR` telling the user to install it.
3. **`create_spark_session` semantics.** The SDK's only public creation path is
   `connect()` (create mode), which provisions the `SparkConnect` CR, waits for
   readiness, and opens a driver connection. The tool provisions the session,
   **immediately releases** the transient `SparkSession`, and returns session
   **metadata + connect info**; the data plane attaches separately. If `connect()`
   fails *after* the CR is created (e.g. the MCP host can't reach the driver), the
   tool still reports the provisioned session with a warning rather than losing it.
   A provision-only SDK method would remove the host-reachability requirement —
   noted as future SDK work.
4. **Namespace safety.** A `None` namespace resolves the effective namespace via
   the same resolver `check_namespace_allowed` uses, so the policy-checked
   namespace always matches the one the client operates in (no allowlist bypass).
5. **Mirror the `trainer` module.** Same package layout, metadata shape
   (`MODULE_INFO`, `TOOLS`, `CLIENT_TOOL_DESCRIPTIONS/ANNOTATIONS`,
   `CLIENT_RESOURCES`, `INSTRUCTION_SECTIONS`, `PHASE_TO_SECTION`) and server
   registration via `CLIENT_MODULES`, selectable with `--clients trainer,spark`.

## Testing

33 unit tests (metadata/annotation consistency, persona + phase + destructive
wiring, `SparkConnectInfo` serialization, and mocked-client behavior for all five
tools incl. not-found → `RESOURCE_NOT_FOUND`, log tailing/truncation, confirm-gate
previews, and namespace resolution). The SDK is mocked, so tests run without the
`kubeflow[spark]` extra. Full unit suite passes and the server loads + registers
all five tools in `full` and `progressive` modes.

## Scope & phasing

- **This PR = SparkConnect only.** It matches the SDK surface that is *already
  released*, keeping the change reviewable and immediately usable.
- **SparkJob is out of scope here.** Spark batch jobs (`SparkApplication`) depend
  on [kubeflow/sdk#521](https://github.com/kubeflow/sdk/pull/521), which is not yet
  in a released `kubeflow` version, so they are left as potential follow-up work
  once that SDK support lands — a parallel submit/list/get/logs/delete tool set
  under the same module, reusing these patterns.
- **Dependencies.** This PR depends only on the released `kubeflow[spark]`
  SparkConnect APIs.

## Future work

- SparkJob / `SparkApplication` tools (after kubeflow/sdk#521).
- Provision-only creation path if/when the SDK offers one (avoids the MCP host
  opening a driver connection).
- Streaming logs and attach-to-existing, if a serializable MCP-friendly shape is
  defined.
