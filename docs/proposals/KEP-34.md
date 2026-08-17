---
title: "Katib (Optimizer) Client Support for kubeflow-mcp-server"
authors:
  - "@krishnagupta"
status: implementable
creation-date: 2025-06-02
last-updated: 2026-08-12
---

# KEP-34: Katib (Optimizer) Client Support for kubeflow-mcp-server

## Summary

This KEP proposes implementing the **Optimizer client module** for
`kubeflow-mcp-server`, exposing Katib's Experiment, Trial, and Suggestion
lifecycle as 17 MCP tools across 5 categories. This implements the
Phase 2 optimizer node (module: `kubeflow_mcp.optimizer`) already identified
in the architecture (see Architecture Context below).

Katib is the natural second client after TrainerClient because hyperparameter
tuning is the immediate next step after training infrastructure is in place,
completing the inner loop of `train -> evaluate -> tune -> retrain` without
leaving the MCP interface.

## Motivation

### Problem Statement

AI IDEs and orchestrator agents currently have no structured way to:

1. Launch Katib HPO experiments from natural language descriptions
2. Inspect experiment progress, individual trial results, or suggestion
   algorithm status
3. Retrieve the best hyperparameter configuration from a completed experiment
4. Integrate HPO into automated ML pipelines managed by agents

The existing stub (`kubeflow_mcp.optimizer`) declares 8 planned tools with
`status: "stub"` but contains no implementations.

### Architecture Context

```
CLIENT_MODULES = {
    "trainer": "kubeflow_mcp.trainer",     # Phase 1: Implemented (23 tools)
    "optimizer": "kubeflow_mcp.optimizer", # Phase 2: This KEP (17 tools)
    "hub": "kubeflow_mcp.hub",            # Phase 3: Planned
}
```

The `--clients optimizer` CLI flag already exists. When the optimizer module
exposes a non-empty `TOOLS` list, `server.py` automatically registers tools
with the existing audit, rate-limiting, circuit-breaking, persona filtering,
and namespace enforcement infrastructure.

### SDK Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| `kubeflow` (unified SDK) | >= 0.4.0 | Provides `kubeflow.optimizer.OptimizerClient`; bundles `kubeflow_katib_api` (v1beta1 CRD types) |
| Kubeflow Trainer | >= v2.2.0 | Required when trial templates use TrainJobs |
| Kubernetes | >= 1.27 | Matches existing MCP server requirement |
| Python | >= 3.10 | Matches existing MCP server requirement |

## Goals

1. Implement 17 MCP tools across 5 categories for Katib experiment, trial,
   and suggestion lifecycle (see MCP Tools tables)
2. Decompose experiment creation into agent-friendly tools, following the
   trainer's `fine_tune`/`run_custom_training`/`run_container_training` pattern
3. Use `kubeflow.optimizer.OptimizerClient` as primary interface, with
   `CustomObjectsApi` + `kubeflow_katib_api` models fallback where the SDK
   lacks coverage (early stopping, raw spec creation, suggestions)
4. Follow TrainerClient structural patterns and integrate with existing server
   infrastructure (personas, audit, rate limiting, circuit breaker, namespaces)
5. Two-phase confirmation for all mutating operations
6. Unit tests with at least 80% coverage, integration tests against live Katib
7. Update `SDK_COMPATIBILITY`, `TOOL_PHASES`, `TOOL_NEXT_HINTS`, and `PERSONAS`

## Non-Goals

- **NAS (Neural Architecture Search)**: Follow-up after HPO.
- **Custom suggestion algorithm deployment**: Users install independently.
- **Katib UI replacement**: MCP tools complement the dashboard.
- **Katib DB Manager direct access**: `get_trial_metrics()` requires gRPC;
  trial metrics are available via CRD status instead.
- **Legacy `tune()` API wrapping**: Was a `KatibClient` method requiring Python
  callables; not serializable via MCP. No equivalent in `OptimizerClient`.
- **Budget patching on running jobs**: `TrialConfig` is set at creation time;
  dynamic budget changes require `CustomObjectsApi` patch. Deferred to a follow-up.

## Proposal

### Module Structure

```
kubeflow_mcp/optimizer/
  __init__.py          # MODULE_INFO, TOOLS, descriptions, annotations, resources, instructions
  api/
    __init__.py        # Re-exports
    planning.py        # katib_pre_flight
    optimization.py    # create_hpo_experiment, create_experiment_from_spec
    discovery.py       # list_experiments, get_experiment, get_experiment_status,
                       # get_trial, get_successful_trials, list_suggestions
    monitoring.py      # get_experiment_trials, get_best_trial, get_suggestion,
                       # wait_for_experiment, get_experiment_trial_logs,
                       # get_experiment_events
    lifecycle.py       # delete_experiment, update_experiment
  constants/
    __init__.py        # Katib CRD constants (group, version, plurals, conditions)
  types/
    __init__.py        # Katib-specific type helpers
  resources/
    hpo-patterns.md    # HPO workflow patterns and algorithm selection guide
    troubleshooting.md # Common Katib errors and fixes
```

> **Note — Stub alignment**: The existing `optimizer/api/__init__.py` stub
> declares planned tools (e.g., `create_optimization_job`) using the confirmed
> pattern. The tool names proposed in this KEP (`create_hpo_experiment`, etc.)
> intentionally replace the stub placeholders to align with Katib SDK semantics.
> The stub's contributor guide and confirmed-pattern example remain valid for
> the implementation phase.

### Key Design Decisions

**Experiment creation cannot go through `OptimizerClient.optimize()`.** The
method takes a `TrainJobTemplate` whose `trainer` must be a `CustomTrainer`
holding a live Python callable in its `func` field, and the Kubernetes backend
unconditionally mutates `trial_template.trainer.func_args` to inject
`${trialParameters.*}` references. Three consequences make it unusable from MCP:

1. A callable cannot cross the MCP/JSON boundary. The image-based
   `CustomTrainerContainer` has no `func_args` attribute, so passing one raises
   `AttributeError` inside the backend.
2. `optimize()` generates its own Experiment name (`uuid4().hex[:11]`), so a
   caller-supplied name cannot be honoured.
3. It sets no metadata labels, so the MCP ownership label that
   `delete_experiment` and `update_experiment` enforce could never be applied.

Both creation tools therefore build a `V1beta1Experiment` with
`kubeflow_katib_api` models and submit it through `CustomObjectsApi`:

- `create_hpo_experiment`: Flat parameters (objective, search space, algorithm,
  trial template, trial budget). The SDK's `Search.uniform/loguniform/choice`
  helpers are still used to encode the feasible space, so parameter
  serialisation stays identical to the SDK's own output.
- `create_experiment_from_spec`: Full `V1beta1Experiment` JSON spec for
  advanced use cases (early stopping, custom metrics collectors, resume
  policies, NAS configs). Validates the spec via
  `kubeflow_katib_api.models.V1beta1Experiment.from_dict()` before submitting.

Both use two-phase confirmation, stamp the MCP ownership label, and produce the
same response schema.

A side benefit: building the CR directly reaches every Katib suggestion
algorithm (`random`, `grid`, `bayesianoptimization`, `tpe`, `multivariate-tpe`,
`cmaes`, `sobol`, `hyperband`), whereas the SDK's typed algorithm classes only
express `RandomSearch` and `GridSearch`.

**Client factory**: `get_optimizer_client()` singleton with per-namespace variant
in `common/utils.py`. Lazy import -- `kubeflow.optimizer` is only imported when
`--clients optimizer` is active.

**RBAC**: Service account needs get/list/create/delete/patch on `experiments`,
`trials`, `suggestions` and get/list on `pods/log`.

### MCP Tools


#### Planning

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `katib_pre_flight` | Verify Katib CRD, controller health, suggestion algorithm availability, and namespace-level Katib config (e.g., default suggestion image, resource quotas) | `CustomObjectsApi` + `CoreV1Api` |

> **Scope note**: The trainer's `pre_flight()` is a compound tool covering
> 4 sub-checks (compatibility, cluster resources, estimate, runtimes). This
> `katib_pre_flight` is narrower by design — it validates Katib-specific
> readiness only. Namespace-level Katib configuration (default suggestion
> images, controller config) is included in the check to ensure experiments
> can be created in the target namespace.

#### Optimization

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `create_hpo_experiment` | Create HPO experiment from flat params (objective, search space, algorithm, trial template, trial budget) | `CustomObjectsApi` + `kubeflow_katib_api` models (see Key Design Decisions) |
| `create_experiment_from_spec` | Create experiment from complete V1beta1Experiment JSON spec, incl. early stopping | `CustomObjectsApi` + `kubeflow_katib_api` models |

#### Discovery

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `get_experiment` | Experiment status, trial counts, embedded trials, search space, algorithm, objectives, trial config, plus conditions, best trial and early stopping read from the CR | `OptimizerClient.get_job()` + `CustomObjectsApi` |
| `list_experiments` | List experiments with optional status filter | `OptimizerClient.list_jobs()` |
| `get_experiment_status` | Lightweight status-only check (status string + trial counts) | `OptimizerClient.get_job()` → extract `.status` + trial counts |
| `get_trial` | Detailed trial status, parameters, and metrics by name | `OptimizerClient.get_job()` → filter `.trials` by name |
| `get_successful_trials` | All succeeded trials with hyperparameters and metrics | `OptimizerClient.get_job()` → filter `.trials` by status |
| `list_suggestions` | List suggestion resources in namespace | `CustomObjectsApi` |

#### Monitoring

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `get_experiment_trials` | List trials with status filter (includes `EarlyStopped`) | `OptimizerClient.get_job()` → `.trials` |
| `get_best_trial` | Best trial with hyperparameters and metrics | `OptimizerClient.get_best_results()` |
| `get_suggestion` | Suggestion algorithm status and count | `CustomObjectsApi` |
| `wait_for_experiment` | Poll until terminal state | `OptimizerClient.wait_for_job_status()` |
| `get_experiment_trial_logs` | Pod logs with failure pattern detection | `OptimizerClient.get_job_logs()` |
| `get_experiment_events` | K8s events for experiment and trials | `OptimizerClient.get_job_events()` |

#### Lifecycle

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `delete_experiment` | Delete experiment and associated trials/suggestions | `OptimizerClient.delete_job()` |
| `update_experiment` | Suspend or resume experiment (patch `parallelTrialCount`) | `CustomObjectsApi` get + patch |

> **Clarification on `update_experiment` semantics**: Katib has no first-class
> `suspend` field, unlike the trainer's `update_training_job(action="suspend")`
> which patches TrainJob's native `suspend`. `update_experiment` therefore
> pauses an experiment by setting `spec.parallelTrialCount` to 0 — running
> trials finish, but no new ones start.
>
> It deliberately does **not** touch `spec.resumePolicy`. That field
> (`Never`, `FromVolume`, `LongRunning`) governs whether an experiment may
> restart *after completion*, which is a different concern from a mid-run
> pause; changing it as a side effect of suspending would silently alter
> post-completion behaviour.
>
> Because `parallelTrialCount` is destroyed by the suspend, the pre-suspend
> value is recorded in the `kubeflow-mcp/pre-suspend-parallel-trial-count`
> annotation so `action="resume"` restores the original concurrency instead of
> guessing. Suspend is idempotent, and resume never restores 0.

All read-only tools: `readOnlyHint=True`, `idempotentHint=True`.
Mutating tools: `create_*` are not idempotent; `delete_experiment` is destructive; `update_experiment` is idempotent.
`wait_for_experiment`: `readOnlyHint=True`, blocking. Default `polling_interval=15s`, `timeout_seconds` capped at 3600s. Agents should prefer `get_experiment_status()` for non-blocking polling.

`create_hpo_experiment`, `create_experiment_from_spec`, `list_suggestions`,
`get_suggestion`, `update_experiment`, and `katib_pre_flight` use
`CustomObjectsApi`/`CoreV1Api` directly where `OptimizerClient` lacks coverage.
`get_experiment_events` is covered by `OptimizerClient.get_job_events()`.

**Bounded collection responses**: every tool that returns a collection
(`list_experiments`, `get_experiment` — its embedded trials —
`get_successful_trials`, `list_suggestions`, `get_experiment_trials`,
`get_experiment_events`) accepts a `limit` parameter, defaulting to 50 and
capped at 500, matching the trainer client. An experiment may run up to 1000
trials, so unbounded responses would exhaust the agent's context window. The
`total` field always reports the true count so truncation stays visible, and
`get_experiment` additionally sets `trials_truncated`.

**Namespace policy**: optimizer tools resolve an omitted namespace through the
*optimizer* client before checking it against the allowlist. Resolving it
through the trainer client would validate a different namespace than the one
the operation targets, which both allows the allowlist to be bypassed and
breaks `--clients optimizer` deployments where no TrainerClient exists.

### Persona Coverage

| Persona | Optimizer Tools |
|---------|----------------|
| `readonly` | All read-only tools (13 tools: planning + discovery + monitoring) |
| `data-scientist` | readonly + `create_hpo_experiment`, `delete_experiment` (MCP-owned only) |
| `ml-engineer` | data-scientist + `create_experiment_from_spec`, `update_experiment` |
| `platform-admin` | all (wildcard) |

### Tool Phase Categories

```python
TOOL_PHASES.update({
    "optimizer_planning": ["katib_pre_flight"],
    "optimizer_discovery": [
        "list_experiments", "get_experiment", "get_experiment_status",
        "get_trial", "get_successful_trials", "list_suggestions",
    ],
    "optimizer_optimization": [
        "create_hpo_experiment", "create_experiment_from_spec",
    ],
    "optimizer_monitoring": [
        "get_experiment_trials", "get_best_trial", "get_suggestion",
        "wait_for_experiment", "get_experiment_trial_logs",
        "get_experiment_events",
    ],
    "optimizer_lifecycle": ["delete_experiment", "update_experiment"],
})

PHASE_TO_SECTION = {
    "optimizer_planning": "optimizer_planning",
    "optimizer_discovery": None,
    "optimizer_optimization": "optimizer_optimization",
    "optimizer_monitoring": "optimizer_monitoring",
    # lifecycle instructions folded into the monitoring section
    "optimizer_lifecycle": "optimizer_monitoring",
}
```

> **Section names must be namespaced.** The trainer module already defines
> sections called `planning`, `monitoring` and `training`; reusing those names
> would make the two modules' instruction sections collide.
>
> Ordering comes from each client's `SECTION_ORDER` export, which `server.py`
> concatenates in `CLIENT_MODULES` order, so a new client never edits core. A
> section from a module that exports no `SECTION_ORDER` still surfaces, sorted,
> keeping the result deterministic. The optimizer therefore exports:
>
> ```python
> SECTION_ORDER = [
>     "optimizer_planning",
>     "optimizer_optimization",
>     "optimizer_monitoring",
> ]
> ```

### Instruction Sections

**Planning** (all personas):
```
PLANNING (always do first):
- katib_pre_flight() -> Verify Katib CRD, controller, suggestion algorithms
- If blockers returned, STOP and inform user
- When both clients active, pre_flight() covers cluster/GPU;
  katib_pre_flight() covers Katib-specific readiness
DISCOVERY:
- list_experiments() -> find existing experiments
- get_experiment(name) -> inspect config, status, trial counts
- list_suggestions() -> inspect Suggestion CRs (algorithm state) when stuck
```

**Optimization** (data-scientist+):
```
TOOL SELECTION:
- Simple HPO -> create_hpo_experiment()
  - Search types: uniform, loguniform, choice
  - Algorithms: random (default), grid, bayesianoptimization, tpe,
    multivariate-tpe, cmaes, sobol, hyperband
  - trial_template is the Katib trialSpec; reference tuned params
    as ${trialParameters.<name>}
- Early stopping / metrics collectors / resume policies / NAS
  -> create_experiment_from_spec()
- Both require confirmed=True to submit; False returns preview
RULES:
- ALWAYS preview first (confirmed=False)
- max_trial_count is required, no unbounded default (capped at 1000)
- parallel_trial_count is capped at 100
- Trial templates can reference TrainJobs — use list_runtimes() first
```

**Monitoring** (all personas):
```
MONITORING AND LIFECYCLE:
- get_experiment_status(name) -> lightweight polling
- get_experiment(name) -> full status with best trial and conditions
- get_experiment_trials(name) -> trials with status filter
- get_best_trial(name) -> optimal hyperparameters
- get_successful_trials(name) -> all succeeded trials for comparison
- get_experiment_trial_logs(experiment, trial) -> pod logs with failure hints
- get_experiment_events(name) -> K8s events for debugging scheduling
- wait_for_experiment(name) -> block until Succeeded/Failed (blocks MCP)
- update_experiment(name, action="suspend"|"resume") -> pause/resume
- delete_experiment(name, confirmed=True) -> remove permanently (preview first)
```

### SDK Compatibility Update

#### SDK Method → MCP Tool Mapping

The following table shows the mapping between `OptimizerClient` SDK methods and
the MCP tools that wrap them:

| SDK Method (`OptimizerClient`) | MCP Tool(s) | Notes |
|-------------------------------|-------------|-------|
| `optimize()` | *(none)* | **Not usable from MCP** — requires a Python callable, generates its own name, sets no labels. See Key Design Decisions |
| `get_job()` | `get_experiment`, `get_experiment_status`, `get_trial`, `get_successful_trials`, `get_experiment_trials` | Single SDK call; MCP tools extract different views (full status, status-only, per-trial, filtered trials) |
| `list_jobs()` | `list_experiments` | |
| `delete_job()` | `delete_experiment` | |
| `get_best_results()` | `get_best_trial` | |
| `wait_for_job_status()` | `wait_for_experiment` | Status set maps to `{Complete, Failed}` |
| `get_job_logs()` | `get_experiment_trial_logs` | Supports per-trial log retrieval |
| `get_job_events()` | `get_experiment_events` | |

#### Tools Beyond SDK Abstraction

The following MCP tools go beyond what `OptimizerClient` exposes directly:

- **`create_hpo_experiment`**: `optimize()` cannot be driven from JSON input
  (see Key Design Decisions), so this tool constructs the `V1beta1Experiment`
  itself. Doing so also lets it honour a caller-supplied name, stamp the MCP
  ownership label, and select any Katib suggestion algorithm rather than only
  the two the SDK's typed classes express.
- **`create_experiment_from_spec`**: `OptimizerClient` has no API for raw
  `V1beta1Experiment` JSON, early stopping, custom metrics collectors, or
  resume policies. This tool uses `CustomObjectsApi` with `kubeflow_katib_api`
  models to create experiments from complete specs.
- **`get_experiment`**: `OptimizationJob` carries no conditions, current optimal
  trial or early-stopping spec, so the tool reads them from the Experiment CR
  and merges them into the response. The CR is also authoritative for trial
  counts: it tracks `trialsEarlyStopped`, a state that never reaches the
  TrainJob the SDK derives its trial view from. The extra read is best effort,
  degrading to `detail_unavailable: true` rather than failing the call.
- **`get_trial`**: While `OptimizerClient.get_job()` returns all trials in
  `OptimizationJob.trials`, MCP provides individual trial lookup by name
  for targeted debugging.
- **`list_suggestions`**: `OptimizerClient` does not expose Suggestion CRs.
  MCP uses `CustomObjectsApi` for debugging suggestion algorithm status.
- **`get_suggestion`**: Same — uses `CustomObjectsApi` for detailed algorithm
  status metadata useful for agent-driven diagnostics.
- **`update_experiment`**: `OptimizerClient` has no update/patch method.
  Uses `CustomObjectsApi` patch.

These tools use `CustomObjectsApi` directly to access Experiment, Trial, and
Suggestion CRs via `kubeflow_katib_api` models.

#### SDK Compatibility Snippet

```python
"optimizer": {
    "status": "implemented",
    "sdk_client": "kubeflow.optimizer.OptimizerClient",
    # OptimizerClient is included in kubeflow >= 0.4.0.
    # CRD types are provided by kubeflow_katib_api >= 0.19.0 (bundled).
    "covered_methods": [
        "get_job", "list_jobs",
        "delete_job", "get_best_results",
        "wait_for_job_status", "get_job_logs",
        "get_job_events",
    ],
    "uncovered_methods": [
        # optimize() requires TrainJobTemplate.trainer to be a CustomTrainer
        # holding a live Python callable (func), which cannot cross the
        # MCP/JSON boundary. It also generates its own Experiment name and
        # sets no labels, so a caller-supplied name and the MCP ownership
        # label are both unreachable through it. create_hpo_experiment
        # builds the V1beta1Experiment CR directly instead.
        "optimize",
    ],
    "k8s_api_operations": [
        "create_hpo_experiment (CustomObjectsApi create)",
        "create_experiment_from_spec (CustomObjectsApi create)",
        "list_suggestions (CustomObjectsApi list)",
        "get_suggestion (CustomObjectsApi get)",
        "update_experiment (CustomObjectsApi get + patch)",
        "katib_pre_flight (ApiextensionsV1Api + CoreV1Api)",
    ],
}
```

### Cross-Client Integration

A typical agent-driven HPO workflow spans both clients:

```
pre_flight()                       # Validate cluster, GPU availability
katib_pre_flight()                 # Validate Katib readiness
list_runtimes()                    # Find training runtimes
create_hpo_experiment()            # Create experiment with TrainJob trial template
wait_for_experiment()              # Wait for completion
get_best_trial()                   # Get optimal hyperparameters
fine_tune()                        # Retrain with best config
```

Key `TOOL_NEXT_HINTS` bridging both clients:

| Tool | Next Hint |
|------|-----------|
| `wait_for_training` | "Use `create_hpo_experiment()` to optimize hyperparameters" |
| `create_hpo_experiment` | "Monitor with `get_experiment_status()` or `wait_for_experiment()`" |
| `wait_for_experiment` | "Use `get_best_trial()` for optimal hyperparameters" |
| `get_best_trial` | "Use `fine_tune()` to retrain with these hyperparameters" |

`katib_pre_flight` reports whether the trainer client is loaded, as a
non-blocking warning — a missing trainer must not prevent optimizer-only use.

The optimizer module does not import the trainer module: shared logic (log
failure-pattern detection) lives in `common/`, so `--clients optimizer` loads
standalone.

> **Deferred**: validating trial-template resource requests against
> `get_cluster_resources()` when both clients are active. Not implemented —
> the trial template is an opaque `trialSpec` that may describe any workload
> kind, so there is no single field to inspect. Follow-up work.

## Risks and Mitigations

### Katib API Stability

Katib CRDs are at `v1beta1`. A future `v1` promotion could break field names.

**Mitigation**: Pin schemas to `v1beta1`. `KATIB_API_GROUP`, `KATIB_API_VERSION`
and the CRD plurals are re-exported from `kubeflow.optimizer.constants` rather
than hardcoded, so they track the SDK. An environment-variable override for
`KATIB_API_VERSION` was considered but is **not implemented** — the SDK models
are generated per API version, so overriding the version alone would not make
the client compatible with a `v1` CRD. Follow-up work if Katib promotes to `v1`.

### Experiment Cost Runaway

An agent could create experiments with excessive `maxTrialCount` or
`parallelTrialCount`, or read back so many trials that it exhausts its own
context window.

**Mitigations**:
- Two-phase confirmation (preview before submit)
- `readonly` persona blocks creation/deletion
- `max_trial_count` is required, no unbounded default, and capped at 1000
- `parallel_trial_count` capped at 100
- Collection responses bounded by `limit` (default 50, max 500)
- `data-scientist` deletion restricted to MCP-owned experiments

### Trial Polling Efficiency

**Mitigations**:
- `polling_interval` default 15s (SDK default), minimum 5s
- `timeout_seconds` capped at 3600s
- Circuit breaker on underlying API calls

### Trial Log Retrieval

Pods may be deleted before log retrieval (`trialTemplate.retain=false`).

**Mitigation**: `create_hpo_experiment` sets `trialTemplate.retain=true` so trial
pods survive for log retrieval. Note the trade-off: a large experiment then
leaves one retained pod per trial, which is why `max_trial_count` is capped.
When no logs are available the SDK yields an empty stream rather than raising,
so the tool returns an empty `logs` field instead of an error.

### OptimizerClient Import Failure

**Mitigation**: Lazy import. `server.py` already catches `ImportError` for
client modules.

### Cross-Client Dependency

Agent may assume trainer tools are available during optimizer workflows.

**Mitigation**: `katib_pre_flight` reports whether the trainer client is
loaded (see Cross-Client Integration). Cross-client hints only injected
when both modules are active.

## Testing Plan

### Unit Tests

- Each tool in isolation with mocked `OptimizerClient`
- SDK method calls, error handling, response schemas, two-phase confirmation,
  persona filtering, namespace enforcement
- `V1beta1Experiment` construction in `create_hpo_experiment`: search-space
  encoding, trial-parameter references, ownership label, caller-supplied name
- `update_experiment` patch generation, including the suspend/resume
  round-trip that must preserve the original `parallelTrialCount`
- `katib_pre_flight` CRD detection and controller **readiness** (a pod in phase
  `Running` whose container fails its readiness probe must not report ready)
- Namespace allowlist enforced by every tool, resolved via the optimizer client
- `limit` bounds every collection response while `total` stays truthful
- Target: at least 80% line coverage for `optimizer/api/`

> Mocks must reproduce the SDK's real failure shape. The Kubernetes backend
> re-raises API errors as `RuntimeError(...) from ApiException`, so a test that
> raises a bare `ApiException` does not exercise the same code path that
> production hits.

### SDK Contract Tests

- `OptimizerClient` methods exist with expected signatures
  (per `trainer/api/sdk_contracts_test.py` pattern)
- `OptimizerClient`, `OptimizationJob`, `Result`, `TrialConfig`, `Objective`,
  `Search`, `RandomSearch`, `GridSearch` importable from `kubeflow.optimizer`
- `V1beta1Experiment`, `V1beta1ExperimentSpec`, `V1beta1EarlyStoppingSpec`
  importable from `kubeflow_katib_api.models`

### Integration Tests

- Kind cluster with Katib: pre-flight, create, wait, get trials, get best
  trial, suspend, resume, delete
- Cross-client: trainer pre-flight then optimizer experiment with TrainJob
  trial template
- Early stopping via `create_experiment_from_spec`: `medianstop` experiment,
  confirm `EarlyStopped` trials

> **Known gap, to be confirmed live**: per-trial status is derived from
> `Trial.trainjob.status`, which reports TrainJob states
> (`Created`/`Running`/`Complete`/`Failed`) and has no `EarlyStopped` value.
> Katib does track early stopping, but on the Experiment CR
> (`status.trialsEarlyStopped` and `status.earlyStoppedTrialList`), which is why
> `get_experiment` reports counts from the CR rather than from the SDK view.
>
> The gap that remains is `get_successful_trials` and the `status` filter on
> `get_experiment_trials`: both match on TrainJob status, so an early-stopped
> trial holding valid metrics is not reported as successful. Resolving it means
> cross-referencing `earlyStoppedTrialList`. Deferred until it can be confirmed
> against a live Katib install.

### Persona Tests

- `readonly` limited to read-only tools
- `data-scientist` blocked from `create_experiment_from_spec`
- `data-scientist` `delete_experiment` enforces MCP ownership label
- `platform-admin` unrestricted

### MCP Inspector

- All 17 tools in registry with correct descriptions and annotations
- Progressive mode discovery matches `TOOL_PHASES`

## References

- [KEP-936: Kubeflow MCP Server](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server)
- [Katib Experiment CRD types](https://github.com/kubeflow/katib/blob/master/pkg/apis/controller/experiments/v1beta1/experiment_types.go)
- [Katib Trial CRD types](https://github.com/kubeflow/katib/blob/master/pkg/apis/controller/trials/v1beta1/trial_types.go)
- [Katib Suggestion CRD types](https://github.com/kubeflow/katib/blob/master/pkg/apis/controller/suggestions/v1beta1/suggestion_types.go)
- [Kubeflow Optimizer SDK -- OptimizerClient](https://github.com/kubeflow/sdk/blob/main/sdk/kubeflow/optimizer/api/optimizer_client.py)
- [kubeflow-mcp-server -- TrainerClient](https://github.com/kubeflow/mcp-server/tree/main/kubeflow_mcp/trainer)
- [kubeflow-mcp-server -- Optimizer stub](https://github.com/kubeflow/mcp-server/tree/main/kubeflow_mcp/optimizer)
- [kubeflow-mcp-server -- Persona policy](https://github.com/kubeflow/mcp-server/blob/main/kubeflow_mcp/core/policy.py)
