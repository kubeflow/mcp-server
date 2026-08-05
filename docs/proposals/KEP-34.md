---
title: "Katib (Optimizer) Client Support for kubeflow-mcp-server"
authors:
  - "@krishnagupta"
status: provisional
creation-date: 2025-06-02
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

**Experiment creation decomposition**: `OptimizerClient.optimize()` accepts
typed Python objects (`TrainJobTemplate`, `Search.*`, `TrialConfig`, `Objective`,
`BaseAlgorithm`) but does not expose early stopping, custom metrics collectors,
or resume policies. This KEP decomposes creation into two tools:

- `create_hpo_experiment`: Flat parameters (objective, search space, algorithm,
  trial template, early stopping). Uses `OptimizerClient.optimize()` for the
  common case; when early stopping is requested, constructs a
  `V1beta1Experiment` via `kubeflow_katib_api` models and creates it through
  `CustomObjectsApi` directly.
- `create_experiment_from_spec`: Full `V1beta1Experiment` JSON spec for
  advanced use cases (custom metrics collectors, resume policies, NAS configs).
  Validates the spec via `kubeflow_katib_api.models.V1beta1Experiment.from_dict()`
  and creates it through `CustomObjectsApi`.

Both use two-phase confirmation and produce the same response schema.

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
| `create_hpo_experiment` | Create HPO experiment from flat params (objective, search space, algorithm, trial template, early stopping) | `OptimizerClient.optimize()` (simple) / `CustomObjectsApi` (with early stopping) |
| `create_experiment_from_spec` | Create experiment from complete V1beta1Experiment JSON spec | `CustomObjectsApi` + `kubeflow_katib_api` models |

#### Discovery

| Tool | Description | Underlying API |
|------|-------------|----------------|
| `get_experiment` | Experiment status, best trial, trial counts, conditions, early stopping config | `OptimizerClient.get_job()` |
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
| `update_experiment` | Suspend or resume experiment (patch resumePolicy) | `CustomObjectsApi` patch |

> **Clarification on `update_experiment` semantics**: Katib's `resumePolicy`
> field (`Never`, `FromVolume`, `LongRunning`) controls whether an experiment
> can restart *after completion*, which differs from the trainer's
> `update_training_job(action="suspend")` that uses TrainJob's native `suspend`
> field for mid-execution pause. The `update_experiment` tool patches
> `spec.resumePolicy` and, when `action="suspend"`, sets the experiment's
> `parallelTrialCount` to 0 to effectively halt new trial creation. This is
> the established Katib pattern for pausing experiments, since Katib lacks a
> first-class suspend field.

All read-only tools: `readOnlyHint=True`, `idempotentHint=True`.
Mutating tools: `create_*` are not idempotent; `delete_experiment` is destructive; `update_experiment` is idempotent.
`wait_for_experiment`: `readOnlyHint=True`, blocking. Default `polling_interval=15s`, `timeout_seconds` capped at 3600s. Agents should prefer `get_experiment_status()` for non-blocking polling.

`list_suggestions`, `get_suggestion`, `update_experiment`, and
`katib_pre_flight` use `CustomObjectsApi`/`CoreV1Api` directly where
`OptimizerClient` lacks coverage. `get_experiment_events` is now covered by
`OptimizerClient.get_job_events()`.

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
    "optimizer_planning": "planning",
    "optimizer_discovery": None,
    "optimizer_optimization": "optimization",
    "optimizer_monitoring": "monitoring",
    "optimizer_lifecycle": "monitoring",  # lifecycle instructions folded into monitoring section
}
```

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
- list_suggestions() -> check available suggestion algorithms
```

**Optimization** (data-scientist+):
```
TOOL SELECTION:
- Simple HPO -> create_hpo_experiment()
- Advanced spec -> create_experiment_from_spec()
- Both require confirmed=True to submit; False returns preview
RULES:
- ALWAYS preview first (confirmed=False)
- maxTrialCount is required, no unbounded default
- Use early_stopping with medianstop for long-running trials
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
| `optimize()` | `create_hpo_experiment` | Flat-param decomposition; simple cases go through SDK |
| `get_job()` | `get_experiment`, `get_experiment_status`, `get_trial`, `get_successful_trials`, `get_experiment_trials` | Single SDK call; MCP tools extract different views (full status, status-only, per-trial, filtered trials) |
| `list_jobs()` | `list_experiments` | |
| `delete_job()` | `delete_experiment` | |
| `get_best_results()` | `get_best_trial` | |
| `wait_for_job_status()` | `wait_for_experiment` | Status set maps to `{Complete, Failed}` |
| `get_job_logs()` | `get_experiment_trial_logs` | Supports per-trial log retrieval |
| `get_job_events()` | `get_experiment_events` | |

#### Tools Beyond SDK Abstraction

The following MCP tools go beyond what `OptimizerClient` exposes directly:

- **`create_experiment_from_spec`**: `OptimizerClient.optimize()` accepts typed
  Python objects and does not support raw `V1beta1Experiment` JSON, custom
  metrics collectors, or resume policies. This tool uses `CustomObjectsApi`
  with `kubeflow_katib_api` models to create experiments from complete specs.
- **`create_hpo_experiment` (early stopping path)**: When early stopping is
  requested, the tool bypasses `optimize()` and constructs a `V1beta1Experiment`
  with `V1beta1EarlyStoppingSpec` via `CustomObjectsApi`.
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
        "optimize", "get_job", "list_jobs",
        "delete_job", "get_best_results",
        "wait_for_job_status", "get_job_logs",
        "get_job_events",
    ],
    "uncovered_methods": [],  # all 8 OptimizerClient methods are covered
    "k8s_api_operations": [
        "create_experiment_from_spec (CustomObjectsApi create)",
        "create_hpo_experiment/early_stopping (CustomObjectsApi create)",
        "list_suggestions (CustomObjectsApi list)",
        "get_suggestion (CustomObjectsApi get)",
        "update_experiment (CustomObjectsApi patch)",
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

`katib_pre_flight` reports whether the trainer client is loaded. Next-step
hints referencing trainer tools are only injected when the trainer module is
active. `create_hpo_experiment` validates trial template resource requests
against `get_cluster_resources` when both clients are active.

## Risks and Mitigations

### Katib API Stability

Katib CRDs are at `v1beta1`. A future `v1` promotion could break field names.

**Mitigation**: Pin schemas to `v1beta1`, use SDK as abstraction layer.
`KATIB_API_VERSION` constant overridable via environment variable.

### Experiment Cost Runaway

An agent could create experiments with excessive `maxTrialCount` or
`parallelTrialCount`.

**Mitigations**:
- Two-phase confirmation (preview before submit)
- `readonly` persona blocks creation/deletion
- `maxTrialCount` is required, no unbounded default
- `data-scientist` deletion restricted to MCP-owned experiments

### Trial Polling Efficiency

**Mitigations**:
- `polling_interval` default 15s (SDK default), minimum 5s
- `timeout_seconds` capped at 3600s
- Circuit breaker on underlying API calls

### Trial Log Retrieval

Pods may be deleted before log retrieval (`trialTemplate.retain=false`).

**Mitigation**: Clear error suggesting `retain=true`. Tool checks trial status
before attempting retrieval.

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
- `optimize()` parameter construction in `create_hpo_experiment`; `V1beta1Experiment` construction for early stopping path
- `update_experiment` patch generation
- `katib_pre_flight` CRD/controller detection
- Target: at least 80% line coverage for `optimizer/api/`

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
- Early stopping: `medianstop` experiment, confirm `EarlyStopped` trials

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
