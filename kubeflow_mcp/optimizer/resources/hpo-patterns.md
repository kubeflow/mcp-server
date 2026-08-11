# HPO Workflow Patterns & Algorithm Selection

## Standard workflow

```
katib_pre_flight() → list_experiments() → create_hpo_experiment(confirmed=False)
                   → create_hpo_experiment(confirmed=True) → wait_for_experiment()
                   → get_best_trial()
```

### 1. Pre-flight

```
katib_pre_flight()
```

Verifies the Experiment CRD exists, the Katib controller pod is **Running and
Ready**, and reports whether the trainer client is loaded. If `blockers` is
non-empty, stop and surface them — creating an experiment against a broken
control plane produces an Experiment that is never reconciled.

### 2. Preview, then submit

`create_hpo_experiment` is two-phase. Call it with `confirmed=False` first, show
the returned `config` (the full Experiment manifest) to the user, and only then
re-call with `confirmed=True`.

```
create_hpo_experiment(
    name="lr-search",
    objective_metric="accuracy",
    objective_type="maximize",
    search_space={
        "lr": {"min": 0.001, "max": 0.1, "type": "loguniform"},
        "batch_size": {"choices": [16, 32, 64]},
    },
    algorithm="random",
    max_trial_count=12,
    parallel_trial_count=3,
    trial_template={
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "spec": {
            "runtimeRef": {"name": "torch-distributed"},
            "trainer": {
                "image": "my-registry/trainer:latest",
                "args": [
                    "--lr=${trialParameters.lr}",
                    "--batch-size=${trialParameters.batch_size}",
                ],
            },
        },
    },
    confirmed=False,
)
```

### 3. Monitor

```
get_experiment_status(name)      # lightweight poll — prefer this
get_experiment_trials(name)      # trials with params/metrics (limit applies)
get_experiment_trial_logs(name)  # logs from the best trial so far
get_experiment_events(name)      # K8s events — scheduling / image pull issues
```

`wait_for_experiment(name)` blocks the MCP connection until the experiment
reaches `Complete` or `Failed`. Prefer polling `get_experiment_status()` unless
the user explicitly wants to wait.

### 4. Results

```
get_best_trial(name)             # optimal hyperparameters + metrics
get_successful_trials(name)      # successful trials, for comparison
```

## `trial_template` — the part that usually goes wrong

`trial_template` is the Katib **trialSpec**: the workload Katib clones once per
trial. It is *not* a Katib-specific wrapper — it is a complete resource
manifest with `apiVersion`, `kind` and `spec`.

Every tuned parameter must be referenced with `${trialParameters.<name>}`, and
the name must match a key in `search_space`. A parameter that is searched but
never referenced is silently ignored — trials all run with the same config and
the results look flat.

`create_hpo_experiment` sets `primaryContainerName` to the Trainer's `node`
container, so the template should describe a **TrainJob**. For a plain
`batch/v1` Job or another kind, use `create_experiment_from_spec()` and set
`primaryContainerName` yourself, otherwise metrics are never collected.

Use `list_runtimes()` (trainer client) to find a valid `runtimeRef`.

## Algorithm selection

All algorithms below are available directly from `create_hpo_experiment` via
the `algorithm=` parameter.

| Algorithm | `algorithm=` | Best for | Notes |
|---|---|---|---|
| Random search | `"random"` | Default. Wide spaces, easy parallelism | Strong baseline; scales with `parallel_trial_count` |
| Grid search | `"grid"` | Small discrete/categorical spaces | Trial count grows multiplicatively — keep the space tiny |
| Bayesian optimization | `"bayesianoptimization"` | Expensive trials, few parameters | Sequential; low `parallel_trial_count` works best |
| TPE | `"tpe"` | Mixed continuous + categorical | Good general-purpose sequential choice |
| Multivariate TPE | `"multivariate-tpe"` | Correlated parameters | Models parameter interactions |
| CMA-ES | `"cmaes"` | Continuous parameters, few dimensions | Continuous only — no categoricals |
| Sobol | `"sobol"` | Even coverage of a continuous space | Quasi-random; deterministic sequence |
| Hyperband | `"hyperband"` | Long trials where early stopping pays | Manages its own budget across brackets |

Rules of thumb: start with `random` to establish a baseline; move to `tpe` or
`bayesianoptimization` when each trial is expensive; use `grid` only when the
full cross-product is small enough to enumerate.

## Search space

| Type | Form | Produces |
|---|---|---|
| Continuous, uniform | `{"min": 0.5, "max": 0.99}` | `double`, uniform distribution |
| Continuous, log-scale | `{"min": 1e-5, "max": 1e-1, "type": "loguniform"}` | `double`, logUniform |
| Categorical | `{"choices": [16, 32, 64]}` | `categorical` (values are stringified) |

`uniform` is the default when `type` is omitted. Use `loguniform` for anything
spanning orders of magnitude — learning rates and weight decay especially;
a uniform range from 1e-5 to 1e-1 spends 90% of its samples above 0.01.

## Budget and response limits

- `max_trial_count` is required, capped at **1000**
- `parallel_trial_count` capped at **100**
- `max_failed_trials` optional — the experiment fails once it is exceeded
- Collection tools accept `limit` (default **50**, max **500**). `total` always
  reports the true count, so compare the two to detect truncation.

Trial pods are retained (`retain=true`) so logs remain readable after a trial
finishes. A large experiment therefore leaves one pod per trial behind — keep
`max_trial_count` proportionate and clean up with `delete_experiment()`.

## Beyond `create_hpo_experiment`

Use `create_experiment_from_spec()` for anything the flat parameters do not
model — **early stopping** (e.g. `medianstop`), custom metrics collectors,
resume policies, or NAS. It takes a complete `V1beta1Experiment` manifest,
validates it against the Katib schema, and applies the same preview/confirm
flow. It is restricted to the `ml-engineer` persona and above.

## Cross-client: optimize, then train

```
best = get_best_trial("lr-search")
# best["parameters"] -> {"lr": "0.0123", "batch_size": "32"}
# feed those into fine_tune() or run_custom_training()
```

Parameter values arrive as **strings** (Katib stores them that way); cast them
before use in a typed training API.
