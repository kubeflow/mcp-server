# HPO Workflow Patterns & Algorithm Selection

## Workflow: Basic HPO with Random Search

```
katib_pre_flight() → list_experiments() → create_hpo_experiment() → wait_for_experiment() → get_best_trial()
```

### Step 1: Pre-flight
```
katib_pre_flight()
```
Checks Katib CRD, controller health, and trainer availability.

### Step 2: Create Experiment
```
create_hpo_experiment(
    name="lr-search",
    objective_metric="accuracy",
    objective_type="maximize",
    search_space={
        "lr": {"min": 0.001, "max": 0.1, "type": "uniform"},
        "batch_size": {"choices": [16, 32, 64]},
    },
    algorithm="random",
    max_trial_count=10,
    parallel_trial_count=2,
    trial_template={...},
    confirmed=False  # Preview first!
)
```

### Step 3: Monitor
```
get_experiment_status(name)     # Lightweight poll
get_experiment_trials(name)     # All trials with params/metrics
get_experiment_trial_logs(name) # Pod logs from best trial
```

### Step 4: Results
```
get_best_trial(name)            # Optimal hyperparameters
get_successful_trials(name)     # All successful trials for comparison
```

## Algorithm Selection Guide

| Algorithm | SDK Support | Best For | `algorithm=` |
|-----------|:-----------:|----------|-------------|
| Random Search | ✅ SDK | Quick exploration, many parameters | `"random"` |
| Grid Search | ✅ SDK | Small discrete spaces, exhaustive | `"grid"` |
| TPE | ❌ Raw spec | Sequential, Bayesian approach | `create_experiment_from_spec()` |
| CMA-ES | ❌ Raw spec | Continuous parameters, few dims | `create_experiment_from_spec()` |
| Hyperband | ❌ Raw spec | Early stopping, resource-aware | `create_experiment_from_spec()` |

## Search Space Types

- **uniform**: Continuous range `{"min": 0.001, "max": 0.1, "type": "uniform"}`
- **loguniform**: Log-scale range `{"min": 0.0001, "max": 0.01, "type": "loguniform"}`
- **choice**: Categorical `{"choices": [16, 32, 64]}`

## Cross-Client Workflow: Optimize → Train

After HPO completes, use best parameters to launch full training:
```
best = get_best_trial(name)
# Extract best hyperparameters from best["parameters"]
# Use them in fine_tune() or run_custom_training()
```
