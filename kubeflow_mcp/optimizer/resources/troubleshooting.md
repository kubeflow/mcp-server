# Katib Troubleshooting Guide

## Start here

```
katib_pre_flight()          # CRD present? controller Running AND Ready?
get_experiment_status(name) # experiment-level state + trial counts
get_experiment_events(name) # K8s events: scheduling, image pull, quota
get_suggestion(name)        # algorithm state when trials never appear
```

## Control-plane problems

| Symptom | Cause | Fix |
|---|---|---|
| `Katib Experiment CRD not found` | Katib not installed | `kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=v0.19.0"` |
| `Katib controller pod not found` | Wrong namespace, or install incomplete | `kubectl get pods -A -l katib.kubeflow.org/component=controller` |
| **`Running but not Ready`** | Controller cannot watch its CRDs — almost always missing RBAC | Verify `kubectl get clusterrole katib-controller`. Installing Katib as a non-cluster-admin silently skips ClusterRoles: Kubernetes refuses to let a user grant permissions it does not itself hold. Re-apply the manifests as cluster-admin. |
| Controller `CrashLoopBackOff` | Config error or OOM | `kubectl logs -n kubeflow -l katib.kubeflow.org/component=controller` |
| `katib-db-manager` CrashLoopBackOff | Cannot reach `katib-mysql` | Check the MySQL pod and its PVC — a `Pending` PVC means no storage class could bind |

A controller in phase `Running` is **not** proof of health; `katib_pre_flight`
checks container readiness for this reason and reports `controller_ready`.

## Experiment problems

| Symptom | Cause | Fix |
|---|---|---|
| `Experiment 'x' already exists` | Name collision | Pick another name, or `delete_experiment(name, confirmed=True)`. Reported as a validation error — it does not trip the circuit breaker |
| `Namespace 'x' not allowed by policy` | Namespace outside the server's allowlist | Pass an allowed `namespace=`, or run the server without the restriction |
| `Experiment 'x' was not created by MCP` | Non-admin persona touching an external experiment | Expected. `data-scientist`/`ml-engineer` may only mutate MCP-created experiments; use `platform-admin` |
| Stuck in `Created`, no trials | Suggestion not ready, or the trial template is invalid | `get_suggestion(name)` for algorithm state, then `get_experiment_events(name)` |
| All trials fail immediately | Bad image, bad command, or unresolvable `runtimeRef` | `get_experiment_trial_logs(name)`; failure patterns (OOM, missing module, permissions) are auto-detected and returned in `failure_hint` |
| Trials run but no metrics | The objective metric name does not match what the script prints, or `primaryContainerName` is wrong for the trial kind | Ensure the script emits the metric named in `objective_metric`; for non-TrainJob templates use `create_experiment_from_spec()` |
| Results look flat / identical | A searched parameter is never referenced in the template | Every `search_space` key needs a `${trialParameters.<key>}` reference in `trial_template` |
| `Timeout after Ns` from `wait_for_experiment` | Trials slower than the timeout | Not a failure — the experiment is still running. Poll `get_experiment_status()`, or raise `timeout_seconds` (max 3600) |

## Status values

Experiment status and trial status come from **different** sources — do not
filter one with the other's vocabulary.

**Experiment** (`get_experiment_status`, `list_experiments`):

| Status | Meaning |
|---|---|
| `Created` | Accepted; no trials running yet |
| `Running` | Trials executing |
| `Complete` | Finished; best result available |
| `Failed` | Failed (e.g. `max_failed_trials` exceeded) |

**Trial** (`get_experiment_trials`, `get_trial`) — derived from the underlying
TrainJob, so it uses TrainJob vocabulary: `Created`, `Running`, `Complete`,
`Failed`, `Suspended`.

Filter succeeded trials with `status="Complete"`, **not** `"Succeeded"`.
`list_experiments(status="Succeeded")` is accepted as an alias for `Complete`
at the experiment level, but `get_experiment_trials` has no such alias.

## No best trial available

`get_best_trial` returns `best_trial: null` rather than an error when Katib has
not yet recorded an optimal trial. Check in order:

1. `get_experiment_status(name)` — still `Created`/`Running`? Nothing to report yet.
2. `get_experiment_trials(name, status="Complete")` — any trial actually finished?
3. `get_experiment_trial_logs(name)` — is the objective metric being printed at all?

If trials complete but no metrics are recorded, the metrics collector is not
seeing the output — verify the metric name and the primary container.

## Suspend and resume

`update_experiment(name, action="suspend")` sets `parallelTrialCount` to 0;
running trials finish, no new ones start. The previous value is saved to the
`kubeflow-mcp/pre-suspend-parallel-trial-count` annotation and restored on
`action="resume"`. Suspending an already-suspended experiment is a no-op.

If someone edits `parallelTrialCount` by hand while suspended, resume restores
the **annotated** value, not the hand-edited one.

## Cleanup

`delete_experiment(name, confirmed=True)` removes the experiment along with its
trials and suggestion. It is irreversible — the first call returns a preview.
Retained trial pods are removed with the experiment.
