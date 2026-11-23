# Katib Troubleshooting Guide

## Start here

```
katib_pre_flight()          # CRD present? controller Running AND Ready?
get_hpo_experiment_status(name) # experiment-level state + trial counts
get_hpo_experiment_events(name) # K8s events: scheduling, image pull, quota
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
| `Experiment 'x' already exists` | Name collision | Pick another name, or `delete_hpo_experiment(name, confirmed=True)`. Reported as a validation error — it does not trip the circuit breaker |
| `Namespace 'x' not allowed by policy` | Namespace outside the server's allowlist | Pass an allowed `namespace=`, or run the server without the restriction |
| `Experiment 'x' was not created by MCP` | Non-admin persona touching an external experiment | Expected. `data-scientist`/`ml-engineer` may only mutate MCP-created experiments; use `platform-admin` |
| Stuck in `Created`, no trials | Suggestion not ready, or the trial template is invalid | `get_suggestion(name)` for algorithm state, then `get_hpo_experiment_events(name)` |
| All trials fail immediately | Bad image, bad command, or unresolvable `runtimeRef` | `get_hpo_experiment_trial_logs(name)`; failure patterns (OOM, missing module, permissions) are auto-detected and returned in `failure_hint` |
| Trials run but no metrics | The objective metric name does not match what the script prints, the derived `primaryContainerName` is not the container that prints it, or the namespace lacks `katib.kubeflow.org/metrics-collector-injection=enabled` | Ensure the script emits the metric named in `objective_metric`; pass `primary_container_name` when a sidecar is listed first; label the namespace |
| `search_space parameters not referenced in trial_template` | A searched parameter is never used in the template; Katib would reject the experiment | Reference it as `${trialParameters.<key>}` in `trial_template`, or remove it from `search_space` |
| `Timeout after Ns` from `wait_for_hpo_experiment` | Trials slower than the timeout | Not a failure — the experiment is still running. Poll `get_hpo_experiment_status()`, or raise `timeout_seconds` (max 3600) |

## Status values

Experiment status and trial status come from **different** sources — do not
filter one with the other's vocabulary.

**Experiment** (`get_hpo_experiment_status`, `list_hpo_experiments`):

| Status | Meaning |
|---|---|
| `Created` | Accepted; no trials running yet |
| `Running` | Trials executing |
| `Complete` | Finished; best result available |
| `Failed` | Failed (e.g. `max_failed_trials` exceeded) |

**Trial** (`get_hpo_experiment_trials`, `get_trial`): read from the Trial's own
conditions, so it is the same whatever the trial runs (TrainJob, Job, ...):
`Created` (pending), `Running`, `Complete`, `Failed`, `Killed`, `EarlyStopped`,
`MetricsUnavailable`.

A succeeded trial or experiment is reported as `Complete`. Both
`list_hpo_experiments` and `get_hpo_experiment_trials` also accept `"Succeeded"`
as an alias for it. `get_successful_trials` returns only `Complete` trials, not
`EarlyStopped` ones.

## No best trial available

`get_best_trial` returns `best_trial: null` rather than an error when Katib has
not yet recorded an optimal trial. Check in order:

1. `get_hpo_experiment_status(name)` — still `Created`/`Running`? Nothing to report yet.
2. `get_hpo_experiment_trials(name, status="Complete")` — any trial actually finished?
3. `get_hpo_experiment_trial_logs(name)` — is the objective metric being printed at all?

If trials complete but no metrics are recorded, the metrics collector is not
seeing the output — verify the metric name and the primary container.

## Adjusting a running experiment

Katib has no pause. Its admission webhook rejects `parallelTrialCount` of 0 and
only lets a live experiment change three fields, which is what
`update_hpo_experiment` exposes:

| Parameter | Use |
|---|---|
| `parallel_trial_count` | Raise it to go faster. Lowering it deletes trials (see below) |
| `max_trial_count` | Extend the budget. Must exceed the trials already created |
| `max_failed_trial_count` | Tolerate more (or fewer) failures before the experiment fails |

The tool previews the change first (`confirmed=False`) and refuses edits Katib
would reject: concurrency above the budget, a budget at or below the trials
already created, or changes to a finished experiment. A finished experiment can
only be extended when it stopped at `max_trial_count` and was created with
`resumePolicy: LongRunning` or `FromVolume` (the default is `Never`), which
needs `create_hpo_experiment_from_spec`. To stop an experiment for good, use
`delete_hpo_experiment`.

**Lowering `parallel_trial_count` is not a gentle throttle.** While more trials
are active (pending or running) than the new value allows, Katib deletes trials
from **all** of the experiment's trials, newest first and whatever their state,
and repeats until the active ones fit. A trial that finished after an older one
started is newer, so it goes first, results and all; the running one goes on the
next round. Deleted trials do not show up in `trialsKilled`. The preview lists
them by name and state in `trials_to_delete`, estimated from the trials at the
time of the call; wait for trials to finish first if their results matter.

## Cleanup

`delete_hpo_experiment(name)` previews the deletion; once the user agrees,
`delete_hpo_experiment(name, confirmed=True)` removes the experiment along with
its trials and suggestion. It is irreversible. Retained trial pods are removed
with the experiment.
