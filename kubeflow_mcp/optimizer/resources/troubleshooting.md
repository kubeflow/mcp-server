# Katib Troubleshooting Guide

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Katib Experiment CRD not found` | Katib not installed | Install Katib: `kubectl apply -k github.com/kubeflow/katib/manifests/v1beta1/installs/katib-standalone` |
| `Controller pod not found` | Controller not running | Check `kubectl get pods -n kubeflow -l katib.kubeflow.org/component=controller` |
| `Controller pod is in 'CrashLoopBackOff'` | Config issue or OOM | Check logs: `kubectl logs -n kubeflow -l katib.kubeflow.org/component=controller` |
| `Experiment stuck in Created` | Suggestion not ready | Check `get_suggestion(name)` and controller logs |
| `All trials failed` | Training script error | Check `get_experiment_trial_logs(name)` for errors |
| `Timeout waiting for experiment` | Too many trials or slow training | Increase timeout or reduce max_trial_count |

## Debugging Steps

### Experiment Not Starting
1. `katib_pre_flight()` — check CRD and controller
2. `get_experiment_events(name)` — K8s events
3. `get_suggestion(name)` — suggestion algorithm status

### Trials Failing
1. `get_experiment_trials(name)` — which trials failed
2. `get_experiment_trial_logs(name, trial=trial_name)` — pod logs
3. `get_experiment_events(name)` — scheduling/image issues

### No Best Trial Available
- Experiment may still be running: `get_experiment_status(name)`
- No trials succeeded yet: `get_experiment_trials(name, status="Succeeded")`
- Metric collection issue: check metrics collector container in trial pods

## Status Values

| Status | Meaning |
|--------|---------|
| `Created` | Experiment accepted, waiting for trials |
| `Running` | Trials are being executed |
| `Complete` | All trials finished, best result available |
| `Failed` | Experiment failed (max_failed_trials exceeded or other error) |
