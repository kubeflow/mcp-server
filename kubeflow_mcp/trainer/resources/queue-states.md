# Queue States and Kueue Admission Guide

Guidance for monitoring, diagnosing, and managing Kubeflow Trainer jobs in clusters running CNCF Kueue.

---

## Overview

When CNCF Kueue manages batch scheduling for Kubeflow Trainer:
- A `TrainJob` is submitted with `labels={"kueue.x-k8s.io/queue-name": "<queue-name>"}`.
- Kueue creates an associated `Workload` resource in the same namespace to track admission and quota.
- Until quota is admitted, the `TrainJob` remains suspended (`spec.suspend: true`) and no worker pods are created.
- `get_training_job` surfaces live queue details under `queue_status`.

---

## Workload State Matrix

| State | Condition & Reason | Description | Agent Action |
|:------|:-------------------|:------------|:-------------|
| **`queued`** | `QuotaReserved=False`<br>`reason: Pending` | Job is waiting in queue for required quota (GPUs, CPU, memory). | **Do NOT delete or resubmit.** Report queue wait to user. Wait for quota to become available. |
| **`admitted`** | `Admitted=True`<br>`QuotaReserved=True` | Quota granted by ClusterQueue. Pods are being scheduled. | Proceed with standard monitoring (`get_training_logs`, `get_training_events`). |
| **`suspended`** | `TrainJob.spec.suspend: true` | TrainJob is suspended. For Kueue-managed jobs, suspension is managed by Kueue. | **Do NOT call `update_training_job(action="resume")`**. Kueue will automatically un-suspend the job upon admission. |
| **`inadmissible`** | `QuotaReserved=False`<br>`reason: Inadmissible` | Configuration mismatch (e.g. LocalQueue does not exist, flavor mismatch, inactive ClusterQueue). | **Do NOT delete the job.** Kueue continuously re-evaluates workloads when cluster resources or queues change. Report the reason to the user to fix queue configuration. |
| **`evicted`** | `Evicted=True`<br>`reason: Preempted` / `PodsReadyTimeout` / `AdmissionCheck` | Workload was preempted by higher priority work or timed out. | Check `queue_status.message` and `requeue_count`. Kueue automatically requeues preempted jobs unless max retries are exceeded. |

---

## State Details and Safe Remediation

### 1. `queued` (Pending Quota)
- **Root Cause**: The cluster or tenant queue currently lacks sufficient resources (typically GPUs) to admit the entire gang at once.
- **Remediation**:
  - Do NOT cancel or resubmit the job; resubmitting resets queue position.
  - If `wait_for_training` times out while queued, advise extending `timeout_seconds` or checking cluster queue utilization with the cluster administrator.

### 2. `suspended` (Controller Managed)
- **Root Cause**: All Kueue-managed `TrainJob`s start with `spec.suspend: true` while waiting in queue.
- **Important**: Do **NOT** use `update_training_job(name, action="resume")` to resume a Kueue-managed job.
  - Manually resuming bypasses queue admission and causes reconciling conflicts with Kueue.
  - Kueue will automatically set `spec.suspend: false` once quota is reserved.

### 3. `inadmissible` (Configuration Mismatch)
- **Root Cause**: The requested `LocalQueue` does not exist in the namespace, the associated `ClusterQueue` is stopped, or requested resources do not match any available `ResourceFlavor`.
- **Remediation**:
  - Inspect `queue_status.message` for the exact configuration issue.
  - Inform the user of the misconfiguration (e.g., "LocalQueue 'my-queue' does not exist in namespace 'team-a'").
  - Do not blindly delete the job; if an administrator creates or fixes the queue, Kueue will automatically re-evaluate and admit the pending workload.

### 4. `evicted` (Preemption or Timeout)
- **Root Causes**:
  - `Preempted`: A higher priority workload needed quota and preempted this job.
  - `PodsReadyTimeout`: Scheduled pods took longer than the configured timeout to reach Ready status.
  - `Deactivated`: The workload was administratively deactivated (`spec.active: false`).
- **Remediation**:
  - Check `requeue_count` in `queue_status`. If `requeue_count > 0`, Kueue has requeued the job to wait for quota again.
  - If repeatedly preempted, inform the user so they can adjust priority or request dedicated resources.
