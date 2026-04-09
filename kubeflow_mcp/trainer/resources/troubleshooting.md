# Troubleshooting

Diagnostic workflows, error-to-fix tables, and known tool limitations.

---

## Diagnostic Workflow

1. `get_training_job(name)` — check status
2. `get_training_events(name)` — check K8s events
3. `get_training_logs(name)` — check container output
4. Match error in tables below

## Error-to-Fix Table

| Error / Event | Cause | Fix |
|--------------|-------|-----|
| OOMKilled | GPU/CPU memory exceeded | Reduce `batch_size`, enable QLoRA (`quantize_base=True`), use gradient checkpointing |
| FailedScheduling | No node matches resource request | Check `get_cluster_resources()`, reduce `gpu_per_node`, add `tolerations` parameter |
| ErrImagePull / ImagePullBackOff | Image not found or auth failed | Verify image name, check `image_pull_secrets` parameter |
| NCCL timeout | Multi-node communication failure | Pass `env={"NCCL_TIMEOUT": "1800"}`, try gloo backend, check network |
| 403 Forbidden (HuggingFace) | Gated model, no token | Accept model license, pass `hf_token` |
| Read-only file system | Platform with read-only root FS | Add emptyDir volumes (see trainer://guides/platform-fixes) |
| Permission denied /.local | Read-only root filesystem | Add dot-local emptyDir volume |
| ProcessGroupNCCL...no GPUs | torchtune on CPU cluster | Use `run_custom_training()` with gloo backend instead |
| BackOff (crash loop) | Container keeps crashing | Check `get_training_logs()` for the root error |
| Script syntax error | Invalid Python in `run_custom_training` | Script is wrapped into function body — no top-level indentation errors, no `if __name__` guards |
| "was not created by MCP" | Deleting externally created job as non-admin | Use `platform-admin` persona, or re-create the job via MCP tools |
| Controller not found | Wrong namespace for controller tools | Set `KUBEFLOW_MCP_CONTROLLER_NAMESPACE` env var or pass `namespace=` |

## GPU Memory Reference

| Model Size | bf16 | int4 (QLoRA) | Recommended GPU |
|------------|------|--------------|-----------------|
| 1-3B | ~6-16GB | ~3-6GB | T4, RTX 3080 |
| 7-8B | ~24GB | ~7GB | A10, RTX 4090 |
| 13B | ~40GB | ~12GB | A100-40GB |
| 70B | ~140GB | ~40GB | A100-80GB x2 |

## Batch Size Guide

| GPU Memory | Recommended batch_size |
|------------|----------------------|
| 8GB | 1-2 |
| 16GB | 2-4 |
| 24GB | 4-8 |
| 40GB+ | 8-16 |

---

## Scheduling and Environment Parameters

All three training tools (`fine_tune`, `run_custom_training`, `run_container_training`) accept these **direct parameters** for pod scheduling:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tolerations` | list of dicts | K8s tolerations for tainted nodes |
| `node_selector` | dict | K8s node selector labels |
| `volumes` | list of dicts | K8s volume definitions (emptyDir, PVC, etc.) |
| `volume_mounts` | list of dicts | K8s volume mounts |
| `affinity` | dict | K8s pod affinity/anti-affinity |
| `service_account_name` | string | K8s service account |
| `image_pull_secrets` | list of dicts | K8s image pull secrets |
| `labels` | dict | Extra labels on the TrainJob |
| `annotations` | dict | Extra annotations on the TrainJob |

**Environment variables (`env`):**
- `run_custom_training()` and `run_container_training()`: pass `env={"KEY": "VALUE"}` directly
- `fine_tune()`: does NOT support `env` — if you need env vars, use `run_custom_training()` with a LoRA script instead

### Common Patterns

**Add GPU tolerations:**
```json
"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
```

**Add OpenShift emptyDir volumes:**
```json
"volumes": [
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}}
]
```

**Set NCCL env vars (run_custom_training or run_container_training):**
```json
"env": {"NCCL_DEBUG": "INFO", "NCCL_P2P_DISABLE": "1", "NCCL_TIMEOUT": "1800"}
```

---

## Known Limitations

### Streaming Logs Not Supported

`get_training_logs(follow=True)` returns a static message. MCP is request/response — no streaming.
**Workaround**: Poll with `get_training_logs(follow=False)` at intervals.

### Suspended Jobs Show "Created" Status

After `update_training_job(name, action="suspend")`, status becomes "Created" not "Suspended". This is controller behavior.
**Workaround**: Check `get_training_events()` for the suspend event.

### list_training_jobs Status Filter is Client-Side

Fetches all jobs then filters. Slow with hundreds of jobs. `limit` truncates after filtering.

### wait_for_training Blocks the Connection

Synchronous blocking call (up to 3600s). No other tool calls during wait.
**Workaround**: Use short timeouts and poll with `get_training_job()`.

### get_runtime(include_packages=True) is Slow

Creates a temporary Pod to run `pip list` (~30-60s). torchtune runtimes always return `packages_error`.

### Script Safety Check is Best-Effort

Scans for dangerous patterns (`os.system()`, `subprocess.run()`, `eval()`, `shutil.rmtree()`, `__import__`). Flagged patterns produce `safety_warnings` but do NOT block submission.

### torchtune Requires GPUs (NCCL Backend)

All torchtune runtimes use NCCL. On CPU-only clusters, use `run_custom_training()` with gloo.

### torchtune Runtimes May Enforce num_nodes=1

Some runtimes enforce `num_nodes=1` via webhook. Check `get_runtime()` before requesting multi-node.

### fine_tune() Does Not Support env Parameter

The `env` parameter is not available on `fine_tune()`. If you need custom environment variables, use `run_custom_training()` with a LoRA script instead (see trainer://guides/training-patterns).

### MCP Ownership Label

All jobs created via MCP tools are labeled `kubeflow-mcp/managed-by=mcp`. Non-admin personas can only delete MCP-labeled jobs. Platform-admin bypasses this check.

## Recovery Actions

```
delete_training_job(name)    # Delete and retry (MCP-created jobs only for non-admin)
update_training_job(name, action="suspend")   # Pause
update_training_job(name, action="resume")   # Resume
```

## Parameter Rules (Non-Inferable)

- `resources_per_node` overrides `gpu_per_node` when both provided
- `packages` requires writable `/.local` — add emptyDir on read-only FS
- `pip_index_urls` only used when `packages` is non-empty
- `command`/`args` in `run_container_training` are separate parameters
- URI formats: `fine_tune()` requires `hf://` prefix; `estimate_resources()` uses bare model IDs
- Volume scope: `fine_tune()` volumes apply to ALL replicated jobs; `run_custom_training()` applies to node only
- `loss` in `fine_tune` only supports `CEWithChunkedOutputLoss`
- `dataset_source`/`dataset_split`/`dataset_column_map` configure instruct-format preprocessing
