# Platform Fixes

Actionable JSON and commands for platform-specific issues.

---

## Platform Detection

| Node label | Platform |
|------------|----------|
| `node.openshift.io/os_id` | OpenShift |
| `eks.amazonaws.com/nodegroup` | Amazon EKS |
| `cloud.google.com/gke-nodepool` | Google GKE |
| None of above | Vanilla K8s / bare-metal |

Error-based detection (from `get_training_logs()`):

| Error pattern | Platform | Fix |
|---------------|----------|-----|
| "Read-only file system" | OpenShift (restricted SCC) | Add emptyDir volumes below |
| "Permission denied" on /.local or /.cache | OpenShift (restricted SCC) | Add emptyDir volumes below |

## OpenShift emptyDir Volumes

ALWAYS pass when `platform=openshift`. Copy-paste ready — pass directly as `volumes` parameter to any training tool:

```json
"volumes": [
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}}
]
```

**Rules**:
- `fine_tune()`: Do NOT add workspace emptyDir — `/workspace` comes from the runtime PVC
- `run_custom_training()`: auto-injects workspace emptyDir at `/workspace`
- `run_container_training()`: add workspace emptyDir only if your image writes to `/workspace`

## OpenShift Non-Root UID

Random UIDs (e.g. 1000660000). Do NOT assume root. Implications:
- HOME may not be writable — use `/tmp` for outputs
- `/.local` emptyDir fixes most pip issues
- Avoid `chmod` / `chown` in training scripts

## GPU Tolerations

Pass as a direct `tolerations` parameter to any training tool:

```json
"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
```

Or shorthand: `gpu_per_node=1` (auto-maps to `nvidia.com/gpu`).

## NCCL Environment (Multi-Node GPU)

For `run_custom_training()` or `run_container_training()`, pass `env` directly:

```json
"env": {
  "NCCL_DEBUG": "INFO",
  "NCCL_P2P_DISABLE": "1",
  "NCCL_TIMEOUT": "1800"
}
```

For `fine_tune()`: the `env` parameter is NOT supported. If you need NCCL tuning, use `run_custom_training()` with a LoRA script and pass env vars there (see trainer://guides/training-patterns).

## OpenShift SCC Workarounds

SCC (Security Context Constraints) cannot be changed via MCP tools. Instead of escalating privileges, use tool parameters to work around the `restricted` SCC:

**1. Read-only filesystem** — pass writable emptyDir volumes to the training tool:

```json
"volumes": [
  {"name": "dot-local", "mount_path": "/.local", "empty_dir": {}},
  {"name": "dot-cache", "mount_path": "/.cache", "empty_dir": {}},
  {"name": "tmp", "mount_path": "/tmp", "empty_dir": {}},
  {"name": "home", "mount_path": "/home", "empty_dir": {}}
]
```

For `fine_tune()`, these volumes apply to ALL replicated jobs (node, dataset-initializer, model-initializer).

**2. Non-root UID issues** — avoid commands that assume root. In scripts:
- Write outputs to `/tmp` instead of `/root` or `/home`
- Do NOT use `chmod`, `chown`, or write to `/etc`
- Use `/.local` for pip user installs (already covered by emptyDir above)

**3. Network restrictions** — some SCCs block host networking. For multi-node training, pass env vars to `run_custom_training()` or `run_container_training()`:

```json
"env": {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
```

**4. Pip package installation permission denied in run_custom_training()** — when using `run_custom_training(packages=[...])` on OpenShift under a restricted SCC, the SDK's pre-script pip install step attempts to run `pip install --user` which writes to `/.local`. Since user-defined emptyDir volumes are not mounted on the training container during this step, the job will fail immediately with:
`PermissionError: [Errno 13] Permission denied: '/.local'`

*Workaround*: Do **NOT** use the `packages` parameter on OpenShift. Instead, write a workaround directly in your training script to install the required packages to `/workspace/lib` (which is a writable emptyDir) and append it to `sys.path`:

```python
import subprocess, sys, os
lib_dir = '/workspace/lib'
os.makedirs(lib_dir, exist_ok=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '--target', lib_dir, '--quiet',
    'transformers', 'peft', 'trl'
], check=True)
sys.path.insert(0, lib_dir)
```

**5. If emptyDirs are not enough** — escalate to cluster admin:

```bash
oc adm policy add-scc-to-user anyuid -z <service-account> -n <namespace>
```

This is a cluster-level change and cannot be done through MCP tools. Inform the user.
