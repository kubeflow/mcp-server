# Kubeflow Pipelines troubleshooting

Error-to-fix reference for the `pipelines` client tools.

| Symptom | Likely cause | Fix |
|---|---|---|
| `SDK_ERROR` "Kubeflow Pipelines support is not installed" | `kfp` missing on the server host | Install `kubeflow[pipelines]` on the server host |
| `RESOURCE_NOT_FOUND` "Pipeline not found" | Wrong name or namespace | `list_pipelines(namespace=...)` and use the exact name |
| `RESOURCE_NOT_FOUND` "Experiment not found" | Experiment does not exist | Leave `experiment` out to use the default, or create it outside MCP |
| `VALIDATION_ERROR` "Multiple pipelines found" | Two pipelines share a name | Rename one in the KFP UI |
| `VALIDATION_ERROR` "looks like a file path" | A `.yaml`/`.zip` path was passed | Upload the pipeline outside MCP, then run it by name |
| `VALIDATION_ERROR` "Invalid run_id" | Not a run UUID | Take the ID from `list_pipeline_runs()` |
| `PERMISSION_DENIED` | Token or RBAC lacks access to KFP | Check `KUBEFLOW_MCP_PIPELINES_TOKEN` and the server's service account |
| Run stuck in `pending` | No capacity, or image pull problems | `get_pipeline_logs(run_id)`; check cluster events |
| `get_pipeline_logs` reports "pod no longer exists" | KFP cleaned up the task pod | Logs are gone from the cluster; check your log store |

## Server configuration

| Variable | Purpose |
|---|---|
| `KUBEFLOW_MCP_PIPELINES_URL` | KFP API server URL; auto-discovered when unset |
| `KUBEFLOW_MCP_PIPELINES_TOKEN` | Bearer token for the KFP API |
| `KUBEFLOW_MCP_PIPELINES_NAMESPACE` | Default namespace for pipelines calls |

## Known limitations

- Runs can only be started from registered pipelines; uploading and deleting pipelines is not exposed.
- KFP runs carry no MCP ownership label, so stop and retry are limited to the `ml-engineer` persona and above.
- `get_pipeline_logs` reads live pods only and returns the last lines of up to 10 pods.
