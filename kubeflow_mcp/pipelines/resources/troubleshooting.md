# Kubeflow Pipelines troubleshooting

Error-to-fix reference for the `pipelines` client tools.

| Symptom | Likely cause | Fix |
|---|---|---|
| `SDK_ERROR` "Kubeflow Pipelines support is not installed" | `kfp` missing on the server host | Install `kubeflow[pipelines]` on the server host |
| `RESOURCE_NOT_FOUND` "Pipeline not found" | Wrong name or namespace | `list_pipelines(namespace=...)` and use the exact name |
| `RESOURCE_NOT_FOUND` "Experiment not found" | Experiment does not exist | Leave `experiment` out to use the default, or `create_experiment(name)` |
| `VALIDATION_ERROR` "Multiple pipelines found" | Two pipelines share a name | Rename one in the KFP UI |
| `VALIDATION_ERROR` "looks like a file path" | A `.yaml`/`.zip` path was passed | Register the pipeline with `create_pipeline`, then run it by name |
| `VALIDATION_ERROR` "pipeline_code is not supported" | The code uses a shape `create_pipeline` refuses | Follow the line and rule in the error; see `pipelines://guides/authoring` |
| `VALIDATION_ERROR` "The pipeline failed to compile" | KFP rejected the code (wrong argument name or type, ...) | Fix the line named in the error and preview again |
| `VALIDATION_ERROR` "Pipeline code blocked: unsafe patterns" | The safety scan flagged `subprocess`, `os.system`, ... | Remove them, or the operator sets `KUBEFLOW_MCP_UNSAFE_SCRIPTS=true` |
| `VALIDATION_ERROR` "Compiling the pipeline took longer than" | The compile did not finish in time | Keep the `@dsl.pipeline` body to component calls and task options |
| Step fails with `403 Forbidden` creating a `TrainJob` | The pipeline's service account lacks Trainer RBAC | Grant `pipeline-runner` create/get on `trainjobs` in the namespace |
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

- Runs can only be started from registered pipelines. `create_pipeline` registers pipelines
  from code; uploading a precompiled YAML file is not exposed.
- `create_pipeline` compiles in a child process on the server host, so that host needs
  `kubeflow-mcp[pipelines]`. The `@dsl.pipeline` body runs there at compile time, which is
  why its shape is restricted; component bodies only run in the cluster.
- On Linux the compile worker is capped at 1 GiB of memory and a 2 MB pipeline spec;
  on Windows only the 120 s timeout applies.
- There is no image allowlist: the preview and the response list every image a pipeline
  uses, so review them before confirming.
- KFP experiments and pipelines carry no MCP ownership label, so `delete_experiment` and
  `delete_pipeline` are `platform-admin` only.
- KFP runs carry no MCP ownership label, so stop and retry are limited to the `ml-engineer` persona and above.
- `get_pipeline_logs` reads live pods only and returns the last lines of up to 10 pods.
