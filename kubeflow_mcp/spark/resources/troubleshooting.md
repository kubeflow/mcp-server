# SparkConnect Troubleshooting

Error-to-fix reference for the `spark` client tools.

## Error-to-fix table

| Symptom | Likely cause | Fix |
|---|---|---|
| `SparkClient requires the Spark extra` | `kubeflow[spark]` not installed on the MCP host | `pip install 'kubeflow[spark]'` (or `'kubeflow-mcp[spark]'`) and restart the server |
| `create_spark_session` returns a `warning` about the driver connection | Session was provisioned, but the MCP host can't reach the driver's gRPC port | Expected off-cluster. Poll `get_spark_session(name)`; attach from a pod that can reach the service |
| `get_spark_session_logs` → "has no driver pod yet" | Session still `Provisioning` | Poll `get_spark_session(name)` until `state == "Ready"`, then retry |
| `RESOURCE_NOT_FOUND` on get/delete/logs | Wrong name or namespace | `list_spark_sessions(namespace=...)` to find the exact name |
| Session stuck in `NotReady` | Insufficient cluster resources or image pull issues | `get_spark_session_logs(name)`; check quotas and executor resource requests |
| Session `Failed` | Driver crashed at startup | Read logs, `delete_spark_session(name, confirmed=True)`, recreate with corrected config |

## Known limitations

- **Streaming logs (`follow=True`) are not exposed.** MCP tools return a bounded
  snapshot; use `tail_lines` to control volume.
- **Attaching to an existing server is not an MCP tool.** `SparkClient.connect(base_url=...)`
  returns a live PySpark `SparkSession`, which is not serializable over MCP. The data
  plane attaches directly using the connect info from `create_spark_session`.
- **`create_spark_session` needs `kubeflow[spark]` and cluster reachability** from the
  server host, because the SDK's only public creation path (`connect()`) also opens a
  driver connection. A provision-only SDK method would remove that requirement — tracked
  as a follow-up.
- **Batch (`SparkApplication`) jobs are out of scope** for this module; the SDK's
  `SparkClient` currently covers SparkConnect sessions only.
