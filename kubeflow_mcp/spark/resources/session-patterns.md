# SparkConnect Session Patterns

Structured context for AI agents helping users run Spark on Kubeflow through the
`spark` client. The MCP server manages the **session lifecycle** (create, inspect,
delete); the actual Spark work runs in the **data plane** — a notebook, job, or
tool that attaches to the session with PySpark. The MCP server never proxies
Spark RPCs.

## End-to-end workflow

```
create_spark_session(confirmed=False)   # preview the resolved config
create_spark_session(confirmed=True)    # provision; returns name + connect info
get_spark_session(name)                 # poll until state == "Ready"
# --- data plane (outside MCP): attach with PySpark ---
#   from pyspark.sql import SparkSession
#   spark = SparkSession.builder.remote("sc://<service>:15002").getOrCreate()
get_spark_session_logs(name)            # inspect driver output if something looks wrong
delete_spark_session(name, confirmed=True)   # tear down when finished
```

## Sizing a session

Pass simple resource dicts — any valid Kubernetes resource name works, including
accelerators:

```python
create_spark_session(
    name="etl-nightly",
    num_executors=5,
    executor_resources={"cpu": "4", "memory": "8Gi"},
    driver_resources={"cpu": "1", "memory": "2Gi"},
    spark_conf={"spark.sql.adaptive.enabled": "true"},
    confirmed=True,
)
```

- Omit `num_executors` / resources to use operator defaults.
- GPUs: add `"nvidia.com/gpu": "1"` to `executor_resources`.
- Names must be RFC 1123 (lowercase alphanumeric and `-`); one is generated when omitted.

## States

`Provisioning` → `Ready`/`Running` are the healthy path. `NotReady` usually means
the driver pod is still starting; keep polling. `Failed` means the session will not
become usable — read the logs, then delete and recreate.

## Attaching from the data plane

`create_spark_session` returns the session `service_name` and namespace. Inside the
cluster, the Spark Connect gRPC endpoint is `sc://<service_name>:15002`. From
outside the cluster, port-forward the driver pod first:

```
kubectl port-forward svc/<service_name> 15002:15002 -n <namespace>
```
