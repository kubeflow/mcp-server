# SparkConnect Session Patterns

Structured context for AI agents helping users run Spark on Kubeflow through the
`spark` client. The MCP server manages the **session lifecycle** (create, inspect,
delete); the actual Spark work runs in the **data plane** — a notebook, job, or
tool that attaches to the session with PySpark. The MCP server never proxies
Spark RPCs.

## End-to-end workflow

```
preview = create_spark_session(confirmed=False)   # preview the resolved config
# Show preview to the user; reuse its generated name and resolved namespace:
create_spark_session(name=preview["config"]["name"],
                     namespace=preview["config"]["namespace"], confirmed=True)
get_spark_session(name)                 # poll until state == "Ready"
# --- data plane (outside MCP): attach with PySpark ---
#   from pyspark.sql import SparkSession
#   spark = SparkSession.builder.remote("sc://<service>:15002").getOrCreate()
get_spark_session_logs(name)            # inspect driver output if something looks wrong
delete_spark_session(name)              # preview the teardown
delete_spark_session(name, confirmed=True)   # tear down when finished
```

## Sizing a session

Pass resource dicts containing only `cpu` and `memory`. SDK 0.5.0's SparkConnect
builder ignores other resource keys, so the MCP tool rejects them before preview:

```python
create_spark_session(
    name="etl-nightly",
    num_executors=5,
    executor_resources={"cpu": "4", "memory": "8Gi"},
    driver_resources={"cpu": "1", "memory": "2Gi"},
    spark_conf={"spark.sql.adaptive.enabled": "true"},
    confirmed=False,  # inspect this preview, then repeat with confirmed=True
)
```

- Omit `num_executors` / resources to use operator defaults.
- CPU and memory must be positive. The SDK rounds fractional CPU up to whole
  cores (for example `500m` becomes `1`); the preview shows that effective CPU value.
- GPUs and other extended resources are not supported by this MCP session path.
- Names must be RFC 1123 (lowercase alphanumeric and `-`); one is generated when omitted.
- Reuse all reviewed configuration, including the generated name, on confirmation.
  `confirmed` must be a JSON boolean; strings such as `"false"` are not confirmation.

## States

`Provisioning` → `Ready`/`Running` are the healthy path. `NotReady` usually means
the driver pod is still starting; keep polling. `Failed` means the session will not
become usable — read the logs, then delete and recreate.

## Driver ServiceAccount and RBAC

Use an existing ServiceAccount in the session namespace with permission to manage
executor pods, services and configmaps. For example, preview
`create_spark_session(name="etl", namespace="spark-workloads", service_account_name="spark")`,
inspect the configuration, then repeat with `confirmed=True`. The SDK maps this
parameter to `spec.server.template.spec.serviceAccountName`, even when no driver
resource overrides are supplied. Omitting it leaves account selection to Kubernetes
(normally the namespace's `default` account); the tool does not grant it permissions.

An administrator can provision a dedicated account and namespace-scoped binding
in an existing workload namespace. Review these permissions for your deployment:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: spark
  namespace: spark-workloads
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: spark-driver
  namespace: spark-workloads
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps"]
    verbs: ["get", "list", "watch", "create", "delete", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: spark-driver
  namespace: spark-workloads
subjects:
  - kind: ServiceAccount
    name: spark
    namespace: spark-workloads
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: spark-driver
```

Verify the driver identity, for example:
`kubectl auth can-i create pods --as=system:serviceaccount:spark-workloads:spark -n spark-workloads`.
The administrator running this check needs permission to impersonate that account.
Check services and configmaps too. Custom storage/features can need additional permissions.

The driver identity is separate from the MCP host's Kubernetes credentials and the
Spark Operator controller's identity. The MCP host needs access to SparkConnect
objects and driver pod logs; the operator needs its own installation RBAC. Setting
the driver's account does not configure either of those identities. The MCP tool
never creates ServiceAccounts, Roles or RoleBindings as a side effect.

The `spark.kubernetes.authenticate.*` Spark configuration keys remain restricted;
use the explicit `service_account_name` parameter for driver identity. Kubernetes
RBAC and admission policy remain authoritative, and a denied create returns
`PERMISSION_DENIED`. On OpenShift, check SCC/admission denials separately from RBAC;
selecting an account does not grant an SCC, and privileged SCC is not assumed.

See the [Spark Kubernetes RBAC guide](https://spark.apache.org/docs/4.0.1/running-on-kubernetes.html#rbac).

## Attaching from the data plane

`create_spark_session` returns the session `service_name` and namespace. Inside the
cluster, use `sc://<service_name>.<namespace>.svc.cluster.local:15002` with the
service name returned by the operator (adjust the cluster domain if customized). From
outside the cluster, port-forward the driver pod first:

```
kubectl port-forward svc/<service_name> 15002:15002 -n <namespace>
```
