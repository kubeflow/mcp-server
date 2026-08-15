# Deploying Kubeflow MCP Server in a cluster

A minimal, self-contained example for running the MCP Server inside Kubernetes so
that agents can reach it over HTTP and drive
[Kubeflow Trainer](https://github.com/kubeflow/trainer) in your own namespace.

It is intentionally small: one Deployment, one ClusterIP Service, a ServiceAccount
and least-privilege RBAC. No Ingress, no OIDC, no Helm chart.

## Prerequisites

- A Kubernetes cluster with Kubeflow Trainer installed
- A namespace to run in. The example uses `kubeflow-user-example-com`, the Profile
  namespace created by a standard Kubeflow installation
- `kubectl` pointed at that cluster

## Quick start

```bash
NAMESPACE=kubeflow-user-example-com

# Generate the bearer token and deploy everything in one step. Piping keeps the
# token out of manifests.yaml, so no credential is ever written to disk.
sed "s/REPLACE_ME/$(openssl rand -hex 32)/" manifests.yaml | kubectl apply -f -

kubectl rollout status deploy/kubeflow-mcp -n "$NAMESPACE"
```

Read the token back when configuring a client:

```bash
kubectl get secret kubeflow-mcp-auth -n "$NAMESPACE" -o jsonpath='{.data.token}' | base64 -d
```

To rotate it later, re-run the command above and restart the pod so the new value
is picked up:

```bash
kubectl rollout restart deploy/kubeflow-mcp -n "$NAMESPACE"
```

If you prefer to manage the Secret separately — for example with an external secret
store — delete the `Secret` document from `manifests.yaml` and create it
imperatively instead. Leaving the document in place while doing this would reset
the token to `REPLACE_ME` on the next apply:

```bash
kubectl create secret generic kubeflow-mcp-auth -n "$NAMESPACE" \
  --from-literal=token="$(openssl rand -hex 32)"
```

## What gets created

| Resource | Purpose |
|---|---|
| `Namespace/kubeflow-user-example-com` | Already present on a standard Kubeflow install, where the Profile controller owns it — applying is then a no-op |
| `ServiceAccount/kubeflow-mcp` | Identity the server uses against the Kubernetes API |
| `ClusterRole/kubeflow-mcp-read` | Read-only: `ClusterTrainingRuntime`, nodes, namespaces, CRDs |
| `Role/kubeflow-mcp-trainjobs` | Full TrainJob lifecycle, in this namespace only |
| `Role/kubeflow-mcp-trainer-version` | Read of the single `kubeflow-trainer-public` ConfigMap in `kubeflow-system`, so the SDK can report the Trainer control-plane version |
| `Secret/kubeflow-mcp-auth` | Bearer token for the HTTP transport |
| `Deployment/kubeflow-mcp` | The server, HTTP transport on port 8000 |
| `Service/kubeflow-mcp` | ClusterIP, reachable at `kubeflow-mcp:8000` in-namespace |

The server can create and delete TrainJobs in its own namespace, and can only read
anything outside it.

## Verify

From inside the cluster:

```bash
kubectl run mcp-check -n "$NAMESPACE" --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://kubeflow-mcp:8000/ready
```

`/health` and `/ready` are served without authentication so the kubelet can probe
them; every other endpoint requires the bearer token.

To reach it from your workstation:

```bash
kubectl port-forward -n "$NAMESPACE" svc/kubeflow-mcp 8000:8000
```

Then point an MCP client at `http://localhost:8000/mcp` with
`Authorization: Bearer <token>`.

`KUBEFLOW_MCP_ALLOWED_HOSTS` lists `localhost` and `127.0.0.1` alongside the Service
names for this reason. Setting the variable replaces the built-in loopback defaults
rather than adding to them, so dropping those two entries makes port-forwarded
clients fail with HTTP 421 while `/health` and `/ready` keep working.

## Deploying into a different namespace

The server is deployed into the user's Profile namespace on purpose: in-cluster the
Kubeflow SDK derives its default namespace from the pod's ServiceAccount namespace.
Running it elsewhere makes every tool default to a namespace that has no TrainJobs
in it.

To use another namespace, replace `kubeflow-user-example-com` everywhere in
`manifests.yaml` — including inside `KUBEFLOW_MCP_ALLOWED_HOSTS`, which spells out
the Service DNS names:

```bash
sed -i 's/kubeflow-user-example-com/my-namespace/g' manifests.yaml
```

## Choosing what the agent can do

Tool exposure is controlled by the persona, which is separate from RBAC — RBAC
decides what the ServiceAccount may touch in the Kubernetes API, the persona
decides which MCP tools exist at all.

| `KUBEFLOW_MCP_PERSONA` | Tools |
|---|---|
| `readonly` (default) | Inspection and monitoring only — cannot submit a TrainJob |
| `data-scientist` (used here) | `readonly` plus `fine_tune`, `run_custom_training`, `wait_for_training`, `delete_training_job` |
| `ml-engineer` | `data-scientist` plus `run_container_training`, `update_training_job`, `inspect_crd`, `inspect_controller` |
| `platform-admin` | Everything |

The example uses `data-scientist` because the default `readonly` persona cannot
submit training jobs, which is the first thing most people want to try. Switch the
`KUBEFLOW_MCP_PERSONA` env var to change it.

`ml-engineer` additionally reads the Trainer controller's pods and logs, which the
RBAC in this example does not grant. If you use it, add a read-only Role for the
controller namespace and set `KUBEFLOW_MCP_CONTROLLER_NAMESPACE`.

For finer control than the built-in personas, mount a policy file at
`$HOME/.kf-mcp-policy.yaml` in the pod — see the main
[README](../../README.md) for the format.

## Troubleshooting

**Requests return HTTP 421** — DNS rebinding protection allows loopback `Host`
headers by default, so anything arriving via the Service is rejected.
`KUBEFLOW_MCP_ALLOWED_HOSTS` must list the exact hostname clients use; only the
`:*` port wildcard is supported, not a host wildcard. Note that `port-forward`
does *not* reproduce this, because it sends a loopback `Host` header.

**Tools report an empty namespace, or a 403 listing runtimes** — the pod is running
outside the namespace whose TrainJobs you expect. See the section above.

**The agent has no tool for submitting a TrainJob** — the persona is `readonly`.
Set `KUBEFLOW_MCP_PERSONA` to `data-scientist` or higher.

**Log line: `Trainer control-plane version info is not available ... (404)`** — the
SDK reads the Trainer version from the `kubeflow-trainer-public` ConfigMap in
`kubeflow-system`, which older Trainer releases do not create. Harmless: every
`check_compatibility` check still passes, including the CRD and API version. If the
same message shows `(403)` instead, the `kubeflow-mcp-trainer-version` Role is
missing or Trainer runs in a different namespace than `kubeflow-system`.

**Every authenticated request returns 401** — the token still reads `REPLACE_ME`,
or the client is sending a different one. Compare against the value stored in the
Secret.

**Pod stuck in `CreateContainerConfigError`** — the `kubeflow-mcp-auth` Secret is
missing, which happens if you removed it from `manifests.yaml` without creating it
imperatively.
