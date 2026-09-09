# Deploying Kubeflow MCP Server in a cluster

A minimal, self-contained example for running the MCP Server inside Kubernetes so
that agents can reach it over HTTP and drive
[Kubeflow Trainer](https://github.com/kubeflow/trainer) in your own namespace.

It contains one Deployment, one ClusterIP Service, a ServiceAccount, and
least-privilege RBAC. Ingress, OIDC, and Helm are outside this profile.

## Quick start

Requires a cluster with Kubeflow Trainer installed, `kubectl`, and an existing
namespace — the example uses `kubeflow-user-example-com`, the Profile namespace a
standard Kubeflow install creates. The bearer token is not part of `manifests.yaml`,
so create it first:

```bash
NAMESPACE=kubeflow-user-example-com

kubectl create secret generic kubeflow-mcp-auth -n "$NAMESPACE" \
  --from-literal=token="$(openssl rand -hex 32)"

kubectl apply -f manifests.yaml
kubectl rollout status deploy/kubeflow-mcp -n "$NAMESPACE"
```

Read the token back when configuring a client:

```bash
kubectl get secret kubeflow-mcp-auth -n "$NAMESPACE" -o jsonpath='{.data.token}' | base64 -d
```

To rotate it, replace the Secret and restart the pod so the new value is picked up.
`replace` rather than `apply` keeps the new token out of the
`last-applied-configuration` annotation:

```bash
kubectl create secret generic kubeflow-mcp-auth -n "$NAMESPACE" \
  --from-literal=token="$(openssl rand -hex 32)" --dry-run=client -o yaml \
  | kubectl replace -f -
kubectl rollout restart deploy/kubeflow-mcp -n "$NAMESPACE"
```

## What gets created

| Resource | Purpose |
|---|---|
| `ServiceAccount/kubeflow-mcp` | Identity the server uses against the Kubernetes API |
| `ClusterRole/kubeflow-mcp-read` | Read-only: `ClusterTrainingRuntime`, nodes, namespaces, CRDs |
| `Role/kubeflow-mcp-trainjobs` | Full TrainJob lifecycle, in this namespace only |
| `Role/kubeflow-mcp-trainer-version` | Read of the single `kubeflow-trainer-public` ConfigMap in `kubeflow-system`, so the SDK can report the Trainer control-plane version |
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
the Service DNS names, and the namespace suffix on both binding names. It leaves
`kubeflow-system` untouched:

```bash
sed -i.bak 's/kubeflow-user-example-com/my-namespace/g' manifests.yaml
```

`-i.bak` rather than a bare `-i`: BSD `sed` on macOS reads the next argument as the
backup suffix and fails without one.
