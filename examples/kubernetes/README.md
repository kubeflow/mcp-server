# Deploying Kubeflow MCP Server in a cluster

A minimal, self-contained example for running the MCP Server inside Kubernetes so
that agents can reach it over HTTP and drive
[Kubeflow Trainer](https://github.com/kubeflow/trainer) in your own namespace.

It is intentionally small: one Deployment, one ClusterIP Service, a ServiceAccount
and least-privilege RBAC. No Ingress, no OIDC, no Helm chart.

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

## Troubleshooting

**Requests return HTTP 421** — DNS rebinding protection allows loopback `Host`
headers by default, so anything arriving via the Service is rejected.
`KUBEFLOW_MCP_ALLOWED_HOSTS` must list the exact hostname clients use; only the
`:*` port wildcard is supported, not a host wildcard. Note that `port-forward`
does *not* reproduce this, because it sends a loopback `Host` header.

**A browser-based client returns 403** — the `Origin` header is not allowed. Add it
to `KUBEFLOW_MCP_ALLOWED_ORIGINS`; setting hosts alone leaves origins at their
loopback-only defaults. Non-browser MCP clients send no `Origin` and are unaffected.

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

**Every authenticated request returns 401** — the client is sending a different
token than the one stored in the Secret.

**Pod stuck in `CreateContainerConfigError`** — the `kubeflow-mcp-auth` Secret does
not exist yet. Create it as shown in the quick start.

**`kubectl apply` fails with `namespaces "kubeflow-system" not found`** — Trainer
runs somewhere else, so the two `kubeflow-mcp-trainer-version` documents have
nowhere to go while the Deployment and Service are still created. Point those two
documents at the Trainer namespace and set `KUBEFLOW_SYSTEM_NAMESPACE` to match.

**The pod is `Ready` but calls still fail** — `/ready` is evaluated once when routes
are registered and never re-checked, so it carries no more signal than `/health`. If
the Kubernetes API becomes unreachable the pod stays `Ready` and keeps taking
traffic from the Service.

**A service mesh blocks traffic to the pod** — the Deployment sets
`sidecar.istio.io/inject: "false"`. If your mesh enforces strict mTLS, remove that
annotation and allow the traffic with a `PeerAuthentication`/`DestinationRule`.
