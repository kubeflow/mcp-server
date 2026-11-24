# Deploying Kubeflow MCP Server in a cluster

A minimal, self-contained example for running the MCP Server inside Kubernetes so
that agents can reach it over HTTP and drive
[Kubeflow Trainer](https://github.com/kubeflow/trainer) in your own namespace.

It contains one Deployment, one ClusterIP Service, a ServiceAccount, and
least-privilege RBAC. Ingress and Helm are outside this profile. It authenticates
clients with a shared bearer token; to use your identity provider instead, see
[Using JWT (OIDC) instead of the shared token](#using-jwt-oidc-instead-of-the-shared-token).

## Quick start

Requires a cluster with Kubeflow Trainer installed, `kubectl`, and an existing
namespace — the example uses `kubeflow-user-example-com`, the Profile namespace a
standard Kubeflow install creates. The bearer token is not part of `manifests.yaml`,
so create it first:

```bash
NAMESPACE=kubeflow-user-example-com

kubectl create secret generic kubeflow-mcp-auth -n "$NAMESPACE" \
  --from-literal=token="$(openssl rand -hex 32)"

kubectl apply -k examples/kubernetes
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

## Using JWT (OIDC) instead of the shared token

For production, have clients send access tokens from your identity provider instead
of sharing one static token. Anyone who can read a token on the wire can replay it
until it expires, so it must stay encrypted all the way to the pod. This example
serves plain HTTP, so before using it in production, encrypt every hop, for example
with a service mesh that uses mTLS between pods. An Ingress that terminates TLS is
not enough on its own, because it forwards the token to the pod in plain text.

The example is deployed with `kubectl apply -k`, so make the change in
`manifests.yaml`. Uncomment `KUBEFLOW_MCP_JWKS_URI`, `KUBEFLOW_MCP_JWT_ISSUER` and
`KUBEFLOW_MCP_JWT_AUDIENCE`, set them to the provider's JWKS URI, its issuer and an
audience reserved for this server, and apply:

```bash
kubectl apply -k examples/kubernetes
kubectl rollout status deploy/kubeflow-mcp -n "$NAMESPACE"
```

All three are required. With only the JWKS URI the pod exits at startup, because the
server would otherwise accept any token the provider signs, including tokens it
issued to other applications. Use the issuer exactly as it appears in the tokens'
`iss` claim, and as the audience the value the provider puts in `aud` for tokens
meant for this server (usually the client ID or API identifier you registered for
it), not one shared with other applications. Tokens must also carry an `exp` claim.
The pod needs network access to the JWKS URI.

The JWKS URI takes precedence over the shared token, so once JWT works, remove the
`KUBEFLOW_MCP_AUTH_TOKEN` entry from `manifests.yaml` and apply again. Delete the
Secret only after that. If you delete it while the Deployment still references it,
any new pod (from the next `kubectl apply -k`, a restart or a reschedule) fails with
`CreateContainerConfigError`. The `&&` stops before the delete if the apply or the
rollout fails:

```bash
kubectl apply -k examples/kubernetes \
  && kubectl rollout status deploy/kubeflow-mcp -n "$NAMESPACE" \
  && kubectl delete secret kubeflow-mcp-auth -n "$NAMESPACE"
```

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
`:*` port wildcard is supported, not a host wildcard. `port-forward` does not
reproduce this because it sends a loopback `Host` header.

**A browser-based client returns 403** — the `Origin` header is not allowed. Add it
to `KUBEFLOW_MCP_ALLOWED_ORIGINS`; setting hosts alone leaves origins at their
loopback-only defaults. Non-browser MCP clients send no `Origin` and are unaffected.

**Tools report an empty namespace or a 403 listing runtimes** — the pod is running
outside the namespace whose TrainJobs you expect. See the namespace section above.

**The agent has no tool for submitting a TrainJob** — the persona is `readonly`.
Set `KUBEFLOW_MCP_PERSONA` to `data-scientist` or higher.

**Every authenticated request returns 401** — the client is sending a different
token than the one stored in the Secret.

**The pod exits at startup with `JWT auth needs an issuer and an audience`** —
`KUBEFLOW_MCP_JWKS_URI` is set without both `KUBEFLOW_MCP_JWT_ISSUER` and
`KUBEFLOW_MCP_JWT_AUDIENCE`. Set both, as shown in the JWT section above.

**JWT requests return 401** — the token was issued for another audience or by
another issuer, has no `exp`, or has expired. Compare its `iss` and `aud` claims
with `KUBEFLOW_MCP_JWT_ISSUER` and `KUBEFLOW_MCP_JWT_AUDIENCE`.

**Pod stuck in `CreateContainerConfigError`** — the `kubeflow-mcp-auth` Secret does
not exist yet. Create it as shown in the quick start.

**Trainer control-plane version information is unavailable** — older Trainer
releases may not create the `kubeflow-trainer-public` ConfigMap in
`kubeflow-system`. This warning is harmless when the CRD and API checks pass. A
403 instead usually means the version Role is missing or Trainer runs in a
different namespace.

**Applying the manifests reports that `kubeflow-system` does not exist** — point
the Trainer version Role and RoleBinding at the namespace where Trainer is
installed, and update `KUBEFLOW_SYSTEM_NAMESPACE` to match.

**The pod is not `Ready` (`/ready` returns 503)** — `/ready` dynamically checks
Kubernetes API reachability, Trainer CRD existence/version, and essential RBAC permissions.
If the pod fails readiness, check `kubectl logs <pod>` to see the specific diagnostic failure.

**The pod is `Ready` but calls still fail** — `/ready` re-checks Kubernetes API
reachability, the Trainer CRD, and essential RBAC on each probe, but it does not cover
every permission or dependency a tool may need. Check the server logs and run
`check_compatibility()` for a full report.

**A service mesh blocks traffic to the pod** — the Deployment disables automatic
sidecar injection. If the mesh enforces strict mTLS, remove that annotation and
allow the traffic with the mesh's peer-authentication and destination rules.
