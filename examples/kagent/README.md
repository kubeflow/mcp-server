# KAgent with Kubeflow MCP Server

This example deploys the Kubeflow MCP Server over authenticated Streamable HTTP and registers it with KAgent. It assumes that KAgent and Kubeflow Trainer are already installed; it does not install KAgent, Trainer, Agentgateway, or a model provider.

## Why use KAgent?

KAgent provides the agent runtime that turns natural-language requests into
controlled Kubeflow operations. It discovers the MCP tools, applies the
configured identity and permissions, and presents training readiness, resource,
submission, and monitoring workflows through one agent interface. Kubeflow
resources remain authoritative in the Kubernetes API.

## Reference workflow

The following captures show the expected user-facing results after registration.

![Kubeflow MCP tools registered alongside KAgent tools](assets/kagent-tool-catalog-overview.png)

*The KAgent catalog shows Kubeflow MCP registered alongside the platform's existing tools.*

![Detailed Trainer readiness check](assets/kagent-trainer-readiness-detail.png)

*A natural-language request invokes MCP tools and returns compatibility checks and available runtimes.*

![Cluster resource constraints reported by the agent](assets/kagent-resource-constraints-detail.png)

*The agent explains the cluster's available resources and recommends an appropriate training path before submission.*

## Prerequisites

- KAgent installed in the `kagent` namespace.
- Kubeflow Trainer installed and its CRDs available.
- `agents.kagent.dev`, `remotemcpservers.kagent.dev`, and Trainer CRDs installed.
- Agentgateway installed with the `agentgateway` GatewayClass and Gateway API CRDs.
- Permission to create the ServiceAccount, cluster-scoped RBAC, Deployment, Service, and RemoteMCPServer.

MCP and its credentials stay in `kagent`; training and package-inspection pods
run in `kubeflow-workloads`. Keep that namespace free of control-plane secrets
and privileged service accounts: pod creators can mount its secrets.

The projected namespace file sets the SDK's workload default without changing
its `kagent:kubeflow-mcp` identity. To rename the workload namespace, update the
Deployment annotation, Namespace, Role, and RoleBinding together.

Install KAgent and Trainer using the [KAgent documentation](https://kagent.dev/docs/kagent/0.x/introduction/installation/) and [Kubeflow Trainer documentation](https://www.kubeflow.org/docs/components/trainer/overview/).

## 1. Create the MCP credentials

Create the token with the secret-management process used by your cluster:

```bash
export MCP_NAMESPACE=kagent
export MCP_TOKEN="$(openssl rand -hex 32)"

kubectl create namespace "$MCP_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
printf '%s' "$MCP_TOKEN" | kubectl -n "$MCP_NAMESPACE" \
  create secret generic kubeflow-mcp-http-auth \
  --from-file=KUBEFLOW_MCP_AUTH_TOKEN=/dev/stdin \
  --dry-run=client -o yaml | kubectl apply -f -
printf 'Bearer %s' "$MCP_TOKEN" | kubectl -n "$MCP_NAMESPACE" \
  create secret generic kubeflow-mcp-http-header \
  --from-file=AUTHORIZATION=/dev/stdin --dry-run=client -o yaml | kubectl apply -f -
```

## 2. Apply the MCP component

### Upgrading an existing installation

Pause submissions and finish or deliberately stop existing training workloads
in `kagent`. Jobs cannot move namespaces; recreate future runs in
`kubeflow-workloads` and retire the old workload pods before resuming access.

Remove the old permissions: applying the new profile does not delete them.

```bash
kubectl -n kagent delete rolebinding kubeflow-mcp-trainer --ignore-not-found
kubectl -n kagent delete role kubeflow-mcp-trainer --ignore-not-found
```

Apply the profile below and pass the namespace checks in **Validate** before
resuming submissions. If `kagent` writes remain allowed, remove the extra grants.

### Deploy

Review [`deployment.yaml`](deployment.yaml) and set the MCP image to the
required image tag for your environment. The example uses the `latest` image
tag; set a release tag in `deployment.yaml` when you need a pinned version.
Apply the component:

```bash
kubectl apply -k examples/kagent
kubectl rollout status deployment/kubeflow-mcp-http -n "$MCP_NAMESPACE"
```

To enable tracing, use the combined [`examples/kagent-observability`](../kagent-observability/README.md) bundle instead. The plain KAgent example does not require the OpenTelemetry Operator.

The internal MCP Service endpoint is:

```text
http://kubeflow-mcp-http.<namespace>:8000/mcp
```

The `RemoteMCPServer` resource points to the Agentgateway endpoint. Apply
`examples/agentgateway` before applying this profile or starting an Agent that
uses this registration.
The Service is an internal Agentgateway target. The NetworkPolicy restricts
application traffic to the Agentgateway namespace.

The gateway requires `kubeflow-mcp-client: "true"` on callers in `kagent`.
Label each trusted KAgent caller Deployment and wait for its rollout:

```bash
kubectl -n kagent patch deployment <trusted-kagent-caller-deployment> --type merge \
  -p '{"spec":{"template":{"metadata":{"labels":{"kubeflow-mcp-client":"true"}}}}}'
kubectl -n kagent rollout status deployment/<trusted-kagent-caller-deployment>
```

Unlabeled callers are denied. Do not label training pods or let workload users
create/patch pods in `kagent`.

For local or development-only stdio registration, KAgent's command form can launch
the package directly with `uvx`. This starts a separate MCP process and is not a
replacement for the deployed HTTP server:

```text
uvx kubeflow-mcp serve --transport stdio --clients trainer --persona ml-engineer --mode full
```

![KAgent command-based MCP registration](assets/kagent-command-registration.png)

## 3. Use the MCP server from KAgent

The `RemoteMCPServer` manifest registers `kubeflow-mcp-http` with KAgent. Once
the resource is accepted, start a new Agent chat and confirm that the MCP tools
are listed. If you manage Agents declaratively, reference this resource from the
Agent definition instead of adding a second registration.

## 4. Validate

```bash
kubectl get remotemcpserver kubeflow-mcp-http -n "$MCP_NAMESPACE"
kubectl get deployment,service,pods -n "$MCP_NAMESPACE" \
  -l app.kubernetes.io/name=kubeflow-mcp-http
```

The RemoteMCPServer must be accepted and expose tools such as `pre_flight`, `list_training_jobs`, `fine_tune`, and `health_check`. Run `pre_flight` before submitting a workload.

Verify the namespace boundary before using real credentials:

```bash
# Expected: yes, no, no, respectively (requires permission to impersonate the SA).
kubectl auth can-i create pods -n kubeflow-workloads --as=system:serviceaccount:kagent:kubeflow-mcp
kubectl auth can-i create pods -n kagent --as=system:serviceaccount:kagent:kubeflow-mcp
kubectl auth can-i create trainjobs.trainer.kubeflow.org -n kagent --as=system:serviceaccount:kagent:kubeflow-mcp
```

With NetworkPolicy enforcement, verify gateway port 80 allows trusted callers
but denies unlabeled `kagent` pods and all `kubeflow-workloads` pods. Submit a CPU
job and inspect packages; both must create pods only in `kubeflow-workloads`.
Audit other RoleBindings and NetworkPolicies for grants that broaden access.
