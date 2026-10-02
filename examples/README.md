# Kubeflow MCP Integration Examples

Each directory is a separate profile. Install the operators and profiles required
by your deployment; some profiles depend on services created by another profile.

## Core profiles

- [`kubernetes`](kubernetes/): standalone Kubeflow MCP Server with Trainer RBAC.
- [`kagent`](kagent/): KAgent RemoteMCPServer integration over Streamable HTTP; requires Agentgateway.
- [`agentgateway`](agentgateway/): optional Agentgateway routing and policy layer.
- [`kagent-observability`](kagent-observability/): KAgent plus an OpenTelemetry
  Collector.
- [`observability`](observability/): standalone OpenTelemetry Collector base.
- [`grafana`](grafana/): optional Grafana Operator dashboards.

Install the operators and CRDs required by a profile separately. Configure
provider credentials, identity policies, public ingress, and storage backends
for your environment before applying the manifests. Read the README in the
selected profile first.

## Recommended composition

For the complete KAgent, MCP, Agentgateway, tracing, and Grafana setup:

1. Install KAgent, Kubeflow Trainer, Agentgateway, OpenTelemetry, and Grafana
   operators with their CRDs.
2. Follow [`kagent`](kagent/) to create the MCP credentials.
3. Create the Agentgateway upstream Secret and apply [`agentgateway`](agentgateway/):

   ```bash
   kubectl apply -k examples/agentgateway
   ```

4. Apply [`kagent-observability`](kagent-observability/), which deploys the MCP
   server, its KAgent registration, and the OpenTelemetry Collector:

   ```bash
   kubectl apply -k examples/kagent-observability
   ```

   Use [`kagent`](kagent/) instead when tracing is not required; do not apply
   both profiles.
5. Configure the Prometheus and Tempo endpoints, then apply [`grafana`](grafana/):

   ```bash
   kubectl apply -k examples/grafana
   ```

The Agentgateway profile provides MCP routing. Model routing and the KAgent
model configuration are separate and require provider-specific values. For a
standalone MCP deployment, use [`kubernetes`](kubernetes/) instead.

## Compatibility

Reference environment used for these tested examples:

| Component | Version or requirement |
|---|---|
| Kubernetes | `v1.32.10` |
| Kubeflow MCP Server | `latest` image tag |
| Kubeflow Trainer | `2.3.0` and Trainer CRDs |
| KAgent | `v1.0.0-alpha4` with `RemoteMCPServer` |
| Agentgateway | `v1.5.0` with Gateway API |
| Observability | OpenTelemetry Operator/Collector and Grafana Operator with Prometheus and Tempo |
