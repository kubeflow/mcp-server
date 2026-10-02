# KAgent with MCP and OpenTelemetry traces

This bundle combines the Kubeflow MCP deployment used by KAgent with an
OpenTelemetry Collector. Use it when the OpenTelemetry Operator is installed
and tracing is required.

## Prerequisites

- The prerequisites from [`examples/kagent`](../kagent/README.md).
- The OpenTelemetry Operator and `OpenTelemetryCollector` CRD.
- A trace backend such as Tempo or Jaeger if traces must be viewed in a dashboard.

Create the MCP credentials from the KAgent profile before applying this bundle.
The collector receives traces only when the MCP image includes its optional
OpenTelemetry dependencies. Build and publish an image with the optional group
enabled, then set that image in `mcp-tracing-patch.yaml` before applying:

```bash
docker build --build-arg INSTALL_OTEL=true -t <registry>/kubeflow-mcp:otel .
docker push <registry>/kubeflow-mcp:otel
```

## Deploy

```bash
kubectl apply -k examples/kagent-observability
kubectl rollout status deployment/kubeflow-mcp-http -n kagent
```

This applies the MCP server, RBAC, Service, `RemoteMCPServer`, collector, and NetworkPolicies. The MCP server exports OTLP/HTTP traces to the in-cluster collector.

Configure the collector's exporter for the Tempo or Jaeger deployment used by
your cluster before enabling trace queries. The collector uses the OpenTelemetry
`debug` exporter by default; replace it with an OTLP exporter when a trace
backend is available. The OpenTelemetry Operator does not provide trace storage
or a dashboard.

## Reference trace view

![Kubeflow MCP tools/list trace in a tracing UI](assets/mcp-tools-list-trace.png)

This capture illustrates the trace detail available after MCP traffic reaches
the collector and trace backend. Service names and available filters depend on
the backend and collector configuration used by the cluster.
