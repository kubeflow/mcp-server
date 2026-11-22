# Kubeflow MCP Server Helm Chart

Official Helm chart for deploying the [Kubeflow MCP Server](https://github.com/kubeflow/mcp-server) into Kubernetes clusters, delivering production-ready, stateless Streamable HTTP model context protocol services for autonomous AI agents.

## Overview

Kubeflow MCP Server exposes tools for managing Kubeflow workloads (Trainer, Spark, Pipelines) over the Model Context Protocol (MCP).

This chart provides:
- **Stateless HTTP Deployment**: Horizontally scalable across multiple replicas backed by Horizontal Pod Autoscaler (HPA).
- **Least-Privilege RBAC**: Scoped ClusterRole for platform inspection and namespace-scoped Role for workload management.
- **Enterprise Security**: Read-only root filesystem, non-root user (UID 1000), dropped Linux capabilities, seccomp runtime default.
- **Authentication**: Token and OIDC/OAuth 2.1 integration via Kubernetes Secrets.
- **Observability**: Readiness (`/ready`), liveness (`/health`), and optional Prometheus ServiceMonitor integration.

## Prerequisites

- Kubernetes 1.28+
- Helm 3.10+
- [Kubeflow Trainer](https://github.com/kubeflow/trainer) CRDs installed in the target cluster

## Quick Start

### 1. Install with an Auth Token

Generate an authentication token and install the chart:

```bash
helm install kubeflow-mcp ./charts/kubeflow-mcp \
  --namespace kubeflow \
  --create-namespace \
  --set auth.token="your-secure-mcp-token"
```

### 2. Connect Your AI Agent

Port-forward the service:

```bash
kubectl port-forward svc/kubeflow-mcp 8000:8000 -n kubeflow
```

Point your MCP client (Claude Code, Cursor, fastmcp, or custom agent runtime) to:

```
http://localhost:8000/mcp
```

With HTTP header:
```
Authorization: Bearer your-secure-mcp-token
```

## Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `replicaCount` | Number of server replicas | `2` |
| `image.repository` | Container image repository | `ghcr.io/kubeflow/mcp-server` |
| `image.tag` | Container image tag | `0.2.0rc1` |
| `server.clients` | List of enabled MCP clients (`trainer`, `spark`, `pipelines`) | `["trainer", "spark", "pipelines"]` |
| `server.persona` | Role persona (`readonly`, `data-scientist`, `ml-engineer`, `platform-admin`) | `ml-engineer` |
| `server.mode` | Tool loading mode (`full`, `progressive`, `semantic`) | `full` |
| `server.transport` | Protocol transport (`http` or `stdio`) | `http` |
| `server.instructionTier` | Instruction detail level (`minimal`, `standard`, `detailed`) | `standard` |
| `auth.enabled` | Enable HTTP authentication | `true` |
| `auth.token` | Literal bearer token string | `""` |
| `auth.existingSecret` | Name of pre-existing Secret containing token | `""` |
| `auth.secretKey` | Key name inside the secret | `token` |
| `auth.oidc.enabled` | Enable OIDC / OAuth 2.1 authentication | `false` |
| `rbac.create` | Create RBAC roles and bindings | `true` |
| `rbac.clusterRole` | Create ClusterRole for platform reads | `true` |
| `autoscaling.enabled` | Enable Horizontal Pod Autoscaler | `true` |
| `autoscaling.minReplicas` | Minimum HPA replicas | `2` |
| `autoscaling.maxReplicas` | Maximum HPA replicas | `10` |
| `probes.liveness.httpGet.path` | Liveness probe endpoint | `/health` |
| `probes.readiness.httpGet.path` | Readiness probe endpoint | `/ready` |
| `monitoring.serviceMonitor.enabled` | Deploy Prometheus Operator ServiceMonitor | `false` |
