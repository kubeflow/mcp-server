# Stateless HTTP Kustomize Overlay

This overlay customizes the base Kubeflow MCP Server manifests for horizontally scalable, production stateless HTTP deployments.

## Features

- **Multiple Replicas**: Scales baseline deployment to at least 2 replicas.
- **Horizontal Pod Autoscaling**: Automatically scales up to 10 replicas under CPU load.
- **Ingress Controller Route**: Provides an HTTP ingress definition for routing MCP agent requests.

## Deployment

Deploy using `kubectl`:

```bash
kubectl apply -k examples/kubernetes/overlays/stateless-http
```
