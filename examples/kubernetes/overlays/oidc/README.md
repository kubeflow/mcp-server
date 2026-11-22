# OIDC / OAuth 2.1 Kustomize Overlay

This overlay customizes the base Kubeflow MCP Server deployment with OpenID Connect (OIDC) authentication environment variables, enabling native token validation and IdP auto-discovery.

## Deployment

Deploy using `kubectl`:

```bash
kubectl apply -k examples/kubernetes/overlays/oidc
```
