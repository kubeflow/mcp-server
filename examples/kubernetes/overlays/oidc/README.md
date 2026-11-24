# OIDC / OAuth 2.1 Kustomize Overlay

This overlay customizes the base Kubeflow MCP Server deployment with OpenID Connect (OIDC) authentication environment variables, enabling native token validation and IdP auto-discovery.

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `KUBEFLOW_MCP_OIDC_ISSUER` | OpenID Connect Issuer URL for OIDC discovery (`.well-known/openid-configuration`) | `https://accounts.google.com` |
| `KUBEFLOW_MCP_OIDC_AUDIENCE` | Expected audience (`aud`) in validated JWT tokens | `kubeflow-mcp` |
| `KUBEFLOW_MCP_JWKS_URI` | Explicit JWKS endpoint URL for token signature verification | `https://www.googleapis.com/oauth2/v3/certs` |
| `KUBEFLOW_MCP_JWT_ISSUER` | Expected token issuer (`iss`) in validated JWT tokens | `https://accounts.google.com` |

## Deployment

Deploy using `kubectl`:

```bash
kubectl apply -k examples/kubernetes/overlays/oidc
```
