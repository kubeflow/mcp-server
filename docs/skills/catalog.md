# Agent Skills catalog

Catalog skeleton for independently released Kubeflow skill plugins.

- `plugins/`: plugin packages.
- `.agents/plugins/marketplace.json`: Codex catalog.
- `.claude-plugin/marketplace.json`: Claude Code catalog.

Register this checkout with either client:

```bash
claude plugin marketplace add .
codex plugin marketplace add .
```

Once published on the default branch, use `kubeflow/mcp-server` instead of `.`.
See [contribution guidance](CONTRIBUTING.md) to add a plugin.
