# Contributing skills

- Add `plugins/<plugin>/plugin.json` using [Agent Plugins v1](https://agent-plugins.org/specification).
- Store skills at `skills/<skill>/SKILL.md` within the plugin, following
  [Agent Skills](https://agentskills.io/specification). Keep support files alongside each skill.
- Register the plugin in both catalogs and document its owner, license, version,
  dependencies and compatible component versions.
- Validate manifests, installation, discovery and skill behavior before maintainer and
  component-owner review.

Release plugins independently with `<plugin>-v<version>` tags. Do not use the MCP server
release workflow; compatibility declarations do not enforce version constraints.
