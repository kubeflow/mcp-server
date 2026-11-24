# Contributing skills

- Add `plugins/<plugin>/plugin.json` using [Agent Plugins v1](https://agent-plugins.org/specification).
- Store skills at `skills/<skill>/SKILL.md` within the plugin, following
  [Agent Skills](https://agentskills.io/specification). Keep support files alongside each skill.
- Register the plugin in `.agents/plugins/marketplace.json` and
  `.claude-plugin/marketplace.json`; document its owner, license, version, dependencies
  and compatible component versions.
- Validate manifests, installation, discovery and skill behavior before maintainer and
  component-owner review.

Bump the version in `plugin.json` before tagging a release as `<plugin>-v<version>`;
do not duplicate the version in catalog entries. Relative plugin sources follow the marketplace
checkout, not release tags. For tag-pinned sources, update the source reference in both catalogs
for each release. Do not use the MCP server release workflow; compatibility declarations
do not enforce version constraints.
