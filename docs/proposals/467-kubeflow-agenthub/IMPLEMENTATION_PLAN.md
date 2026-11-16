# KEP-467: MCP Server and Agent Skills — Implementation Plan

This plan describes how to deliver the [Agent Skills catalog](README.md) alongside the MCP server;
GitHub hosts the initial catalog. A hosted registry is out of scope.

## Design Principles

- Keep one portable source for each skill. The initial Trainer skill works without MCP.
- Skills are standalone workflows; they do not wrap or duplicate MCP tools. If a later skill needs MCP,
  document its dependency and minimum server version. Server authorization and confirmation still
  apply to MCP tools.
- Use Agent Plugins v1 for portable packaging; validate each client's required metadata separately.
- MCP Skills exposure is optional and read-only. Existing tool schemas and permissions stay unchanged.

## Repository Changes

The tree below shows additions and changes relative to the existing `kubeflow/mcp-server` project.
`[new]`, `[update]`, and `[optional]` mark the proposed work; existing MCP code stays unchanged.

```text
kubeflow/mcp-server/
├── .agents/                                                    [new]
│   └── plugins/marketplace.json                                [new; Codex catalog]
├── .claude-plugin/                                             [new]
│   └── marketplace.json                                        [new; Claude Code catalog]
├── plugins/                                                    [new]
│   └── training/                                               [new]
│       ├── plugin.json                                         [Agent Plugins v1]
│       ├── .claude-plugin/plugin.json                          [optional Claude metadata]
│       └── skills/
│           └── trainer-script-adapter/SKILL.md                 [initial skill]
├── docs/                                                       [existing]
│   └── skills/                                                 [new]
│       ├── catalog.md                                          [entries, owners, versions, clients, dependencies, support]
│       └── CONTRIBUTING.md                                     [contribution and maintenance guidance]
├── tests/                                                      [existing]
│   └── skills/                                                 [new]
│       ├── test_skill_validation.py                            [frontmatter, names, links, layout]
│       ├── test_marketplace_manifests.py                       [plugin and catalog schemas]
│       ├── test_package_contents.py                            [packaged skill files]
│       └── evaluations/                                        [representative prompts and output checks]
├── README.md                                                   [update; link to install and fallback guidance]
└── kubeflow_mcp/                                               [existing; unchanged]
```

### Package rules

- Both catalogs point to `./plugins/training/`; `kubeflow-mcp` is the proposed marketplace name and
  `training` is the plugin name. The Claude catalog is `.claude-plugin/marketplace.json`.
- `plugin.json` follows Agent Plugins v1. Add Claude-specific plugin metadata only if needed, keeping
  its name and version aligned with the portable manifest.
- Skill frontmatter requires `name` and `description`, and the skill name must match its directory.
- Keep `SKILL.md` and any referenced support files inside each skill directory, using relative paths
  so users can copy it into a standard skills directory. Skills must not depend on plugin metadata.
- The initial training plugin has no MCP config. If a future plugin bundles MCP configuration, that config connects to the
  server but does not install it; use separate plugins if release schedules differ.

## Phase 0: Scope and Ownership

### Compatibility targets

These are initial validation targets, not prerequisites for installing a standalone skill. SDK and
runtime requirements apply to generated artifacts; MCP compatibility matters only when a skill uses
the server. The catalog release is independent of the MCP Server 0.2 release.

| Component | Initial validation target |
| --- | --- |
| Kubeflow SDK | `>=0.5.0,<0.6.0` |
| Trainer | `>=2.3.0,<2.4.0` |
| Python | 3.10–3.12 |
| Kubernetes | `>=1.27` |

Before each release, recheck the [server requirements](https://github.com/kubeflow/mcp-server/blob/main/README.md#requirements)
and [`SDK_COMPATIBILITY`](https://github.com/kubeflow/mcp-server/blob/main/kubeflow_mcp/common/constants.py).

Record tested versions and skill-specific dependencies in the catalog. If the KFP candidate is
approved, document its compiler version separately.

### Decisions to record

1. Assign skill and catalog maintainers; confirm review across WG Agents, MCP Server, WG ML Experience,
   and component owners. Reconcile scope with KEP-936.
2. Define supported Python script patterns for callable conversion, including explicit inputs, helper
   functions, and data/model setup. Separately define supported PyTorch versions and runtimes for
   optional distributed adaptation. Include the KFP candidate only if an upstream skill meets
   its artifact-only scope and its owner agrees.
3. Select supported clients and marketplace formats. Record tested installs and direct-copy paths in
   `docs/skills/catalog.md`.

## Phase 1: Prepare the Catalog

### Actions

1. Add `plugins/training/plugin.json` and the skill at
   `plugins/training/skills/trainer-script-adapter/`. Add
   `plugins/training/.claude-plugin/plugin.json` only if Claude needs extra metadata. Keep the training
   plugin independent of MCP. In `docs/skills/catalog.md`, record each skill's owner, source, version,
   component, supported clients, dependency mode (`standalone`, `MCP optional`, or `MCP required`), and
   minimum compatible server version when needed.
2. Build the script adapter to enclose supported Python workloads in a self-contained callable with
   explicit inputs, imports, helpers, dependencies, and assumptions. For Trainer, return `train()`
   for `CustomTrainer(func=train)` plus runtime and resource configuration. Add distributed PyTorch
   setup only when requested. Keep KFP component/pipeline generation in Story 2; reject unsupported
   input and do not submit jobs or duplicate the MCP server's
   [`run_custom_training` wrapper](https://github.com/kubeflow/mcp-server/blob/main/kubeflow_mcp/trainer/api/training.py).
3. Add Claude and Codex catalog entries named `training` that point to `./plugins/training/`.
   Validate each catalog against its supported client's schema and record the tested client version.
   Verify the proposal's install and fallback paths; a valid portable manifest alone does not establish
   client compatibility.
4. Prepare matching SemVer metadata in the plugin and both catalogs, with release notes and install,
   update, and removal guidance. Publish only after Phase 2 validation and owner review pass.

## Phase 2: Validate and Release

### Tests

- Validate Agent Skills frontmatter (`name` and `description`), directory/name agreement, internal
  links, the portable manifest, the optional Claude manifest when present, marketplace catalogs, and
  package contents.
- Evaluate supported inputs, unsupported scripts, and missing prerequisites. Run converted functions
  from small controlled fixtures in a clean process; check explicit arguments, enclosed helpers,
  expected outputs, and declared dependencies without notebook state or undeclared globals. Preserve
  observable behavior for callable-only conversion.
- For distributed adaptation, use a small local multi-process CPU fixture to check disjoint data
  shards, epoch reseeding, rank-zero checkpoint writes, and process-group cleanup. Check requested
  nodes/resources and SDK compatibility separately; constructing `CustomTrainer(func=train)` alone
  is insufficient. Record any intended behavior changes. Validate GPU claims with a representative
  multi-node GPU run by maintainers; the skill itself must not submit workloads.
- Verify the GitHub marketplace install and skill discovery in each supported client. Also verify the
  direct-copy fallback for compatible clients without marketplace support. These checks must not
  require a live cluster or installed MCP server; Trainer skill loading and use must not require
  `kubeflow-mcp`.
- If MCP exposure ships, run the relevant
  [SEP-2640 conformance scenarios](https://github.com/modelcontextprotocol/conformance/pull/330) and
  test reads, digests, missing files, path containment, and packaging.

Use [`skills-ref validate`](https://github.com/agentskills/agentskills/tree/main/skills-ref) as a
reference check, not the only quality gate. Validate `plugin.json` against the
[Agent Plugins v1 schema](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json).

### Release

Publish only after the checks above and owner review pass:

- Release the plugin independently from `kubeflow-mcp`, using tags such as `training-v0.1.0` and
  separate release notes. The tag must capture the validated plugin and matching client catalogs;
  a skill-only release must not publish the MCP server to PyPI or a container registry.
- Document installation from a tagged checkout using each client's local marketplace support, or
  direct-copy the complete skill folder from that checkout. Test this pinned installation path so
  users can reproduce a release independently of the default branch.
- Require skill-owner and component-owner review; inspect instructions, scripts, references, and
  dependencies for unsafe behavior; check licensing, attribution, and secret-free assets.
- Document supported Kubeflow/client versions, plugin versioning, MCP dependency declarations, updates,
  and breaking changes.
- Review ownership, conformance, API compatibility, and client support periodically with WG Agents.

### Acceptance criteria

- A versioned catalog is published; bundled skills are distinguished from upstream links. List a KFP
  upstream skill only if it meets the artifact-only contract; bundle it only with owner agreement.
- The portable manifest, optional Claude manifest, and marketplace catalogs validate. Marketplace
  install and skill discovery are verified without requiring the MCP server.
- The Trainer skill returns a self-contained function, arguments, dependencies, runtime, configuration,
  and assumptions. Callable behavior and rejection evaluations pass, SDK compatibility is verified, and
  distributed GPU claims have multi-node execution evidence. The skill works without MCP.
- Ownership, contribution, safety, and support guidance are documented.

### Ongoing maintenance

- Named maintainers and a documented review process are active.
- Supported clients and Kubeflow versions are published and kept current.
- Releases are reproducible, with update/deprecation guidance and no unresolved critical issues.

## Optional Phase 3: Expose Skills Through MCP

This phase is an optional follow-up. Confirm FastMCP and client support before implementation.
Expose the same packaged skills through
[SEP-2640](https://modelcontextprotocol.io/seps/2640-skills-extension), following the negotiated
protocol's capabilities, discovery methods, file manifests, and resource reads. Verify digests,
missing-file handling, and path containment with extension conformance tests. Keep this interface
read-only; it must not install or execute skills.
