# [0.1.1](https://github.com/kubeflow/mcp-server/releases/tag/0.1.1) (2026-08-11)

This is the first official release of the Kubeflow MCP Server.

```
pip install kubeflow-mcp==0.1.1
```

The Kubeflow MCP Server exposes Kubeflow Trainer operations as [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) tools, letting AI agents plan, submit, monitor, and manage training jobs conversationally. Learn more about the highlights in the blog post:

* [Introducing Kubeflow MCP Server](https://blog.kubeflow.org/introducing-kubeflow-mcp/)

## 🚀 Features

- feat: add latency benchmarking and HTML reporting for trainer MCP tools ([#26](https://github.com/kubeflow/mcp-server/pull/26) by @haroon0x)
- feat(ci): add weekly OSV-Scanner version bump workflow ([#90](https://github.com/kubeflow/mcp-server/pull/90) by @Kartikeya-trivedi)
- feat: create initial release workflow for automated pypi publishing ([#28](https://github.com/kubeflow/mcp-server/pull/28) by @Krishna-kg732)
- feat(ci): add MCP protocol conformance tests ([#81](https://github.com/kubeflow/mcp-server/pull/81) by @Kartikeya-trivedi)
- feat(ci): replace pip-audit with OSV-Scanner for CVE detection ([#30](https://github.com/kubeflow/mcp-server/pull/30) by @Kartikeya-trivedi)
- feat: add OpenTelemetry tracing for tool calls ([#21](https://github.com/kubeflow/mcp-server/pull/21) by @priyank766)
- feat: add Dockerfile and container image CI ([#25](https://github.com/kubeflow/mcp-server/pull/25) by @SzymonIwaniuk)
- feat: add core MCP server, TrainerClient tools and Resources ([#2](https://github.com/kubeflow/mcp-server/pull/2) by @abhijeet-dhumal)

## 🐛 Bug Fixes

- fix: prefer project-local configuration ([#151](https://github.com/kubeflow/mcp-server/pull/151) by @pratik-naik003)
- fix: handle non-mapping YAML config ([#142](https://github.com/kubeflow/mcp-server/pull/142) by @pratik-naik003)
- fix: document KUBEFLOW_MCP_INSTRUCTION_TIER environment variable ([#146](https://github.com/kubeflow/mcp-server/pull/146) by @pratik-naik003)
- fix: make CLI serve example copyable ([#144](https://github.com/kubeflow/mcp-server/pull/144) by @pratik-naik003)
- fix: preserve OTLP endpoint query parameters ([#140](https://github.com/kubeflow/mcp-server/pull/140) by @pratik-naik003)
- fix(docs): update SECURITY.md dependency scanning to OSV-Scanner ([#147](https://github.com/kubeflow/mcp-server/pull/147) by @Kartikeya-trivedi)
- fix: clarify update training job confirmation ([#129](https://github.com/kubeflow/mcp-server/pull/129) by @pratik-naik003)
- fix: clarify log format default ([#131](https://github.com/kubeflow/mcp-server/pull/131) by @pratik-naik003)
- fix: nightly security dependency updates ([#137](https://github.com/kubeflow/mcp-server/pull/137) by @github-actions[bot])
- fix: correct OpenTelemetry install warning ([#133](https://github.com/kubeflow/mcp-server/pull/133) by @pratik-naik003)
- fix: correct MCP_TRANSPORT default ([#124](https://github.com/kubeflow/mcp-server/pull/124) by @pratik-naik003)
- fix: correct OpenTelemetry installation command ([#127](https://github.com/kubeflow/mcp-server/pull/127) by @pratik-naik003)
- fix: remove unavailable agent command docs ([#125](https://github.com/kubeflow/mcp-server/pull/125) by @pratik-naik003)
- fix(ci): create GitHub release as draft before uploading assets ([#117](https://github.com/kubeflow/mcp-server/pull/117) by @Krishna-kg732)
- fix(trainer): fallback to previous pod logs on crash and detect OpenS… ([#47](https://github.com/kubeflow/mcp-server/pull/47) by @reckless-sherixx)
- fix(core): pin kubeflow to exact version 0.4.0 ([#107](https://github.com/kubeflow/mcp-server/pull/107) by @myselfkunal)
- fix: codeql issue for incomplete URL substring sanitization ([#108](https://github.com/kubeflow/mcp-server/pull/108) by @jaiakash)
- fix(trainer): inject HF_HOME on fine_tune pods to fix OpenShift PermissionError ([#83](https://github.com/kubeflow/mcp-server/pull/83) by @ptalgulk01)
- fix(core): exempt /health and /ready from DNS rebinding Host/Origin validation ([#94](https://github.com/kubeflow/mcp-server/pull/94) by @Nagasirikancha)
- fix: unify make verify with pre-commit and document uv setup ([#98](https://github.com/kubeflow/mcp-server/pull/98) by @Solaris-star)
- fix(core): mask bare token fields in audit logs ([#93](https://github.com/kubeflow/mcp-server/pull/93) by @FAUST-BENCHOU)
- fix(core): add HTTP health probes ([#88](https://github.com/kubeflow/mcp-server/pull/88) by @floze-the-genius)
- fix(trainer): detect OpenShift pip permission error and document work… ([#46](https://github.com/kubeflow/mcp-server/pull/46) by @priyank766)
- fix: suggest valid HuggingFace model IDs when the format check fails ([#43](https://github.com/kubeflow/mcp-server/pull/43) by @AhmedDlshad007)
- fix(trainer): workaround torchtune top-level HF dataset paths ([#45](https://github.com/kubeflow/mcp-server/pull/45) by @reckless-sherixx)
- fix(trainer): execute top-level train calls and strictly validate func_args ([#31](https://github.com/kubeflow/mcp-server/pull/31) by @reckless-sherixx)

## ⚙️ Miscellaneous Tasks

- chore: expand unit coverage for monitoring tools and is_mcp_managed (#68) ([#114](https://github.com/kubeflow/mcp-server/pull/114) by @ptalgulk01)
- chore(ci): bump actions/checkout from 4.3.1 to 7.0.1 ([#113](https://github.com/kubeflow/mcp-server/pull/113) by @dependabot[bot])
- chore: add unit tests for training tools confirmed paths (#68) ([#150](https://github.com/kubeflow/mcp-server/pull/150) by @ptalgulk01)
- chore: add unit tests for common utils, types, and constants (#68) ([#149](https://github.com/kubeflow/mcp-server/pull/149) by @ptalgulk01)
- chore(ci): bump actions/download-artifact from 7 to 8 ([#136](https://github.com/kubeflow/mcp-server/pull/136) by @dependabot[bot])
- chore(ci): bump actions/setup-python from 5 to 7 ([#102](https://github.com/kubeflow/mcp-server/pull/102) by @dependabot[bot])
- chore(ci): bump github/codeql-action from 4 to 4.37.4 in the actions group ([#135](https://github.com/kubeflow/mcp-server/pull/135) by @dependabot[bot])
- chore(test): add unit test scaffold with testcase harness ([#6](https://github.com/kubeflow/mcp-server/pull/6) by @abhijeet-dhumal)
- chore(ci): prepare 0.1.0rc1 and MCP Registry metadata ([#119](https://github.com/kubeflow/mcp-server/pull/119) by @abhijeet-dhumal)
- chore(ci): add git-cliff changelog generation and release ([#118](https://github.com/kubeflow/mcp-server/pull/118) by @Raghul-M)
- chore: add jaiakash and Krishna-kg732 as reviewers ([#55](https://github.com/kubeflow/mcp-server/pull/55) by @abhijeet-dhumal)
- chore(docs): add SECURITY.md and expand security model in ARCHITECTURE.md ([#14](https://github.com/kubeflow/mcp-server/pull/14) by @abhijeet-dhumal)
- chore(ci): add workflow to validate uv.lock and check for regressions ([#73](https://github.com/kubeflow/mcp-server/pull/73) by @priyank766)
- chore(docs): add AGENTS.md and Copilot review instructions ([#70](https://github.com/kubeflow/mcp-server/pull/70) by @Raghul-M)
- chore(ci): align pre-commit config with Kubeflow-SDK Introduced check-added-large-files ([#71](https://github.com/kubeflow/mcp-server/pull/71) by @Raghul-M)
- chore(ci): bump actions/setup-python from 5 to 6 ([#77](https://github.com/kubeflow/mcp-server/pull/77) by @dependabot[bot])
- chore(ci): bump actions/github-script from 8 to 9 ([#76](https://github.com/kubeflow/mcp-server/pull/76) by @dependabot[bot])
- chore(ci): bump docker/setup-qemu-action from 3 to 4 ([#75](https://github.com/kubeflow/mcp-server/pull/75) by @dependabot[bot])
- chore(ci): bump astral-sh/setup-uv from 4 to 7 ([#74](https://github.com/kubeflow/mcp-server/pull/74) by @dependabot[bot])
- chore(ci): bump docker/setup-buildx-action from 3 to 4 ([#78](https://github.com/kubeflow/mcp-server/pull/78) by @dependabot[bot])
- chore: added pre-commit to dev dependency group ([#86](https://github.com/kubeflow/mcp-server/pull/86) by @Shaurya2k06)
- chore(docs): add ADOPTERS.md ([#69](https://github.com/kubeflow/mcp-server/pull/69) by @Raghul-M)
- chore: add demo video, Coveralls badge, and DeepWiki badge to README ([#64](https://github.com/kubeflow/mcp-server/pull/64) by @nabsei)
- chore(ci): bump GitHub Actions to latest major versions ([#49](https://github.com/kubeflow/mcp-server/pull/49) by @priyank766)
- chore: add Kubeflow MCP server roadmap ([#13](https://github.com/kubeflow/mcp-server/pull/13) by @abhijeet-dhumal)
- chore: add CI and community hygiene ([#3](https://github.com/kubeflow/mcp-server/pull/3) by @abhijeet-dhumal)
- chore: initialize kubeflow-mcp repository skeleton ([#1](https://github.com/kubeflow/mcp-server/pull/1) by @abhijeet-dhumal)

## New Contributors

- @pratik-naik003 made their first contribution in [#151](https://github.com/kubeflow/mcp-server/pull/151)
- @ptalgulk01 made their first contribution in [#114](https://github.com/kubeflow/mcp-server/pull/114)
- @dependabot[bot] made their first contribution in [#113](https://github.com/kubeflow/mcp-server/pull/113)
- @haroon0x made their first contribution in [#26](https://github.com/kubeflow/mcp-server/pull/26)
- @Kartikeya-trivedi made their first contribution in [#90](https://github.com/kubeflow/mcp-server/pull/90)
- @github-actions[bot] made their first contribution in [#137](https://github.com/kubeflow/mcp-server/pull/137)
- @abhijeet-dhumal made their first contribution in [#6](https://github.com/kubeflow/mcp-server/pull/6)
- @Raghul-M made their first contribution in [#118](https://github.com/kubeflow/mcp-server/pull/118)
- @Krishna-kg732 made their first contribution in [#117](https://github.com/kubeflow/mcp-server/pull/117)
- @reckless-sherixx made their first contribution in [#47](https://github.com/kubeflow/mcp-server/pull/47)
- @myselfkunal made their first contribution in [#107](https://github.com/kubeflow/mcp-server/pull/107)
- @jaiakash made their first contribution in [#108](https://github.com/kubeflow/mcp-server/pull/108)
- @Nagasirikancha made their first contribution in [#94](https://github.com/kubeflow/mcp-server/pull/94)
- @Solaris-star made their first contribution in [#98](https://github.com/kubeflow/mcp-server/pull/98)
- @FAUST-BENCHOU made their first contribution in [#93](https://github.com/kubeflow/mcp-server/pull/93)
- @floze-the-genius made their first contribution in [#88](https://github.com/kubeflow/mcp-server/pull/88)
- @priyank766 made their first contribution in [#46](https://github.com/kubeflow/mcp-server/pull/46)
- @Shaurya2k06 made their first contribution in [#86](https://github.com/kubeflow/mcp-server/pull/86)
- @nabsei made their first contribution in [#64](https://github.com/kubeflow/mcp-server/pull/64)
- @AhmedDlshad007 made their first contribution in [#43](https://github.com/kubeflow/mcp-server/pull/43)
- @SzymonIwaniuk made their first contribution in [#25](https://github.com/kubeflow/mcp-server/pull/25)
