# Upgrading from 0.1.x to 0.2.0

0.2.0 is the first release since 0.1.1 (2026-08-11). The 0.1.x tags were cut from the 0.1.0rc1 base, so every change merged to `main` since then ships in 0.2.0. Most of it needs no action. Read section 1 before upgrading.

## 1. Action required

| Change | PR | What to do |
|---|---|---|
| Kubeflow SDK moves from `==0.4.0` to `>=0.5.0,<0.6.0`. The server now needs Kubeflow Trainer `>=2.3.0,<2.4.0` (0.1.x used Trainer 2.2.1). | #201 | Upgrade Trainer on the cluster before upgrading the server. |
| FastMCP is pinned to `==4.0.10` (MCP SDK v2). | #292 | Install the server in its own environment if another package needs a different FastMCP version. |
| The `sse` transport is deprecated and logs a warning at startup. It still works in this release. | #292 | Switch `--transport sse` / `MCP_TRANSPORT=sse` to `http` (Streamable HTTP). |
| The `mcp.session.id` span attribute is removed. The protocol is now sessionless. | #292 | Update trace queries or dashboards that group by `mcp.session.id`. |
| The server's own tool span is now an INTERNAL child named `execute_tool <tool>` under FastMCP's `tools/call <tool>` SERVER span. 0.1.x opened a second SERVER span with the same name as FastMCP's. The attributes are unchanged. | #423 | Update trace queries that read the audit attributes from the `tools/call` span. |
| Tool tags are no longer in `ToolAnnotations`. Clients get them as `_meta.fastmcp.tags`. | #292 | Clients that filter tools by tag read `_meta.fastmcp.tags`. |
| A policy file that exists but can't be loaded (bad YAML, or PyYAML missing) now stops the server with `RuntimeError`. In 0.1.x it was ignored, which meant no restrictions. A missing file is still fine. | #285 | Check that `~/.kf-mcp-policy.yaml` loads before upgrading. |
| Policy values with the wrong shape also stop the server, and the error names the file and the key: a single string where a list is expected (`deny: risk:destructive`, `allow: list_*`, `namespaces: ml-team-dev`), a quoted `read_only: "false"`, or an unknown `category:` name. In 0.1.x these were misread. A single string in `deny` denied nothing and one in `allow` allowed every tool. | #409, #451 | Write `allow`, `deny` and `namespaces` as YAML lists and `read_only` as a plain `true` / `false`. |
| A config file that exists but can't be loaded (bad YAML, a root that isn't a mapping, or PyYAML missing) now stops the server. In 0.1.x it logged a warning and started with the defaults. | #431, #394 | Check that `.kubeflow-mcp.yaml` loads before upgrading. |
| `wait_for_training` on a job that fails returns a normal response with `status: "Failed"` and `reached: false`. In 0.1.x it returned `SDK_ERROR`. | #399 | Agents that treated the error as "job failed" check `reached` instead. |
| `update_training_job` takes `confirmed=False` by default. Without `confirmed=True` it returns a preview and does not suspend or resume the job. | #198 | Scripts and agents that suspend or resume jobs pass `confirmed=True`. |
| Tool errors are returned with `isError: true` (was `false`). The error body is unchanged. | #282 | Clients that raise on `isError` (FastMCP's `Client.call_tool` does by default) now raise on tool errors. Check your error handling. |
| A custom persona that inherits from an unrestricted parent (`tools: "*"`, such as `platform-admin`) now gets all tools. In 0.1.x a bug limited it to its own tool list. | #274 | Review custom personas in your policy file. Some get more access after the upgrade. |
| `./.kubeflow-mcp.yaml` in the working directory now takes priority over the file in your home directory. | #151 | If you have both, the local file now applies. |
| `cb_failure_threshold` must be greater than 0 and `cb_recovery_timeout` can't be negative. Other values now stop the server at startup. | #351 | Check the `resilience` section of your config and `KUBEFLOW_MCP_CB_*` env vars. |
| `rate_limit` and `rate_capacity` must be greater than 0. Other values now stop the server at startup. | #332 | Check the `resilience` section of your config. |

## 2. Changed responses

No action needed, but clients that check error codes or response fields may notice these.

**Error codes**

- `get_training_events` and `wait_for_training` on a missing job: `SDK_ERROR` → `RESOURCE_NOT_FOUND`. These no longer trip the circuit breaker. (#248)
- `delete_training_job` / `update_training_job` on a missing job, for non-admin personas: `VALIDATION_ERROR` → `RESOURCE_NOT_FOUND`. (#158)
- `estimate_resources` with a malformed or unknown model id: `SDK_ERROR` → `VALIDATION_ERROR`. A 401 is still `SDK_ERROR`. (#258)
- `inspect_controller` when the pod lookup fails: a 403 now returns `PERMISSION_DENIED` and other failures return `KUBERNETES_ERROR` (was always `RESOURCE_NOT_FOUND`). (#246)
- `execute_tool` in progressive and semantic modes with invalid arguments: `VALIDATION_ERROR`, and it no longer trips the circuit breaker. (#241)
- `fine_tune`, `run_custom_training` and `run_container_training` when the cluster denies the request with a 403 (RBAC or an admission policy): `SDK_ERROR` → `PERMISSION_DENIED`. (#318)

**Stricter input validation.** These inputs now return `VALIDATION_ERROR` instead of reaching the SDK: discovery names (#163), monitoring job names (#165), runtime names in `create/patch/delete_runtime` (#236), the `inspect_controller` namespace (#256), the namespace on every other tool that takes one, including an empty string (#344), `wait_for_training` `target_statuses` (#213), `list_training_jobs` status (#281), `get_server_logs` level (#215, `WARN` used to be read as INFO), `resources_per_node` (#276), an empty `--clients` / `KUBEFLOW_MCP_CLIENTS` (#329), `service_account_name` (#318), the container `image` in `run_custom_training` and `run_container_training` (#319, #443), and a `wait_for_training` `polling_interval` that isn't less than `timeout_seconds` (#399).

`fine_tune` checks `hf://` references before submitting. A dataset needs `hf://<org>/<dataset>`; a model may be unscoped, like `hf://gpt2`. On a confirmed submission, a repo the Hub reports as missing is a `VALIDATION_ERROR`. Private, gated or unreachable repos don't block, and the check needs `HF_TOKEN` on the server host. (#320)

Dotted runtime names such as `torchtune-llama3.2-1b` are now accepted. (#233)

**Response fields**

- `list_training_jobs` and `get_training_events` add `returned` and `has_more`. (#208, #244)
- `get_training_job` adds `steps` (failed steps first, at most 32), `step_summary` and `num_nodes`. (#425)
- `get_training_events` with a `limit` keeps the most recent events. 0.1.x kept the oldest. (#439)
- JSON logs and `get_server_logs` entries carry `trace_id` and `span_id` while a span is active. (#423)
- `get_training_logs` reports `lines: 0` for empty logs (was 1). (#210)
- Truncated logs keep the tail, with the marker at the front. 0.1.x kept the head and lost the traceback. (#220)
- `serverInfo.version` is the package version. 0.1.x reported FastMCP's version. (#262)
- `_meta.next` is withheld when `pre_flight` / `check_compatibility` report blockers (#237), and on confirm-gate previews (#296).

## 3. New

- `get_training_logs(since_line=N, max_lines=M)` reads logs incrementally and returns `next_offset`, `total_lines` and `log_reset`. Without `since_line` it behaves as before. (#294)
- `examples/kubernetes/manifests.yaml` for running the server in-cluster, with RBAC, an auth Secret and `/health` / `/ready` probes. (#160)
- MCP SDK v2 and the 2026-07-28 MCP spec. (#292)
- A custom persona from the policy file works from `--persona`, `KUBEFLOW_MCP_PERSONA` and the config file. An unknown persona from any of them stops `serve` with a usage error. (#411)
- [`docs/quickstart.md`](../quickstart.md) covers install, client setup, transports and troubleshooting. (#328)

Only two tool signatures changed: `update_training_job` gained `confirmed`, and `get_training_logs` gained `since_line` and `max_lines`. No tools were added or removed.

For all other fixes, see the [changelog](../../CHANGELOG).
