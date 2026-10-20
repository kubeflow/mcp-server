# Mutation authorization and safety model

Rules every tool that changes cluster state must follow, in any client (`trainer`,
`optimizer`, `hub`, ...). It describes what the server does today, what a new mutating
tool must do, and the gaps that still need follow-up work.

See also: [CONVENTIONS.md](../CONVENTIONS.md) for module layout and naming,
[`core/policy.py`](../../kubeflow_mcp/core/policy.py) and
[`core/security.py`](../../kubeflow_mcp/core/security.py).

## Mutating tools today

| Tool | Minimum persona | Namespace check | Ownership check | Preview type | `destructiveHint` |
|---|---|---|---|---|---|
| `fine_tune` | data-scientist | yes | n/a (creates, adds MCP label) | `PreviewResponse` | no |
| `run_custom_training` | data-scientist | yes | n/a (creates, adds MCP label) | `PreviewResponse` | no |
| `run_container_training` | ml-engineer | yes | n/a (creates, adds MCP label) | `PreviewResponse` | no |
| `update_training_job` | ml-engineer | yes | yes, unless platform-admin | `PreviewResponse` | no |
| `delete_training_job` | data-scientist | yes | yes, unless platform-admin | `PreviewResponse` | yes |
| `create_runtime` | platform-admin | n/a (cluster-scoped) | n/a | `ToolResponse` (legacy) | no |
| `patch_runtime` | platform-admin | n/a (cluster-scoped) | n/a | `ToolResponse` (legacy) | no |
| `delete_runtime` | platform-admin | n/a (cluster-scoped) | n/a | `ToolResponse` (legacy) | yes |

## Order of checks

This is the intended order for a mutating call. Not every tool follows it yet:
`fine_tune()` checks namespace policy before validating its other inputs, and the
training tools still run some input checks after the namespace check (#319 moves the
image check earlier). See the known gaps below.

The authorization checks (exposure, namespace policy and ownership) fail closed: if they
cannot decide, the call is refused. Some informational checks are best effort and do not
block; see [Best-effort checks](#best-effort-checks).

1. **Exposure.** The tool is only registered if the persona allows it
   (`get_allowed_tools`), the policy file's `allow`/`deny` keep it
   (`apply_policy_filters`), and `read_only` is off. Under `read_only`, any tool without
   `readOnlyHint: True` is removed, so a new tool with no annotations is hidden rather
   than exposed.
2. **Input validation.** Names, image references, resources and similar inputs are
   checked before any Kubernetes call and return `VALIDATION_ERROR`.
3. **Namespace policy.** Namespaced tools call `check_namespace_allowed()`. An omitted
   namespace is resolved to the effective default and checked too, so the implicit
   default cannot bypass the allowlist.
4. **Ownership.** For tools that change an existing namespaced resource, non-admin
   personas may only act on resources that carry the MCP managed label
   (`_require_mcp_ownership`). A missing resource is `RESOURCE_NOT_FOUND`; an unmanaged
   one is `VALIDATION_ERROR`. If ownership cannot be read, the call is refused.
5. **Confirm gate.** With `confirmed=False` (the default) the tool returns a preview and
   changes nothing. Only `confirmed=True` executes. Steps 1 to 4 run before the preview,
   so a preview means those checks passed when it was generated. The confirmed call can
   still be refused later, for example by RBAC or an admission policy (see below), or
   because the resource changed in between.
6. **Execution and audit.** Every call is wrapped by `_audit_wrap`: rate limit, circuit
   breaker, an audit log entry with masked parameters, and a trace span. Kubernetes errors
   are returned as `ToolError` with a stable `error_code`, never as a traceback.

### Best-effort checks

These add information to a preview but do not stop a call when they fail:

- **`delete_runtime` dependency check.** The preview lists TrainJobs that reference the
  runtime. If listing TrainJobs fails, the list is treated as empty, so the preview says
  no jobs depend on it and a `confirmed=True` delete still goes ahead. #340 makes the
  preview report that the check failed instead of showing an empty list; whether the
  confirmed delete should then be refused is still open.

The Kubernetes API is the final authority. RBAC or an admission policy can still deny a
call that passed all of the above, and that denial should be reported as
`PERMISSION_DENIED`, not as a generic error.

## Rules for a new mutating tool

- Accept `confirmed: bool = False` and return a `PreviewResponse` when it is false. The
  preview should say exactly what will change and, for deletes, what depends on it.
- Run validation, namespace and ownership checks before building the preview.
- Set `readOnlyHint: False`. Set `destructiveHint: True`, add `[DESTRUCTIVE]` to the
  description and add the tool to `DESTRUCTIVE_TOOLS` if it deletes or overwrites data
  that cannot be recovered.
- Add the tool to the lowest persona that should have it in `core/policy.py`, and extend
  the persona matrix tests.
- Label created resources with `MCP_MANAGED_LABEL` and check ownership before changing or
  deleting existing ones, unless the tool is platform-admin only.
- Never log raw arguments. `_audit_wrap` logs them through `mask_sensitive_data`; do not
  add extra logging of secrets, tokens or full specs.
- Do not retry a mutation automatically. A retry can overwrite someone else's change.

## Known gaps and proposed follow-ups

1. **The preview-then-confirm order is not enforced.** The gate is checked per call, but
   the server cannot tell whether a preview was ever shown. In #311 a 3B model called
   `delete_training_job(..., confirmed=True)` first in 5 of 5 runs, and the server
   executed it as designed. Options: a short-lived token returned by the preview that the
   confirmed call must send back, or native MCP elicitation (ROADMAP Phase 4). A token
   changes tool schemas, so it needs maintainer agreement first. Tracked in #433, which
   needs a decision and an owner before this gap counts as resolved for v0.2.
2. **`patch_runtime` is not marked destructive.** A bad patch can break every job that
   uses the runtime, and without a `metadata.resourceVersion` it silently overwrites
   concurrent changes. #422 adds conflict detection; the annotation question remains.
3. **Runtime previews use `ToolResponse`.** `create_runtime`, `patch_runtime` and
   `delete_runtime` return `{"action": "preview"}` instead of `PreviewResponse`, so
   clients have to special-case them. Moving them to `PreviewResponse` changes the
   response shape and needs a deprecation note.
4. **Audit entries in progressive and semantic modes name the meta-tool.** A mutation
   run through `execute_tool` is logged with `tool: "execute_tool"`; the real tool name
   is only inside `parameters`. Logging the inner tool name would make audits easier to
   search.
5. **A failed dependency check does not block `delete_runtime`.** See
   [Best-effort checks](#best-effort-checks). #340 fixes what the preview reports; the
   confirmed delete still proceeds.
6. **Not every tool validates input before the namespace check.** `fine_tune()` calls
   `check_namespace_allowed()` before validating its name, dtype, model and dataset
   references and bounds. With an allowlist and no namespace, that check resolves the
   default through the Trainer SDK, so malformed input can trigger an SDK call before it
   is rejected. #318 and #319 fix this order for `run_custom_training` and
   `run_container_training`.
7. **Persona state is process-global.** `get_effective_persona()` reads a global set at
   startup, which matters for the ownership check when several servers share a process.
   #348 scopes it per server.

Each item should become its own issue before it is implemented, so the behavior change
can be discussed on its own.
