# Agent-Facing Error Contract

How a failure reaches an MCP client, and what the client is expected to do with it.

**Reference:** [`kubeflow_mcp/common/types.py`](../../kubeflow_mcp/common/types.py) ·
[`kubeflow_mcp/common/constants.py`](../../kubeflow_mcp/common/constants.py) ·
[CONVENTIONS.md](../CONVENTIONS.md)

---

## What the contract covers

The contract covers every failure this server produces: errors a tool returns at
its own boundary, and errors the audit wrapper or `execute_tool` returns on a
tool's behalf. Those are the paths listed under
[Where the contract is produced](#where-the-contract-is-produced), and each one
is a `ToolError` dump.

It does not cover failures raised by the MCP framework before a tool is reached.
In full mode FastMCP validates arguments against the tool schema and rejects a
bad call itself, so the server never runs and has nothing to report. Those
errors have a different shape, described in
[Framework validation errors](#framework-validation-errors) below. The dynamic
modes do not have this split: `execute_tool` performs the same pydantic
validation in-process (#484) and returns `VALIDATION_ERROR` through the contract.

## The shape

A failure covered by the contract is a `ToolError` dump:

```json
{
  "success": false,
  "error": "Invalid arguments for 'get_training_job': name: Field required",
  "error_code": "VALIDATION_ERROR",
  "details": null,
  "hint": "Call describe_tools(['get_training_job']) for the parameter schema, then retry with corrected arguments.",
  "tool": "get_training_job"
}
```

| Field | Guarantee |
|-------|-----------|
| `success` | Always `false` on failure. A client may branch on this field alone. |
| `error_code` | Stable. One of `ErrorCode`. This is the field to switch on. |
| `error` | Human-readable and safe. Free-form: never match on it. |
| `details` | Structured context. `exception_details()` output, or extra data such as `available`. |
| `hint` | Recovery guidance when the server has something useful to say. |
| `tool` | Set when a wrapper reports the failure, so the agent knows which call failed. |

The MCP result is also marked `isError: true`, so a client that only reads the
protocol-level flag still sees the failure. The fields above matter to clients
that read the structured body.

Stack traces never appear in `error`. `details["traceback"]` is populated only
when the `kubeflow_mcp` logger is at DEBUG.

## Framework validation errors

In full mode every tool is registered with FastMCP, which validates arguments
against the tool's schema before dispatch. A call that fails that check never
reaches the tool or the audit wrapper, so no `ToolError` is produced. Calling
`get_training_job` without its required `name` returns:

```
isError:            true
structuredContent:  absent
content:            "1 validation error for call[get_training_job]
                     name
                       Missing required argument [type=missing_argument, ...]"
```

There is no `success` field, no `error_code`, and no `hint`, because this is the
MCP framework reporting a malformed call rather than the server reporting a
failed one.

**How a client should handle it.** Treat a result with `isError: true` and no
structured body as a caller error, equivalent to `VALIDATION_ERROR`: the request
was malformed, retrying it unchanged will fail the same way, and the text
content names the offending field. A client that switches on `error_code` needs
a branch for a missing structured body, and that branch should correct the
request rather than retry it.

In short:

| Where it failed | `isError` | Structured body | `error_code` |
|---|---|---|---|
| FastMCP schema validation, full mode | `true` | absent | none |
| Tool, audit wrapper, or `execute_tool` | `true` | `ToolError` dump | one of `ErrorCode` |

## The codes

| `error_code` | Means | What the agent should do |
|--------------|-------|--------------------------|
| `VALIDATION_ERROR` | The request is malformed or rejected by a guard | Correct the input and retry |
| `RESOURCE_NOT_FOUND` | The named Kubernetes resource does not exist | Pick another resource, or create it |
| `TOOL_NOT_FOUND` | The named tool is not registered on this server | Re-discover with the mode's discovery tool, which `hint` names |
| `PERMISSION_DENIED` | The persona or RBAC forbids the call | Ask the user for permission or a different persona |
| `QUOTA_EXCEEDED` | A cluster quota blocks the request | Reduce the request, or ask the user |
| `KUBERNETES_ERROR` | The Kubernetes API failed | Back off and retry |
| `SDK_ERROR` | The Kubeflow SDK or the wrapped tool raised | Back off and retry |
| `TIMEOUT` | The operation did not finish in time | Back off and retry |
| `RATE_LIMITED` | The server's own rate limiter rejected the call | Back off and retry |
| `CIRCUIT_OPEN` | The breaker is open for this tool | Back off; it closes after the recovery timeout |

Two frozensets make that table executable: `RETRIABLE_ERROR_CODES` for the
rows where retrying unchanged can succeed, and `CALLER_ERROR_CODES` for the
rows where it cannot. `is_retriable_error(result)` is the one-call form.

The split is deliberately not the same as the breaker's notion of failure.
`is_infrastructure_error()` decides what trips the circuit and covers only
`KUBERNETES_ERROR`, `SDK_ERROR` and `TIMEOUT`: `RATE_LIMITED` and
`CIRCUIT_OPEN` are retriable by the client but must not themselves count as
failures, or the breaker would feed on its own output.

## Where the contract is produced

Tools return `ToolError(...).model_dump()` at their own boundary. Two wrappers
produce it on the tool's behalf:

| Path | Location | Code |
|------|----------|------|
| Rate limiter rejected the call | `_audit_wrap`, `core/server.py` | `RATE_LIMITED` |
| Breaker open for the tool | `_audit_wrap`, `core/server.py` | `CIRCUIT_OPEN` |
| Tool name not registered | `execute_tool`, `core/dynamic_tools.py` | `TOOL_NOT_FOUND` |
| Arguments fail pydantic validation | `execute_tool`, `core/dynamic_tools.py` | `VALIDATION_ERROR` |
| Breaker open inside `execute_tool` | `execute_tool`, `core/dynamic_tools.py` | `CIRCUIT_OPEN` |
| Wrapped tool raised | `execute_tool`, `core/dynamic_tools.py` | `SDK_ERROR` |
| Too many names for `describe_tools` | `core/dynamic_tools.py` | `VALIDATION_ERROR` |
| Query over `MAX_QUERY_LENGTH` | `find_tools`, `core/dynamic_tools.py` | `VALIDATION_ERROR` |

`kubeflow_mcp/core/error_contract_test.py` asserts the shape on each of these.

### Hints name a tool the agent can call

The two dynamic modes register different meta-tools: `list_tools` and
`describe_tools` only in progressive, `find_tools` only in semantic. A hint is
useless if it names a tool the agent cannot call, so hints are built from the
mode recorded by `init_dynamic_tools()` and a test asserts that every tool named
in a hint is registered in that mode.

Semantic mode registers nothing that returns a parameter schema, so its
bad-argument hint points at the field named in the message instead of at
`describe_tools`.

## Adding a code

New codes are additive. A client that does not know a code must still be able
to act, so put it in `RETRIABLE_ERROR_CODES` or `CALLER_ERROR_CODES`, add a row
above, and extend the contract test. Removing or renaming a code breaks every
agent that branches on it and needs maintainer approval, the same as a tool
rename.
