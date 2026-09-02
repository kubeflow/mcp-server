# MCP Client Module Conventions

Conventions every client module (`trainer`, `optimizer`, `hub`, …) must follow.
**Reference:** [`kubeflow_mcp/trainer/`](../kubeflow_mcp/trainer/) · [AGENTS.md](../AGENTS.md)

---

## Module contract

```
kubeflow_mcp/<client>/
├── __init__.py       # exports below
├── api/              # one public function per tool + *_test.py
├── resources/        # optional
└── constants/ types/ # optional
```

**Register** (follow trainer): `CLIENT_MODULES` · `TOOL_PHASES` / `TOOL_NEXT_HINTS` ·
`core/policy.py` · `SDK_COMPATIBILITY`. Enable via `--clients` or `server.clients` in config.

| Export | Required when |
|--------|----------------|
| `MODULE_INFO` | Always (`name`, `description`, `status`) |
| `TOOLS` | Always (ordered callables; name = MCP tool name) |
| `CLIENT_TOOL_DESCRIPTIONS` | First tool ships |
| `CLIENT_TOOL_ANNOTATIONS` | First tool ships |
| `CLIENT_RESOURCES` | When guides exist |
| `INSTRUCTION_SECTIONS` | When agent workflow text exists |
| `PHASE_TO_SECTION` | When instructions map from `TOOL_PHASES` |

Stub: `MODULE_INFO` + `TOOLS = []`. Implemented: all exports above + tests.

---

## Tools

**Naming:** `{verb}_{noun}` in `snake_case`. Standard verbs: `list`, `get`, `create`,
`delete`, `update`, `run`, `wait_for`, `inspect`, `patch`. Domain verbs OK when clearer
(`fine_tune`). **Renames break agents** — KEP or maintainer approval required.

**Phases:** `planning`, `discovery`, domain (`training`, `optimization`, …), `monitoring`,
`lifecycle`, `platform`. `health` is server-only (`core/health.py`).

**Responses** (`common/types.py`, always `.model_dump()`):

| Model | Use |
|-------|-----|
| `ToolResponse` | Success (`data={...}`) |
| `PreviewResponse` | Mutation preview (`status="preview"`) |
| `ToolError` | Failure (`error_code`, optional `hint`) |

**Confirm gate:** Mutators accept `confirmed: bool = False` — preview first, execute at
`confirmed=True`. `delete_*`: `[DESTRUCTIVE]` in description, `destructiveHint: True`,
entry in `DESTRUCTIVE_TOOLS`. Legacy exceptions: trainer runtime
previews use `ToolResponse` not `PreviewResponse` — do not extend.

**Annotations** (every tool): `title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`,
`openWorldHint`, `tags` (include phase).

---

## Personas

Update `core/policy.py` when adding tools.

| Persona | Adds |
|---------|------|
| `readonly` | `list_*`, `get_*`, planning, monitoring |
| `data-scientist` | + `create_*`, `run_*`, delete own MCP resources |
| `ml-engineer` | + `update_*`, platform inspect, advanced submit |
| `platform-admin` | `*` |

Copy trainer's persona matrix tests (`trainer/api/architecture_test.py`).

---

## Security

- Names: `validate_k8s_name()` · Namespaces: `check_namespace_allowed()` (fail closed on
  implicit default).
- Client-specific default namespace: wrapper or `check_namespace_allowed(..., resolver=...)`.
- Created resources: label `kubeflow-mcp/managed-by=mcp` (`MCP_MANAGED_LABEL` in
  `common/utils.py`). Non-admin may only mutate labeled resources.
- Missing resource: `RESOURCE_NOT_FOUND`. Exists but not MCP-owned: `VALIDATION_ERROR`.
- 404s: `is_k8s_not_found()` — never substring-match `"not found"`.
- Errors: return `ToolError` at tool boundary; log WARNING + `exc_info`; no stack traces in
  responses (use `exception_details()`).

---

## Agent guidance

- **Resources:** `{client}://guides/{topic}` maps to markdown in `resources/`.
- **Instructions:** `INSTRUCTION_SECTIONS` (`"full"` tier); map phases via `PHASE_TO_SECTION`.
- **Steering:** `data["next_steps"]` on key responses; server injects `_meta.phase` /
  `_meta.next` from `TOOL_TO_PHASE` / `TOOL_NEXT_HINTS`. Cross-client hints encouraged.

---

## SDK & tests

- Factory: `get_<client>_client()` in `common/utils.py`.
- Coverage: `SDK_COMPATIBILITY` + `sdk_contracts_test.py`. SDK gaps: build CR directly,
  document under `uncovered_methods`.
- Before `status: implemented`: co-located unit tests, architecture tests (metadata +
  personas), namespace sweep; `make verify`, `make test-python`, `make conformance`; update
  schema snapshot if parameters change (`make update-schema-snapshot`).

---

## PR checklist

- [ ] `CLIENT_MODULES`, exports, `TOOL_PHASES`, `TOOL_NEXT_HINTS`, personas, `DESTRUCTIVE_TOOLS`
- [ ] Confirm gate, shared response types, namespace + ownership checks
- [ ] Architecture + unit tests; `SDK_COMPATIBILITY`; README; schema snapshot if schemas change
- [ ] `make verify` + `make test-python`

---

## Do not

- Register tools only in README — must be in `TOOLS` and policy.
- Skip namespace check when `namespace` is omitted.
- Weaken SDK contract or schema snapshot tests to green CI.
- Add new mutators without `confirmed` preview.
