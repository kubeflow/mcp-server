---
name: review-changes
description: Self-review of local changes against main before committing or opening a pull request, using specialized agents.
allowed-tools: ["Bash", "Glob", "Grep", "Read", "Task"]
---

# Comprehensive Change Review

Run a comprehensive review of the branch's changes using multiple specialized agents, each focusing on a different aspect of code quality.

The scope is everything the branch changes relative to `main`, committed or not. Run it before you commit or open a pull request so findings get fixed in the editor. Findings are reported where the skill is running — the terminal or the agent UI — and nothing is posted anywhere.

## Output

Report every finding directly where the user is working: the `file:line`, why it matters, and a fenced `diff` block with the change to make, so the user applies it in their editor. Write findings as notes to the author of the work, not as replies to a reviewer. Keep each one concise, respectful, and actionable: assume good intent and explain why a change matters without scolding or exaggerating severity. Use an encouraging tone, especially for new contributors.

**Review Aspects (optional):** "$ARGUMENTS"

## Review Workflow:

1. **Gather Context and Determine Review Scope**
   - Parse arguments to see if the user requested specific review aspects
   - The change is the primary context: `git diff main` shows everything the branch changes
     relative to `main`, including work already committed to the branch. Update your local
     `main` first if it is stale.
   - `git status --porcelain` covers what the diff cannot see — untracked files. Read those
     files directly; they are part of the change.
   - `git log main..HEAD --oneline` gives the commits on the branch and their rationale

2. **Available Review Aspects:**
   - **code** - General code review for project guidelines (always runs)
   - **mcp** - MCP-specific contract checks: tool schema stability, confirm gate,
     persona permissions, auth and namespace validation, error responses, SDK
     contracts, protocol conformance
   - **comments** - Analyze code comment accuracy and maintainability
   - **tests** - Review test coverage quality and completeness
   - **errors** - Check error handling for silent failures
   - **types** - Analyze type design and invariants
   - **simplify** - Separate opt-in step that edits code, never part of a review
     run (see [Opt-in Simplify Pass](#opt-in-simplify-pass))
   - **all** - Run every except simplify

3. **Identify Changed Files**
   - Get the changed file list from `git diff main --name-only`, plus the untracked files
     reported by `git status --porcelain`
   - Identify file types and what reviews apply

4. **Route Reviews by Changed Files**

   The general reviewer is the default and always runs. Everything else runs
   only when the diff calls for it. Route on the file map:

   | Changed files | Reviewer to run |
   |---|---|
   | anything under `kubeflow_mcp/` | code-reviewer (default) |
   | tool modules, `core/policy.py`, `core/security.py`, `core/auth.py`, `core/server.py`, `common/types.py` | code-reviewer + mcp-contract-reviewer |
   | `*_test.py` or no tests touched by production changes | pr-test-analyzer |
   | docstrings, comments, docs, `resources/` markdown | comment-analyzer |
   | try/except, `ToolError`, fallback logic | silent-failure-hunter |
   | new or modified type classes (`common/types.py`, `*/types/`) | type-design-analyzer |

   No aspect-specific reviewer runs for a diff that does not touch its files.
   That keeps findings unique and review time proportional to the change.

5. **Launch Review Agents**

   Each agent's instructions live in [agents/](agents/) next to this file (e.g. `agents/code-reviewer.md`).
   If an agent with that name is registered in your environment, use it. Otherwise, launch a
   general-purpose subagent and pass the body of the matching file as its prompt, followed by the
   review scope (the changed files and the base branch).

   **Sequential approach** (one at a time):
   - Easier to understand and act on
   - Each report is complete before next
   - Good for interactive review

   **Parallel approach** (user can request):
   - Launch all agents simultaneously
   - Faster for comprehensive review
   - Results come back together

6. **Aggregate and Deduplicate Results**

   After agents complete, combine overlapping feedback before reporting: when two
   agents report the same issue on the same lines, keep one finding and cite
   both agents in it. Never report two findings for the same issue. Summarize:
   - **Critical Issues** (must fix before merge)
   - **Important Issues** (should fix)
   - **Suggestions** (nice to have)
   - **Positive Observations** (what's good)

7. **Provide Action Plan**

   Organize findings:

   ```markdown
   # Change Review Summary

   ## Critical Issues (X found)

   - [agent-name]: Issue description [file:line]

   ## Important Issues (X found)

   - [agent-name]: Issue description [file:line]

   ## Suggestions (X found)

   - [agent-name]: Suggestion [file:line]

   ## Strengths

   - What's well-done in this change

   ## Recommended Action

   1. Fix critical issues first
   2. Address important issues
   3. Consider suggestions
   4. Re-run review after fixes
   ```

8. **Report the Findings**

   Report the summary and the per-finding `diff` blocks where the skill is running, then
   stop there. Nothing is posted.

## Opt-in Simplify Pass

The simplify aspect is not part of a review run. Unlike the reviewers, it edits
code, so it never runs alongside a review run by default. To use it, run it
deliberately after review findings are resolved:

```
/review-changes simplify
```

Then:

1. Show every proposed edit as a diff to the user for approval and never change behavior
2. Re-run the review workflow on the simplified code before committing

## Usage Examples:

**Self-review local changes before committing:**

```
/review-changes
```

Findings are printed where the skill is running, with `file:line` and a `diff` block to apply.

**Specific aspects:**

```
/review-changes tests errors
# Reviews only test coverage and error handling

/review-changes mcp
# Reviews only the MCP contract checks

/review-changes comments
# Reviews only code comments

/review-changes simplify
# Separate opt-in step: edits code, never part of a review
```

**Parallel review:**

```
/review-changes all parallel
# Launches the review agents in parallel (never includes simplify)
```

## Agent Descriptions:

**code-reviewer**:

- Checks AGENTS.md compliance
- Detects bugs and issues
- Reviews general code quality
- Always runs; the default reviewer

**mcp-contract-reviewer**:

- Verifies tool schema stability and the confirm gate
- Checks persona allowlists and namespace/name validation
- Reviews auth fail-closed behavior and ToolError responses
- Guards SDK contract tests and protocol conformance

**comment-analyzer**:

- Verifies comment accuracy vs code
- Identifies comment rot
- Checks documentation completeness

**pr-test-analyzer**:

- Reviews behavioral test coverage
- Identifies critical gaps
- Evaluates test quality

**silent-failure-hunter**:

- Finds silent failures
- Reviews catch blocks
- Checks error logging

**type-design-analyzer**:

- Analyzes type encapsulation
- Reviews invariant expression
- Rates type design quality

**code-simplifier** (opt-in, outside review):

- Simplifies complex code
- Improves clarity and readability
- Applies project standards
- Preserves functionality

## Tips:

- **Run early**: before you commit and open a pull request
- **Focus on changes**: the scope is `git diff main`, so committed and uncommitted work are both covered
- **Address critical first**: Fix high-priority issues before lower priority
- **Re-run after fixes**: Verify issues are resolved
- **Use specific reviews**: Target specific aspects when you know the concern

## Workflow Integration:

**Before committing:**

```
1. Write code
2. Run: /review-changes code errors
3. Fix any critical issues
4. Commit
```

**Before creating PR:**

```
1. Stage all changes
2. Run: /review-changes all
3. Address all critical and important issues
4. Run specific reviews again to verify
5. Create PR
```

**After review feedback:**

```
1. Make requested changes
2. Run targeted reviews based on feedback
3. Verify issues are resolved
4. Commit and push updates
```

## Notes:

- Agents run autonomously and return detailed reports
- Each agent focuses on its specialty for deep analysis
- Results are actionable with specific file:line references
- Agents use appropriate models for their complexity
- Agent prompts are bundled in `agents/` next to this skill
- code-simplifier is opt-in and lives outside review runs because it edits code
