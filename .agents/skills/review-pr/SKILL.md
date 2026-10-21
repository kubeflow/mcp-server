---
name: review-pr
description: Comprehensive PR review using specialized agents. Use when asked to review a pull request or local changes before committing or creating a PR.
allowed-tools: ["Bash", "Glob", "Grep", "Read", "Task"]
---

# Comprehensive PR Review

Run a comprehensive pull request review using multiple specialized agents, each focusing on a different aspect of code quality.

In the review body message keep it minimal: thank the contributor for their work and note that the review was done using AI tools. Do not include summaries or aggregated findings in the review body, only post inline comments on specific lines.

Keep inline comments concise, respectful, and actionable. Assume good intent and explain why a change matters without scolding or exaggerating severity. Use an encouraging tone, especially for new contributors, so they feel welcome to contribute again.

**Prefer GitHub suggestions over prose comments.** Whenever a finding can be expressed as a concrete code change, post it as a committable GitHub `suggestion` block so the author can apply it in one click. Do NOT describe the change in words. See [Comment Format: Prefer Suggestions](#comment-format-prefer-suggestions) for the mechanics. Only fall back to a plain prose comment when a suggestion is not feasible (see that section for when).

Before posting on GitHub ALWAYS print all proposed comments and the organized summary in the terminal for human review. The summary is for local reference only and must NOT be posted to GitHub.

Post the review as a **pending** GitHub review. Never submit it yourself: the user inspects the draft and submits it in GitHub. See [Posting the Review](#posting-the-review).

## Posting the Review

Create the review with the [create-a-review API](https://docs.github.com/en/rest/pulls/reviews#create-a-review-for-a-pull-request) and **omit the `event` field**. Without `event`, GitHub creates the review in the `PENDING` state, which is visible only to the user until they submit it.

```bash
gh api repos/{owner}/{repo}/pulls/{pull_number}/reviews --method POST --input review.json
```

Where `review.json` contains:

```json
{
  "commit_id": "<head commit SHA of the PR>",
  "body": "Thanks for your work on this! This review was performed using AI tools.",
  "comments": [
    {
      "path": "kubeflow_mcp/core/server.py",
      "line": 42,
      "side": "RIGHT",
      "body": "<comment text or suggestion block>"
    }
  ]
}
```

- Do NOT set `event` (`APPROVE`, `REQUEST_CHANGES`, or `COMMENT`) and do NOT call the submit-review endpoint.
- GitHub allows only one pending review per user per PR. If the request fails because a pending review already exists, stop and ask the user whether to delete it or add to it.
- After the review is created, print its `html_url` and tell the user to open the PR's **Files changed** tab, adjust or remove comments as needed, and submit the review there.

## Comment Format: Prefer Suggestions

The default and preferred format for every actionable inline comment is a GitHub suggested change. A suggestion renders a "Commit suggestion" button, so the author applies the fix without hand-editing.

**Rules for writing a suggestion:**

- The comment body must contain a fenced block using the ` ```suggestion ` language tag. The lines inside the block REPLACE the anchored line(s) verbatim.
- Anchor the comment to the EXACT line(s) the block replaces. For a single line, set `line` (and `side`). For a multi-line replacement, set `start_line` + `line` (and `start_side` + `side`) so the range covers every line being replaced.
- The block content must be the complete replacement for the anchored range, with correct indentation: GitHub swaps in exactly what you write.
- Keep a one-line explanation (the "why") above the block; keep it short.

Example comment body posted via `gh api .../reviews`:

````markdown
Rename the import to match the module under test:

```suggestion
from kubeflow_mcp.core import dynamic_tools
```
````

**When to use a plain prose comment instead** (suggestion not feasible):

- The fix spans multiple files, or lines not present in the diff hunk.
- The change is conceptual/architectural (e.g. "extract this into a helper", "add a test for X") with no single obvious replacement.
- You are uncertain of the exact replacement and would be guessing at indentation or surrounding context.

In these cases, post a small prose comment with a clear action item instead of a low-confidence suggestion.

**Review Aspects (optional):** "$ARGUMENTS"

## Review Workflow:

1. **Gather Context and Determine Review Scope**
   - Parse arguments to see if the user requested specific review aspects
   - For a PR, build the review on the PR's actual state, not on local uncommitted
     work: `gh pr view <num>` (body, linked issues), `gh pr diff <num>` (full diff
     including committed changes), existing review threads
     (`gh api repos/{owner}/{repo}/pulls/<num>/comments`), and CI results
     (`gh pr checks <num>`). A local `git diff` misses changes already committed
     to the PR branch.
   - Only fall back to `git status` / `git diff` when reviewing local uncommitted
     work before pushing
   - `gh pr checks` failures are findings in their own right: report what failed,
     why it matters for this diff, and the fix

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
   - Get the changed file list from `gh pr diff --name-only <num>` for a PR, or
     `git diff --name-only` for local work
   - Identify file types and what reviews apply
   - Check if a PR already exists: `gh pr view`

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

5. **Run Automated Checks**

   Before or alongside the agents, run the repo's own lint and quality gates so
   mechanical findings carry over into the summary instead of being re-discovered
   by the agents: `make verify` and `make test-python`. Record both results and
   include them in the aggregate summary as an "Automated checks" section.

6. **Launch Review Agents**

   Each agent's instructions live in [agents/](agents/) next to this file (e.g. `agents/code-reviewer.md`).
   If an agent with that name is registered in your environment, use it. Otherwise, launch a
   general-purpose subagent and pass the body of the matching file as its prompt, followed by the
   review scope (PR number or changed files).

   **Sequential approach** (one at a time):
   - Easier to understand and act on
   - Each report is complete before next
   - Good for interactive review

   **Parallel approach** (user can request):
   - Launch all agents simultaneously
   - Faster for comprehensive review
   - Results come back together

7. **Aggregate and Deduplicate Results**

   After agents complete, combine overlapping feedback before posting: when two
   agents report the same issue on the same lines, keep one comment and cite
   both agents in it. Never post two comments for the same finding. Summarize:
   - **Automated checks** (make verify, make test-python results)
   - **Critical Issues** (must fix before merge)
   - **Important Issues** (should fix)
   - **Suggestions** (nice to have)
   - **Positive Observations** (what's good)

8. **Provide Action Plan**

   Organize findings:

   ```markdown
   # PR Review Summary

   ## Automated checks

   - make verify: <pass/fail>
   - make test-python: <pass/fail>

   ## Critical Issues (X found)

   - [agent-name]: Issue description [file:line]

   ## Important Issues (X found)

   - [agent-name]: Issue description [file:line]

   ## Suggestions (X found)

   - [agent-name]: Suggestion [file:line]

   ## Strengths

   - What's well-done in this PR

   ## Recommended Action

   1. Fix critical issues first
   2. Address important issues
   3. Consider suggestions
   4. Re-run review after fixes
   ```

9. **Post a Pending Review**

   For a PR, create a pending GitHub review with the inline comments as described in
   [Posting the Review](#posting-the-review), then share its link so the user can submit it.

## Opt-in Simplify Pass

The simplify aspect is not part of a review run. Unlike the reviewers, it edits
code, so it must never contribute comments to a pending review and never runs
alongside one by default. To use it, run it deliberately after review findings
are resolved:

```
/review-pr simplify
```

Then:

1. Show every proposed edit as a diff to the user for approval and never change behavior
2. Re-run the review workflow on the simplified code before submitting
3. Run `make verify` and `make test-python` again before pushing

## Usage Examples:

**Default review (code reviewer, plus aspect reviewers when the diff calls for them):**

```
/review-pr
```

**Review a specific PR:**

```
/review-pr https://github.com/kubeflow/mcp-server/pull/1234
```

**Specific aspects:**

```
/review-pr tests errors
# Reviews only test coverage and error handling

/review-pr mcp
# Reviews only the MCP contract checks

/review-pr comments
# Reviews only code comments

/review-pr simplify
# Separate opt-in step: edits code, never part of a review
```

**Parallel review:**

```
/review-pr all parallel
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

- **Run early**: Before creating PR, not after
- **Focus on changes**: For a PR the agents analyze the full PR diff, not local uncommitted work
- **Address critical first**: Fix high-priority issues before lower priority
- **Re-run after fixes**: Verify issues are resolved
- **Use specific reviews**: Target specific aspects when you know the concern

## Workflow Integration:

**Before committing:**

```
1. Write code
2. Run: make verify
3. Run: /review-pr code errors
4. Fix any critical issues
5. Commit
```

**Before creating PR:**

```
1. Stage all changes
2. Run: make verify && make test-python
3. Run: /review-pr all
4. Address all critical and important issues
5. Run specific reviews again to verify
6. Create PR
```

**After PR feedback:**

```
1. Make requested changes
2. Run targeted reviews based on feedback
3. Verify issues are resolved
4. Push updates
```

## Notes:

- Agents run autonomously and return detailed reports
- Each agent focuses on its specialty for deep analysis
- Results are actionable with specific file:line references
- Agents use appropriate models for their complexity
- Agent prompts are bundled in `agents/` next to this skill
- code-simplifier is opt-in and lives outside review runs because it edits code
