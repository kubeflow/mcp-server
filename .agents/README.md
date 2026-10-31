# Agent Resources

Shared resources for AI coding agents working on Kubeflow MCP Server.

```
.agents/
└── skills/     # Agent skills
    └── review-pr/
```

## review-pr

Runs a comprehensive review with specialized subagents. It is meant for self-review of local
changes before you commit and open a pull request: findings come back in the terminal as
`file:line` plus a `diff` block you apply in your editor. Pointing it at a pull request reviews
the PR's diff instead and posts the inline comments as a pending GitHub review, which is visible
only to you until you submit it in the **Files changed** tab.

```
skills/review-pr/
├── SKILL.md                       # Review workflow
└── agents/                        # Subagent prompts
    ├── code-reviewer.md
    ├── code-simplifier.md         # Opt-in, outside review (edits code)
    ├── comment-analyzer.md
    ├── mcp-contract-reviewer.md
    ├── pr-test-analyzer.md
    ├── silent-failure-hunter.md
    └── type-design-analyzer.md
```

### Usage

```
/review-pr                      # Self-review local changes: file:line plus a diff block per finding
/review-pr <PR_LINK>           # Review an existing PR, e.g. /review-pr https://github.com/kubeflow/mcp-server/pull/1234
/review-pr tests errors        # Review only test coverage and error handling
/review-pr mcp                 # Review only the MCP contract checks
/review-pr all parallel        # Launch the review agents in parallel
/review-pr simplify            # Separate opt-in step that edits code, never part of a review
```

The local run needs no GitHub access and covers unpushed work. The PR run reviews the PR's
actual state: the linked issue, the full PR diff (committed work included), existing review
threads, and CI checks. Overlapping findings from different agents are combined into one
comment before posting.

Available aspects: `code`, `mcp`, `comments`, `tests`, `errors`, `types`,
`simplify` (opt-in, separate step), `all`.

### Configure for Claude Code

Claude Code discovers skills from `.claude/skills/` and subagents from `.claude/agents/`. Symlink
them from the repository root:

```bash
mkdir -p .claude/skills .claude/agents
ln -s ../../.agents/skills/review-pr .claude/skills/review-pr
for f in .agents/skills/review-pr/agents/*.md; do
  ln -s "../../$f" ".claude/agents/$(basename "$f")"
done
```

If the subagents are not registered, the skill falls back to launching a general-purpose subagent
with the prompt from `agents/<name>.md`.
