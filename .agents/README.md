# Agent Resources

Shared resources for AI coding agents working on Kubeflow MCP Server.

```
.agents/
└── skills/     # Agent skills
    └── review-changes/
```

## review-changes

Runs a comprehensive review with specialized subagents. It is meant for self-review of your
branch before you commit and open a pull request: the scope is everything the branch changes
since it diverged from `main` (`git diff $(git merge-base main HEAD)`), and findings come back
where the skill runs — the terminal or the agent UI — as `file:line` plus a `diff` block you
apply in your editor. Nothing is posted anywhere.

```
skills/review-changes/
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
/review-changes                 # Review the branch against main: file:line plus a diff block per finding
/review-changes tests errors    # Review only test coverage and error handling
/review-changes mcp             # Review only the MCP contract checks
/review-changes all parallel    # Launch the review agents in parallel
/review-changes simplify        # Separate opt-in step that edits code, never part of a review
```

The run stays local: `git diff $(git merge-base main HEAD)` covers committed and uncommitted
work, and `git status --porcelain` surfaces untracked files for the reviewers to read (the
`.claude/` symlinks from the setup below are skipped). Overlapping findings from different
agents are combined into one finding before they are reported.

Available aspects: `code`, `mcp`, `comments`, `tests`, `errors`, `types`,
`simplify` (opt-in, separate step), `all`.

### Configure for Claude Code

Claude Code discovers skills from `.claude/skills/` and subagents from `.claude/agents/`. Symlink
them from the repository root:

```bash
mkdir -p .claude/skills .claude/agents
ln -s ../../.agents/skills/review-changes .claude/skills/review-changes
for f in .agents/skills/review-changes/agents/*.md; do
  ln -s "../../$f" ".claude/agents/$(basename "$f")"
done
```

If the subagents are not registered, the skill falls back to launching a general-purpose subagent
with the prompt from `agents/<name>.md`.

`.claude/` is local tooling: it stays untracked (don't commit it) and the skill skips it, so it
never becomes part of a review. On Windows, run these commands from WSL or enable Developer Mode
— Git Bash otherwise copies the files instead of creating symlinks.
