# Platform Compatibility

These skills are written for both Claude Code and Codex-style skill consumers.
Use the equivalent capability exposed by your runtime; do not depend on a single
tool name when another local equivalent is available.

| Capability | Claude Code surface | Codex surface |
|---|---|---|
| Read files | `Read` | shell reads such as `sed`, `rg`, or equivalent file APIs |
| Search files | `Grep`, `Glob` | `rg`, `rg --files`, or equivalent search APIs |
| Edit files | `Edit`, `Write` | `apply_patch` for manual edits, or runtime file APIs for generated artifacts |
| Run scripts | `Bash` | local shell execution |
| Literature/web lookup | `WebSearch`, `WebFetch` | web/search tools available in the runtime |
| Parallel workers | `Agent` or `Task` | subagents only when the runtime policy and user authorization allow them |
| External review bridge | configured MCP/tool bridge | configured MCP/tool bridge, or self-review fallback |

## Frontmatter

Some skill folders keep Claude-compatible optional metadata such as
`argument-hint` and `allowed-tools`. Codex-compatible consumers should rely on
`name` and `description` for activation and may ignore optional metadata fields.

## Fallback Rule

If a specific parallel-worker or external-review tool is unavailable, use the
sequential or self-review fallback described in the skill. Missing one optional
tool name should not stop the workflow when the core file, search, shell, and
editing capabilities are available.
