---
title: Scripts
---

# Scripts

Small Python utilities for maintaining the wiki. No third-party dependencies — stdlib only. Run with the system Python (3.9+).

## `unwrap_markdown.py`

Joins hard-wrapped paragraph lines into single long lines. Preserves frontmatter, code fences, headings, tables, list markers, blockquotes, and horizontal rules. Lists and blockquotes have their continuation lines merged into the parent item/quote.

```bash
# Preview (no changes written)
python scripts/unwrap_markdown.py wiki/ --dry-run

# Apply to the whole wiki
python scripts/unwrap_markdown.py wiki/

# Apply to a single file or a subtree, excluding drafts
python scripts/unwrap_markdown.py wiki/sources/papers/ --exclude draft

# Quiet summary only
python scripts/unwrap_markdown.py wiki/ --quiet
```

Exit code is `0` on success, `1` if any file errored.

**Scope:** run on `wiki/` freely (LLM-owned). Do **not** run on `docs/` without confirming with the user — those files are user-authored raw sources and their wrapping style may be intentional.

## Conventions for adding scripts here

- Stdlib only. If a dep is unavoidable, document it at the top of the script.
- One script = one job. Compose at the shell, not in Python.
- Provide `--dry-run` for any script that writes files.
- Print a one-line `Done. changed: N. unchanged: M. errors: K.` summary at the end.
- Document the script in this README with an example invocation.
