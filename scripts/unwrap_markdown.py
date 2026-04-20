#!/usr/bin/env python3
"""
unwrap_markdown.py — join hard-wrapped paragraph lines in Markdown files.

Usage:
    python scripts/unwrap_markdown.py <path> [--dry-run] [--exclude SUBSTR]

Preserves (never merges):
  - YAML frontmatter (leading ---...---)
  - Fenced code blocks (``` or ~~~)
  - Headings (ATX: #, ##, ...)
  - Horizontal rules (---, ***, ___)
  - Tables (| ... |)
  - List item markers (new bullets / numbered items stay on new lines)
  - Blank lines (paragraph boundaries)
  - HTML block tags on their own line

Joins (merges consecutive lines into one):
  - Plain prose paragraphs
  - List-item continuation lines (indented or unindented, below a bullet)
  - Blockquote continuation lines within a single quoted paragraph

Exit code 0 on success; 1 if any file failed to read/parse.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# --- Line classifiers -------------------------------------------------------

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
HR_RE = re.compile(
    r"^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})\s*$"
)
TABLE_ROW_RE = re.compile(r"^\s{0,3}\|")
UNORDERED_LIST_RE = re.compile(r"^\s{0,3}[-+*]\s+")
ORDERED_LIST_RE = re.compile(r"^\s{0,3}\d+[.)]\s+")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")
FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
HTML_BLOCK_RE = re.compile(r"^\s{0,3}</?[a-zA-Z][^>]*>\s*$")


def is_atomic_line(line: str) -> bool:
    return bool(
        HEADING_RE.match(line)
        or HR_RE.match(line)
        or TABLE_ROW_RE.match(line)
        or HTML_BLOCK_RE.match(line)
    )


def starts_list_item(line: str) -> bool:
    return bool(UNORDERED_LIST_RE.match(line) or ORDERED_LIST_RE.match(line))


def is_blockquote(line: str) -> bool:
    return bool(BLOCKQUOTE_RE.match(line))


def is_blank(line: str) -> bool:
    return line.strip() == ""


def is_break_for_paragraph(line: str) -> bool:
    """Any line that terminates an ongoing prose/list paragraph."""
    if is_blank(line):
        return True
    if is_atomic_line(line):
        return True
    if starts_list_item(line):
        return True
    if is_blockquote(line):
        return True
    if FENCE_RE.match(line):
        return True
    return False


# --- Core unwrap ------------------------------------------------------------


def unwrap(content: str) -> str:
    trailing_newline = content.endswith("\n")
    lines = content.split("\n")
    if trailing_newline and lines and lines[-1] == "":
        lines.pop()

    out: List[str] = []
    i, n = 0, len(lines)

    # Pass through leading YAML frontmatter verbatim
    if n > 0 and lines[0].strip() == "---":
        out.append(lines[0])
        i = 1
        while i < n:
            out.append(lines[i])
            if lines[i].strip() == "---":
                i += 1
                break
            i += 1

    while i < n:
        line = lines[i]

        # Fenced code block — pass through verbatim
        m = FENCE_RE.match(line)
        if m:
            fence = m.group(1)
            out.append(line)
            i += 1
            while i < n:
                out.append(lines[i])
                if lines[i].lstrip().startswith(fence):
                    i += 1
                    break
                i += 1
            continue

        if is_blank(line):
            out.append("")
            i += 1
            continue

        if is_atomic_line(line):
            out.append(line)
            i += 1
            continue

        # Blockquote — merge consecutive `>` lines into one paragraph per
        # blockquote block, splitting on empty-within-quote lines.
        if is_blockquote(line):
            bq_para: List[str] = []
            while i < n and is_blockquote(lines[i]):
                stripped = BLOCKQUOTE_RE.sub("", lines[i]).rstrip()
                if stripped == "":
                    if bq_para:
                        out.append("> " + " ".join(bq_para))
                        bq_para = []
                    out.append(">")
                else:
                    bq_para.append(stripped.strip())
                i += 1
            if bq_para:
                out.append("> " + " ".join(bq_para))
            continue

        # List item (plus its continuation lines)
        if starts_list_item(line):
            item_chunks = [line.rstrip()]
            i += 1
            while i < n and not is_break_for_paragraph(lines[i]):
                item_chunks.append(lines[i].strip())
                i += 1
            out.append(" ".join(item_chunks))
            continue

        # Plain prose paragraph
        para = [line.rstrip()]
        i += 1
        while i < n and not is_break_for_paragraph(lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(" ".join(s.strip() for s in para))

    result = "\n".join(out)
    if trailing_newline:
        result += "\n"
    return result


# --- CLI --------------------------------------------------------------------


def process_file(path: Path, dry_run: bool) -> Tuple[str, bool]:
    """Return (status, changed?).  status ∈ {updated, would-change, unchanged, error:<msg>}."""
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as e:
        return (f"error:read:{e}", False)
    try:
        updated = unwrap(original)
    except Exception as e:
        return (f"error:unwrap:{e}", False)
    if updated == original:
        return ("unchanged", False)
    if dry_run:
        return ("would-change", True)
    path.write_text(updated, encoding="utf-8")
    return ("updated", True)


def _excluded(path: Path, substrings: List[str]) -> bool:
    s = str(path).replace("\\", "/")
    return any(sub in s for sub in substrings)


def iter_markdown(paths: Iterable[Path], excludes: List[str]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            if p.suffix.lower() == ".md" and not _excluded(p, excludes):
                yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.md")):
                if not _excluded(f, excludes):
                    yield f
        else:
            print(f"warning: not found: {p}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unwrap hard-wrapped paragraphs in Markdown files."
    )
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Files or directories (recursive for dirs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Skip paths containing this substring (repeatable)")
    ap.add_argument("--quiet", action="store_true",
                    help="Only print the summary line")
    args = ap.parse_args()

    changed = unchanged = errors = 0
    for path in iter_markdown(args.paths, args.exclude):
        status, did_change = process_file(path, dry_run=args.dry_run)
        if status.startswith("error:"):
            print(f"ERROR {path}: {status[6:]}", file=sys.stderr)
            errors += 1
        elif did_change:
            if not args.quiet:
                label = "would-change" if args.dry_run else "updated"
                print(f"{label}: {path}")
            changed += 1
        else:
            unchanged += 1

    verb = "would change" if args.dry_run else "changed"
    print(f"\nDone. {verb}: {changed}. unchanged: {unchanged}. errors: {errors}.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
