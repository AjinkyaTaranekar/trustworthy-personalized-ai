#!/usr/bin/env python3
"""Validate and append a completed training example to train_v3.jsonl.

Usage:
    python save_example.py /tmp/example.json
"""
import json, sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_FILE = DATA_DIR / "train_v3.jsonl"

REQUIRED_META = {"question_id", "category", "tool_profile", "pipeline"}
REQUIRED_TOP = {"messages", "metadata"}


def validate(ex: dict) -> str | None:
    for k in REQUIRED_TOP:
        if k not in ex:
            return f"missing top-level key: {k}"
    meta = ex["metadata"]
    for k in REQUIRED_META:
        if k not in meta:
            return f"missing metadata key: {k}"
    msgs = ex["messages"]
    if not msgs or msgs[0].get("role") != "system":
        return "first message must be system"
    if not any(m.get("role") == "assistant" for m in msgs):
        return "no assistant message"
    last = msgs[-1]
    if last.get("role") != "assistant":
        return "last message must be assistant"
    if "<answer>" not in last.get("content", ""):
        return "last assistant message missing <answer> tag"
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: save_example.py <path_to_example.json>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    ex = json.loads(path.read_text(encoding="utf-8"))
    err = validate(ex)
    if err:
        print(f"Validation failed: {err}", file=sys.stderr)
        sys.exit(1)

    qid = ex["metadata"]["question_id"]

    # Check for duplicate
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
                if existing["metadata"]["question_id"] == qid:
                    print(f"Already saved: {qid} — skipping.")
                    sys.exit(0)
            except Exception:
                pass

    with OUT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = sum(1 for l in OUT_FILE.read_text().splitlines() if l.strip())
    print(f"Saved {qid} ({ex['metadata']['category']}, {ex['metadata']['tool_profile']}) — total lines: {total}")


if __name__ == "__main__":
    main()
