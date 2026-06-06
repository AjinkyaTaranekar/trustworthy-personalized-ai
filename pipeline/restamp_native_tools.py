#!/usr/bin/env python
"""restamp_native_tools.py — re-stamp metadata.native_tools to the canonical served schema.

PURE TRANSFORM (no GPU, no teacher). The native tool schemas are rendered into the model's prompt
via apply_chat_template(tools=metadata["native_tools"]), so they are part of the train/serve
contract. Over time the registry descriptions drifted and the served tool SET (which includes the
always-on user_memory_*/scratchpad_* tools) grew beyond what older datasets baked — e.g.
train_sft_v3.jsonl trained on 1–4 tools per row while the server now serves 7–10. That is a
train/serve schema mismatch that degrades tool use.

This script rewrites each row's metadata.native_tools to exactly what the inference server would
serve for that row's `tool_profile`: registry.to_openai_schemas(TOOL_PROFILES[profile]). The
conversation `messages` are NOT touched — only the advertised schema. Run it after any registry
description change, or to repair the tool-set drift, then retrain.

Usage
  python restamp_native_tools.py --data data/train_sft_v3.jsonl              # writes in place, keeps .bak
  python restamp_native_tools.py --data data/train_sft_v3.jsonl --dry_run    # report only
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from pipeline_tools import ToolRegistry
from tool_io import TOOL_PROFILES

_REGISTRY = ToolRegistry()
# Cache the canonical schema per profile (registration order == served order).
_CANON = {p: _REGISTRY.to_openai_schemas(tools) for p, tools in TOOL_PROFILES.items()}
_DEFAULT_PROFILE = "all_tools"


def _names(schemas: list) -> tuple:
    return tuple(s["function"]["name"] for s in schemas)


def restamp(path: str, dry_run: bool = False) -> int:
    p = Path(path)
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    before = Counter()
    after = Counter()
    changed = 0
    for r in rows:
        meta = r.setdefault("metadata", {})
        profile = meta.get("tool_profile") or _DEFAULT_PROFILE
        canon = _CANON.get(profile, _CANON[_DEFAULT_PROFILE])
        old = meta.get("native_tools") or []
        before[(profile, len(old))] += 1
        if json.dumps(old) != json.dumps(canon):
            changed += 1
        meta["native_tools"] = canon
        after[(profile, len(canon))] += 1

    print(f"  rows: {len(rows)}  changed native_tools: {changed}")
    print("  BEFORE (profile, n_tools): " + ", ".join(f"{k}={v}" for k, v in sorted(before.items())))
    print("  AFTER  (profile, n_tools): " + ", ".join(f"{k}={v}" for k, v in sorted(after.items())))
    for profile in sorted({pr for pr, _ in after}):
        print(f"    {profile:18s} -> {list(_names(_CANON.get(profile, _CANON[_DEFAULT_PROFILE])))}")

    if dry_run:
        print("  [dry-run] no file written.")
        return 0

    bak = p.with_suffix(p.suffix + ".pretoolstamp.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backup → {bak}")
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  re-stamped → {p}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-stamp metadata.native_tools to the canonical served schema.")
    ap.add_argument("--data", required=True, help="SFT JSONL with metadata.tool_profile per row.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    restamp(args.data, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
