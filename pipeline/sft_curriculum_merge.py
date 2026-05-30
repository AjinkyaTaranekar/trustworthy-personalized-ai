#!/usr/bin/env python
"""sft_curriculum_merge.py — interleave Branch B clarification rows into the factored
Thinker set for curriculum ordering (Thinker–Executor experiment §7.5, §7.9 #5, §7.10).

The factored Thinker set (`train_sft_thinker.jsonl`, from sft_trajectory_splitter.py) is
entirely Branch A/C — every row eventually ACTS. Trained on that alone, the Thinker learns
to always delegate and never stops to ask. The synthesised Branch B rows
(`train_sft_thinker_branch_b.jsonl`, from `sft_v3_generator.py --branch_b`) are the only
examples of the `<ask>` decision (plus its `B_negative` don't-over-clarify counterparts).

This step merges them with curriculum ordering: one Branch B row roughly every N factored
rows (default auto = len(factored)//len(branch_b), so ALL Branch B rows are placed and
evenly spread), producing the Thinker trainer input. It re-stamps `THINKER_STUDENT_PROMPT`
onto every row so both sources carry a byte-identical system prompt.

Runnable before Branch B exists: if the Branch B file is missing or empty, it warns loudly
and writes the factored set through unchanged (so Thinker training can still proceed on
A/C alone) — the merge is re-run once Branch B is generated (pending supervisor sign-off).

Usage:
  python sft_curriculum_merge.py                         # auto ratio, default paths
  python sft_curriculum_merge.py --ratio 11              # force 1 Branch B per 11 factored
  python sft_curriculum_merge.py --no_restamp            # keep each row's own system prompt
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sft_v3_generator import THINKER_STUDENT_PROMPT


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _restamp(row: dict) -> dict:
    """Overwrite the system message with THINKER_STUDENT_PROMPT (insert if absent) so every
    merged row carries the exact served Thinker prompt."""
    msgs = row.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": THINKER_STUDENT_PROMPT}
    else:
        msgs = [{"role": "system", "content": THINKER_STUDENT_PROMPT}, *msgs]
    out = dict(row)
    out["messages"] = msgs
    return out


def interleave(factored: list[dict], branch_b: list[dict], ratio: int) -> list[dict]:
    """Place one Branch B row every `ratio` factored rows; append any remainder so all
    Branch B rows are included. Factored order is preserved (it already encodes the
    Part A -> Part B curriculum)."""
    out: list[dict] = []
    bi = 0
    for i, f in enumerate(factored):
        out.append(f)
        if (i + 1) % ratio == 0 and bi < len(branch_b):
            out.append(branch_b[bi])
            bi += 1
    out.extend(branch_b[bi:])   # leftover B rows (if B denser than the ratio allows)
    return out


def _branch_dist(rows: list[dict]) -> dict:
    c = Counter((r.get("metadata") or {}).get("branch", "factored_AC") for r in rows)
    return dict(c)


def run(factored_path: str, branch_b_path: str, output: str,
        ratio: int = 0, seed: int = 42, restamp: bool = True) -> None:
    factored = _load(Path(factored_path))
    branch_b = _load(Path(branch_b_path))

    if not factored:
        raise SystemExit(f"[merge] factored Thinker set is empty/missing: {factored_path}\n"
                         f"        run sft_trajectory_splitter.py first.")

    if restamp:
        factored = [_restamp(r) for r in factored]
        branch_b = [_restamp(r) for r in branch_b]

    if not branch_b:
        print(f"  [merge] WARNING: Branch B file is missing/empty ({branch_b_path}).")
        print( "          Writing the factored A/C set through unchanged — the Thinker will")
        print( "          have NO <ask> examples and will learn to always delegate. Re-run this")
        print( "          merge once Branch B is generated (pending supervisor sign-off).")
        merged = factored
        used_ratio = None
    else:
        random.Random(seed).shuffle(branch_b)   # variety in placement, deterministic
        used_ratio = ratio if ratio > 0 else max(1, len(factored) // len(branch_b))
        merged = interleave(factored, branch_b, used_ratio)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Curriculum merge summary ===")
    print(f"  factored A/C rows : {len(factored)}")
    print(f"  branch B rows     : {len(branch_b)}  {_branch_dist(branch_b) if branch_b else ''}")
    if used_ratio:
        print(f"  interleave ratio  : 1 Branch B per {used_ratio} factored"
              f"{' (auto)' if ratio == 0 else ''}")
    else:
        print( "  interleave ratio  : n/a (no Branch B)")
    print(f"  merged total      : {len(merged)}")
    print(f"  restamp prompt    : {'on' if restamp else 'off'}")
    print(f"\n  wrote {output}")
    print("\nNext: train the Thinker on the merged curriculum:")
    print(f"  python 2_model_trainer.py --mode sft --dataset {output} --output_name checkpoint_thinker")


def main() -> None:
    p = argparse.ArgumentParser(description="Interleave Branch B clarification rows into the factored Thinker set.")
    p.add_argument("--factored", default="data/train_sft_thinker.jsonl")
    p.add_argument("--branch_b", default="data/train_sft_thinker_branch_b.jsonl")
    p.add_argument("--output", default="data/train_sft_thinker_curriculum.jsonl")
    p.add_argument("--ratio", type=int, default=0, help="Factored rows per 1 Branch B row (0 = auto: len(factored)//len(branch_b)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_restamp", action="store_true", help="Keep each row's own system prompt instead of THINKER_STUDENT_PROMPT.")
    args = p.parse_args()
    run(args.factored, args.branch_b, args.output,
        ratio=args.ratio, seed=args.seed, restamp=not args.no_restamp)


if __name__ == "__main__":
    main()
