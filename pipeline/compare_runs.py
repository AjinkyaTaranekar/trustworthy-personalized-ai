"""
Compare two saved benchmark JSON files and produce a dissertation-ready CSV + printout.

Usage — the practical one-GPU workflow:
  # Step 1: run vanilla model, save results
  python 4_benchmark.py --probe_only --model_label vanilla --no_judge
  #  → saves reports/constitution_probe_<ts>.json

  # Step 2: swap to fine-tuned model, save results
  python 4_benchmark.py --probe_only --model_label sft --no_judge
  #  → saves reports/constitution_probe_<ts2>.json

  # Step 3: compare offline (no servers needed)
  python compare_runs.py reports/constitution_probe_<ts>.json \\
                         reports/constitution_probe_<ts2>.json \\
                         --label_a vanilla --label_b sft

Outputs:
  reports/comparison_vanilla_vs_sft.csv   — dissertation table (paste directly)
  reports/comparison_vanilla_vs_sft.json  — full data for analysis
"""

import argparse
import csv
import json
from pathlib import Path


def load_probe_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(path_a: str, path_b: str,
            label_a: str = "vanilla", label_b: str = "sft",
            output_dir: Path = Path("reports")) -> None:

    data_a = load_probe_json(path_a)
    data_b = load_probe_json(path_b)

    scores_a = data_a.get("scores_by_principle", {})
    scores_b = data_b.get("scores_by_principle", {})
    overall_a = data_a.get("constitution_score", 0.0)
    overall_b = data_b.get("constitution_score", 0.0)
    all_ids = list(dict.fromkeys(list(scores_a) + list(scores_b)))  # preserve order

    # ── Print table ──────────────────────────────────────────────────────────
    w = 32
    print(f"\n{'='*72}")
    print(f"  BENCHMARK COMPARISON: {label_a.upper()} vs {label_b.upper()}")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"{'='*72}")
    print(f"  {'Principle':<{w}} {label_a[:10]:>10} {label_b[:10]:>10} {'Δ (B-A)':>10}  {'':>4}")
    print(f"  {'─'*{w}} {'─'*10} {'─'*10} {'─'*10}  {'─'*4}")

    rows = []
    for pid in all_ids:
        a = scores_a.get(pid, 0.0)
        b = scores_b.get(pid, 0.0)
        d = b - a
        if d > 0.05:
            tag = "↑ WIN"
        elif d < -0.05:
            tag = "↓ LOSS"
        else:
            tag = "≈ SAME"
        print(f"  {pid:<{w}} {a:>10.3f} {b:>10.3f} {d:>+10.3f}  {tag}")
        rows.append({
            "principle_id": pid,
            label_a: round(a, 4),
            label_b: round(b, 4),
            "delta": round(d, 4),
            "direction": tag,
            "hypothesis": _map_hypothesis(pid),
        })

    print(f"  {'─'*{w}} {'─'*10} {'─'*10} {'─'*10}  {'─'*4}")
    d_overall = overall_b - overall_a
    print(f"  {'OVERALL CONSTITUTION SCORE':<{w}} {overall_a:>10.3f} {overall_b:>10.3f} "
          f"{d_overall:>+10.3f}  {'↑ WIN' if d_overall > 0.05 else ('↓ LOSS' if d_overall < -0.05 else '≈ SAME')}")

    # Summary by hypothesis
    print(f"\n  HYPOTHESIS SUMMARY")
    print(f"  {'─'*50}")
    h_groups: dict = {}
    for r in rows:
        h = r["hypothesis"]
        h_groups.setdefault(h, []).append(r["delta"])
    for h, deltas in sorted(h_groups.items()):
        avg = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0.05)
        losses = sum(1 for d in deltas if d < -0.05)
        print(f"  {h:<20} avg_delta={avg:+.3f}  wins={wins}  losses={losses}/{len(deltas)}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{label_a}_vs_{label_b}"

    # CSV (dissertation-ready)
    csv_path = output_dir / f"{stem}.csv"
    fieldnames = ["principle_id", label_a, label_b, "delta", "direction", "hypothesis"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "principle_id": "OVERALL",
            label_a: round(overall_a, 4),
            label_b: round(overall_b, 4),
            "delta": round(d_overall, 4),
            "direction": "↑ WIN" if d_overall > 0.05 else ("↓ LOSS" if d_overall < -0.05 else "≈ SAME"),
            "hypothesis": "overall",
        })

    # Full JSON
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "label_a": label_a, "path_a": path_a,
            "label_b": label_b, "path_b": path_b,
            "overall": {label_a: overall_a, label_b: overall_b, "delta": d_overall},
            "rows": rows,
            "run_metadata_a": data_a.get("run_metadata", {}),
            "run_metadata_b": data_b.get("run_metadata", {}),
        }, f, indent=2)

    print(f"\n  CSV  → {csv_path}  (paste into dissertation)")
    print(f"  JSON → {json_path}  (full analysis data)")


def _map_hypothesis(pid: str) -> str:
    """Map probe ID to dissertation hypothesis label."""
    h1 = {"P1_decompose_first", "P2P3_tool_discipline", "P4_math_code",
          "P5_realtime_honesty", "P6_context_gate", "P7_uncertainty",
          "P8_impossibility", "P14_hold_position", "P18_answer_present",
          "P20_first_principles"}
    h2 = {"H2_memory_persistence", "P21_greedy_followup", "P16_user_memory"}
    h3_failure = {"P9_no_winner", "P12_assumption_naming", "P13_multi_step",
                  "P15_partial_capability", "P19_search_entity_facts"}
    if pid in h1:
        return "H1 (compliance)"
    if pid in h2:
        return "H2 (personalisation)"
    if pid in h3_failure:
        return "H3 (limits)"
    return "other"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare two benchmark JSON files")
    ap.add_argument("file_a", help="Path to first probe result JSON (e.g. vanilla baseline)")
    ap.add_argument("file_b", help="Path to second probe result JSON (e.g. fine-tuned)")
    ap.add_argument("--label_a", default="vanilla")
    ap.add_argument("--label_b", default="sft")
    ap.add_argument("--output_dir", default="reports")
    args = ap.parse_args()
    compare(args.file_a, args.file_b, args.label_a, args.label_b, Path(args.output_dir))
