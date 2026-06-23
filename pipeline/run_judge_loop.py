#!/usr/bin/env python3
"""
run_judge_loop.py — unattended VM runner for the LLM judge
==========================================================

Runs 5_judgement_day.py repeatedly until every item is judged, optionally committing
and pushing reports/ after each pass (mini-checkpoints, off-box). Safe to kill and
restart at any time: the judge resumes (skips already-judged items) and checkpoints
to disk every N items, and its RobustPool keeps going through rate limits (key
rotation + adaptive workers + near-unlimited retries). "Doesn't matter how many
retries it takes — it keeps going."

Usage on the VM (from pipeline/):
    python run_judge_loop.py --push                       # default: minimax-m3, all 5 conditions
    JUDGE_MODEL=nvidia_nim/minimaxai/minimax-m3 nohup python run_judge_loop.py --push &
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LABELS = ["vanilla_base", "vanilla_tools", "sft_template",
                  "sft_constitution", "thinker_executor"]
DEFAULT_SUITES = ["constitution", "categories", "drift", "persona"]


def _load_judge():
    spec = importlib.util.spec_from_file_location("judge5", "5_judgement_day.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # runs load_dotenv() in a real-script frame
    return m


def _remaining(judge, reports_dir: str, labels, suites) -> int:
    """Count items still missing a judge score across the latest report per suite."""
    paths = judge.discover_reports(Path(reports_dir), labels, latest_only=True)
    n = 0
    for p in paths:
        suite = judge.detect_suite(p)
        if suite not in suites:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if suite in ("constitution", "categories"):
            key = "probe_results" if suite == "constitution" else "category_results"
            for g in d.get(key, []):
                n += sum(1 for q in g.get("question_results", []) if q.get("llm_score") is None)
        elif suite == "drift":
            n += sum(1 for t in d.get("turn_results", []) if t.get("llm_score") is None)
        elif suite == "persona":
            for r in d.get("persona_results", []):
                j = r.get("judge")
                ok = isinstance(j, dict) and not j.get("judge_failed") and j.get("overall") is not None
                if not ok:
                    n += 1
    return n


def _git(*args, cwd="."):
    try:
        subprocess.run(["git", *args], cwd=cwd, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"  git {' '.join(args)} failed: {e}", flush=True)


def main() -> None:
    import os
    ap = argparse.ArgumentParser(description="Loop the judge until complete; checkpoint + push.")
    ap.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL", "nvidia_nim/minimaxai/minimax-m3"))
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=36.0)
    ap.add_argument("--checkpoint_every", type=int, default=10)
    ap.add_argument("--max_passes", type=int, default=50)
    ap.add_argument("--sleep", type=int, default=10, help="Seconds between passes.")
    ap.add_argument("--push", action="store_true", help="git add/commit/push reports/ after each pass.")
    args = ap.parse_args()

    judge = _load_judge()
    for i in range(1, args.max_passes + 1):
        rem = _remaining(judge, args.reports_dir, args.labels, args.suites)
        print(f"\n=== judge pass {i}/{args.max_passes} — {rem} items remaining ===", flush=True)
        if rem == 0:
            print("All items judged. Done.", flush=True)
            break
        subprocess.run([sys.executable, "5_judgement_day.py",
                        "--judge_model", args.judge_model,
                        "--labels", *args.labels, "--suites", *args.suites,
                        "--workers", str(args.workers), "--rpm", str(args.rpm),
                        "--checkpoint_every", str(args.checkpoint_every), "--report"],
                       check=False)
        if args.push:
            _git("add", args.reports_dir)
            _git("commit", "-m", f"judge: checkpoint pass {i} ({args.judge_model})")
            _git("push")
        time.sleep(args.sleep)
    else:
        print(f"Reached max_passes={args.max_passes}; {_remaining(judge, args.reports_dir, args.labels, args.suites)} still unjudged.", flush=True)

    print("\nNext: python analyze_experiments.py --labels "
          + " ".join(args.labels) + " --figures", flush=True)


if __name__ == "__main__":
    main()
