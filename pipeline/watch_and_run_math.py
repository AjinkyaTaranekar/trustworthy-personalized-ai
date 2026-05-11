"""
watch_and_run_math.py
---------------------
1. Polls pipeline/pipeline/data/train_partA.jsonl every INTERVAL seconds.
2. When line count exceeds THRESHOLD:
     a. Commits train_partA.jsonl.
     b. Launches sft_math_pipeline.py (passes through any --math_args).
3. While the math pipeline runs, monitors pipeline/data/train_partB.jsonl
   and commits every COMMIT_EVERY new lines.
4. Does a final commit of any remaining uncommitted partB lines when done.

Usage:
    python watch_and_run_math.py
    python watch_and_run_math.py --threshold 950 --interval 30 --commit_every 100
    python watch_and_run_math.py --math_args "--gsm8k_count 300 --math_count 700"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
GIT_ROOT   = SCRIPT_DIR.parent                            # repo root

PART_A     = SCRIPT_DIR / "data" / "train_partA.jsonl"
PART_B     = SCRIPT_DIR / "data" / "train_partB.jsonl"
MATH_PIPELINE = SCRIPT_DIR / "sft_math_pipeline.py"

# Paths relative to git root — used in git add / commit messages
PART_A_REL = PART_A.relative_to(GIT_ROOT)
PART_B_REL = PART_B.relative_to(GIT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"  [count_lines] ERROR: {e}")
        return 0


def git(*args: str) -> bool:
    """Run a git command from the repo root. Returns True on success."""
    cmd = ["git", "-C", str(GIT_ROOT), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [git] ERROR: {result.stderr.strip()}")
        return False
    return True


def commit_file(rel_path: Path, message: str) -> bool:
    """Stage a single file and commit it. Skips if nothing to commit."""
    git("add", str(rel_path))
    # Check if there is actually something staged
    status = subprocess.run(
        ["git", "-C", str(GIT_ROOT), "diff", "--cached", "--name-only"],
        capture_output=True, text=True,
    )
    if not status.stdout.strip():
        print(f"  [git] Nothing new to commit for {rel_path.name}")
        return False
    ok = git("commit", "-m", message)
    if ok:
        print(f"  [git] Committed: {message}")

    # Push
    push_ok = git("push")
    if not push_ok:
        print("  [git] WARNING: Failed to push commit. Please check your git status and push manually.")

    return ok


# ---------------------------------------------------------------------------
# Phase 1 — watch partA
# ---------------------------------------------------------------------------

def watch_part_a(threshold: int, interval: int, commit_every: int) -> int:
    print(f"[Phase 1] Watching {PART_A}")
    print(f"          Threshold : {threshold} lines  |  Poll : {interval}s\n")
    while True:
        lines = count_lines(PART_A)
        print(f"[{time.strftime('%H:%M:%S')}] train_partA  {lines} lines", end="")
        if lines >= threshold:
            print(f"  → threshold exceeded!")
            return lines
        print(f"  (need {threshold - lines} more)")
        if lines > 0 and lines % commit_every == 0:
            commit_file(PART_A_REL, f"chore: auto-commit train_partA.jsonl ({lines} lines)")
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Phase 2 — commit partA, launch math pipeline, watch partB
# ---------------------------------------------------------------------------

def run_with_part_b_commits(
    math_args: list[str],
    commit_every: int,
    interval: int,
) -> int:
    # --- commit partA ---
    lines_a = count_lines(PART_A)
    print(f"\n[Phase 2] Committing train_partA.jsonl ({lines_a} lines)...")
    commit_file(PART_A_REL, f"chore: auto-commit train_partA.jsonl ({lines_a} lines)")

    # --- launch math pipeline (inherits stdout/stderr so its logs appear here) ---
    cmd = [sys.executable, str(MATH_PIPELINE)] + math_args + ["--resume"]
    print(f"\n[Phase 2] Launching: {' '.join(cmd)}\n{'='*60}")
    proc = subprocess.Popen(cmd)

    # --- watch partB while pipeline runs ---
    committed_up_to = count_lines(PART_B)   # baseline (lines already on disk before run)
    print(f"\n[Phase 3] Monitoring train_partB.jsonl (committing every {commit_every} new lines)")
    print(f"          Baseline : {committed_up_to} lines already on disk\n")

    def flush_part_b(label: str) -> None:
        """Commit any uncommitted partB lines, used on normal finish and Ctrl+C."""
        final = count_lines(PART_B)
        remaining = final - committed_up_to
        if remaining > 0:
            print(f"\n[Phase 3] {label} — committing {remaining} new partB lines...")
            commit_file(PART_B_REL, f"chore: auto-commit train_partB.jsonl {label} ({final} lines, +{remaining})")
        else:
            print(f"\n[Phase 3] {label} — no uncommitted partB lines.")

    try:
        while proc.poll() is None:
            time.sleep(interval)
            current = count_lines(PART_B)
            new_lines = current - committed_up_to
            print(f"[{time.strftime('%H:%M:%S')}] train_partB  {current} lines  (+{new_lines} since last commit)")
            if new_lines >= commit_every:
                commit_file(PART_B_REL, f"chore: auto-commit train_partB.jsonl ({current} lines, +{new_lines})")
                committed_up_to = current

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Stopping math pipeline...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        flush_part_b("interrupted")
        print("[Ctrl+C] Exiting.")
        sys.exit(1)

    flush_part_b("finished")
    return proc.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Watch partA, commit it when ready, run math pipeline, commit partB every N lines"
    )
    parser.add_argument("--threshold", type=int, default=1020,
                        help="partA line count that triggers the math pipeline (default: 950)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Poll interval in seconds (default: 60s)")
    parser.add_argument("--commit_every", type=int, default=50,
                        help="Commit partB every N new lines written (default: 100)")
    parser.add_argument("--math_args", type=str, default="",
                        help="Extra args forwarded to sft_math_pipeline.py, e.g. '--gsm8k_count 300'")
    args = parser.parse_args()

    watch_part_a(args.threshold, args.interval, args.commit_every)

    extra = args.math_args.split() if args.math_args.strip() else []
    rc = run_with_part_b_commits(
        math_args    = extra,
        commit_every = args.commit_every,
        interval     = args.interval,
    )

    print(f"\nDone. Math pipeline exited with code {rc}.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
