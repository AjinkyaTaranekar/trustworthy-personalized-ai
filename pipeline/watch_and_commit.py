"""
Watch a file and commit+push every N new lines.
Usage: python pipeline/watch_and_commit.py <file> [--threshold 100] [--interval 30]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8") if _.strip())
    except FileNotFoundError:
        return 0


def git_commit_push(watched: Path, count: int) -> None:
    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=watched.parent)
        .decode()
        .strip()
    )
    rel_path = watched.relative_to(repo_root)
    msg = f"data: auto-checkpoint — {count} lines in {watched.name}"
    try:
        subprocess.run(["git", "add", str(rel_path)], cwd=repo_root, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=repo_root, check=True)
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        print(f"[watcher] committed + pushed at {count} lines")
    except subprocess.CalledProcessError as exc:
        print(f"[watcher] git error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="File to watch")
    parser.add_argument("--threshold", type=int, default=100, help="Commit every N new lines")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    watched: Path = args.file.resolve()
    baseline = count_lines(watched)
    last_committed = (baseline // args.threshold) * args.threshold
    print(f"[watcher] started — file: {watched}, current lines: {baseline}, last checkpoint: {last_committed}")

    while True:
        time.sleep(args.interval)
        current = count_lines(watched)
        checkpoint = (current // args.threshold) * args.threshold
        if checkpoint > last_committed:
            git_commit_push(watched, current)
            last_committed = checkpoint


if __name__ == "__main__":
    main()
