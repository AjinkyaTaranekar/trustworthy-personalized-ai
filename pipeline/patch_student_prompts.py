"""Patch all existing entries in train_partA_v3.jsonl to use the updated student prompts.

Each entry's system message is replaced with the current STUDENT_PROMPTS[tool_profile]
from sft_v3_generator.py — which now includes First Principles + 5W+H + greedy follow-up.

Usage:
    python patch_student_prompts.py                              # dry-run (preview counts)
    python patch_student_prompts.py --apply                     # write changes
    python patch_student_prompts.py --apply --input data/train_partA_v3.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sft_v3_generator as gen


def patch_file(input_path: Path, output_path: Path, dry_run: bool = True) -> None:
    old_prompts = set(gen.STUDENT_PROMPTS.values())

    total = updated = skipped = already_current = parse_errors = 0

    out_lines: list[str] = []
    with open(input_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [line {lineno}] JSON parse error — skipping: {e}")
                parse_errors += 1
                out_lines.append(raw)
                continue

            total += 1
            messages = entry.get("messages", [])
            meta = entry.get("metadata", {})
            profile_label = meta.get("tool_profile", "")

            # Locate system message
            sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
            if sys_idx is None:
                print(f"  [line {lineno}] no system message — skipping")
                skipped += 1
                out_lines.append(raw)
                continue

            new_prompt = gen.STUDENT_PROMPTS.get(profile_label)
            if new_prompt is None:
                print(f"  [line {lineno}] unknown tool_profile={profile_label!r} — skipping")
                skipped += 1
                out_lines.append(raw)
                continue

            current_content = messages[sys_idx].get("content", "")
            if current_content == new_prompt:
                already_current += 1
                out_lines.append(raw)
                continue

            # Replace the system prompt
            messages[sys_idx] = {**messages[sys_idx], "content": new_prompt}
            entry["messages"] = messages
            updated += 1
            out_lines.append(json.dumps(entry, ensure_ascii=False))

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total entries  : {total}")
    print(f"Already current: {already_current}")
    print(f"Will update    : {updated}")
    print(f"Skipped        : {skipped}")
    print(f"Parse errors   : {parse_errors}")

    if dry_run:
        print("\nDry run — no files written. Pass --apply to write changes.")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"\nWritten {len(out_lines)} lines to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Patch student prompts in JSONL training data")
    p.add_argument("--input", default="data/train_partA_v3.jsonl")
    p.add_argument("--output", default=None, help="Output path (defaults to overwriting input)")
    p.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    patch_file(input_path, output_path, dry_run=not args.apply)


if __name__ == "__main__":
    main()
