"""
SFT Dataset Assembler
======================
Merges Part A (constitution + teacher) and Part B (rejection sampling) datasets
into a single training file. Applies quality filters, deduplicates, and produces
the final train/eval split.

Outputs:
    data/train_sft_v2.jsonl  — training set
    data/eval_sft_v2.jsonl   — eval set (10% held out)
    data/sft_v2_stats.json   — dataset statistics

Usage:
    python sft_dataset_assembler.py
    python sft_dataset_assembler.py --part_a data/train_partA.jsonl \\
                                     --part_b data/train_partB.jsonl \\
                                     --output_dir data/
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

MIN_RESPONSE_LENGTH = 80   # chars — too short = probably wrong
MAX_RESPONSE_LENGTH = 6000 # chars — too long = probably runaway
REQUIRED_TAGS = [ "<answer>", "</answer>"]
CAPABILITY_CHECK_REQUIRED = True


def passes_quality_filter(example: dict) -> tuple[bool, str]:
    """Return (passes, reason_if_failed)."""
    messages = example.get("messages", [])

    if len(messages) < 3:
        return False, "too_few_messages"

    # Find last assistant message
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False, "no_assistant_message"

    last_response = assistant_msgs[-1].get("content", "")

    if len(last_response) < MIN_RESPONSE_LENGTH:
        return False, f"response_too_short_{len(last_response)}"

    # if len(last_response) > MAX_RESPONSE_LENGTH:
    #     return False, f"response_too_long_{len(last_response)}"

    # Check required structural tags
    for tag in REQUIRED_TAGS:
        if tag not in last_response:
            return False, f"missing_tag_{tag}"

    # Check CAPABILITY_CHECK is present in think block.
    # appraisal_empathy examples also require an <appraisal> block.
    category = example.get("metadata", {}).get("category", "")
    if CAPABILITY_CHECK_REQUIRED:
        think_match = last_response.find("<think>")
        think_end = last_response.find("</think>")
        if think_match != -1 and think_end != -1:
            think_content = last_response[think_match:think_end]
            if "CAPABILITY_CHECK" not in think_content:
                return False, "missing_capability_check"
            if category == "appraisal_empathy" and "<appraisal>" not in think_content:
                return False, "missing_appraisal_block"

    # Check system message is present
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if not system_msgs:
        return False, "no_system_message"

    return True, "ok"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def get_question(example: dict) -> str:
    """Extract the user question from an example."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicate questions. Returns (deduped, n_removed)."""
    seen: set[str] = set()
    result = []
    n_removed = 0

    for ex in examples:
        q = get_question(ex)
        if not q:
            n_removed += 1
            continue
        if q in seen:
            n_removed += 1
            continue
        seen.add(q)
        result.append(ex)

    return result, n_removed


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


def balance_categories(examples: list[dict], max_per_category: int = 400) -> list[dict]:
    """Cap each category at max_per_category to prevent imbalance."""
    by_category = defaultdict(list)

    for ex in examples:
        cat = ex.get("metadata", {}).get("category") or ex.get("metadata", {}).get("question_type", "unknown")
        by_category[cat].append(ex)

    result = []
    for cat, cat_examples in by_category.items():
        random.shuffle(cat_examples)
        kept = cat_examples[:max_per_category]
        result.extend(kept)
        if len(cat_examples) > max_per_category:
            print(f"  Capped {cat}: {len(cat_examples)} → {max_per_category}")

    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(examples: list[dict]) -> dict:
    category_counts = Counter()
    source_counts = Counter()
    pipeline_counts = Counter()
    revised_count = 0
    avg_response_length = 0
    has_tool_call = 0

    for ex in examples:
        meta = ex.get("metadata", {})
        cat = meta.get("category") or meta.get("question_type", "unknown")
        category_counts[cat] += 1
        source_counts[meta.get("source", "unknown")] += 1
        pipeline_counts[meta.get("pipeline", "unknown")] += 1
        if meta.get("revised"):
            revised_count += 1

        # Find last assistant response
        assistant_msgs = [m for m in ex.get("messages", []) if m.get("role") == "assistant"]
        if assistant_msgs:
            resp = assistant_msgs[-1].get("content", "")
            avg_response_length += len(resp)
            if "<tool>" in resp:
                has_tool_call += 1

    n = len(examples)
    avg_response_length = avg_response_length // max(n, 1)

    return {
        "total": n,
        "by_category": dict(category_counts.most_common()),
        "by_source": dict(source_counts),
        "by_pipeline": dict(pipeline_counts),
        "revised_count": revised_count,
        "revised_pct": f"{100*revised_count/max(n,1):.1f}%",
        "avg_response_length_chars": avg_response_length,
        "has_tool_call_count": has_tool_call,
        "has_tool_call_pct": f"{100*has_tool_call/max(n,1):.1f}%",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Assemble final SFT v2 dataset from Part A + Part B")
    parser.add_argument("--part_a", type=str, default="pipeline/data/train_partA.jsonl")
    parser.add_argument("--part_b", type=str, default="pipeline/data/train_partB.jsonl")
    parser.add_argument("--output_dir", type=str, default="pipeline/data")
    parser.add_argument("--eval_frac", type=float, default=0.10,
                        help="Fraction held out for eval (default: 0.10)")
    parser.add_argument("--max_per_category", type=int, default=400,
                        help="Max examples per category (prevents imbalance)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both parts
    all_examples = []
    for filepath, label in [(args.part_a, "Part A"), (args.part_b, "Part B")]:
        p = Path(filepath)
        if not p.exists():
            print(f"Warning: {filepath} not found — skipping {label}")
            continue

        loaded = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    all_examples.append(ex)
                    loaded += 1
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {loaded} examples from {label} ({filepath})")

    print(f"\nTotal before filtering: {len(all_examples)}")

    # Quality filter
    passed = []
    failed_reasons: Counter = Counter()
    for ex in all_examples:
        ok, reason = passes_quality_filter(ex)
        if ok:
            passed.append(ex)
        else:
            failed_reasons[reason] += 1

    print(f"After quality filter: {len(passed)} ({len(all_examples)-len(passed)} removed)")
    if failed_reasons:
        print("  Filter reasons:", dict(failed_reasons.most_common(10)))

    # Deduplicate
    deduped, n_dupes = deduplicate(passed)
    print(f"After dedup: {len(deduped)} ({n_dupes} duplicates removed)")

    # Balance categories
    balanced = balance_categories(deduped, args.max_per_category)
    print(f"After balancing: {len(balanced)}")

    # Shuffle
    random.shuffle(balanced)

    # Train / eval split
    n_eval = max(1, int(len(balanced) * args.eval_frac))
    n_train = len(balanced) - n_eval
    eval_set = balanced[:n_eval]
    train_set = balanced[n_eval:]

    print(f"\nSplit: {n_train} train / {n_eval} eval")

    # Write output files
    train_path = output_dir / "train_sft_v2.jsonl"
    eval_path = output_dir / "eval_sft_v2.jsonl"
    stats_path = output_dir / "sft_v2_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "train": compute_stats(train_set),
        "eval": compute_stats(eval_set),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train: {n_train} examples → {train_path}")
    print(f"Eval:  {n_eval} examples  → {eval_path}")
    print(f"Stats: {stats_path}")
    print("\nCategory distribution (train):")
    for cat, count in stats["train"]["by_category"].items():
        pct = 100 * count / max(n_train, 1)
        print(f"  {cat:<35} {count:>5}  ({pct:.1f}%)")
    print(f"\nPipeline split:")
    for pipeline, count in stats["train"]["by_pipeline"].items():
        pct = 100 * count / max(n_train, 1)
        print(f"  {pipeline:<20} {count:>5}  ({pct:.1f}%)")
    print(f"\nRevised by self-critique: {stats['train']['revised_count']} ({stats['train']['revised_pct']})")
    print(f"With tool calls: {stats['train']['has_tool_call_count']} ({stats['train']['has_tool_call_pct']})")
    print(f"Avg response length: {stats['train']['avg_response_length_chars']} chars")


if __name__ == "__main__":
    main()
