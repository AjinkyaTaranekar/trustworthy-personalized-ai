#!/usr/bin/env python
"""executor_ablation.py — does the Executor SFT earn its place?

The Executor's whole job is: ONE plain-language instruction → ONE native tool call, copying the
instruction's concrete argument (code / query / url) verbatim. Base Qwen3-0.6B is already
post-trained for Hermes function calling, so before committing a second checkpoint to the
Thinker–Executor system we measure whether SFT actually beats the base model at this narrow task.

Two metrics, both verifiable (no judge model):
  • tool_choice_acc — emitted tool name == gold tool name
  • copy_fidelity   — the gold salient argument (code/query/url) is reproduced in the emitted call.
                       Reported as exact-match rate AND mean char-similarity (difflib ratio), since
                       a 0.6B paraphrasing long code is the dominant Executor failure mode.

Decoding is the FIXED Executor regime (clean greedy: do_sample=False, NO repetition_penalty /
no_repeat_ngram — see P0.2; anti-repetition forbids the model from copying its own target).

Usage
  # CPU, no model — validates parsing/scoring on canned outputs
  python executor_ablation.py --self_test

  # GPU — compare base vs the SFT'd Executor on a held-out slice
  python executor_ablation.py \
      --base unsloth/Qwen3-0.6B \
      --sft  models/checkpoint_executor \
      --data data/train_sft_executor.jsonl --n 150 --holdout_seed 42

Output: a side-by-side table + reports/executor_ablation.json. The decision rule is printed:
keep the SFT Executor only if it beats base by a real margin on BOTH metrics; otherwise drop the
Executor checkpoint and serve the base model behind the same EXECUTOR_STUDENT_PROMPT + schema.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path
from typing import Optional

from sft_v3_generator import EXECUTOR_STUDENT_PROMPT
from sft_trajectory_splitter import EXECUTOR_OWNED, EXECUTOR_SCHEMAS, _load_hermes, _HERMES_RE


# --- scoring ---------------------------------------------------------------------------
def _salient_arg(name: str, args: dict) -> str:
    if name == "python_execute":
        return (args.get("code") or "").strip()
    if name == "web_search":
        return (args.get("query") or "").strip()
    if name == "read_url":
        return (args.get("url") or "").strip()
    return ""


def _parse_call(text: str) -> Optional[tuple[str, dict]]:
    m = _HERMES_RE.search(text or "")
    if not m:
        return None
    return _load_hermes(m.group(1).strip())


def score_one(gold_name: str, gold_arg: str, emitted_text: str) -> dict:
    parsed = _parse_call(emitted_text)
    if parsed is None:
        return {"parsed": False, "tool_ok": False, "exact": False, "sim": 0.0, "emitted_name": None}
    name, args = parsed
    emitted_arg = _salient_arg(name, args)
    sim = difflib.SequenceMatcher(None, gold_arg, emitted_arg).ratio() if gold_arg else (1.0 if not emitted_arg else 0.0)
    return {
        "parsed": True,
        "tool_ok": name == gold_name,
        "exact": (name == gold_name) and (emitted_arg.strip() == gold_arg.strip()),
        "sim": round(sim, 4),
        "emitted_name": name,
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores) or 1
    return {
        "rows":             len(scores),
        "parse_rate":       round(sum(s["parsed"] for s in scores) / n, 4),
        "tool_choice_acc":  round(sum(s["tool_ok"] for s in scores) / n, 4),
        "copy_exact_rate":  round(sum(s["exact"] for s in scores) / n, 4),
        "copy_sim_mean":    round(sum(s["sim"] for s in scores) / n, 4),
    }


# --- held-out slice --------------------------------------------------------------------
def load_holdout(path: str, n: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        msgs = r["messages"]
        instr = msgs[1]["content"]
        parsed = _parse_call(msgs[2]["content"])
        if parsed is None:
            continue
        name, args = parsed
        out.append({"instruction": instr, "gold_name": name, "gold_arg": _salient_arg(name, args)})
    return out


# --- model backend ---------------------------------------------------------------------
class _HF:
    """Minimal Executor backend: EXECUTOR_STUDENT_PROMPT + instruction + native schema, clean greedy."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def emit(self, instruction: str, max_new_tokens: int = 512) -> str:
        conv = [
            {"role": "system", "content": EXECUTOR_STUDENT_PROMPT},
            {"role": "user", "content": instruction},
        ]
        prompt = self.tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
            tools=EXECUTOR_SCHEMAS, enable_thinking=True,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            # FIXED Executor decoding (P0.2): clean greedy, NO anti-repetition.
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True)


def run_backend(label: str, model_id: str, holdout: list[dict]) -> dict:
    print(f"\n[{label}] loading {model_id} …")
    be = _HF(model_id)
    scores = []
    for i, ex in enumerate(holdout, 1):
        emitted = be.emit(ex["instruction"])
        scores.append(score_one(ex["gold_name"], ex["gold_arg"], emitted))
        if i % 25 == 0:
            print(f"  [{label}] {i}/{len(holdout)}")
    agg = aggregate(scores)
    print(f"[{label}] {agg}")
    return agg


# --- self test (CPU, no model) ---------------------------------------------------------
def _self_test() -> None:
    print("=== executor_ablation self-test (CPU, no model) ===")
    gold = ('python_execute', 'print(17 * 23)')
    perfect = '<tool_call>\n{"name":"python_execute","arguments":{"code":"print(17 * 23)"}}\n</tool_call>'
    wrong_tool = '<tool_call>\n{"name":"web_search","arguments":{"query":"17 times 23"}}\n</tool_call>'
    paraphrase = '<tool_call>\n{"name":"python_execute","arguments":{"code":"print(17*23)"}}\n</tool_call>'
    no_call = "I will run the code now."

    s = score_one(*gold, perfect)
    assert s["tool_ok"] and s["exact"] and s["sim"] == 1.0, s
    s = score_one(*gold, wrong_tool)
    assert not s["tool_ok"] and not s["exact"], s
    s = score_one(*gold, paraphrase)
    assert s["tool_ok"] and not s["exact"] and 0.8 < s["sim"] < 1.0, s
    s = score_one(*gold, no_call)
    assert not s["parsed"] and s["sim"] == 0.0, s

    agg = aggregate([score_one(*gold, perfect), score_one(*gold, paraphrase),
                     score_one(*gold, wrong_tool), score_one(*gold, no_call)])
    assert agg["tool_choice_acc"] == 0.5 and agg["copy_exact_rate"] == 0.25, agg
    assert agg["parse_rate"] == 0.75, agg
    print(f"  scoring sanity: {agg}")
    print("  [pass] parse / tool-choice / copy-fidelity scoring correct")
    print("ALL SELF-TESTS PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description="Base vs SFT Executor ablation (copy-fidelity + tool choice).")
    p.add_argument("--base", default="unsloth/Qwen3-0.6B", help="Base model id (no SFT).")
    p.add_argument("--sft", default="models/checkpoint_executor", help="SFT'd Executor checkpoint.")
    p.add_argument("--data", default="data/train_sft_executor.jsonl")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--holdout_seed", type=int, default=42)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--out", default="reports/executor_ablation.json")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    holdout = load_holdout(args.data, args.n, args.holdout_seed)
    print(f"Held-out Executor rows: {len(holdout)} (seed={args.holdout_seed})")
    base_agg = run_backend("base", args.base, holdout)
    sft_agg = run_backend("sft", args.sft, holdout)

    print("\n=== Executor ablation ===")
    print(f"  {'metric':18s} {'base':>10s} {'sft':>10s} {'Δ(sft-base)':>12s}")
    for k in ("parse_rate", "tool_choice_acc", "copy_exact_rate", "copy_sim_mean"):
        print(f"  {k:18s} {base_agg[k]:10.4f} {sft_agg[k]:10.4f} {sft_agg[k]-base_agg[k]:12.4f}")

    # Decision rule
    margin = 0.03
    keep = (sft_agg["tool_choice_acc"] - base_agg["tool_choice_acc"] > margin) and \
           (sft_agg["copy_exact_rate"] - base_agg["copy_exact_rate"] > margin)
    verdict = ("KEEP the SFT Executor — it beats base by a real margin on both metrics."
               if keep else
               "DROP the SFT Executor — base is within margin; serve base + EXECUTOR_STUDENT_PROMPT + schema.")
    print(f"\n  Decision (margin {margin}): {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base": base_agg, "sft": sft_agg, "n": len(holdout),
                   "holdout_seed": args.holdout_seed, "keep_sft": keep, "verdict": verdict}, f, indent=2)
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
