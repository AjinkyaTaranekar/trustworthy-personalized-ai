#!/usr/bin/env python
"""test_tool_calling.py — do the published checkpoints emit parseable native tool calls?

Loads each model directly (no inference server) and reproduces the *exact* training / eval
rendering path — system prompt + that row's own metadata.native_tools, rendered via
apply_chat_template(tools=..., enable_thinking=True), greedy decode (do_sample=False).
A PASS means the weights learned the native <tool_call>{"name":...,"arguments":...}</tool_call>
format, so any empty tool_trace at benchmark time is a *serving* problem, not the model.

Tests every published ablation model against the data it actually trained on:
  vanilla_base       unsloth/Qwen3-0.6B                              (base, zero-shot)
  sft_template       AjinkyaTaranekar/trustworthy-ai-sft-template    train_interleaved_native.jsonl
  sft_constitution   AjinkyaTaranekar/trustworthy-ai-sft-constitution train_sft_v3.jsonl
  thinker            AjinkyaTaranekar/trustworthy-ai-thinker         train_sft_thinker_curriculum.jsonl  (delegates via <act>)
  executor           AjinkyaTaranekar/trustworthy-ai-executor        train_sft_executor.jsonl

  REPLAY — replays N real training rows (prompt up to the first assistant turn), the faithful
           per-model test. For single/executor models it expects a parseable <tool_call>;
           the thinker delegates, so it expects an <act> block instead.
  FRESH  — single models only: canonical STUDENT_PROMPTS[profile] + a brand-new question.

Usage:
    python test_tool_calling.py                       # all published models, 3 replay rows each
    python test_tool_calling.py --only constitution    # just one
    python test_tool_calling.py --models ./models/checkpoint_sft_constitution
    python test_tool_calling.py --discover             # enumerate AjinkyaTaranekar/* from the HF Hub
"""
import argparse
import gc
import json
import re
import sys

# repo_id, data_file, kind, label
PUBLISHED = [
    ("unsloth/Qwen3-0.6B",                              "data/train_sft_v3.jsonl",                 "single",   "vanilla_base"),
    ("AjinkyaTaranekar/trustworthy-ai-sft-template",    "data/train_interleaved_native.jsonl",     "single",   "sft_template"),
    ("AjinkyaTaranekar/trustworthy-ai-sft-constitution","data/train_sft_v3.jsonl",                 "single",   "sft_constitution"),
    ("AjinkyaTaranekar/trustworthy-ai-thinker",         "data/train_sft_thinker_curriculum.jsonl", "thinker",  "thinker"),
    ("AjinkyaTaranekar/trustworthy-ai-executor",        "data/train_sft_executor.jsonl",           "executor", "executor"),
]


# Mirrors 3_infererence.py:_parse_native_tool_call exactly (tag-delimited, strict=False).
def parse_native_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1), strict=False)
        if isinstance(obj, list):
            obj = next((o for o in obj if isinstance(o, dict)), None)
        if not isinstance(obj, dict):
            return None
        return {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def _tool_names(native_tools):
    out = []
    for t in native_tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if fn.get("name"):
            out.append(fn["name"])
    return out


def _prompt_messages(row):
    """Messages up to (not including) the first assistant turn — the model's input."""
    out = []
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            break
        out.append({"role": m["role"], "content": m.get("content", "")})
    return out


def _load_rows(path, kind, n):
    """First n rows whose first assistant turn carries the expected marker for this kind."""
    marker = "<act>" if kind == "thinker" else "<tool_call>"
    rows = []
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return rows
    with fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = row.get("messages", [])
            asst = next((m for m in msgs if m.get("role") == "assistant"), None)
            if not asst or marker not in (asst.get("content") or ""):
                continue
            if not row.get("metadata", {}).get("native_tools"):
                continue
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def test_model(repo_id, data_file, kind, label, args, STUDENT_PROMPTS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}\n  {label}  →  {repo_id}   (kind={kind})\n{'='*70}", flush=True)
    rec = {"label": label, "repo": repo_id, "loaded": False, "replay": "0/0", "fresh": "-", "note": ""}

    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        rec["loaded"] = True
    except Exception as e:  # noqa: BLE001
        rec["note"] = f"load failed: {type(e).__name__}: {str(e)[:120]}"
        print(f"  [FAIL] {rec['note']}")
        print("         (published models are 16-bit merged; if this is a LoRA adapter, point at the merged repo)")
        return rec

    def generate(messages, native_tools):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       tools=native_tools, enable_thinking=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---- REPLAY: faithful per-model test on its own training rows ----
    rows = _load_rows(data_file, kind, args.n)
    if not rows:
        rec["note"] = f"no replay rows in {data_file}"
        print(f"  [skip] {rec['note']}")
    else:
        passed = 0
        for i, row in enumerate(rows, 1):
            native_tools = row["metadata"]["native_tools"]
            gen = generate(_prompt_messages(row), native_tools)
            if kind == "thinker":
                ok = "<act>" in gen or "<ask>" in gen or "<answer>" in gen
                tag = "<act>/<ask>/<answer>" if ok else "none of <act>/<ask>/<answer>"
            else:
                call = parse_native_tool_call(gen)
                ok = bool(call and call["name"] in set(_tool_names(native_tools)))
                tag = f"{call['name']}(...)" if call else ("<tool_call> unparseable" if "<tool_call>" in gen else "no <tool_call>")
            passed += ok
            print(f"  replay {i}/{len(rows)}: {'PASS' if ok else 'FAIL'}  → {tag}")
            if not ok:
                print("    ---- generation (first 500 chars) ----")
                print("    " + gen[:500].replace("\n", "\n    "))
        rec["replay"] = f"{passed}/{len(rows)}"

    # ---- FRESH: single models only (canonical prompt + new question) ----
    if kind == "single":
        native_tools = rows[0]["metadata"]["native_tools"] if rows else None
        if native_tools:
            sys_prompt = STUDENT_PROMPTS.get(args.profile, STUDENT_PROMPTS["all_tools"])
            gen = generate([{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": args.question}], native_tools)
            call = parse_native_tool_call(gen)
            ok = bool(call and call["name"] in set(_tool_names(native_tools)))
            rec["fresh"] = "PASS" if ok else "FAIL"
            print(f"  FRESH (canonical {args.profile} + new question): {rec['fresh']}"
                  + (f"  → {call['name']}(...)" if call else ""))
            if not ok:
                print("    ---- generation (first 500 chars) ----")
                print("    " + gen[:500].replace("\n", "\n    "))

    # free the GPU before the next model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="Explicit repo ids / local dirs (kind inferred from name)")
    ap.add_argument("--only", default=None, help="Substring filter on label/repo (e.g. 'constitution')")
    ap.add_argument("--discover", action="store_true", help="Enumerate {hf_user}/trustworthy-ai-* from the HF Hub")
    ap.add_argument("--hf_user", default="AjinkyaTaranekar")
    ap.add_argument("--profile", default="compute_only", choices=["all_tools", "compute_only", "compute_and_search", "no_tools"])
    ap.add_argument("--question", default="What is 9847 * 23.5 + 1284? Compute it exactly.")
    ap.add_argument("--n", type=int, default=3, help="Replay rows per model")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    # Importing STUDENT_PROMPTS IS the fix under test — must succeed in the no-litellm env.
    try:
        from sft_v3_generator import STUDENT_PROMPTS
        print("[ok] STUDENT_PROMPTS imported from sft_v3_generator (canonical source).")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] cannot import STUDENT_PROMPTS ({type(e).__name__}: {e}) — server would use the DRIFTED fallback.")
        return 2

    def infer_kind(name):
        low = name.lower()
        if "executor" in low:
            return "executor", "data/train_sft_executor.jsonl"
        if "thinker" in low:
            return "thinker", "data/train_sft_thinker_curriculum.jsonl"
        if "template" in low:
            return "single", "data/train_interleaved_native.jsonl"
        return "single", "data/train_sft_v3.jsonl"

    if args.models:
        targets = [(m, *infer_kind(m), m.rstrip("/").split("/")[-1]) for m in args.models]
    elif args.discover:
        from huggingface_hub import HfApi
        ids = [m.id for m in HfApi().list_models(author=args.hf_user) if "trustworthy-ai" in m.id]
        print(f"[discover] {len(ids)} repo(s): {ids}")
        targets = [(i, *infer_kind(i), i.split("/")[-1]) for i in ids]
    else:
        targets = PUBLISHED

    if args.only:
        targets = [t for t in targets if args.only.lower() in t[0].lower() or args.only.lower() in t[3].lower()]
    if not targets:
        print("[FAIL] no models matched.")
        return 2

    results = []
    for repo_id, data_file, kind, label in targets:
        results.append(test_model(repo_id, data_file, kind, label, args, STUDENT_PROMPTS))

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  {'model':<18} {'loaded':<7} {'REPLAY':<8} {'FRESH':<6} note")
    for r in results:
        print(f"  {r['label']:<18} {str(r['loaded']):<7} {r['replay']:<8} {r['fresh']:<6} {r['note']}")
    print("\n  REPLAY = parseable tool call (or <act> for thinker) on real training rows, greedy.")
    print("  All REPLAY passing → weights are fine; empty benchmark tool_trace is serving plumbing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
