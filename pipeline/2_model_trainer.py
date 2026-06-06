"""
Model Trainer
=============
Phase 1 — SFT (Supervised Fine-Tuning):
    python 2_model_trainer.py --mode sft

Phase 2 — GRPO (Group Relative Policy Optimisation, DAPO improvements):
    python 2_model_trainer.py --mode grpo --sft_checkpoint models/checkpoint_sft \
        --reward_type d --output_name checkpoint_grpo_d

Reward types:
    c  format + accuracy only         (Ablation C)
    d  format + accuracy + tool + constitution  (Ablation D — full thesis contribution)

DAPO improvements applied over vanilla GRPO:
    - Token-level loss normalisation (Dr.GRPO): loss_type='dr_grpo' in GRPOConfig
    - Clip-Higher: asymmetric ε (0.2 low, 0.28 high) via epsilon_high
    - Dynamic sampling: skip zero-variance groups (monkey-patched on GRPOTrainer)
    - Truncated completion masking: mask_truncated_completions=True
    - Reference policy = SFT checkpoint (not base model)
"""

import json
import os
import re
import argparse
import subprocess
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Python 3.14 + dill compatibility patch
#
# Root cause: Python 3.14 changed pickle.save_dict to call
#   self._batch_setitems(obj.items(), obj)   ← 2 positional args
# but _batch_setitems only accepts (self, items).
# Dill does NOT define _batch_setitems in its own Pickler.__dict__ — it
# inherits from pickle.Pickler — so a __dict__.get() check returns None
# and the previous patch was silently skipped.
#
# Fix: walk the full MRO to find the real implementation, then define a new
# override DIRECTLY on dill.Pickler that accepts and ignores the extra arg.
# This ensures Python's dispatch finds our override before the broken one.
# ---------------------------------------------------------------------------
def _apply_py314_dill_patch():
    import sys
    if sys.version_info < (3, 14):
        return
    try:
        import dill._dill as _dill
        # Walk MRO (skipping dill.Pickler itself) to find the defining class
        _real_bsi = None
        for _base in type.mro(_dill.Pickler):
            if _base is _dill.Pickler:
                continue
            _bsi = _base.__dict__.get("_batch_setitems")
            if _bsi is not None:
                _real_bsi = _bsi
                break
        if _real_bsi is None:
            return  # Nothing found; nothing to fix
        # Define it on dill.Pickler so dispatch hits our version first
        def _patched_bsi(self, items, obj=None):
            return _real_bsi(self, items)
        _dill.Pickler._batch_setitems = _patched_bsi
    except Exception:
        pass

_apply_py314_dill_patch()

try:
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    from datasets import load_dataset, Dataset
    from transformers import TrainerCallback
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    TrainerCallback = object  # fallback base so the monitor class still defines


# ---------------------------------------------------------------------------
# Model + training configuration
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    # unsloth/Qwen3-0.6B IS the instruct/chat model — there is no separate -Instruct variant.
    # Qwen3 naming: the main model (pretraining + post-training) ships under the plain name;
    # the raw base (pretraining only) is under unsloth/Qwen3-0.6B-Base.
    # Using the plain model is correct — it already has instruction-following from post-training.
    "base_model":     "unsloth/Qwen3-0.6B",
    "max_seq_length": 4096,  # p95 of training examples is ~3530 tokens; 4096 covers ~98% without truncation
    # 16-bit LoRA: load the 0.6B instruct model in bf16 (~1.2 GB) instead of 4-bit.
    # Eliminates QLoRA's quantisation noise at negligible VRAM cost on 16 GB.
    # Full-precision base + LoRA adapters ≈ 3-4 GB vs 4-bit QLoRA ≈ 1.5 GB — both
    # fit comfortably; the quality gain justifies the small extra overhead.
    "load_in_4bit":   False,
    # r=64, α=16 matches the community consensus for complex multi-behaviour tasks
    # (distillabs benchmark that beat a 120B teacher used r=64, α=16 at 5e-5 LR).
    # Previous r=16 gave the model insufficient capacity to internalise First
    # Principles + 5W+H + greedy follow-up as a unified behavioural pattern.
    "lora_r":         64,
    "lora_alpha":     16,
}

SFT_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    # 3 epochs: SFT history showed eval loss plateauing at epoch 2.5-3.0 (~1.32)
    # while training loss kept falling — textbook overfitting. The extra epoch at
    # r=16 added noise, not generalisation. At r=64 the model learns faster per
    # epoch; 3 epochs gives ~729 gradient updates on 1944 examples (eff. batch 8),
    # matching the gradient-step budget where the previous run's best checkpoint fell.
    "num_train_epochs":            3,
    # 1e-4 (was 2e-4): the 2026-05-25 benchmark showed SFT collapsing the base model's
    # reasoning (think_empty 0%→95%, P1/P15/P20 1.0→0.0) — capacity displacement from too
    # aggressive an update on 0.6B. Halving the LR preserves more of the base thinking
    # pathway while still learning the constitutional behaviour. Re-benchmark on GPU to confirm.
    "learning_rate":               1e-4,
    "warmup_steps":                50,
    "logging_steps":               10,
    "save_steps":                  25,
    "eval_steps":                  25,
    "save_total_limit":            4,
    "bf16":                        True,
    "optim":                       "adamw_8bit",
    "weight_decay":                0.01,
    "lr_scheduler_type":           "cosine",  # cosine decay outperforms linear for behavioural SFT; matches GRPO phase
    "packing":                     False,  # disabled: packing can split multi-turn tool-call sequences at pack boundaries
}

GRPO_CONFIG = {
    # Group size G — number of completions sampled per prompt
    # 4 is the safe limit for 0.6B + 4-bit + 16 GB VRAM (A4000); 8 needs 24 GB+
    "num_generations":             4,
    # Learning rate — lower than SFT, fine-tuning a fine-tuned model
    "learning_rate":               1e-6,
    # KL coefficient β — anchors the policy to the SFT checkpoint (reference policy)
    # 0.001 is the R1 stage-1 value; increase to 0.01 if constitutional drift detected
    "kl_coef":                     0.001,
    # DAPO Clip-Higher: asymmetric clipping
    #   ε_low  = standard lower clip (same as vanilla GRPO ε=0.2)
    #   ε_high = looser upper clip — lets high-reward completions update more freely,
    #             preventing entropy collapse where all G completions become identical
    "clip_range_ratio":            0.2,    # ε_low
    "clip_range_ratio_high":       0.28,   # ε_high (DAPO Clip-Higher)
    # Generation settings
    "temperature":                 1.0,    # rollout temperature — must be >0 for diversity
    "max_new_tokens":              2048,   # was 768; model needs ≥2048 to produce full tool-call + reasoning responses
    # Prompt length cap — system prompt (constitution, 23 principles) + user question
    # can reach ~700 tokens; 1536 gives headroom for longer user messages.
    # Without this, TRL defaults to 512 and silently truncates the system prompt.
    "max_prompt_length":           1536,
    # Training loop — 2 epochs gives ~300 gradient steps on a 1200-row dataset
    # (batch=1, grad_accum=8 → ~150 steps/epoch); Unsloth recommends 300+ for RL signal.
    "num_train_epochs":            2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps":               5,
    "save_steps":                  100,
    "bf16":                        True,
    "optim":                       "adamw_8bit",
    "weight_decay":                0.01,
    "warmup_ratio":                0.05,   # short warmup; GRPO LR is already low (1e-6)
    "lr_scheduler_type":           "cosine",
    "max_grad_norm":               1.0,    # gradient clipping — prevents reward spikes destabilising training
    # DAPO dynamic sampling: skip prompts where all G completions score identically
    # (zero-gradient batches waste compute and reward signal)
    "dynamic_sampling":            True,
}

# Reward component weights — must sum to 1.0
REWARD_WEIGHTS = {
    "format":           0.20,  # structural quality: think content + answer tag (was 0.25; 0.05 reallocated to greedy_followup)
    "accuracy":         0.35,  # correctness: math code execution
    "tool_integrity":   0.10,  # no hallucinated/unavailable tools (P3)
    "tool_quality":     0.15,  # correct tool for question type + non-empty params (was 0.20; 0.05 reallocated)
    "constitution":     0.10,  # broader rule check: P1+P4+P14+P18
    "greedy_followup":  0.10,  # <answer> ends with a 5W+H follow-up question (First Principles / personalisation principle P21)
}


# ---------------------------------------------------------------------------
# Curriculum learning — three-stage data split
# ---------------------------------------------------------------------------

def _split_curriculum_stages(
    examples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into three curriculum stages.

    Stage 1: short, no-tool examples — format establishment.
    Stage 2: all examples — complexity scaling.
    Stage 3: all + 20% Stage-1 replay — anti-drift.
    Returns (stage1, stage2, stage3).
    """
    import random as _random
    stage1 = []
    stage2 = list(examples)

    for ex in examples:
        msgs = ex.get("messages", [])
        has_tool = any(m.get("role") == "tool" for m in msgs)
        asst_text = " ".join(
            m.get("content", "") or ""
            for m in msgs if m.get("role") == "assistant"
        )
        if not has_tool and len(asst_text) < 600:
            stage1.append(ex)

    replay_n = min(len(stage1), max(1, len(stage2) // 5))
    replay = _random.sample(stage1, replay_n) if stage1 else []
    stage3 = stage2 + replay
    return stage1, stage2, stage3


def _write_temp_jsonl(examples: list[dict]) -> str:
    """Write examples to a temp .jsonl and return its path (caller-owned)."""
    import tempfile as _tempfile
    tmp = _tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for ex in examples:
        tmp.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# ROUGE evaluation helper
# ---------------------------------------------------------------------------

def compute_rouge(hypotheses: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L. Returns precision/recall/fmeasure per metric.

    Args:
        hypotheses: Model-generated responses.
        references: Gold reference responses.
    Returns:
        Dict mapping metric name → {precision, recall, fmeasure}, all rounded to 4 dp.
        Returns zeros for all metrics if rouge_score is not installed.
    """
    _empty = {k: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for k in ("rouge1", "rouge2", "rougeL")}
    try:
        from rouge_score import rouge_scorer as _rs
    except ImportError:
        print("  [WARN] rouge_score not installed — ROUGE skipped. pip install rouge-score")
        return _empty
    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: dict = {"rouge1": [], "rouge2": [], "rougeL": []}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for key in agg:
            agg[key].append(scores[key])
    if not hypotheses:
        return {k: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for k in agg}
    return {
        key: {
            "precision": round(sum(s.precision for s in vals) / len(vals), 4),
            "recall":    round(sum(s.recall    for s in vals) / len(vals), 4),
            "fmeasure":  round(sum(s.fmeasure  for s in vals) / len(vals), 4),
        }
        for key, vals in agg.items()
    }


def _retry_hf_push(fn, *args, max_retries: int = 3, **kwargs) -> None:
    """Call a HuggingFace push function with exponential backoff retries.

    Protects against transient network failures after long training runs.
    Raises the last exception if all retries are exhausted.
    """
    import time
    for attempt in range(1, max_retries + 1):
        try:
            fn(*args, **kwargs)
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"  [publish] HF push failed (attempt {attempt}/{max_retries}): {e}")
            print(f"  [publish] Retrying in {wait}s... (local files are safe)")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# GRPO reward functions (all verifiable — no judge model needed)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS_GRPO = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})


def _safe_execute(code: str, timeout: int = 10) -> tuple:
    """Run code after import validation. Returns (success, output_str)."""
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS_GRPO:
                    return False, f"blocked_import: {alias.name}"
        elif isinstance(node, _ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS_GRPO:
                return False, f"blocked_import: {node.module}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _extract_tool_calls(response: str) -> list[dict]:
    """Extract all tool calls from a response — handles both XML and native formats.

    Returns list of {"name": str, "kwargs": dict}.
    """
    results = []
    # Native format: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        try:
            obj = json.loads(m.group(1).strip())
            results.append({
                "name":   obj.get("name", ""),
                "kwargs": obj.get("arguments", {}),
            })
        except (json.JSONDecodeError, TypeError):
            pass
    # XML format (legacy): <tool>name(args)</tool> — kept for backwards compat
    for m in re.finditer(r"<tool>(\w+)\(", response):
        name = m.group(1)
        if not any(r["name"] == name for r in results):
            # Only add XML-format calls not already captured by native parser
            results.append({"name": name, "kwargs": {}})
    return results


def _extract_code_from_response(response: str) -> list[str]:
    blocks = []
    # Native format: <tool_call>{"name": "python_execute", "arguments": {"code": "..."}}</tool_call>
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        try:
            obj = json.loads(m.group(1).strip())
            if obj.get("name") == "python_execute":
                code = obj.get("arguments", {}).get("code", "")
                if code:
                    blocks.append(code)
        except (json.JSONDecodeError, TypeError):
            pass
    # XML format (legacy)
    if not blocks:
        p1 = r'<tool>\s*python_execute\s*\(\s*code\s*=\s*["\']+(.*?)["\']+\s*\)\s*</tool>'
        for m in re.finditer(p1, response, re.DOTALL):
            code = m.group(1).replace("\\n", "\n").replace('\\"', '"')
            blocks.append(code)
        p2 = r'<tool>\s*python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)\s*</tool>'
        for m in re.finditer(p2, response, re.DOTALL):
            blocks.append(m.group(1).strip())
    return blocks


def _last_number(text: str) -> str | None:
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def _answers_match(a: str, b: str, tol: float = 0.01) -> bool:
    try:
        af, bf = float(a), float(b)
        if abs(bf) < 1e-9:
            return abs(af) < 1e-6
        return abs(af - bf) / abs(bf) < tol
    except (ValueError, TypeError):
        return a.strip() == b.strip()


# v3 training data has no CAPABILITY_CHECK — format reward only requires <think> + <answer>
_V3_FORMAT_MODE: bool = True

# All tools the model is permitted to call (includes always-on tools like user_memory/scratchpad)
_ALL_KNOWN_TOOLS = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime",
    # always-on memory/scratchpad tools — must not be treated as hallucinated
    "user_memory_sections", "user_memory_read", "user_memory_update",
    "scratchpad_sections", "scratchpad_read", "scratchpad_update",
})

# Math category names used in both v3 training data and GRPO dataset
_MATH_CATEGORIES = frozenset({
    "arithmetic", "algebra", "geometry", "statistics",
    "unit_conversion", "word_problems", "trigonometry", "calculus", "advanced_geometry",
    # v3 prefixed variants
    "math_arithmetic", "math_algebra", "math_geometry", "math_statistics",
    "math_trigonometry", "math_word_problems", "math_calculus",
    # gsm8k / partB labels
    "math_word_problems", "gsm8k",
})


def _extract_think(response: str) -> str:
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _coerce_text(response) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="ignore")
    if isinstance(response, dict):
        for key in ("content", "text", "generated_text"):
            val = response.get(key)
            if isinstance(val, str):
                return val
    if isinstance(response, (list, tuple)):
        parts = []
        for item in response:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, bytes):
                parts.append(item.decode("utf-8", errors="ignore"))
            elif isinstance(item, dict):
                for key in ("content", "text", "generated_text"):
                    val = item.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                        break
        if parts:
            return "".join(parts)
    return str(response)


def _format_reward(response: str) -> float:
    """Graded structural quality: think block must be present AND substantive."""
    response = _coerce_text(response)
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))

    if not (has_think and has_ans):
        return 0.0   # missing core structure

    think_text = _extract_think(response)

    if len(think_text) == 0:
        return 0.15  # has tags but zero thinking — strongly discourage

    if len(think_text) < 40:
        return 0.4   # trivial think block (e.g. one word)

    if _V3_FORMAT_MODE and "CAPABILITY_CHECK" in response:
        return 0.6   # v2 artifact leaked into v3 response

    if len(think_text) < 100:
        return 0.75  # thinking present but shallow

    return 1.0       # substantive thinking + correct structure


def _accuracy_reward(response: str, expected_answer: str | None,
                     question_type: str) -> float:
    """For verifiable math: execute code and check answer.
    For behavioural: neutral 0.5 (no ground truth to verify against)."""
    response = _coerce_text(response)
    if not expected_answer or question_type not in _MATH_CATEGORIES:
        return 0.5  # neutral — behavioural examples have no single correct answer

    code_blocks = _extract_code_from_response(response)
    if not code_blocks:
        # No code at all — check if mental-math answer is at least correct
        answer_m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if answer_m:
            num = _last_number(answer_m.group(1))
            if num and _answers_match(num, expected_answer):
                return 0.3  # right answer, wrong method (mental math)
        return 0.0

    # Penalise empty code blocks explicitly — worse than no tool call
    # because the model learned a broken pattern (call but don't fill)
    for code in code_blocks:
        if len(code.strip()) < 5:
            return 0.0  # empty / trivial code is a hard failure

    all_output = ""
    for code in code_blocks:
        ok, out = _safe_execute(code)
        if not ok:
            return 0.0
        all_output += out + "\n"

    computed = _last_number(all_output)
    if computed and _answers_match(computed, expected_answer):
        return 1.0
    return 0.0


def _tool_integrity_reward(response: str, active_tools: set) -> float:
    """P3: no calls to completely unknown tools.
    Always-on tools (user_memory_*, scratchpad_*) are never hallucinated."""
    response = _coerce_text(response)
    called = {tc["name"] for tc in _extract_tool_calls(response)}
    hallucinated = called - _ALL_KNOWN_TOOLS
    profile_restricted = {"python_execute", "web_search", "read_url", "get_datetime"}
    unavailable = (called & profile_restricted) - active_tools
    return 0.0 if (hallucinated or unavailable) else 1.0


def _tool_quality_reward(response: str, question_type: str, active_tools: set) -> float:
    """Reward correct tool selection and non-empty parameters.

    Checks per question type:
      real_time_dependent  → must call get_datetime (or web_search if no datetime)
      user_context_*       → should call user_memory_read / user_memory_sections
      entity_facts_*       → must call web_search
      math categories      → python_execute must have non-empty, non-trivial code
    """
    response = _coerce_text(response)
    tool_calls = _extract_tool_calls(response)
    called = {tc["name"] for tc in tool_calls}

    # ── Hard failure: empty python_execute code ───────────────────────────────
    for tc in tool_calls:
        if tc["name"] != "python_execute":
            continue
        code_text = tc["kwargs"].get("code", "").strip()
        if len(code_text) < 5:
            return 0.0  # called python_execute but left code empty

    score = 1.0

    # ── Real-time questions: must use get_datetime or web_search ─────────────
    _REALTIME = {"real_time_dependent", "real_time_data"}
    if question_type in _REALTIME:
        time_tools = {"get_datetime", "web_search"} & active_tools
        if time_tools and not (called & time_tools):
            score *= 0.1   # strong signal: tool available but not used

    # ── User-context questions: should read user memory first ─────────────────
    _USER_CTX = {"user_context_behavioral", "verbose_context_behavioral"}
    if question_type in _USER_CTX:
        memory_tools = {"user_memory_read", "user_memory_sections"}
        if not (called & memory_tools):
            score *= 0.4   # model skipped personalisation step

    # ── Entity fact questions: must search ────────────────────────────────────
    _ENTITY = {"entity_facts_web_search"}
    if question_type in _ENTITY:
        if "web_search" in active_tools and "web_search" not in called:
            score *= 0.05  # near-zero: hallucinating entity facts is dangerous

    # ── Math questions: must use python_execute (already covered by accuracy,
    #    but doubling signal accelerates learning of the tool habit) ──────────
    if question_type in _MATH_CATEGORIES:
        if "python_execute" in active_tools and "python_execute" not in called:
            score *= 0.2

    return score


_GREEDY_CATEGORIES = frozenset({
    "first_principles_questioning", "user_context_behavioral",
    "ambiguous_underspecified", "multi_step_clarification",
    "appraisal_empathy", "subjective_tradeoffs",
})

# Two-regex approach so the uppercase-label check stays case-sensitive.
# Without this split, re.IGNORECASE causes \b(WHO|WHAT|...)\b to match plain
# "What do you think?" — a generic filler that is NOT a targeted user question.

# Case-sensitive: only matches ALL-CAPS dimension labels the model uses to
# explicitly name a 5W+H axis (e.g. "To understand your WHY better: ...").
_WPLUS_H_UPPERCASE = re.compile(r"\b(WHO|WHAT|WHEN|WHERE|WHY|HOW)\b")

# Case-insensitive: context-specific patterns that require the dimension word
# to appear in a phrase that targets the USER's specific situation.
_WPLUS_H_CONTEXT = re.compile(
    r"your\s+\b(why|who|what|when|where|how|situation|context|background|goal"
    r"|motivation|reason|timeline|use\s+case|role|setup)\b"
    r"|\bwhy\s+(are\s+you|do\s+you|did\s+you|would\s+you|is\s+this)\b"
    r"|\bwho\s+(are\s+you|is\s+this|is\s+the)\b"
    r"|\bwhat\s+(is\s+your|are\s+you|does\s+your|specifically|exactly)\b"
    r"|\bwhen\s+(do\s+you|are\s+you|is\s+this|is\s+your)\b"
    r"|\bwhere\s+(are\s+you|do\s+you|is\s+this|does\s+this)\b"
    r"|\bhow\s+(do\s+you|are\s+you|does\s+your|would\s+you|much)\b"
    r"|tell\s+me\s+(more\s+)?(about\s+)?(your|who|why|what|how|when|where)\b"
    r"|to\s+(give|help)\s+(you\s+)?(more|better|a\s+sharper|sharper)"
    r"|\b5w\+?h\b",
    re.IGNORECASE,
)

# Sentence splitter — split on . ! ? followed by whitespace or end of string
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _last_sentence(text: str) -> str:
    """Return the last non-empty sentence of text."""
    parts = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return parts[-1] if parts else ""


def _greedy_followup_reward(response: str, category: str) -> float:
    """P21 — greedy personalisation: <answer> must end with a 5W+H follow-up question.

    Grading:
      1.0 — last sentence of <answer> ends with '?' AND names a 5W+H user dimension
      0.7 — last sentence ends with '?' but no specific 5W+H dimension named
      0.3 — last sentence is not a question but there is a '?' somewhere in <answer>
      0.0 — no question at all in <answer>
      0.5 — neutral for categories where a closing question is not required
    """
    response = _coerce_text(response)

    if category not in _GREEDY_CATEGORIES:
        return 0.5

    answer_m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if not answer_m:
        return 0.0

    answer_text = answer_m.group(1).strip()
    last = _last_sentence(answer_text)

    # Primary check: last sentence ends with ?
    has_trailing_question = last.endswith("?")

    if not has_trailing_question:
        return 0.3 if "?" in answer_text else 0.0

    # Trailing question present — check if it targets a 5W+H user dimension
    has_5wh_signal = bool(_WPLUS_H_UPPERCASE.search(last) or _WPLUS_H_CONTEXT.search(last))
    return 1.0 if has_5wh_signal else 0.7


def _constitution_reward(response: str, question: str,
                          category: str, tool_profile: dict) -> float:
    """Structural constitution reward (GRPO).

    Scores presence of a substantive <think> block and an <answer> tag. The previous
    implementation imported rule_check_response from sft_gold_response_generator, which
    was removed when sft_v3_generator.py replaced that script — the import always fell
    through to this format check, so it is now inlined directly (no dead import)."""
    response = _coerce_text(response)
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))
    if _V3_FORMAT_MODE:
        return 1.0 if (has_think and has_ans) else 0.5 if (has_think or has_ans) else 0.0
    has_cap = "CAPABILITY_CHECK" in response
    n_ok = sum([has_think, has_cap, has_ans])
    return n_ok / 3.0


def _profile_to_set(label: str) -> set:
    profiles = {
        "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime"},
        "compute_only":       {"python_execute"},
        "compute_and_search": {"python_execute", "web_search", "read_url"},
        "no_tools":           set(),
    }
    return profiles.get(label, {"python_execute"})


def make_reward_fns(reward_type: str = "d") -> list:
    """Return a list of per-component reward functions for GRPOTrainer.

    TRL sums the outputs of all functions to produce the training signal and
    automatically logs each under rewards/{fn_name}_mean — giving per-component
    breakdown in grpo_loss_history.json at zero extra cost.

    Each function returns its *weighted* component score so the sum equals the
    original composite reward. Training dynamics are identical to the old single
    function; only the logging granularity changes.

    reward_type 'c': format + accuracy only (Ablation C — two functions)
    reward_type 'd': full composite         (Ablation D — five functions)
    """
    def format_reward(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        w = REWARD_WEIGHTS["format"]
        return [float(w * _format_reward(c)) for c in completions]

    def accuracy_reward(
        prompts: list[str], completions: list[str],
        question_type: list[str] | None = None,
        expected_answer: list[str] | None = None,
        category: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        qt_list = question_type or category or ["unknown"] * n
        ea_list = expected_answer or [None] * n
        w = REWARD_WEIGHTS["accuracy"]
        return [
            float(w * _accuracy_reward(c, ea, qt))
            for c, ea, qt in zip(completions, ea_list, qt_list)
        ]

    if reward_type == "c":
        # Ablation C: renormalise so the two weights still sum to 1
        total_w = REWARD_WEIGHTS["format"] + REWARD_WEIGHTS["accuracy"]

        def format_reward_c(
            prompts: list[str], completions: list[str], **kwargs
        ) -> list[float]:
            w = REWARD_WEIGHTS["format"] / total_w
            return [float(w * _format_reward(c)) for c in completions]

        def accuracy_reward_c(
            prompts: list[str], completions: list[str],
            question_type: list[str] | None = None,
            expected_answer: list[str] | None = None,
            category: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            n = len(completions)
            qt_list = question_type or category or ["unknown"] * n
            ea_list = expected_answer or [None] * n
            w = REWARD_WEIGHTS["accuracy"] / total_w
            return [
                float(w * _accuracy_reward(c, ea, qt))
                for c, ea, qt in zip(completions, ea_list, qt_list)
            ]

        return [format_reward_c, accuracy_reward_c]

    # Ablation D — full composite (five functions)
    def tool_integrity_reward(
        prompts: list[str], completions: list[str],
        tool_profile_label: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        tp_list = tool_profile_label or ["compute_only"] * n
        w = REWARD_WEIGHTS["tool_integrity"]
        return [
            float(w * _tool_integrity_reward(c, _profile_to_set(tpl)))
            for c, tpl in zip(completions, tp_list)
        ]

    def tool_quality_reward(
        prompts: list[str], completions: list[str],
        question_type: list[str] | None = None,
        tool_profile_label: list[str] | None = None,
        category: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        qt_list = question_type or category or ["unknown"] * n
        tp_list = tool_profile_label or ["compute_only"] * n
        w = REWARD_WEIGHTS["tool_quality"]
        return [
            float(w * _tool_quality_reward(c, qt, _profile_to_set(tpl)))
            for c, qt, tpl in zip(completions, qt_list, tp_list)
        ]

    def constitution_reward(
        prompts: list[str], completions: list[str],
        question: list[str] | None = None,
        question_type: list[str] | None = None,
        expected_answer: list[str] | None = None,
        tool_profile_label: list[str] | None = None,
        category: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        q_list  = question or [""] * n
        qt_list = question_type or category or ["unknown"] * n
        tp_list = tool_profile_label or ["compute_only"] * n
        w = REWARD_WEIGHTS["constitution"]
        results = []
        for comp, q, qt, tpl in zip(completions, q_list, qt_list, tp_list):
            active_tools = _profile_to_set(tpl)
            tool_profile_dict = {
                "context": " | ".join(
                    f"{t} {'✓' if t in active_tools else '✗'}"
                    for t in ["python_execute", "web_search", "read_url", "get_datetime"]
                ),
                "label": tpl,
            }
            results.append(float(w * _constitution_reward(comp, q, qt, tool_profile_dict)))
        return results

    def greedy_followup_reward(
        prompts: list[str], completions: list[str],
        category: list[str] | None = None,
        question_type: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        cat_list = category or question_type or ["unknown"] * n
        w = REWARD_WEIGHTS["greedy_followup"]
        return [
            float(w * _greedy_followup_reward(c, cat))
            for c, cat in zip(completions, cat_list)
        ]

    return [format_reward, accuracy_reward, tool_integrity_reward,
            tool_quality_reward, constitution_reward, greedy_followup_reward]


def make_reward_fn(reward_type: str = "d"):
    """Single composite reward function — used only for held-out evaluation in publish().

    Training uses make_reward_fns() (plural) so TRL logs each component separately.
    """
    fns = make_reward_fns(reward_type)

    def reward_fn(prompts, completions, **kwargs):
        totals = [0.0] * len(completions)
        for fn in fns:
            for i, v in enumerate(fn(prompts, completions, **kwargs)):
                totals[i] += v
        return totals

    return reward_fn


# ---------------------------------------------------------------------------
# GRPO dataset builder
# ---------------------------------------------------------------------------

def build_grpo_dataset(sft_jsonl_path: str) -> "Dataset":
    """Convert the SFT JSONL to GRPO prompt format.

    TRL's GRPOTrainer expects each row to have a 'prompt' key (list of messages
    ending with the user turn) plus any metadata fields the reward function needs.
    """
    rows = []
    with open(sft_jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = ex.get("messages", [])
            meta = ex.get("metadata", {})

            # Extract prompt = system + user messages only (no assistant)
            prompt = [m for m in messages if m["role"] in ("system", "user")]
            if not prompt:
                continue

            # The question is the last user message
            user_msgs = [m for m in messages if m["role"] == "user"]
            question  = user_msgs[-1]["content"] if user_msgs else ""

            # question_type: part-B math rows store it under "question_type"; part-A rows use "category"
            q_type = meta.get("question_type") or meta.get("category", "unknown")
            # expected_answer: populated by sft_rejection_sampler for math rows; empty for behavioural rows
            expected = meta.get("expected_answer", "")
            rows.append({
                "prompt":            prompt,
                "question":          question,
                "category":          meta.get("category") or meta.get("question_type", "unknown"),
                "question_type":     q_type,
                "tool_profile_label": meta.get("tool_profile", "compute_only"),
                "expected_answer":   expected,
                "constitution_score": meta.get("constitution_score", 0.5),
            })

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# SFT helpers
# ---------------------------------------------------------------------------

def messages_to_text(example, tokenizer):
    # Native examples store their OpenAI-schema tool list in metadata so
    # apply_chat_template renders tool definitions identically to inference.
    native_tools = example.get("metadata", {}).get("native_tools") or None
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            tools=native_tools,
            # enable_thinking is a no-op for training renders: with add_generation_prompt=False
            # the Qwen3-0.6B template emits identical text for True and False (verified
            # empirically 2026-05-29), because the assistant turn already contains explicit
            # <think>...</think> blocks. It only matters at inference (add_generation_prompt=True),
            # where True lets the model emit its own <think> (matching this data) and False would
            # inject an empty <think></think> forcing non-thinking mode. Inference uses True.
            enable_thinking=False,
        )
    }


# ---------------------------------------------------------------------------
# In-training collapse monitor
# ---------------------------------------------------------------------------

class CollapseMonitorCallback(TrainerCallback):
    """Generate on a few held-out prompts at each eval, PRINT the sample generations, and
    report think-empty rate + mean tool-call count.

    This surfaces the think-collapse failure mode (reasoning displaced by tool-calls)
    *during* training instead of only after a full multi-hour run + benchmark, and lets you
    eyeball what the model actually produces at each eval. Generation matches inference
    (greedy, enable_thinking=True, native tools rendered). Fully guarded — any generation
    error is swallowed so it can never abort training. Samples are also appended to
    reports/training/<run>/eval_samples.jsonl for later inspection.

    The probe set is FIXED (a seed-stable sample of the held-out eval set) and decoding is greedy,
    so the same questions are shown at every eval ON PURPOSE — that is what lets you watch one
    example improve across steps. To vary which examples are shown, change PIPELINE_EVAL_N / the
    seed or set PIPELINE_EVAL_SHUFFLE=1.

    Env knobs:
      PIPELINE_EVAL_N            — how many held-out prompts to generate on per eval (default 5).
                                   This also caps how many can be printed.
      PIPELINE_EVAL_SHOW_SAMPLES — how many of those to print per eval (default 2; 0 = none)
      PIPELINE_EVAL_SAMPLE_CHARS — truncation length per printed generation (default 700)
      PIPELINE_EVAL_SHUFFLE      — 1 to pick a random (seed-42) sample instead of the first N
    """

    def __init__(self, tokenizer, eval_raw, n: int = 5, max_new_tokens: int = 512,
                 samples_path: "Optional[Path]" = None):
        self._tok = tokenizer
        self._max_new = max_new_tokens
        n = int(os.environ.get("PIPELINE_EVAL_N", str(n)))
        self._show = int(os.environ.get("PIPELINE_EVAL_SHOW_SAMPLES", "2"))
        self._chars = int(os.environ.get("PIPELINE_EVAL_SAMPLE_CHARS", "700"))
        self._samples_path = samples_path
        pool = list(eval_raw or [])
        if os.environ.get("PIPELINE_EVAL_SHUFFLE") == "1":
            import random as _r
            _r.Random(42).shuffle(pool)   # seed-stable so the chosen set is still fixed across evals
        self._prompts: list[tuple[list, object, str]] = []
        for ex in pool[:n]:
            msgs = [m for m in ex.get("messages", []) if m.get("role") in ("system", "user")]
            if msgs:
                question = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
                self._prompts.append((msgs, ex.get("metadata", {}).get("native_tools"), question))

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        model = kwargs.get("model")
        if model is None or not self._prompts:
            return
        try:
            import torch as _torch
            empties = total_calls = n = 0
            printed = 0
            records = []
            for msgs, native_tools, question in self._prompts:
                text = self._tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    tools=native_tools, enable_thinking=True,
                )
                inputs = self._tok(text, return_tensors="pt").to(model.device)
                with _torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=self._max_new, do_sample=False)
                gen = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                tm = re.search(r"<think>(.*?)</think>", gen, re.DOTALL | re.IGNORECASE)
                think = tm.group(1).strip() if tm else ""
                if len(think) < 10:
                    empties += 1
                total_calls += len(re.findall(r"<tool_call>|<tool>", gen))
                n += 1
                records.append({"step": state.global_step, "question": question, "generation": gen})
                # Print the first few generations so you can SEE the model at each eval.
                if printed < self._show:
                    printed += 1
                    snippet = gen if len(gen) <= self._chars else gen[:self._chars] + " …[truncated]"
                    print(f"\n  ┌─ [eval-sample {printed}] step={state.global_step}", flush=True)
                    print(f"  │ Q: {question[:200]}", flush=True)
                    print(f"  │ A: {snippet}".replace("\n", "\n  │    "), flush=True)
                    print("  └─", flush=True)
            if n:
                print(
                    f"  [collapse-monitor] step={state.global_step} "
                    f"think_empty={empties}/{n} ({100 * empties / n:.0f}%) "
                    f"mean_tool_calls={total_calls / n:.2f}",
                    flush=True,
                )
            # Persist all sampled generations for later inspection.
            if self._samples_path is not None and records:
                try:
                    self._samples_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._samples_path, "a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        except Exception as e:  # never break training over a monitoring read
            print(f"  [collapse-monitor] skipped ({type(e).__name__}: {e})", flush=True)


class LogStreamCallback(TrainerCallback):
    """Append every trainer log record (loss, lr, eval_loss, …) to a JSONL the moment it is
    produced, so metrics persist LIVE — a crash/disconnect on a remote GPU loses nothing instead
    of losing the whole log (which was previously dumped only after training finished). Fully
    guarded; a write failure never breaks training."""

    def __init__(self, path: Path):
        self._path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"step": state.global_step, "epoch": state.epoch, **logs}
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Trains Qwen3-0.6B via SFT then GRPO (DAPO improvements)."""

    def __init__(self, data_dir: str, output_dir: str,
                 output_name: str = "checkpoint_sft",
                 hf_username: str = "AjinkyaTaranekar",
                 no_publish: bool = False,
                 skip_gguf: bool = False):
        self.data_dir    = Path(data_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name  = output_name
        self._hf_username = hf_username
        self._no_publish  = no_publish
        self._skip_gguf   = skip_gguf
        self.model        = None
        self.tokenizer    = None
        # Populated during training; used by publish()
        self._eval_raw          = []
        self._grpo_eval_dataset = None

    def load_base_model(self):
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=MODEL_CONFIG["base_model"],
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            load_in_4bit=MODEL_CONFIG["load_in_4bit"],
            dtype=None,
        )
        return self

    def load_checkpoint(self, checkpoint_path: str):
        """Load an existing LoRA checkpoint (e.g. SFT checkpoint for GRPO phase)."""
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            load_in_4bit=MODEL_CONFIG["load_in_4bit"],
            dtype=None,
        )
        return self

    def apply_lora(self):
        self.model = FastModel.get_peft_model(
            self.model,
            r=MODEL_CONFIG["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=MODEL_CONFIG["lora_alpha"],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return self

    # ── Phase 1: SFT ────────────────────────────────────────────────────────

    def train_sft(self, dataset_path: str, output_name: str = "checkpoint_sft",
                  resume_from_checkpoint=False, eval_records: list | None = None):
        """Train one SFT run.

        eval_records — an externally-held-out eval set (P1.2). When provided, the WHOLE of
        dataset_path is used for training and these records are the eval set, so a multi-stage
        curriculum measures every stage against the SAME held-out set (comparable eval_loss, no
        per-stage re-split, no train/eval leakage). When None, the legacy internal 90/10 split is
        used (single-run path). Publishing is NOT done here (P1.1) — the caller publishes once.
        """
        print(f"  SFT dataset   : {dataset_path}")
        print(f"  Output dir    : {self.output_dir / output_name}")
        print(f"  Resume        : {resume_from_checkpoint}")
        print(
            "  Config        : epochs="
            f"{SFT_CONFIG['num_train_epochs']} "
            f"batch={SFT_CONFIG['per_device_train_batch_size']} "
            f"grad_accum={SFT_CONFIG['gradient_accumulation_steps']} "
            f"lr={SFT_CONFIG['learning_rate']} "
            f"max_seq={MODEL_CONFIG['max_seq_length']}"
        )
        # Load directly to bypass DatasetBuilder fingerprinting (breaks on Python 3.14
        # due to dill/pickle._batch_setitems signature change in 3.14).
        with open(dataset_path, encoding="utf-8") as _f:
            _records = [json.loads(l) for l in _f if l.strip()]
        if eval_records is not None:
            # P1.2: externally-held-out eval — train on ALL of dataset_path, eval on the shared set.
            train_records = _records
            eval_raw = list(eval_records)
            print(f"  Eval set      : external held-out ({len(eval_raw)} rows, shared across stages)")
        else:
            # Legacy single-run path: internal 90/10 split.
            raw_split = Dataset.from_list(_records).train_test_split(test_size=0.10, seed=42)
            train_records = list(raw_split["train"])
            eval_raw = [dict(ex) for ex in raw_split["test"]]
        train_dataset = Dataset.from_list(train_records).map(
            messages_to_text, fn_kwargs={"tokenizer": self.tokenizer},
        )
        eval_dataset = Dataset.from_list(eval_raw).map(
            messages_to_text, fn_kwargs={"tokenizer": self.tokenizer},
        )
        # Keep raw eval records (with 'messages' key) for ROUGE computation in publish()
        self._eval_raw = [dict(ex) for ex in eval_raw]
        split = {"train": train_dataset, "test": eval_dataset}
        print(f"  Train: {len(split['train'])}  |  Eval: {len(split['test'])}")

        training_args = SFTConfig(
            output_dir=str(self.output_dir / output_name),
            per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
            num_train_epochs=SFT_CONFIG["num_train_epochs"],
            learning_rate=SFT_CONFIG["learning_rate"],
            warmup_steps=SFT_CONFIG["warmup_steps"],
            logging_steps=SFT_CONFIG["logging_steps"],
            save_steps=SFT_CONFIG["save_steps"],
            eval_steps=SFT_CONFIG["eval_steps"],
            save_total_limit=SFT_CONFIG["save_total_limit"],
            eval_strategy="steps",
            bf16=SFT_CONFIG["bf16"],
            optim=SFT_CONFIG["optim"],
            weight_decay=SFT_CONFIG["weight_decay"],
            lr_scheduler_type=SFT_CONFIG["lr_scheduler_type"],
            packing=SFT_CONFIG["packing"],
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
            # Save the checkpoint with lowest eval_loss, not just the final one.
            # The previous SFT run showed overfitting: eval plateau at epoch 2.5
            # while training loss kept falling — the best model was mid-run, not at end.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            args=training_args,
        )

        # Mask loss on system + user tokens — gradients flow only from assistant responses.
        # Fixes the large train/eval gap caused by computing loss over the ~400-token
        # system prompt (CAPABILITY_CHECK template + 23 principles) on every example.
        # Qwen3 chat format: assistant turns start with <|im_start|>assistant\n
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        # Surface the think-collapse failure mode during training (not just after).
        _live_dir = self.output_dir.parent / "reports" / "training" / output_name
        _samples_path = _live_dir / "eval_samples.jsonl"
        trainer.add_callback(CollapseMonitorCallback(
            self.tokenizer, self._eval_raw, samples_path=_samples_path))
        # Stream loss/metrics to disk as they happen (live persistence for long remote runs).
        trainer.add_callback(LogStreamCallback(_live_dir / "loss_live.jsonl"))
        print(f"  Live logs     : {_live_dir / 'loss_live.jsonl'}  (+ eval_samples.jsonl)")

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save loss history in checkpoint dir (for recovery) AND under reports/training/<name>/
        import datetime as _dt
        _ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _reports_dir = self.output_dir.parent / "reports" / "training" / output_name
        _reports_dir.mkdir(parents=True, exist_ok=True)
        loss_path     = self.output_dir / output_name / "loss_history.json"
        reports_path  = _reports_dir / f"loss_history_{_ts}.json"
        loss_payload  = {
            "model":     output_name,
            "phase":     "sft",
            "timestamp": _ts,
            "config":    {**SFT_CONFIG, **MODEL_CONFIG},
            "log":       trainer.state.log_history,
        }
        for p in (loss_path, reports_path):
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(loss_payload, f, indent=2)
        print(f"  Loss history  : {loss_path}")
        print(f"  Loss report   : {reports_path}")

        trainer.save_model(str(self.output_dir / output_name))
        print(f"  SFT checkpoint saved → {self.output_dir / output_name}")
        # P1.1: publishing is the caller's responsibility (publish ONCE after the final stage),
        # so a multi-stage curriculum does not merge/export/upload after every stage.
        return self

    # ── Phase 2: GRPO ───────────────────────────────────────────────────────

    def train_grpo(
        self,
        sft_checkpoint: str,
        dataset_path: str,
        output_name: str = "checkpoint_grpo_d",
        reward_type: str = "d",
        resume_from_checkpoint=False,
    ):
        """GRPO RL training starting from the SFT checkpoint.

        reward_type 'c': format + accuracy only  (Ablation C — does RL improve correctness?)
        reward_type 'd': full composite           (Ablation D — full thesis contribution)

        DAPO improvements applied:
          - Token-level loss normalisation (set via use_vllm or loss scaling)
          - Clip-Higher via custom clip_range_ratio_high
          - Dynamic sampling: groups with zero reward variance are skipped
          - Reference policy: SFT checkpoint (not base model)
        """
        print(f"\n  Loading SFT checkpoint as starting point: {sft_checkpoint}")
        self.load_checkpoint(sft_checkpoint)
        # Re-enable training (FastModel.from_pretrained sets for inference)
        FastModel.for_training(self.model)

        print(f"  Output dir    : {self.output_dir / output_name}")
        print(f"  Resume        : {resume_from_checkpoint}")
        print(f"  GRPO dataset  : {dataset_path}")
        print(
                "  Config        : epochs="
                f"{GRPO_CONFIG['num_train_epochs']} "
                f"G={GRPO_CONFIG['num_generations']} "
                f"lr={GRPO_CONFIG['learning_rate']} "
                f"kl={GRPO_CONFIG['kl_coef']} "
                f"temp={GRPO_CONFIG['temperature']} "
                f"max_new={GRPO_CONFIG['max_new_tokens']}"
        )
        print(f"  Dynamic sampling: {GRPO_CONFIG['dynamic_sampling']}")

        print(f"  Building GRPO dataset from {dataset_path}...")
        full_grpo = build_grpo_dataset(dataset_path)
        grpo_split = full_grpo.train_test_split(test_size=0.10, seed=42)
        dataset = grpo_split["train"]
        self._grpo_eval_dataset = grpo_split["test"]
        print(f"  GRPO Train: {len(dataset)}  |  Held-out: {len(self._grpo_eval_dataset)}")

        reward_fns = make_reward_fns(reward_type)

        # GRPOConfig — DAPO settings where supported by TRL
        # If your TRL version does not have clip_range_ratio_high, it falls back
        # to symmetric clipping (vanilla GRPO).  Pin trl>=0.13.0 for best support.
        # kl_coef was renamed to beta in TRL >=0.14; try beta first, fall back to kl_coef
        import inspect as _inspect
        _grpo_params = set(_inspect.signature(GRPOConfig.__init__).parameters)
        _kl_key = "beta" if "beta" in _grpo_params else "kl_coef"

        grpo_kwargs = dict(
            output_dir=str(self.output_dir / output_name),
            num_generations=GRPO_CONFIG["num_generations"],
            learning_rate=GRPO_CONFIG["learning_rate"],
            **{_kl_key: GRPO_CONFIG["kl_coef"]},
            epsilon=GRPO_CONFIG["clip_range_ratio"],          # clip_range_ratio → epsilon
            temperature=GRPO_CONFIG["temperature"],
            max_prompt_length=GRPO_CONFIG["max_prompt_length"],
            max_completion_length=GRPO_CONFIG["max_new_tokens"],  # max_new_tokens → max_completion_length
            num_train_epochs=GRPO_CONFIG["num_train_epochs"],
            per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
            logging_steps=GRPO_CONFIG["logging_steps"],
            save_steps=GRPO_CONFIG["save_steps"],
            bf16=GRPO_CONFIG["bf16"],
            optim=GRPO_CONFIG["optim"],
            weight_decay=GRPO_CONFIG["weight_decay"],
            warmup_ratio=GRPO_CONFIG["warmup_ratio"],
            lr_scheduler_type=GRPO_CONFIG["lr_scheduler_type"],
            max_grad_norm=GRPO_CONFIG["max_grad_norm"],
            report_to="none",
            # Dr.GRPO: token-level loss normalisation (divide by completion length)
            # Prevents longer completions from dominating the gradient signal.
            loss_type="dr_grpo",
            # Mask completions that hit max_completion_length — avoids penalising
            # correct think+tool+answer responses truncated mid-generation.
            mask_truncated_completions=True,
        )
        print(f"  KL penalty key : {_kl_key}={GRPO_CONFIG['kl_coef']}")
        print(f"  epsilon        : {GRPO_CONFIG['clip_range_ratio']}")
        print(f"  max_prompt     : {GRPO_CONFIG['max_prompt_length']}")
        print(f"  max_completion : {GRPO_CONFIG['max_new_tokens']}")
        print(f"  scheduler      : {GRPO_CONFIG['lr_scheduler_type']}  warmup={GRPO_CONFIG['warmup_ratio']}")
        print(f"  max_grad_norm  : {GRPO_CONFIG['max_grad_norm']}")

        # Attempt DAPO Clip-Higher (epsilon_high) if TRL supports it
        try:
            config = GRPOConfig(
                **grpo_kwargs,
                epsilon_high=GRPO_CONFIG["clip_range_ratio_high"],  # clip_range_ratio_high → epsilon_high
            )
            print("  DAPO Clip-Higher active (epsilon_high="
                  f"{GRPO_CONFIG['clip_range_ratio_high']})")
        except TypeError:
            config = GRPOConfig(**grpo_kwargs)
            print("  NOTE: epsilon_high not supported by this TRL version — symmetric clipping.")

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_fns,
            args=config,
            train_dataset=dataset,
        )

        # DAPO dynamic sampling: hook to skip zero-variance groups
        # This runs after rollout, before the policy gradient update.
        if GRPO_CONFIG["dynamic_sampling"]:
            _patch_dynamic_sampling(trainer)

        print(f"\n  Starting GRPO training (reward_type={reward_type})...")
        print(f"  Reward weights: {REWARD_WEIGHTS}")
        print(f"  G={GRPO_CONFIG['num_generations']}  β={GRPO_CONFIG['kl_coef']}  "
              f"lr={GRPO_CONFIG['learning_rate']}")

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save GRPO loss history in checkpoint dir AND under reports/training/<name>/
        import datetime as _dt
        _ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _reports_dir = self.output_dir.parent / "reports" / "training" / output_name
        _reports_dir.mkdir(parents=True, exist_ok=True)
        loss_path     = self.output_dir / output_name / "grpo_loss_history.json"
        reports_path  = _reports_dir / f"grpo_loss_history_{_ts}.json"
        loss_payload  = {
            "model":        output_name,
            "phase":        "grpo",
            "timestamp":    _ts,
            "reward_type":  reward_type,
            "config":       {**GRPO_CONFIG},
            "reward_weights": REWARD_WEIGHTS,
            "log":          trainer.state.log_history,
        }
        for p in (loss_path, reports_path):
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(loss_payload, f, indent=2)
        print(f"  GRPO loss history: {loss_path}")
        print(f"  GRPO loss report : {reports_path}")

        trainer.save_model(str(self.output_dir / output_name))
        print(f"  GRPO checkpoint saved → {self.output_dir / output_name}")
        # P1.1: caller publishes once (consistent with train_sft).
        return self

    # ── Convenience: run full SFT pipeline ──────────────────────────────────

    def train(self):
        self.load_base_model()
        self.apply_lora()
        dataset_path = self.data_dir / "train_sft_v3.jsonl"
        self.train_sft(str(dataset_path), self.output_name)

    def _local_generate(self, prompt_msgs: list, max_new_tokens: int = 1024) -> str:
        """Greedy-decode one prompt using the in-memory model (inference mode assumed).

        Used only during publish() — model must already be switched to inference mode
        via FastModel.for_inference() before calling this.
        """
        import torch
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)

    def _compute_rouge_report(self, output_name: str, baseline_path: str = "reports/constitution_baseline.json", max_eval_examples: int = 50) -> dict:
        """Generate hypotheses, compute ROUGE-1/2/L, save reports/rouge_{output_name}.json.

        Two reference sources:
          1. eval split gold responses (stored in self._eval_raw during train_sft)
          2. constitution probe baseline (reports/constitution_baseline.json)
        """
        reports_dir = self.output_dir.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        out_path = reports_dir / f"rouge_{output_name}.json"

        print(f"  Computing ROUGE for {output_name}...")

        # ── Eval split ROUGE ────────────────────────────────────────────────
        eval_rouge = None
        eval_raw = getattr(self, "_eval_raw", [])
        if eval_raw:
            sample = eval_raw[:max_eval_examples]
            print(f"  Eval split examples: {len(sample)}/{len(eval_raw)}")
            hypotheses, references = [], []
            for i, ex in enumerate(sample):
                print(f"    Generating eval example {i + 1}/{len(sample)}...", flush=True)
                msgs = ex.get("messages", [])
                gold = next(
                    (m["content"] for m in reversed(msgs) if m["role"] == "assistant"),
                    None,
                )
                if gold is None:
                    continue
                prompt_msgs = [m for m in msgs if m["role"] in ("system", "user")]
                if not prompt_msgs:
                    continue
                try:
                    hyp = self._local_generate(prompt_msgs)
                except Exception as e:
                    print(f"    [WARN] Eval example generation failed: {e}")
                    continue
                hypotheses.append(hyp)
                references.append(gold)
            if hypotheses:
                eval_rouge = compute_rouge(hypotheses, references)
                print(f"    Eval split ROUGE-1 F1: {eval_rouge['rouge1']['fmeasure']:.4f}")

        # ── Probe baseline ROUGE ────────────────────────────────────────────
        probe_rouge = None
        bp = Path(baseline_path)
        if bp.exists():
            with open(bp, encoding="utf-8") as f:
                baseline = json.load(f)
            probe_results = baseline.get("probe_results", [])
            print(f"  Probe baseline examples: {len(probe_results)}")
            hypotheses, references = [], []
            for pr in probe_results:
                q = pr.get("question", "")
                if isinstance(q, list):
                    q = q[-1]   # last turn of multi-turn probe
                ref = pr.get("response", "")
                if not q or not ref:
                    continue
                try:
                    hyp = self._local_generate([{"role": "user", "content": q}])
                except Exception as e:
                    print(f"    [WARN] Probe generation failed: {e}")
                    continue
                hypotheses.append(hyp)
                references.append(ref)
            if hypotheses:
                probe_rouge = compute_rouge(hypotheses, references)
                print(f"    Probe baseline ROUGE-1 F1: {probe_rouge['rouge1']['fmeasure']:.4f}")
        else:
            print(f"    [ROUGE] No probe baseline at {baseline_path} — skipping probe ROUGE.")

        # ── GRPO held-out reward ─────────────────────────────────────────────
        grpo_reward = None
        grpo_eval = getattr(self, "_grpo_eval_dataset", None)
        if grpo_eval is not None and len(grpo_eval) > 0:
            reward_fn = make_reward_fn("d")   # always use full reward for evaluation
            sample = grpo_eval.select(range(min(50, len(grpo_eval))))
            print(f"  GRPO held-out examples: {len(sample)}/{len(grpo_eval)}")
            all_rewards = []
            for row in sample:
                prompt = row["prompt"]
                try:
                    hyp = self._local_generate(prompt)
                except Exception as e:
                    print(f"    [WARN] GRPO generation failed: {e}")
                    continue
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,
                )
                rewards = reward_fn(
                    prompts=[prompt_text],
                    completions=[hyp],
                    question=[row.get("question", "")],
                    question_type=[row.get("question_type", "unknown")],
                    expected_answer=[row.get("expected_answer")],
                    tool_profile_label=[row.get("tool_profile_label", "compute_only")],
                )
                all_rewards.extend(rewards)
            if all_rewards:
                grpo_reward = round(sum(all_rewards) / len(all_rewards), 4)
                print(f"    GRPO held-out reward: {grpo_reward:.4f}")

        result = {
            "checkpoint":           output_name,
            "eval_split_rouge":     eval_rouge,
            "probe_baseline_rouge": probe_rouge,
            "grpo_held_out_reward": grpo_reward,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"    ROUGE report → {out_path}")
        return result

    def publish(
        self,
        output_name: str,
        hf_username: str,
        baseline_path: str = "reports/constitution_baseline.json",
    ) -> None:
        """Merge LoRA → safetensors, export GGUF, push both to HuggingFace, compute ROUGE.

        Requires HF_TOKEN environment variable for HuggingFace upload.
        Skips upload (but still saves locally and computes ROUGE) if HF_TOKEN is unset.

        Repo naming:
          checkpoint_sft       → {hf_username}/trustworthy-ai-sft
          checkpoint_grpo_c    → {hf_username}/trustworthy-ai-grpo-c
          checkpoint_grpo_d    → {hf_username}/trustworthy-ai-grpo-d
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "publish() requires model and tokenizer to be loaded. "
                "Call train_sft(), train_grpo(), or load_base_model() first."
            )

        # When called via --mode publish, train_sft() was never run so _eval_raw is empty.
        # Load the eval split from disk so ROUGE has real references.
        if not self._eval_raw:
            dataset_path = self.data_dir / "train_sft_v3.jsonl"
            if dataset_path.exists():
                try:
                    with open(dataset_path, encoding="utf-8") as _f:
                        _records = [json.loads(l) for l in _f if l.strip()]
                    raw_split = Dataset.from_list(_records).train_test_split(test_size=0.10, seed=42)
                    self._eval_raw = [dict(ex) for ex in raw_split["test"]]
                    print(f"  [publish] Loaded {len(self._eval_raw)} eval examples from {dataset_path.name}")
                except Exception as e:
                    print(f"  [publish] Could not load eval split: {e} — eval ROUGE skipped")
            else:
                print(f"  [publish] {dataset_path.name} not found — eval ROUGE skipped")

        import os
        from unsloth import FastModel

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            merged_dir_preview = str(self.output_dir / f"{output_name}_merged")
            gguf_dir_preview   = str(self.output_dir / f"{output_name}_gguf")
            print(f"  [publish] HF_TOKEN not set — models will be saved locally "
                  f"({merged_dir_preview}, {gguf_dir_preview}) but HuggingFace upload will be skipped.")

        if not output_name.startswith("checkpoint_"):
            raise ValueError(
                f"output_name must start with 'checkpoint_', got '{output_name}'. "
                "Expected values: checkpoint_sft, checkpoint_grpo_c, checkpoint_grpo_d"
            )
        repo_suffix = output_name.replace("checkpoint_", "").replace("_", "-")
        repo_id     = f"{hf_username}/trustworthy-ai-{repo_suffix}"
        merged_dir  = str(self.output_dir / f"{output_name}_merged")
        gguf_dir    = str(self.output_dir / f"{output_name}_gguf")

        print(f"\n=== Publishing {output_name} ===")
        print(f"  Repo ID      : {repo_id}")

        # 1. Switch to inference mode for ROUGE generation
        FastModel.for_inference(self.model)

        # 2. Compute ROUGE (uses in-memory model — must happen before merge/save)
        self._compute_rouge_report(
            output_name=output_name,
            baseline_path=baseline_path,
        )

        # 3. Merge LoRA adapters → full 16-bit safetensors
        print(f"  Merging LoRA → {merged_dir}")
        self.model.save_pretrained_merged(merged_dir, self.tokenizer, save_method="merged_16bit")

        # 4. Push merged model to HuggingFace (with retry — protects against transient failures)
        if hf_token:
            print(f"  Pushing merged model → {repo_id}")
            try:
                _retry_hf_push(
                    self.model.push_to_hub_merged,
                    repo_id,
                    self.tokenizer,
                    save_method="merged_16bit",
                    token=hf_token,
                    commit_message=f"train: {output_name} checkpoint",
                )
            except Exception as e:
                print(f"  [ERROR] HF merged upload failed after retries: {e}")
                print(f"  [ERROR] Local merged model saved at: {merged_dir}")
                print(f"  [ERROR] Re-run upload manually: python 2_model_trainer.py --mode sft --no_publish")
                print(f"  [ERROR] Or push directly: huggingface-cli upload {repo_id} {merged_dir}")

        # 5. Export GGUF (Q4_K_M quantisation) — skipped if llama.cpp unavailable
        if self._skip_gguf:
            print("  Skipping GGUF export (--skip_gguf set — llama.cpp / sudo not available)")
        else:
            print(f"  Exporting GGUF → {gguf_dir}")
            try:
                self.model.save_pretrained_gguf(gguf_dir, self.tokenizer, quantization_method="q4_k_m")
            except Exception as e:
                print(f"  [WARNING] GGUF export failed: {e}")
                print("  [WARNING] Re-run with --skip_gguf to bypass this step.")

        # 6. Push GGUF to same HuggingFace repo (with retry)
        if hf_token and not self._skip_gguf:
            print(f"  Pushing GGUF → {repo_id}")
            try:
                _retry_hf_push(
                    self.model.push_to_hub_gguf,
                    repo_id,
                    self.tokenizer,
                    quantization_method="q4_k_m",
                    token=hf_token,
                )
            except Exception as e:
                print(f"  [ERROR] HF GGUF upload failed after retries: {e}")
                print(f"  [ERROR] Local GGUF saved at: {gguf_dir}")
                print(f"  [ERROR] Re-run: huggingface-cli upload {repo_id} {gguf_dir}")

        print(f"  Done. Local merged: {merged_dir}  |  Local GGUF: {gguf_dir}")


# ---------------------------------------------------------------------------
# DAPO dynamic sampling patch
# ---------------------------------------------------------------------------

def _patch_dynamic_sampling(trainer: "GRPOTrainer") -> None:
    """Monkey-patch GRPOTrainer to skip zero-variance reward groups.

    After rollout, if all G completions for a prompt receive the same reward,
    the policy gradient is zero — the batch is wasted compute. DAPO discards it.
    """
    _orig_step = trainer.training_step

    def _patched_step(model, inputs, num_items_in_batch=None):
        # TRL versions can wrap the batch in a list; find the dict if present.
        input_dict = None
        if isinstance(inputs, dict):
            input_dict = inputs
        elif isinstance(inputs, (list, tuple)) and inputs:
            if len(inputs) == 1 and isinstance(inputs[0], dict):
                input_dict = inputs[0]
            else:
                for item in inputs:
                    if isinstance(item, dict) and "rewards" in item:
                        input_dict = item
                        break

        if input_dict is None:
            return _orig_step(model, inputs, num_items_in_batch)

        rewards = input_dict.get("rewards")
        if rewards is not None:
            # rewards shape: (batch, num_generations)
            import torch
            if not isinstance(rewards, torch.Tensor):
                return _orig_step(model, inputs, num_items_in_batch)
            variance = rewards.var(dim=-1)
            mask = variance > 0
            if mask.ndim == 0:
                if int(mask.item()) == 0:
                    return torch.tensor(0.0, device=model.device, requires_grad=True)
                return _orig_step(model, inputs, num_items_in_batch)
            if int(mask.sum().item()) == 0:
                # Every group has zero variance — skip entire batch
                return torch.tensor(0.0, device=model.device, requires_grad=True)
            # Filter to non-zero-variance groups only
            for key in list(input_dict.keys()):
                if isinstance(input_dict[key], torch.Tensor) and input_dict[key].shape[0] == mask.shape[0]:
                    input_dict[key] = input_dict[key][mask]
        return _orig_step(model, inputs, num_items_in_batch)

    trainer.training_step = _patched_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SFT + GRPO trainer for Qwen3-0.6B"
    )
    parser.add_argument("--mode", choices=["sft", "grpo", "publish"], default="sft",
                        help="Training mode: 'sft' (Phase 1), 'grpo' (Phase 2), or 'publish' (upload existing checkpoint)")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--output_name", default=None,
                        help="Checkpoint directory name (auto-set if not given)")
    # SFT args
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="Skip if checkpoint already exists")
    parser.add_argument("--hf_username", default="AjinkyaTaranekar",
                        help="HuggingFace username for model repo (e.g. AjinkyaTaranekar)")
    parser.add_argument("--no_publish", action="store_true",
                        help="Skip HuggingFace upload, GGUF export, and ROUGE computation")
    parser.add_argument("--skip_gguf", action="store_true", default=True,
                        help="Skip GGUF export and GGUF HF push (default: True; use --no_skip_gguf to enable)")
    parser.add_argument("--no_skip_gguf", dest="skip_gguf", action="store_false",
                        help="Enable GGUF export (requires llama.cpp and sudo access)")
    # GRPO args
    parser.add_argument("--sft_checkpoint", default="./models/checkpoint_sft",
                        help="Path to SFT checkpoint (starting point for GRPO)")
    parser.add_argument("--reward_type", choices=["c", "d"], default="d",
                        help="c=format+accuracy only  d=full composite (default)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir/output_name")
    parser.add_argument(
        "--from_checkpoint", type=str, default=None,
        help="Path to a prior SFT checkpoint to resume from (for curriculum staging)",
    )
    parser.add_argument(
        "--curriculum_stage", type=int, choices=[1, 2, 3], default=None,
        help="Train a SINGLE curriculum stage for SFT: 1=short format, 2=all examples, "
             "3=anti-drift replay mix. For manual stage-by-stage control / checkpoint chaining.",
    )
    parser.add_argument(
        "--no_curriculum", action="store_true",
        help="Disable the default 3-stage SFT curriculum (format -> complexity -> replay) "
             "and train once on the full dataset. Ignored if --curriculum_stage is set.",
    )
    parser.add_argument(
        "--v3_format", action="store_true",
        help="Use v3 format rewards (no CAPABILITY_CHECK requirement) for GRPO on v3-trained models",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to SFT training JSONL (default: <data_dir>/train_sft_v3.jsonl, "
             "produced by sft_dataset_assembler.py).",
    )

    args = parser.parse_args()

    if hasattr(args, "v3_format") and args.v3_format:
        global _V3_FORMAT_MODE
        _V3_FORMAT_MODE = True
        print("GRPO format reward: v3 mode (no CAPABILITY_CHECK requirement)")

    if not HAS_LIBS:
        print("Required libraries not installed.")
        print("pip install unsloth trl transformers datasets accelerate bitsandbytes")
        return

    # Auto-set output name
    if args.output_name is None:
        if args.mode == "sft":
            args.output_name = "checkpoint_sft"
        elif args.mode == "grpo":
            args.output_name = f"checkpoint_grpo_{args.reward_type}"
        else:  # publish — default to sft checkpoint
            args.output_name = "checkpoint_sft"

    # Skip if exists
    checkpoint_path = Path(args.output_dir) / args.output_name
    if args.skip_if_exists and (checkpoint_path / "adapter_config.json").exists():
        print(f"Checkpoint exists, skipping: {checkpoint_path}")
        return

    trainer = ModelTrainer(
        args.data_dir,
        args.output_dir,
        args.output_name,
        hf_username=args.hf_username,
        no_publish=args.no_publish,
        skip_gguf=args.skip_gguf,
    )

    if args.mode == "sft":
        print("\n=== Phase 1: SFT ===")
        _base_to_load = (args.from_checkpoint
                         if (hasattr(args, "from_checkpoint") and args.from_checkpoint and args.mode == "sft")
                         else MODEL_CONFIG["base_model"])
        if _base_to_load != MODEL_CONFIG["base_model"]:
            print(f"  Loading from checkpoint: {_base_to_load}")
            trainer.load_checkpoint(_base_to_load)
        else:
            trainer.load_base_model()
        trainer.apply_lora()
        dataset_path = Path(args.dataset) if args.dataset else Path(args.data_dir) / "train_sft_v3.jsonl"

        def _load_all(path):
            rows = []
            with open(path, encoding="utf-8") as _f:
                for _line in _f:
                    try:
                        rows.append(json.loads(_line))
                    except json.JSONDecodeError:
                        pass
            return rows

        # P1.2: carve ONE held-out eval set from the full dataset (seed 42), shared by every
        # curriculum stage. Stage pools are built from the TRAIN remainder only, so no eval row
        # leaks into any stage and eval_loss is comparable across stages. The fixed seed also makes
        # the eval set identical across separate --curriculum_stage processes (manual chaining).
        import random as _random_main
        _all_rows = _load_all(dataset_path)
        _shuf = list(_all_rows)
        _random_main.Random(42).shuffle(_shuf)
        _n_eval = max(1, len(_shuf) // 10)
        eval_pool, train_pool = _shuf[:_n_eval], _shuf[_n_eval:]
        print(f"  Held-out eval : {len(eval_pool)} rows (shared, seed 42)  |  train pool: {len(train_pool)}")

        if args.curriculum_stage:
            # Manual single-stage control (for checkpoint chaining via --from_checkpoint).
            _s1, _s2, _s3 = _split_curriculum_stages(train_pool)
            _stage = {1: _s1, 2: _s2, 3: _s3}[args.curriculum_stage]
            print(f"Curriculum stage {args.curriculum_stage}: {len(_stage)} examples "
                  f"(S1={len(_s1)} S2={len(_s2)} S3={len(_s3)})")
            trainer.train_sft(_write_temp_jsonl(_stage), args.output_name,
                              resume_from_checkpoint=args.resume, eval_records=eval_pool)
        elif not args.no_curriculum:
            # Default: 3-stage curriculum (format -> full complexity -> anti-drift replay),
            # trained sequentially on the SAME in-memory model so each stage continues from
            # the previous stage's best checkpoint. Per-stage epoch schedule keeps total
            # training moderate (1+2+1 = 4 effective epochs) to avoid the over-training that
            # contributed to the reasoning collapse. Tune on GPU; use --no_curriculum to skip.
            _s1, _s2, _s3 = _split_curriculum_stages(train_pool)
            if len(_s1) < 8 or len(_s2) < 8:
                print(f"  [curriculum] degenerate split (S1={len(_s1)} S2={len(_s2)}) — "
                      f"falling back to single train-pool run.")
                trainer.train_sft(_write_temp_jsonl(train_pool), args.output_name,
                                  resume_from_checkpoint=args.resume, eval_records=eval_pool)
            else:
                stages = [("1-format", _s1, 1), ("2-complexity", _s2, 2), ("3-replay", _s3, 1)]
                print(f"  [curriculum] 3 stages: S1={len(_s1)} S2={len(_s2)} S3={len(_s3)}")
                _orig_epochs = SFT_CONFIG["num_train_epochs"]
                try:
                    for _i, (_label, _stage, _epochs) in enumerate(stages, 1):
                        SFT_CONFIG["num_train_epochs"] = _epochs
                        print(f"\n--- SFT curriculum stage {_i}/3 ({_label}): "
                              f"{len(_stage)} examples, {_epochs} epoch(s) ---")
                        trainer.train_sft(
                            _write_temp_jsonl(_stage), args.output_name,
                            resume_from_checkpoint=(args.resume and _i == 1),
                            eval_records=eval_pool,
                        )
                finally:
                    SFT_CONFIG["num_train_epochs"] = _orig_epochs
        else:
            trainer.train_sft(_write_temp_jsonl(train_pool), args.output_name,
                              resume_from_checkpoint=args.resume, eval_records=eval_pool)

        # P1.1: publish ONCE after all training completes (model is in memory). For a curriculum
        # this replaces the old per-stage publish (which merged/exported/uploaded 3×).
        if args.no_publish:
            print("  Publish       : skipped (--no_publish)")
        else:
            trainer.publish(output_name=args.output_name, hf_username=args.hf_username)
        print(f"\nNext step → run GRPO training from this checkpoint:")
        print(f"  python 2_model_trainer.py --mode grpo --sft_checkpoint {checkpoint_path}")
        print(f"  # Or serve the SFT model to save a constitution baseline first:")
        print(f"  python 3_infererence.py --model_dir {checkpoint_path}")
        print(f"  # To re-upload this checkpoint later (if publish failed):")
        print(f"  python 2_model_trainer.py --mode publish --output_name {args.output_name} --hf_username {args.hf_username}")

    elif args.mode == "grpo":
        print(f"\n=== Phase 2: GRPO (reward_type={args.reward_type}) ===")
        if not (Path(args.sft_checkpoint) / "adapter_config.json").exists():
            print(f"ERROR: SFT checkpoint not found at {args.sft_checkpoint}")
            print("Run SFT first: python 2_model_trainer.py --mode sft")
            return
        dataset_path = Path(args.dataset) if args.dataset else Path(args.data_dir) / "train_sft_v3.jsonl"
        trainer.train_grpo(
            sft_checkpoint=args.sft_checkpoint,
            dataset_path=str(dataset_path),
            output_name=args.output_name,
            reward_type=args.reward_type,
            resume_from_checkpoint=args.resume,
        )
        # P1.1: publish once after GRPO completes (was previously auto-published inside train_grpo).
        if args.no_publish:
            print("  Publish       : skipped (--no_publish)")
        else:
            trainer.publish(output_name=args.output_name, hf_username=args.hf_username)
        print(f"\nNext step → serve the GRPO checkpoint:")
        print(f"  python 3_infererence.py --model_dir {checkpoint_path}")
        print(f"  # To re-upload this checkpoint later (if publish failed):")
        print(f"  python 2_model_trainer.py --mode publish --output_name {args.output_name} --hf_username {args.hf_username}")

    elif args.mode == "publish":
        print(f"\n=== Publish: uploading existing checkpoint to HuggingFace ===")
        if not (checkpoint_path / "adapter_config.json").exists():
            print(f"ERROR: No LoRA checkpoint found at {checkpoint_path}")
            print(f"  Expected file: {checkpoint_path / 'adapter_config.json'}")
            print(f"  Check --output_name and --output_dir point to a trained checkpoint.")
            print(f"  Available checkpoints:")
            print(f"    python 2_model_trainer.py --mode publish --output_name checkpoint_sft")
            print(f"    python 2_model_trainer.py --mode publish --output_name checkpoint_grpo_c")
            print(f"    python 2_model_trainer.py --mode publish --output_name checkpoint_grpo_d")
            return
        print(f"  Checkpoint    : {checkpoint_path}")
        print(f"  HF username   : {args.hf_username}")
        trainer.load_checkpoint(str(checkpoint_path))
        trainer.publish(
            output_name=args.output_name,
            hf_username=args.hf_username,
        )
        print(f"\nDone. Model pushed to HuggingFace.")
        print(f"  View at: https://huggingface.co/{args.hf_username}/trustworthy-ai-{args.output_name.replace('checkpoint_', '').replace('_', '-')}")


if __name__ == "__main__":
    main()
