"""
Model Trainer
=============
Phase 1 — SFT (Supervised Fine-Tuning):
    python pipeline/2_model_trainer.py --mode sft

Phase 2 — GRPO (Group Relative Policy Optimisation, DAPO improvements):
    python pipeline/2_model_trainer.py --mode grpo --sft_checkpoint models/checkpoint_sft \
        --reward_type d --output_name checkpoint_grpo_d

Reward types:
    c  format + accuracy only         (Ablation C)
    d  format + accuracy + tool + constitution  (Ablation D — full thesis contribution)

DAPO improvements applied over vanilla GRPO:
    - Token-level loss normalisation (Dr.GRPO): divide by completion length
    - Clip-Higher: asymmetric ε (0.2 low, 0.28 high)
    - Dynamic sampling: skip zero-variance groups
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
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    from datasets import load_dataset, Dataset
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False


# ---------------------------------------------------------------------------
# Model + training configuration
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "base_model":     "unsloth/Qwen3-0.6B",
    "max_seq_length": 2048,  # 4096 is marginal on 16 GB VRAM (A4000); raise if you have 24 GB+
    "load_in_4bit":   True,
    "lora_r":         16,
    "lora_alpha":     32,
}

SFT_CONFIG = {
    "per_device_train_batch_size": 1,  # 2 risks OOM with packing on 16 GB
    "gradient_accumulation_steps": 8,  # effective batch = 8 (same as before: 2*4)
    "num_train_epochs":            3,
    "learning_rate":               2e-4,
    "warmup_steps":                100,
    "logging_steps":               10,
    "save_steps":                  50,
    "eval_steps":                  50,
    "bf16":                        True,
    "optim":                       "adamw_8bit",
    "weight_decay":                0.01,
    "packing":                     True,
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
    "max_new_tokens":              512,
    # Training loop
    "num_train_epochs":            1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps":               5,
    "save_steps":                  100,
    "bf16":                        True,
    "optim":                       "adamw_8bit",
    # DAPO dynamic sampling: skip prompts where all G completions score identically
    # (zero-gradient batches waste compute and reward signal)
    "dynamic_sampling":            True,
}

# Reward component weights — must sum to 1.0
REWARD_WEIGHTS = {
    "format":        0.30,  # structural: think + CAPABILITY_CHECK + answer
    "accuracy":      0.40,  # correctness: math code execution
    "tool_integrity": 0.15, # no hallucinated/unavailable tools (P3)
    "constitution":  0.15,  # broader rule check: P1+P4+P14+P18
}


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


def _extract_code_from_response(response: str) -> list[str]:
    blocks = []
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


def _format_reward(response: str) -> float:
    """P1 structural check: <think> + CAPABILITY_CHECK + <answer> all present."""
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_cap   = "CAPABILITY_CHECK" in response
    has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))
    return 1.0 if (has_think and has_cap and has_ans) else 0.0


def _accuracy_reward(response: str, expected_answer: str | None,
                     question_type: str) -> float:
    """For verifiable math categories: execute code and check answer.
    For behavioural categories: neutral 0.5 (no ground truth)."""
    math_types = {"arithmetic", "algebra", "geometry", "statistics",
                  "unit_conversion", "word_problems",
                  "trigonometry", "calculus", "advanced_geometry"}
    if not expected_answer or question_type not in math_types:
        return 0.5  # neutral — behavioural examples have no single correct answer

    code_blocks = _extract_code_from_response(response)
    if not code_blocks:
        # Gave numeric answer without code — wrong method when code is expected
        answer_m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if answer_m:
            num = _last_number(answer_m.group(1))
            if num and _answers_match(num, expected_answer):
                return 0.3  # partial credit — right answer, wrong method (mental math)
        return 0.0

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
    """P3: no calls to non-existent or session-unavailable tools."""
    _ALL_TOOLS = frozenset({
        "python_execute", "web_search", "read_url",
        "get_datetime",
    })
    called = set(re.findall(r"<tool>(\w+)\(", response))
    hallucinated = called - _ALL_TOOLS
    unavailable  = (called & _ALL_TOOLS) - active_tools
    return 0.0 if (hallucinated or unavailable) else 1.0


def _constitution_reward(response: str, question: str,
                          category: str, tool_profile: dict) -> float:
    """Broader rule check using Blocker 2's rule_check_response.
    Falls back to a subset check if import fails."""
    try:
        from sft_gold_response_generator import rule_check_response
        violations = rule_check_response(response, question, category, tool_profile)
        n = len(violations)
        return max(0.0, (5 - n) / 5)  # 5 = max checkable principles
    except ImportError:
        # Minimal fallback: just check P1 + P18
        has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
        has_cap   = "CAPABILITY_CHECK" in response
        has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))
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
    reward_type 'd': full composite         (Ablation D — four functions)
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

    # Ablation D — full composite (four functions)
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

    return [format_reward, accuracy_reward, tool_integrity_reward, constitution_reward]


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
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Trains Qwen3-0.6B via SFT then GRPO (DAPO improvements)."""

    def __init__(self, data_dir: str, output_dir: str,
                 output_name: str = "checkpoint_sft",
                 hf_username: str = "AjinkyaTaranekar",
                 no_publish: bool = False):
        self.data_dir    = Path(data_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name  = output_name
        self._hf_username = hf_username
        self._no_publish  = no_publish
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
                  resume_from_checkpoint=False):
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
        raw = load_dataset("json", data_files=dataset_path)
        # Split the raw dataset BEFORE text-formatting so we keep 'messages' for ROUGE
        raw_split = raw["train"].train_test_split(test_size=0.10, seed=42)
        train_dataset = raw_split["train"].map(
            messages_to_text, fn_kwargs={"tokenizer": self.tokenizer},
        )
        eval_dataset = raw_split["test"].map(
            messages_to_text, fn_kwargs={"tokenizer": self.tokenizer},
        )
        # Keep raw eval records (with 'messages' key) for ROUGE computation in publish()
        self._eval_raw = [dict(ex) for ex in raw_split["test"]]
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
            eval_strategy="steps",
            bf16=SFT_CONFIG["bf16"],
            optim=SFT_CONFIG["optim"],
            weight_decay=SFT_CONFIG["weight_decay"],
            packing=SFT_CONFIG["packing"],
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            args=training_args,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save loss history for thesis charts
        loss_path = self.output_dir / output_name / "loss_history.json"
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"  Loss history  : {loss_path}")

        trainer.save_model(str(self.output_dir / output_name))
        print(f"  SFT checkpoint saved → {self.output_dir / output_name}")

        # Auto-publish unless suppressed
        if self._no_publish:
            print("  Publish       : skipped (--no_publish)")
        else:
            self.publish(
                output_name=output_name,
                hf_username=self._hf_username,
            )
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
        grpo_kwargs = dict(
            output_dir=str(self.output_dir / output_name),
            num_generations=GRPO_CONFIG["num_generations"],
            learning_rate=GRPO_CONFIG["learning_rate"],
            kl_coef=GRPO_CONFIG["kl_coef"],
            clip_range_ratio=GRPO_CONFIG["clip_range_ratio"],
            temperature=GRPO_CONFIG["temperature"],
            max_new_tokens=GRPO_CONFIG["max_new_tokens"],
            num_train_epochs=GRPO_CONFIG["num_train_epochs"],
            per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
            logging_steps=GRPO_CONFIG["logging_steps"],
            save_steps=GRPO_CONFIG["save_steps"],
            bf16=GRPO_CONFIG["bf16"],
            optim=GRPO_CONFIG["optim"],
            report_to="none",
        )

        # Attempt DAPO Clip-Higher if TRL supports it
        try:
            config = GRPOConfig(
                **grpo_kwargs,
                clip_range_ratio_high=GRPO_CONFIG["clip_range_ratio_high"],
            )
            print("  DAPO Clip-Higher active (clip_range_ratio_high="
                  f"{GRPO_CONFIG['clip_range_ratio_high']})")
        except TypeError:
            # Older TRL — fall back to symmetric clipping
            config = GRPOConfig(**grpo_kwargs)
            print("  NOTE: TRL version does not support clip_range_ratio_high. "
                  "Using symmetric clipping. Update to trl>=0.13.0 for DAPO Clip-Higher.")

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

        # Save GRPO loss history
        loss_path = self.output_dir / output_name / "grpo_loss_history.json"
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"  GRPO loss history: {loss_path}")

        trainer.save_model(str(self.output_dir / output_name))
        print(f"  GRPO checkpoint saved → {self.output_dir / output_name}")

        # Auto-publish unless suppressed
        if self._no_publish:
            print("  Publish       : skipped (--no_publish)")
        else:
            self.publish(
                output_name=output_name,
                hf_username=self._hf_username,
            )
        return self

    # ── Convenience: run full SFT pipeline ──────────────────────────────────

    def train(self):
        self.load_base_model()
        self.apply_lora()
        dataset_path = self.data_dir / "train_sft_v2.jsonl"
        self.train_sft(str(dataset_path), self.output_name)

    def _local_generate(self, prompt_msgs: list, max_new_tokens: int = 256) -> str:
        """Greedy-decode one prompt using the in-memory model (inference mode assumed).

        Used only during publish() — model must already be switched to inference mode
        via FastModel.for_inference() before calling this.
        """
        import torch
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
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
            for ex in sample:
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

        # 5. Export GGUF (Q4_K_M quantisation)
        print(f"  Exporting GGUF → {gguf_dir}")
        try:
            self.model.save_pretrained_gguf(gguf_dir, self.tokenizer, quantization_method="q4_k_m")
        except Exception as e:
            print(f"  [WARNING] GGUF export failed: {e}")

        # 6. Push GGUF to same HuggingFace repo (with retry)
        if hf_token:
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
        rewards = inputs.get("rewards")
        if rewards is not None:
            # rewards shape: (batch, num_generations)
            import torch
            variance = rewards.var(dim=-1)
            mask = variance > 0
            if mask.sum() == 0:
                # Every group has zero variance — skip entire batch
                return torch.tensor(0.0, device=model.device, requires_grad=True)
            # Filter to non-zero-variance groups only
            for key in list(inputs.keys()):
                if isinstance(inputs[key], torch.Tensor) and inputs[key].shape[0] == mask.shape[0]:
                    inputs[key] = inputs[key][mask]
        return _orig_step(model, inputs, num_items_in_batch)

    trainer.training_step = _patched_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SFT + GRPO trainer for Qwen3-0.6B"
    )
    parser.add_argument("--mode", choices=["sft", "grpo"], default="sft",
                        help="Training mode: 'sft' (Phase 1) or 'grpo' (Phase 2)")
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
    # GRPO args
    parser.add_argument("--sft_checkpoint", default="./models/checkpoint_sft",
                        help="Path to SFT checkpoint (starting point for GRPO)")
    parser.add_argument("--reward_type", choices=["c", "d"], default="d",
                        help="c=format+accuracy only  d=full composite (default)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir/output_name")

    args = parser.parse_args()

    if not HAS_LIBS:
        print("Required libraries not installed.")
        print("pip install unsloth trl transformers datasets accelerate bitsandbytes")
        return

    # Auto-set output name
    if args.output_name is None:
        if args.mode == "sft":
            args.output_name = "checkpoint_sft"
        else:
            args.output_name = f"checkpoint_grpo_{args.reward_type}"

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
    )

    if args.mode == "sft":
        print("\n=== Phase 1: SFT ===")
        trainer.load_base_model()
        trainer.apply_lora()
        dataset_path = Path(args.data_dir) / "train_sft_v2.jsonl"
        trainer.train_sft(str(dataset_path), args.output_name,
                          resume_from_checkpoint=args.resume)
        print(f"\nNext step → run GRPO training from this checkpoint:")
        print(f"  python pipeline/2_model_trainer.py --mode grpo --sft_checkpoint {checkpoint_path}")
        print(f"  # Or serve the SFT model to save a constitution baseline first:")
        print(f"  python pipeline/3_infererence.py --model_dir {checkpoint_path}")

    elif args.mode == "grpo":
        print(f"\n=== Phase 2: GRPO (reward_type={args.reward_type}) ===")
        if not (Path(args.sft_checkpoint) / "adapter_config.json").exists():
            print(f"ERROR: SFT checkpoint not found at {args.sft_checkpoint}")
            print("Run SFT first: python 2_model_trainer.py --mode sft")
            return
        dataset_path = Path(args.data_dir) / "train_sft_v2.jsonl"
        trainer.train_grpo(
            sft_checkpoint=args.sft_checkpoint,
            dataset_path=str(dataset_path),
            output_name=args.output_name,
            reward_type=args.reward_type,
            resume_from_checkpoint=args.resume,
        )
        print(f"\nNext step → serve the GRPO checkpoint:")
        print(f"  python pipeline/3_infererence.py --model_dir {checkpoint_path}")


if __name__ == "__main__":
    main()
