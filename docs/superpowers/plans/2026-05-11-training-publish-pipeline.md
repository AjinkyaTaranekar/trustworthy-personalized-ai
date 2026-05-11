# Training + Publish Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `2_model_trainer.py` to use a 90/10 train/eval split, automatically publish each checkpoint to HuggingFace (safetensors + GGUF), compute ROUGE scores; update `3_infererence.py` to support GGUF loading; add ROUGE and loss-curve sections to `analysis.ipynb`.

**Architecture:** All post-training steps (merge, HF push, GGUF export, ROUGE) live inside a new `ModelTrainer.publish()` method that is called automatically at the end of `train_sft()` and `train_grpo()`. The inference server adds a `--gguf` flag that dispatches generation to `llama-cpp-python` while keeping all existing endpoints unchanged. The notebook gains two new self-contained sections that load from `reports/rouge_*.json` and `models/*/loss_history.json`.

**Tech Stack:** Unsloth (merge/GGUF export/HF push), `rouge-score` (ROUGE computation), `llama-cpp-python` (GGUF inference), `huggingface_hub` (GGUF download), HuggingFace Hub, Plotly (notebook charts).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pipeline/2_model_trainer.py` | Modify | 90/10 splits; `compute_rouge()`; `ModelTrainer._local_generate()`, `._compute_rouge_report()`, `.publish()`; `--hf_username`/`--no_publish` CLI args |
| `pipeline/3_infererence.py` | Modify | `_USE_GGUF` / `_GGUF_MODEL` globals; `_resolve_gguf_path()`; `_generate_gguf()`; `--gguf` CLI arg; guard updates |
| `pipeline/analysis.ipynb` | Modify | Append Section 8 (ROUGE charts) and Section 9 (loss curves) cells |
| `pipeline/tests/__init__.py` | Create | Makes `tests/` a package |
| `pipeline/tests/test_rouge.py` | Create | Unit tests for `compute_rouge()` |

---

## Task 1 — 90/10 SFT split and store eval records

**Files:**
- Modify: `pipeline/2_model_trainer.py` — `ModelTrainer.train_sft()` (lines 425–471)

- [ ] **Step 1: Open `pipeline/2_model_trainer.py` and locate `train_sft()`**

Find these lines (around line 426):
```python
raw = load_dataset("json", data_files=dataset_path)
full_dataset = raw["train"].map(
    messages_to_text, fn_kwargs={"tokenizer": self.tokenizer},
)
split = full_dataset.train_test_split(test_size=0.05, seed=42)
print(f"  Train: {len(split['train'])}  |  Eval: {len(split['test'])}")
```

Replace with:
```python
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
```

- [ ] **Step 2: Update `SFTTrainer` call to use the new split variables**

Find (around line 455):
```python
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            args=training_args,
        )
```

This is already correct — `split["train"]` and `split["test"]` still work. No change needed here.

- [ ] **Step 3: Verify the split size change is the only behavioural difference**

Run (requires no GPU — just parsing):
```bash
python -c "
import ast, pathlib
src = pathlib.Path('pipeline/2_model_trainer.py').read_text()
assert 'test_size=0.10' in src, 'Split not updated'
assert 'self._eval_raw' in src, 'eval_raw not stored'
print('OK')
"
```
Expected: `OK`

---

## Task 2 — 90/10 GRPO split and store held-out set

**Files:**
- Modify: `pipeline/2_model_trainer.py` — `ModelTrainer.train_grpo()` (around lines 499–504)

- [ ] **Step 1: Locate the dataset-loading block inside `train_grpo()`**

Find (around line 499):
```python
        print(f"  Building GRPO dataset from {dataset_path}...")
        dataset = build_grpo_dataset(dataset_path)
        print(f"  GRPO dataset: {len(dataset)} prompts")
```

Replace with:
```python
        print(f"  Building GRPO dataset from {dataset_path}...")
        full_grpo = build_grpo_dataset(dataset_path)
        grpo_split = full_grpo.train_test_split(test_size=0.10, seed=42)
        dataset = grpo_split["train"]
        self._grpo_eval_dataset = grpo_split["test"]
        print(f"  GRPO Train: {len(dataset)}  |  Held-out: {len(self._grpo_eval_dataset)}")
```

- [ ] **Step 2: Update the `GRPOTrainer` call to use the train subset**

Find (around line 540):
```python
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=dataset,
        )
```

`dataset` now refers to the 90% train subset — no change needed.

- [ ] **Step 3: Verify**

```bash
python -c "
import ast, pathlib
src = pathlib.Path('pipeline/2_model_trainer.py').read_text()
assert 'self._grpo_eval_dataset' in src, 'GRPO eval not stored'
assert 'grpo_split' in src, 'GRPO split missing'
print('OK')
"
```
Expected: `OK`

---

## Task 3 — `compute_rouge()` module-level function and unit tests

**Files:**
- Modify: `pipeline/2_model_trainer.py` — add function after `REWARD_WEIGHTS` block (around line 106)
- Create: `pipeline/tests/__init__.py`
- Create: `pipeline/tests/test_rouge.py`

- [ ] **Step 1: Write the failing tests first**

Create `pipeline/tests/__init__.py` (empty):
```python
```

Create `pipeline/tests/test_rouge.py`:
```python
"""Unit tests for compute_rouge() in 2_model_trainer.py.

Heavy ML deps (unsloth, trl, torch, datasets) are mocked so this runs
without a GPU and without installing the full training stack.
"""
import importlib.util
import sys
import unittest.mock as mock
from pathlib import Path

_TRAINER_PATH = Path(__file__).resolve().parent.parent / "2_model_trainer.py"

_HEAVY = [
    "unsloth", "trl", "torch", "datasets", "accelerate",
    "bitsandbytes", "transformers", "peft",
]


def _load_compute_rouge():
    mocks = {m: mock.MagicMock() for m in _HEAVY}
    with mock.patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location("trainer", _TRAINER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.compute_rouge


compute_rouge = _load_compute_rouge()


def test_perfect_match():
    r = compute_rouge(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"],
    )
    assert r["rouge1"]["fmeasure"] == 1.0, f"Expected 1.0, got {r['rouge1']['fmeasure']}"
    assert r["rouge2"]["fmeasure"] == 1.0
    assert r["rougeL"]["fmeasure"] == 1.0


def test_zero_overlap():
    r = compute_rouge(
        ["apple orange banana"],
        ["the cat sat on the mat"],
    )
    assert r["rouge1"]["fmeasure"] == 0.0, f"Expected 0.0, got {r['rouge1']['fmeasure']}"
    assert r["rouge2"]["fmeasure"] == 0.0
    assert r["rougeL"]["fmeasure"] == 0.0


def test_partial_overlap():
    r = compute_rouge(
        ["the cat sat"],
        ["the cat sat on the mat"],
    )
    assert 0.0 < r["rouge1"]["fmeasure"] < 1.0
    assert r["rougeL"]["fmeasure"] > 0.0


def test_multi_pair_averaging():
    r = compute_rouge(
        ["the cat sat on the mat", "apple orange"],
        ["the cat sat on the mat", "apple orange"],
    )
    assert r["rouge1"]["fmeasure"] == 1.0
```

- [ ] **Step 2: Run tests — expect ImportError (function not yet defined)**

```bash
cd pipeline && python -m pytest tests/test_rouge.py -v 2>&1 | head -30
```
Expected: `AttributeError: module 'trainer' has no attribute 'compute_rouge'` (or similar — proves tests are live)

- [ ] **Step 3: Add `compute_rouge()` to `2_model_trainer.py`**

Find the block ending with `REWARD_WEIGHTS = { ... }` (around line 105). Insert immediately after:

```python
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
    """
    from rouge_score import rouge_scorer as _rs
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd pipeline && python -m pytest tests/test_rouge.py -v
```
Expected output:
```
tests/test_rouge.py::test_perfect_match PASSED
tests/test_rouge.py::test_zero_overlap PASSED
tests/test_rouge.py::test_partial_overlap PASSED
tests/test_rouge.py::test_multi_pair_averaging PASSED
4 passed
```

---

## Task 4 — `ModelTrainer._local_generate()` and `._compute_rouge_report()`

**Files:**
- Modify: `pipeline/2_model_trainer.py` — add two methods to `ModelTrainer` class

- [ ] **Step 1: Add `_local_generate()` method to `ModelTrainer`**

Add after the `train()` convenience method (around line 578), inside the `ModelTrainer` class:

```python
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
```

- [ ] **Step 2: Add `_compute_rouge_report()` method to `ModelTrainer`**

Add directly after `_local_generate()`:

```python
    def _compute_rouge_report(
        self,
        output_name: str,
        baseline_path: str = "reports/constitution_baseline.json",
        max_eval_examples: int = 50,
    ) -> dict:
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
                hyp = self._local_generate(prompt_msgs)
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
            hypotheses, references = [], []
            for pr in probe_results:
                q = pr.get("question", "")
                if isinstance(q, list):
                    q = q[-1]   # last turn of multi-turn probe
                ref = pr.get("response", "")
                if not q or not ref:
                    continue
                hyp = self._local_generate([{"role": "user", "content": q}])
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
            all_rewards = []
            for row in sample:
                prompt = row["prompt"]
                hyp = self._local_generate(prompt)
                rewards = reward_fn(
                    prompts=[str(prompt)],
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
```

---

## Task 5 — `ModelTrainer.publish()` and CLI args

**Files:**
- Modify: `pipeline/2_model_trainer.py` — `ModelTrainer` class and `main()`

- [ ] **Step 1: Add `publish()` method to `ModelTrainer`**

Add after `_compute_rouge_report()`:

```python
    def publish(
        self,
        output_name: str,
        hf_username: str,
        dataset_path: str,
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
        import os
        from unsloth import FastModel

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("  [publish] HF_TOKEN not set — local merge/GGUF will proceed but HF upload will be skipped.")

        repo_suffix = output_name.replace("checkpoint_", "").replace("_", "-")
        repo_id     = f"{hf_username}/trustworthy-ai-{repo_suffix}"
        merged_dir  = str(self.output_dir / f"{output_name}_merged")
        gguf_dir    = str(self.output_dir / f"{output_name}_gguf")

        print(f"\n=== Publishing {output_name} ===")

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

        # 4. Push merged model to HuggingFace
        if hf_token:
            print(f"  Pushing merged model → {repo_id}")
            self.model.push_to_hub_merged(
                repo_id,
                self.tokenizer,
                save_method="merged_16bit",
                token=hf_token,
                commit_message=f"train: {output_name} checkpoint",
            )

        # 5. Export GGUF (Q4_K_M quantisation)
        print(f"  Exporting GGUF → {gguf_dir}")
        self.model.save_pretrained_gguf(gguf_dir, self.tokenizer, quantization_method="q4_k_m")

        # 6. Push GGUF to same HuggingFace repo
        if hf_token:
            print(f"  Pushing GGUF → {repo_id}")
            self.model.push_to_hub_gguf(
                repo_id,
                self.tokenizer,
                quantization_method="q4_k_m",
                token=hf_token,
            )

        print(f"  Done. Local merged: {merged_dir}  |  Local GGUF: {gguf_dir}")
```

- [ ] **Step 2: Store `_hf_username` and `_no_publish` on the `ModelTrainer.__init__`**

Find `__init__` (around line 384):
```python
    def __init__(self, data_dir: str, output_dir: str,
                 output_name: str = "checkpoint_sft"):
        self.data_dir   = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name = output_name
        self.model       = None
        self.tokenizer   = None
```

Replace with:
```python
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
```

- [ ] **Step 3: Wire `publish()` into `train_sft()`**

At the end of `train_sft()`, after `trainer.save_model(...)` and the print statement, add:

```python
        # Auto-publish unless suppressed
        if not self._no_publish:
            self.publish(
                output_name=output_name,
                hf_username=self._hf_username,
                dataset_path=dataset_path,
            )
        return self
```

- [ ] **Step 4: Wire `publish()` into `train_grpo()`**

At the end of `train_grpo()`, after `trainer.save_model(...)` and the print statement, add:

```python
        # Auto-publish unless suppressed
        if not self._no_publish:
            self.publish(
                output_name=output_name,
                hf_username=self._hf_username,
                dataset_path=dataset_path,
            )
        return self
```

- [ ] **Step 5: Add `--hf_username` and `--no_publish` to `main()`**

Find (around line 616):
```python
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="Skip if checkpoint already exists")
```

Add after that line:
```python
    parser.add_argument("--hf_username", default="AjinkyaTaranekar",
                        help="HuggingFace username for model repo (e.g. AjinkyaTaranekar)")
    parser.add_argument("--no_publish", action="store_true",
                        help="Skip HuggingFace upload, GGUF export, and ROUGE computation")
```

- [ ] **Step 6: Pass the new args when constructing `ModelTrainer`**

Find (around line 653):
```python
    trainer = ModelTrainer(args.data_dir, args.output_dir, args.output_name)
```

Replace with:
```python
    trainer = ModelTrainer(
        args.data_dir,
        args.output_dir,
        args.output_name,
        hf_username=args.hf_username,
        no_publish=args.no_publish,
    )
```

- [ ] **Step 7: Verify CLI args are wired**

```bash
python pipeline/2_model_trainer.py --help 2>&1 | grep -E "hf_username|no_publish"
```
Expected:
```
  --hf_username HF_USERNAME
  --no_publish
```

---

## Task 6 — GGUF inference in `3_infererence.py`

**Files:**
- Modify: `pipeline/3_infererence.py`

- [ ] **Step 1: Add GGUF globals and `_resolve_gguf_path()` after the model-state globals**

Find (around line 397):
```python
_MODEL = None
_TOKENIZER = None
_MODEL_LABEL = "not_loaded"
```

Replace with:
```python
_MODEL = None
_TOKENIZER = None
_MODEL_LABEL = "not_loaded"
_USE_GGUF = False
_GGUF_MODEL = None   # llama_cpp.Llama instance — populated when --gguf is passed


def _resolve_gguf_path(gguf_arg: str) -> str:
    """Return a local path to a .gguf file.

    Accepts a local file path or a HuggingFace repo ID.
    For HF repos, downloads the q4_k_m GGUF file (or the first .gguf found) via
    huggingface_hub.hf_hub_download which caches to ~/.cache/huggingface.
    """
    p = Path(gguf_arg)
    if p.exists():
        return str(p)
    # HuggingFace repo ID (contains '/' but not a local path)
    from huggingface_hub import hf_hub_download, list_repo_files
    print(f"  Fetching file list from HF repo: {gguf_arg}")
    repo_files = list(list_repo_files(gguf_arg))
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
    if not gguf_files:
        raise ValueError(f"No .gguf files found in HuggingFace repo '{gguf_arg}'")
    preferred = [f for f in gguf_files if "q4_k_m" in f.lower()]
    target = preferred[0] if preferred else gguf_files[0]
    print(f"  Downloading {target} from {gguf_arg}...")
    return hf_hub_download(repo_id=gguf_arg, filename=target)
```

- [ ] **Step 2: Add `_generate_gguf()` function after `_raw_generate()`**

Find (around line 513):
```python
def _raw_generate(prompt: str, max_new_tokens: int = 256) -> str:
```

Add a new function immediately AFTER `_raw_generate()` (after the closing of that function):

```python
def _generate_gguf(conversation: list, max_new_tokens: int, temperature: float,
                   greedy: bool = False) -> tuple:
    """Generation via llama-cpp-python for GGUF models.

    Returns the same (response_text, n_input_tokens, n_output_tokens, elapsed_s)
    tuple as _generate() so all callers stay compatible.
    """
    t0 = time.perf_counter()
    result = _GGUF_MODEL.create_chat_completion(
        messages=conversation,
        max_tokens=max_new_tokens,
        temperature=0.0 if greedy else max(temperature, 1e-6),
        top_p=0.9 if not greedy else 1.0,
    )
    elapsed = time.perf_counter() - t0
    text  = result["choices"][0]["message"]["content"]
    n_in  = result["usage"]["prompt_tokens"]
    n_out = result["usage"]["completion_tokens"]
    return text, n_in, n_out, elapsed
```

- [ ] **Step 3: Update `_generate()` to dispatch to GGUF backend**

Find the `_generate()` function signature (around line 482):
```python
def _generate(conversation: list, max_new_tokens: int, temperature: float,
              greedy: bool = False) -> tuple:
    """One generation step. Returns (response_text, n_input_tokens, n_output_tokens, elapsed_s)."""
    prompt = _TOKENIZER.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

Replace the entire function body (but keep signature + docstring) with:
```python
def _generate(conversation: list, max_new_tokens: int, temperature: float,
              greedy: bool = False) -> tuple:
    """One generation step. Returns (response_text, n_input_tokens, n_output_tokens, elapsed_s)."""
    if _USE_GGUF:
        return _generate_gguf(conversation, max_new_tokens, temperature, greedy)
    prompt = _TOKENIZER.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = _TOKENIZER(prompt, return_tensors="pt").to("cuda")
    n_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    gen_kwargs: Dict[str, Any] = dict(inputs, max_new_tokens=max_new_tokens)
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.no_grad():
        out = _MODEL.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0
    tokens = out[0][n_in:]
    return _TOKENIZER.decode(tokens, skip_special_tokens=True), n_in, len(tokens), elapsed
```

- [ ] **Step 4: Update `_raw_generate()` to support GGUF**

Find `_raw_generate()` (around line 500):
```python
def _raw_generate(prompt: str, max_new_tokens: int = 256) -> str:
    ...
    if _MODEL is None or _TOKENIZER is None:
        return ""
```

Replace the guard at the top of that function with:
```python
    if _USE_GGUF:
        if _GGUF_MODEL is None:
            return ""
        result = _GGUF_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": "Respond only with the requested JSON. No prose, no markdown fences."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        return result["choices"][0]["message"]["content"]
    if _MODEL is None or _TOKENIZER is None:
        return ""
```

- [ ] **Step 5: Update `/health` endpoint to reflect GGUF mode**

Find (around line 580):
```python
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": _MODEL_LABEL, "loaded": _MODEL is not None}
```

Replace with:
```python
@app.get("/health")
def health() -> Dict[str, Any]:
    loaded = (_MODEL is not None) or (_USE_GGUF and _GGUF_MODEL is not None)
    return {
        "status": "ok",
        "model":  _MODEL_LABEL,
        "loaded": loaded,
        "mode":   "gguf" if _USE_GGUF else "lora",
    }
```

- [ ] **Step 6: Update `chat_completions` guard**

Find (around line 624):
```python
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
```

Replace with:
```python
    if _MODEL is None and not (_USE_GGUF and _GGUF_MODEL is not None):
        raise HTTPException(status_code=503, detail="Model not loaded")
```

- [ ] **Step 7: Add `--gguf` argument and GGUF loading to `main()`**

Find (around line 831):
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Trustworthy AI Inference Server")
    parser.add_argument("--model_dir", default="./models/checkpoint_sft",
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                        help="HuggingFace model ID used when --model_dir does not exist")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (overrides PIPELINE_* env vars)")
    args = parser.parse_args()

    global cfg, _MODEL, _TOKENIZER, _MODEL_LABEL, _GRAPH_CLIENT, _ONTO_GRAPH
```

Replace with:
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Trustworthy AI Inference Server")
    parser.add_argument("--model_dir", default="./models/checkpoint_sft",
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                        help="HuggingFace model ID used when --model_dir does not exist")
    parser.add_argument("--gguf", default=None,
                        help="Load a GGUF model instead of LoRA. Accepts a local .gguf file "
                             "path or a HuggingFace repo ID (downloads q4_k_m variant).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (overrides PIPELINE_* env vars)")
    args = parser.parse_args()

    global cfg, _MODEL, _TOKENIZER, _MODEL_LABEL, _GRAPH_CLIENT, _ONTO_GRAPH, _USE_GGUF, _GGUF_MODEL
```

- [ ] **Step 8: Add the GGUF loading branch inside `main()`**

Find the model loading block (around line 858):
```python
    # ── Load model ──────────────────────────────────────────────────────────
    from unsloth import FastModel  # deferred so the module imports without GPU

    model_path = Path(args.model_dir)
    source = str(model_path) if model_path.exists() else args.base_model
    _MODEL_LABEL = source
    print(f"Loading model: {source}")
    _MODEL, _TOKENIZER = FastModel.from_pretrained(
        model_name=source, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
    )
    FastModel.for_inference(_MODEL)
```

Replace with:
```python
    # ── Load model ──────────────────────────────────────────────────────────
    if args.gguf:
        _USE_GGUF = True
        gguf_path = _resolve_gguf_path(args.gguf)
        from llama_cpp import Llama
        print(f"Loading GGUF model: {gguf_path}")
        _GGUF_MODEL = Llama(
            model_path=gguf_path,
            n_ctx=args.max_seq_length,
            n_gpu_layers=-1,    # offload all layers to GPU; falls back to CPU automatically
            verbose=False,
        )
        _MODEL_LABEL = Path(gguf_path).stem
        print(f"GGUF model ready: {_MODEL_LABEL}")
    else:
        from unsloth import FastModel  # deferred so the module imports without GPU
        model_path = Path(args.model_dir)
        source = str(model_path) if model_path.exists() else args.base_model
        _MODEL_LABEL = source
        print(f"Loading model: {source}")
        _MODEL, _TOKENIZER = FastModel.from_pretrained(
            model_name=source, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
        )
        FastModel.for_inference(_MODEL)
```

- [ ] **Step 9: Verify `--gguf` appears in help**

```bash
python pipeline/3_infererence.py --help 2>&1 | grep gguf
```
Expected: `--gguf GGUF`

---

## Task 7 — Analysis notebook: Section 8 (ROUGE charts)

**Files:**
- Modify: `pipeline/analysis.ipynb`

Use the `NotebookEdit` tool (load schema with `ToolSearch` query `"select:NotebookEdit"` first). Add all cells after the last existing cell (`cell-7c`).

- [ ] **Step 1: Add Section 8 markdown heading cell**

Use `NotebookEdit` with `new_source`:
```markdown
## Section 8 — ROUGE Scores

Compares ROUGE-1, ROUGE-2, ROUGE-L F1 across checkpoints for two reference sources:
eval split gold responses and constitution probe baseline.
Load `reports/rouge_*.json` produced by `2_model_trainer.py publish()`.
```

- [ ] **Step 2: Add cell 8a — load ROUGE reports**

```python
# --- 8a: Load ROUGE reports ---
rouge_reports = []
for p in sorted(REPORTS_DIR.glob("rouge_*.json")):
    with open(p, encoding="utf-8") as f:
        rouge_reports.append(json.load(f))

print(f"ROUGE reports: {len(rouge_reports)}")
for r in rouge_reports:
    ev  = "✓" if r.get("eval_split_rouge")     else "✗"
    pb  = "✓" if r.get("probe_baseline_rouge") else "✗"
    rwd = f"{r['grpo_held_out_reward']:.4f}" if r.get("grpo_held_out_reward") is not None else "—"
    print(f"  {r['checkpoint']:35}  eval={ev}  probe={pb}  grpo_reward={rwd}")
```

- [ ] **Step 3: Add cell 8b — ROUGE F1 bar chart (eval split)**

```python
# --- 8b: ROUGE F1 — eval split gold responses ---
def _rouge_to_df(reports, key):
    rows = []
    for r in reports:
        block = r.get(key)
        if not block:
            continue
        rows.append({
            "checkpoint": r["checkpoint"],
            "ROUGE-1":    block["rouge1"]["fmeasure"],
            "ROUGE-2":    block["rouge2"]["fmeasure"],
            "ROUGE-L":    block["rougeL"]["fmeasure"],
        })
    return pd.DataFrame(rows)

eval_df = _rouge_to_df(rouge_reports, "eval_split_rouge")

if not eval_df.empty:
    melted = eval_df.melt(id_vars="checkpoint", var_name="Metric", value_name="F1")
    fig = px.bar(
        melted, x="checkpoint", y="F1", color="Metric", barmode="group",
        title="ROUGE F1 — Eval Split (Gold Responses)",
        color_discrete_sequence=PALETTE, text_auto=".3f",
        labels={"checkpoint": "Checkpoint", "F1": "F1 Score"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    save_fig(fig, "17_rouge_eval_split")
else:
    print("No eval-split ROUGE data yet — run 2_model_trainer.py to generate reports/rouge_*.json")
```

- [ ] **Step 4: Add cell 8c — ROUGE F1 bar chart (probe baseline)**

```python
# --- 8c: ROUGE F1 — probe baseline (constitution drift indicator) ---
probe_df = _rouge_to_df(rouge_reports, "probe_baseline_rouge")

if not probe_df.empty:
    melted = probe_df.melt(id_vars="checkpoint", var_name="Metric", value_name="F1")
    fig = px.bar(
        melted, x="checkpoint", y="F1", color="Metric", barmode="group",
        title="ROUGE F1 — Probe Baseline (Constitution Drift Indicator)",
        color_discrete_sequence=PALETTE, text_auto=".3f",
        labels={"checkpoint": "Checkpoint", "F1": "F1 Score"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    save_fig(fig, "18_rouge_probe_baseline")
else:
    print("No probe-baseline ROUGE data yet.")
```

- [ ] **Step 5: Add cell 8d — GRPO held-out reward bar chart**

```python
# --- 8d: GRPO held-out reward ---
reward_rows = [
    {"checkpoint": r["checkpoint"], "Held-out Reward": r["grpo_held_out_reward"]}
    for r in rouge_reports
    if r.get("grpo_held_out_reward") is not None
]
if reward_rows:
    rdf = pd.DataFrame(reward_rows)
    fig = px.bar(
        rdf, x="checkpoint", y="Held-out Reward",
        title="GRPO Held-Out Reward Score (10% held-out prompts, full reward function)",
        color_discrete_sequence=[PALETTE[2]], text_auto=".4f",
        labels={"checkpoint": "Checkpoint"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    save_fig(fig, "19_grpo_held_out_reward")
else:
    print("No GRPO held-out reward data yet — run GRPO training to generate.")
```

---

## Task 8 — Analysis notebook: Section 9 (Training loss curves)

**Files:**
- Modify: `pipeline/analysis.ipynb`

Continue appending cells after the Section 8 cells added in Task 7.

- [ ] **Step 1: Add Section 9 markdown heading cell**

```markdown
## Section 9 — Training Loss Curves

SFT loss curves loaded from `models/checkpoint_sft*/loss_history.json`.
GRPO reward/loss curves loaded from `models/checkpoint_grpo*/grpo_loss_history.json`.
```

- [ ] **Step 2: Add cell 9a — load loss histories**

```python
# --- 9a: Discover and load loss/reward history files ---
models_dir = Path("models")

sft_histories  = {}
grpo_histories = {}

for p in sorted(models_dir.glob("checkpoint_sft*/loss_history.json")):
    with open(p, encoding="utf-8") as f:
        sft_histories[p.parent.name] = json.load(f)

for p in sorted(models_dir.glob("checkpoint_grpo*/grpo_loss_history.json")):
    with open(p, encoding="utf-8") as f:
        grpo_histories[p.parent.name] = json.load(f)

print(f"SFT checkpoints:  {list(sft_histories.keys())  or ['none found']}")
print(f"GRPO checkpoints: {list(grpo_histories.keys()) or ['none found']}")
```

- [ ] **Step 3: Add cell 9b — SFT train + eval loss line charts**

```python
# --- 9b: SFT loss curves ---
for label, history in sft_histories.items():
    train_rows = [h for h in history if "loss" in h and "eval_loss" not in h]
    eval_rows  = [h for h in history if "eval_loss" in h]

    if not train_rows:
        print(f"  {label}: no training-loss entries found in history")
        continue

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h["step"] for h in train_rows],
        y=[h["loss"]  for h in train_rows],
        mode="lines", name="Train Loss", line_color=PALETTE[0],
    ))
    if eval_rows:
        fig.add_trace(go.Scatter(
            x=[h["step"]      for h in eval_rows],
            y=[h["eval_loss"] for h in eval_rows],
            mode="lines+markers", name="Eval Loss", line_color=PALETTE[1],
        ))
    fig.update_layout(
        title=f"SFT Loss Curves — {label}",
        xaxis_title="Step", yaxis_title="Cross-Entropy Loss",
        legend_title="Split",
    )
    safe = label.replace("/", "_")
    save_fig(fig, f"20_sft_loss_{safe}")
```

- [ ] **Step 4: Add cell 9c — GRPO policy loss + mean reward**

```python
# --- 9c: GRPO training curves ---
for label, history in grpo_histories.items():
    loss_rows   = [h for h in history if "loss" in h]
    reward_rows = [h for h in history if "rewards/mean" in h or "reward" in h]

    if not loss_rows and not reward_rows:
        print(f"  {label}: no usable entries in grpo_loss_history.json")
        continue

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Policy Loss", "Mean Reward"))

    if loss_rows:
        fig.add_trace(go.Scatter(
            x=[h["step"] for h in loss_rows],
            y=[h["loss"] for h in loss_rows],
            mode="lines", name="Policy Loss", line_color=PALETTE[0], showlegend=True,
        ), row=1, col=1)

    if reward_rows:
        rkey = "rewards/mean" if "rewards/mean" in reward_rows[0] else "reward"
        fig.add_trace(go.Scatter(
            x=[h["step"]  for h in reward_rows],
            y=[h[rkey]    for h in reward_rows],
            mode="lines", name="Mean Reward", line_color=PALETTE[2], showlegend=True,
        ), row=1, col=2)

    fig.update_layout(title=f"GRPO Training — {label}", height=420)
    safe = label.replace("/", "_")
    save_fig(fig, f"21_grpo_curves_{safe}")
```

---

## Self-Review

**Spec coverage:**
- ✓ 90/10 SFT split — Task 1
- ✓ 90/10 GRPO split + held-out reward — Tasks 2, 4
- ✓ HuggingFace push (safetensors) — Task 5 `publish()`
- ✓ GGUF export + push — Task 5 `publish()`
- ✓ ROUGE against eval split — Task 4 `_compute_rouge_report()`
- ✓ ROUGE against probe baseline — Task 4 `_compute_rouge_report()`
- ✓ `--gguf` inference server — Task 6
- ✓ Section 8 ROUGE charts — Task 7
- ✓ Section 9 loss curves — Task 8

**Placeholder scan:** No TBDs. All code blocks are complete. All file paths are exact.

**Type consistency:**
- `compute_rouge()` is module-level in Task 3 and called as `compute_rouge(...)` (not `self.compute_rouge`) in Task 4 — consistent.
- `_generate_gguf()` returns `(text, n_in, n_out, elapsed)` — same tuple as `_generate()` — consistent with all call sites in `chat_completions`.
- `self._eval_raw` stored in Task 1, accessed in Task 4 via `getattr(self, "_eval_raw", [])` — safe.
- `self._grpo_eval_dataset` stored in Task 2, accessed in Task 4 via `getattr(self, "_grpo_eval_dataset", None)` — safe.
- `publish()` called from `train_sft()` and `train_grpo()` with same signature — consistent.
