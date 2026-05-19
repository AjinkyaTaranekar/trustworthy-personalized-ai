---
  Unified Pipeline — End to End

  Two data sources feed the assembler:
    Part A  Behavioural/conversational examples (sft_v3_generator.py → train_v3.jsonl)
    Part B  Verified math examples            (sft_math_pipeline.py → train_partB.jsonl)

  ---
  Step 1a — Generate Behavioural Questions

  python sft_question_generator.py \
      --count 200 --type all \
      --output data/questions_v3.jsonl
  What happens: Produces a JSONL of diverse questions across all categories including the two
  negative trajectory types (inventory_constraint, environment_timeout).

  ---
  Step 1b — Generate Math Training Data (Part B)

  python sft_math_pipeline.py
  What happens: Loads verified Q+A from GSM8K and MATH datasets, has the teacher LLM write
  Python code to solve each question, executes it and checks against the trusted answer.
  Outputs data/train_partB.jsonl (794 examples, no hallucinated answers).

  Optional flags:
    --gsm8k_count 300 --math_count 700   custom split
    --math_max_level 2                   easier questions only
    --smoke                              5 questions quick test
    --resume                             skip already-completed questions

  ---
  Step 2 — Generate Behavioural Gold Responses (Teacher → Student Distillation)

  python sft_v3_generator.py \
      --questions data/questions_v3.jsonl \
      --output data/train_v3.jsonl \
      --model nvidia_nim/moonshotai/kimi-k2.6
  What happens per question:
  1. Teacher (Kimi/Minimax) generates with full 25-principle constitution — flowing narrative
     <think>, no checklists
  2. When the model emits <tool>, generation halts (stop=["</tool>"])
  3. Tool executes live — exa.ai for web search, subprocess for python
  4. Real [TOOL_RESULT] appended, generation resumes
  5. Before saving to JSONL, teacher system prompt swapped → ≤50-word student prompt

  Run again for negative trajectories:
  python sft_v3_generator.py \
      --questions data/questions_v3.jsonl \
      --type inventory_constraint \
      --output data/train_v3_negative.jsonl

  ---
  Step 3 — Validate Before Assembly

  python validate_sft_data.py --input data/train_v3.jsonl
  What happens: Checks every row against 5 invariants (system prompt length, think block
  length, banned placeholders, tool sequence integrity, final answer tag). Exits with
  error if >5% fail. Run with --fix to drop bad rows and continue.

  ---
  Step 4 — Assemble Dataset

  python sft_dataset_assembler.py \
      --part_a data/train_v3.jsonl \
      --part_b data/train_partB.jsonl \
      --output_dir data/
  What happens: Loads both parts, quality-filters, deduplicates, balances categories,
  splits train/eval (90/10), adds robustness variants (minimal/brief/no_principles).
  Outputs data/train_sft_v3.jsonl.

  Defaults: --part_a already defaults to train_v3.jsonl; --part_b to train_partB.jsonl.
  So the above command is equivalent to: python sft_dataset_assembler.py

  Note: CAPABILITY_CHECK filtering is OFF by default (v3 data uses narrative think blocks).
  For legacy v2 data add --capability_check to re-enable the structured-think filter.

  ---
  Step 5 — Curriculum SFT Training (3 separate runs)

  # Stage 1: short no-tool examples → teaches <think>...<answer> syntax
  python 2_model_trainer.py --mode sft \
      --curriculum_stage 1 --output_name checkpoint_sft_s1

  # Stage 2: all examples → complex multi-tool reasoning
  python 2_model_trainer.py --mode sft \
      --curriculum_stage 2 \
      --from_checkpoint models/checkpoint_sft_s1 \
      --output_name checkpoint_sft_s2

  # Stage 3: all + 20% stage-1 replay → prevents anti-drift
  python 2_model_trainer.py --mode sft \
      --curriculum_stage 3 \
      --from_checkpoint models/checkpoint_sft_s2 \
      --output_name checkpoint_sft
  What happens each stage: Loads base/prior checkpoint, filters data to the right subset,
  trains with Unsloth 4-bit LoRA, saves checkpoint.

  ---
  Step 6 — GRPO Reinforcement

  python 2_model_trainer.py --mode grpo \
      --sft_checkpoint models/checkpoint_sft \
      --v3_format
  What --v3_format does: Disables the CAPABILITY_CHECK requirement in the format reward —
  v3 models produce narrative think blocks, not structured checklists.

  ---
  Step 7 — Serve and Benchmark

  python 3_infererence.py --model_dir models/checkpoint_grpo
  python 4_benchmark.py --server_url http://localhost:8000

  ---
  Script Reference

  | Script                    | Role                              | Step |
  |---------------------------|-----------------------------------|------|
  | sft_question_generator.py | Generate behavioural questions    | 1a   |
  | sft_math_pipeline.py      | Generate verified math data       | 1b   |
  | sft_v3_generator.py       | Teacher→student distillation      | 2    |
  | validate_sft_data.py      | Quality gate (5 invariants)       | 3    |
  | sft_dataset_assembler.py  | Filter, dedupe, augment, split    | 4    |
  | 2_model_trainer.py        | SFT curriculum + GRPO             | 5-6  |
  | 3_infererence.py          | FastAPI inference server          | 7    |
  | 4_benchmark.py            | Constitutional benchmark client   | 7    |
  | 5_context_degradation.py  | Ablation: context window stress   | opt  |
  | config.py                 | Shared pipeline configuration     | all  |
  | pipeline_tools.py         | Tool registry (python/web/etc.)   | all  |
  | constitutional_harness.py | Real-time constitution scoring    | all  |
  | scratchpad.py             | Session scratchpad store          | all  |
  | user_memory.py            | Persistent user memory store      | all  |
  | empathy.py                | Appraisal analysis module         | all  |
  | appraisal_labeller.py     | Automated appraisal labelling     | all  |
  | ontology_verifier.py      | Ontology-grounded response check  | all  |
  | user_modelling.py         | GraphRAG user model               | all  |
  | watch_and_commit.py       | Auto-checkpoint watcher           | util |
  | experiment0_reasoning_comparison.py | Pre-training baseline  | done |

  ---
  Data Files

  | File                        | Producer              | Consumer             |
  |-----------------------------|-----------------------|----------------------|
  | data/questions_v3.jsonl     | sft_question_generator| sft_v3_generator     |
  | data/train_v3.jsonl         | sft_v3_generator      | sft_dataset_assembler|
  | data/train_partB.jsonl      | sft_math_pipeline     | sft_dataset_assembler|
  | data/train_sft_v3.jsonl | sft_dataset_assembler | 2_model_trainer  |
