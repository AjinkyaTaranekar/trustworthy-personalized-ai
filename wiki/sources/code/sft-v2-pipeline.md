---
title: SFT v2/v3 Pipeline (constitution-driven, domain-unbounded)
type: source
kind: code
tags: [code, sft, pipeline, constitution, security, tool-use]
sources:
  - pipeline/sft_question_generator.py
  - pipeline/sft_gold_response_generator.py
  - pipeline/sft_math_pipeline.py
  - pipeline/sft_dataset_assembler.py
  - pipeline/transform_sft_tool_format.py
  - pipeline/sft_add_robustness_variants.py
  - pipeline/sft_add_native_tool_examples.py
  - pipeline/constitution.md
  - README.md
updated: 2026-05-12
status: current
---

# SFT v2/v3 Pipeline

**Replaces the v1 42-template scenario approach with a constitution-driven, LLM-teacher system. Two parallel tracks (Part A behavioural, Part B verifiable-math) feed into a merged train set. Post-assembly, three transformation scripts convert the dataset to the multi-turn tool format expected by the inference server (`train_sft_v3_robust.jsonl`).**

## Flow

```
Part A (behavioural)                         Part B (math)
─────────────────────────────────────        ──────────────────────────────
sft_question_generator.py                    sft_math_pipeline.py
  13 categories, ~2,000 Qs                     GSM8K + MATH datasets
        │                                       code executed locally
        ▼                                       answer verified vs expected
sft_gold_response_generator.py                        │
  draft → rule_check_response() [Blocker 2]            │
  → LLM critique (23 principles)                       │
  → _merge_violations() [Blocker 2]                    │
  → revise on merged violations                        │
        │                                              │
        └────────────────┬─────────────────────────────┘
                         ▼
                sft_dataset_assembler.py
               • filter: CAPABILITY_CHECK present (v3-aware)
               • dedupe · category balance · 90/10 split
                         ▼
               train_sft_v2.jsonl  (1,450 — single-turn tool calls)
                         │
                         ▼
          transform_sft_tool_format.py      ← re-executes python_execute code
               • split <tool>+<answer> → multi-turn [assistant→tool→assistant]
               • drops 34 malformed examples
                         ▼
               train_sft_v3.jsonl  (1,416)
                         │
           ┌─────────────┴──────────────────┐
           ▼                                ▼
  sft_add_native_tool_examples.py    sft_add_robustness_variants.py
    20% of tool examples →              15% minimal prompt variants
    native JSON tool_calls format       10% brief prompt variants
    (uses apply_chat_template           5%  no-principles variants
     tools= at training time)
           │                                │
           └─────────────┬──────────────────┘
                         ▼
          train_sft_v3_with_native.jsonl (1,549)
                         ▼
          sft_add_robustness_variants.py
                         ▼
          train_sft_v3_robust.jsonl  (1,983 — final training set)
```

## Categories

**Part A (13 behavioural — updated 2026-05-06):** user-context, real-time, impossible tasks, subjective tradeoffs, adversarial pressure, knowledge boundary, multi-step clarification, ambiguous requests, entity facts requiring web search, verbose-context behavioural (paragraph-length user input), multi-turn conversation (3–5 turn scaffolds), **appraisal-empathy** (loaded from offline AppraisePLM labels — no LLM generation), **interleaved-tool-reasoning** (questions requiring web_search → python_execute chains; trains P23).

**Part B (7 math):** arithmetic, algebra, geometry, statistics, unit conversions, word problems, no-tool control.

## Security hardening in this pipeline (2026-05-02)

**Blocker 1** (code sandbox): `sft_rejection_sampler.py` and `sft_math_question_generator.py` now AST-validate LLM-generated verification code before `subprocess.run`. Blocked: `os`, `sys`, `subprocess`, `socket`, `requests`, `eval`, `exec`, `open`. Allowed: `math`, `statistics`, `decimal`, `fractions`, and other pure-computation stdlib modules.

**Blocker 2** (independent verifier): `sft_gold_response_generator.py` now runs `rule_check_response()` on every draft before the LLM critique. This deterministic check covers P1 (CAPABILITY_CHECK present), P3 (no hallucinated tools), P4 (math without code flagged), P14 (adversarial capitulation detected), P18 (answer tag present). `_merge_violations()` ensures rule violations survive even if the LLM critic returns `NO_VIOLATIONS`. A warning is printed if `--critic_model` is not set (self-critique SPOF mode).

## Tool surface taught to the model

| Tool | Purpose |
| ---- | ------- |
| `python_execute` | Precision arithmetic |
| `web_search` | Real-time / entity facts |
| `read_url` | Follow up a search hit |
| `get_datetime` | Time-aware responses |

## Question generator diversity and dedup (2026-05-04)

Each batch call in `sft_question_generator.py` now receives two additional constraints that address Western/US bias and cross-batch repetition.

**Axis rotation:** A `DIVERSITY_AXES` list of 20 geographic/cultural/demographic slots (South Asia, East Africa, Southeast Asia, Latin America, Middle East, East Asia, West Africa, Eastern Europe, North Africa, etc.) is cycled sequentially across batches. Each batch prompt mandates ≥60% of questions reflect the assigned region, cultural background, and demographic. Country-specific details are explicitly required: local currencies, financial instruments (chit funds, stokvel, M-Pesa, halal finance), healthcare systems, social norms, and naming conventions.

**Dedup injection:** Already-generated question strings for the current category are tracked in memory and injected into subsequent batch prompts. Single-turn categories: last 30 questions shown verbatim. Verbose/multi-turn: last 10, truncated to 100 chars each (to preserve token budget). The model is instructed to avoid repeating or paraphrasing any listed question.

**Temperature:** Set explicitly to `0.9` for all generation calls (previously relied on provider default, which was often low and contributed to near-identical batches).

## LiteLLM design point

All v2 scripts use [`litellm`](https://github.com/BerriAI/litellm) — the `--model` arg swaps providers without code changes. Confirmed working providers:

| Provider | Model string | Key env var | Notes |
|---|---|---|---|
| **NVIDIA NIM** | `nvidia_nim/moonshotai/kimi-k2.6` | `NVIDIA_NIM_API_KEY` | ✅ confirmed; free tier; 1T-param MoE VLM; **default for math pipeline** |
| **NVIDIA NIM** | `nvidia_nim/minimaxai/minimax-m2.7` | `NVIDIA_NIM_API_KEY` | ✅ confirmed; free tier; used as independent critic |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` | Free tier; ~5,100 calls for full run |
| Groq | `groq/gemma2-9b-it` | `GROQ_API_KEY` | Good independent critic (different family) |
| Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | ~$10–15 for 1,500 examples |
| Ollama | `ollama/llama3.2` | `OLLAMA_API_BASE` | Fully local; no key needed |

Recommended setup: NVIDIA NIM Kimi K2.6 as generator + Minimax M2.7 as independent critic — both frontier models, both free, different architectures (genuine critic independence). Also used as comparison models in [[experiments/frontier-model-comparison]].

## Math pipeline configuration (updated 2026-05-07)

Part B uses `sft_math_question_generator.py`. Dataset source switched to **EleutherAI/hendrycks_math** (replaces the original MATH dataset source). Default model is now **`nvidia_nim/moonshotai/kimi-k2.6`** — the same provider used for Part A, simplifying the `.env` setup to a single API key. Previous default was OpenAI-compatible endpoint; Kimi K2.6 is confirmed working on NVIDIA NIM free tier.

## Scoring in rejection sampling (Part B)

- `+1` code executes and answer matches expected
- ` 0` no code (mental approximation)
- `-1` code fails or wrong answer
- For `no_tool_control` questions, +1 is reserved for honest refusal.

## Dataset composition — train_sft_v3_robust.jsonl (1,983 examples)

| Slice | Count | % | Purpose |
|---|---|---|---|
| XML multi-turn tool | 679 | 34.2 | Trained tool-call behaviour (4 known tools) |
| Native JSON tool | 133 | 6.7 | Qwen3 pre-training generalisation — new tools without retraining |
| No-tool | 737 | 37.2 | Constitution reasoning, refusals, empathy |
| Robustness variants | 434 | 21.9 | Minimal/brief system prompts — trains intrinsic behaviour |

## Tool format: v2 vs v3

**v2 (single-turn, historically broken):** The LLM teacher produced `<tool>name(args)</tool><answer>…</answer>` in one assistant message. Earlier inference loop logic checked for `<answer>` before parsing tools, so tool calls were skipped and answers were hallucinated. The server now parses tool calls before the answer check, so single-turn outputs execute, but v3 remains preferred for clearer turn structure and alignment with tool-result envelopes.

**v3 (multi-turn, correct):** Each tool call occupies its own `[assistant]→[tool]→[assistant]` turn triplet. `python_execute` results are real stdout from local re-execution. The `[TOOL_RESULT: name]…[/TOOL_RESULT]` envelope matches what `_sanitise_tool_output()` in the inference server injects at runtime.

**Server enforcement:** when a tool call is present, the server strips any `<answer>…</answer>` wrapper in that same assistant turn (preserving the tool call) so the tool result always appears in its own turn before the final answer.

## System prompt consistency (2026-05-12)

All three contexts now use the same format — critical for the model to activate constitution behaviour at inference without re-instruction:

| Context | Prompt | 23 principles listed |
|---|---|---|
| Part A training | `TRAINING_SYSTEM_PROMPT_TEMPLATE` in `sft_gold_response_generator.py` | ✓ |
| Inference | `_system_prompt_for_profile()` in `3_infererence.py` | ✓ (synced 2026-05-12) |
| Part B training (math) | Short minimal prompt | ✗ (by design — math examples teach tool execution format, not constitution reasoning) |
| Robustness variants | Minimal/brief/no-principles | ✗ (by design — trains intrinsic behaviour) |

The `_CONSTITUTION` constant in `3_infererence.py` is a verbatim copy of the 23-principle block from `TRAINING_SYSTEM_PROMPT_TEMPLATE`. Both must be updated together when principles change.

## Native JSON tool calling (2026-05-12)

Qwen3-0.6B has native function-calling capability from pre-training (0.880 score, tied #1 in tool-calling benchmark). When the inference server is called with `tool_mode="native"`, it passes `tools=[…]` (OpenAI schemas) to `apply_chat_template` — the model reads the schema and calls any described tool without SFT examples for that specific tool. Training examples with `metadata.native_tools` are rendered with the same `apply_chat_template(tools=…)` call, keeping training and inference text identical.

**Adding a new tool (no retraining):** register in `3_infererence.py` → add schema to `sft_add_native_tool_examples.py:TOOL_SCHEMAS` → call endpoint with `tool_mode="native"`.

## Related

- [[sources/code/constitution-document]] — the 23 principles critiqued against
- [[entities/constitution]] — entity-level summary
- [[sources/code/training-and-benchmark]] — the training stage that consumes this output
- [[topics/tool-use-and-verification]] — PAL/ReAct delegation; native vs custom tool formats
- [[sources/papers/auto-cot]] — methodological precedent for automated exemplars
- [[sources/papers/deepseek-r1]] — overall SFT→RL recipe

## Raw

- `pipeline/sft_question_generator.py`
- `pipeline/sft_gold_response_generator.py`
- `pipeline/sft_math_pipeline.py`
- `pipeline/sft_dataset_assembler.py`
- `pipeline/transform_sft_tool_format.py`
- `pipeline/sft_add_robustness_variants.py`
- `pipeline/sft_add_native_tool_examples.py`
- `pipeline/constitution.md`
- `README.md` §"SFT v2/v3 Pipeline"
