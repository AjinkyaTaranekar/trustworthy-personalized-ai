---
title: SFT v2 Pipeline (constitution-driven, domain-unbounded)
type: source
kind: code
tags: [code, sft, pipeline, constitution, security]
sources:
  - pipeline/sft_question_generator.py
  - pipeline/sft_gold_response_generator.py
  - pipeline/sft_math_question_generator.py
  - pipeline/sft_rejection_sampler.py
  - pipeline/sft_dataset_assembler.py
  - pipeline/constitution.md
  - README.md
updated: 2026-05-04
status: current
---

# SFT v2 Pipeline

**Replaces the v1 42-template scenario approach with a constitution-driven, LLM-teacher system. Two parallel tracks (Part A behavioural, Part B verifiable-math) feed into a merged train set.**

## Flow

```
Part A (behavioural)                         Part B (math)
─────────────────────────────────────        ──────────────────────────────
sft_question_generator.py                    sft_math_question_generator.py
  11 categories, ~1,700 Qs                     7 types, ~1,050 Qs
        │                                              │
        ▼                                              ▼
sft_gold_response_generator.py               sft_rejection_sampler.py
  draft → rule_check_response() [Blocker 2]    N candidates → score
  → LLM critique (19 principles)               keep +1 (code + correct)
  → _merge_violations() [Blocker 2]            AST code sandbox [Blocker 1]
  → revise on merged violations
        │                                              │
        └────────────────┬─────────────────────────────┘
                         ▼
                sft_dataset_assembler.py
               • filter: CAPABILITY_CHECK present
               • dedupe · category balance · split
                         ▼
               train_interleaved.jsonl (~2,700)
               eval_interleaved.jsonl  (5%)
```

## Categories

**Part A (11 behavioural — updated 2026-05-01):** user-context, real-time, impossible tasks, subjective tradeoffs, adversarial pressure, knowledge boundary, multi-step clarification, ambiguous requests, entity facts requiring web search, **verbose-context behavioural** (paragraph-length user input), **multi-turn conversation** (3–5 turn scaffolds).

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
| **NVIDIA NIM** | `nvidia_nim/moonshotai/kimi-k2.6` | `NVIDIA_NIM_API_KEY` | ✅ confirmed; free tier; 1T-param MoE VLM |
| **NVIDIA NIM** | `nvidia_nim/minimaxai/minimax-m2.7` | `NVIDIA_NIM_API_KEY` | ✅ confirmed; free tier; used as independent critic |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` | Free tier; ~5,100 calls for full run |
| Groq | `groq/gemma2-9b-it` | `GROQ_API_KEY` | Good independent critic (different family) |
| Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | ~$10–15 for 1,500 examples |
| Ollama | `ollama/llama3.2` | `OLLAMA_API_BASE` | Fully local; no key needed |

Recommended setup: NVIDIA NIM Kimi K2.6 as generator + Minimax M2.7 as independent critic — both frontier models, both free, different architectures (genuine critic independence). Also used as comparison models in [[experiments/frontier-model-comparison]].

## Scoring in rejection sampling (Part B)

- `+1` code executes and answer matches expected
- ` 0` no code (mental approximation)
- `-1` code fails or wrong answer
- For `no_tool_control` questions, +1 is reserved for honest refusal.

## Related

- [[sources/code/constitution-document]] — the 19 principles critiqued against
- [[entities/constitution]] — entity-level summary
- [[sources/code/training-and-benchmark]] — the training stage that consumes this output
- [[sources/papers/auto-cot]] — methodological precedent for automated exemplars
- [[sources/papers/deepseek-r1]] — overall SFT→RL recipe

## Raw

- `pipeline/sft_question_generator.py`
- `pipeline/sft_gold_response_generator.py`
- `pipeline/sft_math_question_generator.py`
- `pipeline/sft_rejection_sampler.py`
- `pipeline/sft_dataset_assembler.py`
- `pipeline/constitution.md`
- `README.md` §"SFT v2 Pipeline"
