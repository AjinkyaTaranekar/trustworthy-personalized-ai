---
title: Automatic Chain of Thought Prompting
type: source
arxiv_id: 2210.03493v1
authors: Zhang, Zhang, Li, Smola (Amazon)
year: 2022
tags: [reasoning, prompting, cot, automation]
sources:
  - docs/Assets/Automatic Chain of Thought Prompting in Large Language Models (2210.03493v1).pdf
  - docs/Literature Notes/Automatic Chain of Thought Prompting in Large Language Models (2210.03493v1).md
updated: 2026-04-19
status: current
---

# Auto-CoT

**Sidesteps hand-crafted CoT exemplars by letting the model generate its own reasoning chains — "let's think step by step" plus a diversity-sampled seed question set. Matches or exceeds manual CoT on 10 benchmarks.**

## What it does
Two stages: (1) cluster questions for diversity, (2) apply zero-shot CoT per cluster, keep as few-shot exemplars. Robustness comes from diversity, not per-example correctness.

## Why it matters for this thesis
Methodological template for the repo's [[entities/constitution|constitution-driven SFT]]: automate the exemplar-creation step, but enforce quality via the constitution/critique loop rather than raw diversity. Connects to the `sft_question_generator.py` / `sft_gold_response_generator.py` design — same "generate → critique" pattern, scaled with explicit principles.

## Related

- [[topics/reasoning]]
- [[sources/papers/chain-of-thought-prompting]] — what Auto-CoT automates
- [[entities/constitution]]

## Sources

- `docs/Assets/Automatic Chain of Thought Prompting in Large Language Models (2210.03493v1).pdf`
- `docs/Literature Notes/Automatic Chain of Thought Prompting in Large Language Models (2210.03493v1).md`
