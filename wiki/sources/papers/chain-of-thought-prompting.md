---
title: Chain-of-Thought Prompting Elicits Reasoning in LLMs
type: source
arxiv_id: 2201.11903v6
authors: Wei et al.
year: 2022
venue: NeurIPS
tags: [reasoning, prompting, cot]
sources:
  - docs/Assets/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2201.11903v6).pdf
  - docs/Literature Notes/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2201.11903v6).md
updated: 2026-04-19
status: current
---

# Chain-of-Thought Prompting

**Prompting with a few worked examples of step-by-step reasoning unlocks arithmetic, commonsense, and symbolic reasoning in sufficiently large LLMs.**

## What it does
Shows that at scale (~100B+), prepending a handful of "think step by step" demonstrations dramatically lifts GSM8K and similar benchmarks — 540B PaLM with 8 CoT exemplars beats verifier-finetuned GPT-3 on math word problems.

## Why it matters for this thesis
CoT is the **baseline** from which the thesis departs. It delivers readable reasoning traces, but those traces are **correlation, not process**: the model emits a plausible-looking derivation without any guarantee the answer was derived from it. This is the "post-hoc rationalisation" problem central to [[topics/reasoning]]. Later work in this batch ([[sources/papers/tree-of-thoughts]], [[sources/papers/deepseek-r1]], [[sources/papers/seed15-thinking]]) tries to fix that by adding search, RL-trained thought-process rewards, or explicit verification.

## Related

- [[topics/reasoning]]
- [[sources/papers/tree-of-thoughts]]
- [[sources/papers/pal]] — replaces fragile CoT arithmetic with code
- [[sources/papers/react]] — adds acting to reasoning

## Sources

- `docs/Assets/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2201.11903v6).pdf`
- `docs/Literature Notes/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2201.11903v6).md`
