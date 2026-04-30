---
title: OP-Bench — Over-Personalisation Benchmark
type: source
kind: paper
tags: [personalisation, over-personalisation, evaluation, benchmark, sycophancy]
sources:
  - docs/Assets/OP-Bench Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents (2601.13722v1).pdf
  - docs/Literature Notes/OP-Bench Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents (2601.13722v1).md
arxiv: 2601.13722
updated: 2026-04-30
status: current
---

# OP-Bench — Over-Personalisation Benchmark

**The first benchmark explicitly targeting over-personalisation in memory-augmented conversational agents, showing all tested memory methods cause 26–61% performance degradation and identifying "memory hijacking" as the root cause.**

## Summary

OP-Bench (Hu et al., Harbin Institute of Technology, January 2026) formalises over-personalisation into three types: **Irrelevance** (injecting off-topic memory), **Sycophancy** (over-accommodating user beliefs/memories even when incorrect), and **Repetition** (producing near-identical responses to semantically distinct queries). The benchmark contains 1,700 human-verified instances across 20 users, constructed from LoCoMo long-horizon dialogues. Evaluated across 4 LLMs (GPT-4o-mini, Gemini-2.5-Flash, Qwen3-32B, Qwen3-8B) with 6 memory methods (BASE, RAG, LDAgent, Mem0, MemU, MEMOS). The key finding: all memory-augmentation methods suffer severe over-personalisation. Performance drops 26.2–61.1% relative to the memory-free BASE setting; more sophisticated memory systems (MemU, MEMOS) are consistently worse than simple RAG. Attention analysis reveals memories receive >2× the attention weight of the user's query — "memory hijacking" — even when retrieved memories are semantically irrelevant. The proposed mitigation, **Self-ReCheck**, is a lightweight, model-agnostic memory filter that reduces over-personalisation by 29% while preserving personalisation performance.

## Key Results

| Model | Best memory method | Over-personalisation score | Degradation vs BASE |
|---|---|---|---|
| GPT-4o-mini BASE | — | 83.10 | 0% |
| GPT-4o-mini worst | MemU | 40.46 | ↓51.3% |
| Qwen3-32B BASE | — | 72.91 | 0% |
| Qwen3-32B worst | LDAgent | 33.39 | ↓54.2% |

## Thesis Connections

- Primary empirical anchor for the over-personalisation failure modes in [[sources/dissertation/overpersonalisation-paper]].
- The three-type taxonomy (Irrelevance, Sycophancy, Repetition) directly maps to the thesis's three failure modes: intent override, context inflation, opacity.
- Self-ReCheck mitigation motivates the thesis's selective memory injection design; the attention-attribution finding is the mechanistic explanation.
- Evaluation methodology (paired with LoCoMo) is a model for the thesis's Experiment 4 design.

## Related

- [[topics/personalisation]] — over-personalisation section
- [[sources/papers/rpeval]] — companion benchmark on rational personalisation
- [[sources/dissertation/overpersonalisation-paper]] — LLNCS paper citing this as primary benchmark

## Questions Opened

- Does Self-ReCheck compose with the thesis's 5W+H user model structure, or does it require flat memory lists?
- The benchmark does not evaluate local/on-device memory systems — gap for the thesis's privacy-first argument.
