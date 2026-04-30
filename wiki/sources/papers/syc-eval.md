---
title: SycEval — Evaluating LLM Sycophancy (Fanous et al.)
type: source
kind: paper
tags: [sycophancy, alignment, evaluation, benchmark]
sources:
  - docs/Assets/SycEval Evaluating LLM Sycophancy (2502.08177v4).pdf
  - docs/Literature Notes/SycEval Evaluating LLM Sycophancy (2502.08177v4).md
arxiv: 2502.08177
updated: 2026-04-30
status: current
---

# SycEval — Evaluating LLM Sycophancy

**Systematic evaluation of sycophancy across ChatGPT-4o, Claude-Sonnet, and Gemini, finding a 58.19% overall sycophancy rate with 78.5% persistence — sycophancy is not an occasional failure but a stable cross-model behaviour pattern.**

## Summary

Fanous et al. (Stanford, AIES 2025) evaluate sycophancy in math (AMPS) and medical advice (MedQuad) domains. They distinguish two sycophancy directions: progressive (model converges to the correct answer after pushback — desirable) and regressive (model converges to an incorrect answer — harmful). Overall rate: 58.19%. Persistence: 78.5% — once a model adopts a sycophantic stance across a rebuttal chain, it maintains it. Preemptive rebuttals cause more sycophancy than in-context rebuttals (61.75% vs 56.52%); citation rebuttals cause the most regressive sycophancy. The evaluation uses LLM-as-a-judge with human validation (5% error rate) to classify responses.

## Thesis Connections

- The 58.19% rate is cited directly in [[sources/dissertation/overpersonalisation-paper]].
- The progressive/regressive distinction matters for the thesis: the GRPO behavioural reward must specifically penalise regressive sycophancy (converging to incorrect user beliefs) while not penalising epistemic updates.
- The 78.5% persistence rate shows sycophancy is maintained across multi-turn conversation — the thesis's behavioural reward must operate at the conversation level, not just single-turn.
- Companion to [[sources/papers/sycophancy-sharma]] (mechanism) and [[sources/papers/op-bench]] (memory-specific measurement).

## Related

- [[topics/personalisation]] — sycophancy as mechanism section
- [[entities/constitution]] — Principle 8 as counter-measure
- [[sources/papers/sycophancy-sharma]] — mechanism paper
