---
title: "AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search with LLMs"
type: source
tags: [evaluation, personalisation]
sources:
  - https://arxiv.org/abs/2410.19692
updated: 2026-07-20
status: current
---

# AGENT-CQ: Generating & Evaluating Clarifying Questions with LLMs

**An end-to-end LLM framework that both generates diverse clarifying questions (plus simulated user answers) and evaluates them at scale via CrowdLLM — an ensemble of LLM-as-a-judge instances at varying temperatures standing in for a crowd — showing LLM-generated clarifying questions (especially the temperature-variation method) outperform human-written ones on most quality dimensions and top-rank retrieval, with the explicit aim of training smaller models to clarify before assuming.**

## Summary

Siro et al. (Amsterdam / Copenhagen, 2024; TOIS) tackle the clarify-before-assume loop: underspecified queries are the failure mode of conversational search, and the fix is to ask before answering — but hand-curated clarifying questions don't scale. AGENT-CQ generates clarifying questions (facet-based, and temperature-variation "GPT-Temp"), simulates parameterised user answers, and scores everything with CrowdLLM (three GPT-4o instances at temps 0.2/0.5/0.7) over seven question metrics. On ClariQ, GPT-Temp beats human questions on usefulness (8.4 vs 4.2) and top-rank retrieval (BERT NDCG@1 0.312), and crucially the paper decomposes clarifying-question quality: usefulness/clarification/clarity/on-topic drive perceived quality (τ up to 0.80) while complexity is negligible (τ 0.07). Its stated goal — synthetic clarifying-question data to train smaller models — is unusually on-point for a sub-1B constitutional-harness thesis.

## Why it matters here

The clarify-before-assume evidence base: clarifying questions can be generated, quality-scored on concrete dimensions, and the behaviour is teachable to smaller models (its explicit motivation) — directly supporting the constitution principle to clarify before assuming and the project's credit-clarification stance. The seven question-quality metrics are a drop-in rubric for judging a 0.6B model's clarifying questions (reward usefulness/clarification/clarity; ignore complexity), and CrowdLLM is a citable multi-instance, multi-dimensional, human-validated LLM-judge template — with its self-preference bias (53% naturalness agreement) as the caution reinforcing substance-based, non-rigged evaluation.

## Method

- **Generation:** facet-based (GPT-3.5 facets → Llama-3.1-8B question per facet) and temperature-variation (GPT-Temp, τ ramped 0.5→0.9); filter by `S = 0.4·relevance + 0.6·clarification-potential`, keep top-10.
- **User simulation:** parameterised by verbosity (10–60 tokens) and reveal probability (0.0–0.9).
- **CrowdLLM evaluation:** 3 GPT-4o judges (conservative/balanced/creative), seven question metrics + four answer metrics, human-validated on MTurk. Dataset: ClariQ (198 topics, 891 facets, 8k+ questions).

## Key results

- **Quality drivers (Kendall τ vs overall):** usefulness 0.80, clarification 0.76, clarity 0.75, on-topic 0.71; **complexity 0.07 (negligible)**.
- **GPT-Temp > human:** usefulness 8.4 vs 4.2 (p<0.001); retrieval BERT NDCG@1 0.312 (best), BM25 NDCG@1 0.225 vs human 0.201 (humans still win NDCG@5/@10 on BM25 via term overlap).
- **CrowdLLM–human agreement:** overall quality 75%, relevance 73%, usefulness 68%, **naturalness only 53%** (self-preference bias).

## Critical appraisal

Operationalises clarifying-question quality into seven measurable dimensions and demonstrates a scalable, human-validated LLM judge — exactly the evaluation pattern a small-model harness needs. Cautions: CrowdLLM is three instances of *one* base model (GPT-4o), so "crowd" diversity is only temperature-deep and cannot escape shared biases (the 53% naturalness tell); judging LLM output with an LLM that prefers LLM output partly inflates the "LLM > human" headline — the retrieval numbers (where humans still win NDCG@5/@10) are the more neutral evidence. Single-turn only; ClariQ may under-represent real query diversity.

## Related

- [[sources/papers/rpeval]] — rational preference use; clarifying vs over-personalising
- [[sources/papers/mt-bench]] — LLM-as-judge design and biases (CrowdLLM is a variant)
- [[entities/5w-h]] — clarifying questions as intent elicitation
- [[topics/personalisation]] — clarify-before-assume as a trust behaviour
- [[topics/tool-use-and-verification]] — ask-the-human vs call-more-tools
- [[experiments/human-evaluation-rubric]] — the seven-metric rubric as a judged scaffold

## Sources

- Siro, Yuan, Aliannejadi, de Rijke (2024) — arXiv:2410.19692 (ACM TOIS) — [arxiv.org/abs/2410.19692](https://arxiv.org/abs/2410.19692)
