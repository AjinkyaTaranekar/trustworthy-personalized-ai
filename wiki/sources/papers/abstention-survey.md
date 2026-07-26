---
title: "Know Your Limits: A Survey of Abstention in Large Language Models"
type: source
tags: [abstention, evaluation, security]
sources:
  - https://arxiv.org/abs/2407.18418
updated: 2026-07-21
status: current
---

# Know Your Limits: A Survey of Abstention in Large Language Models

**Abstention — the LLM's act of withholding or refusing to answer — should be treated as a first-class, unifying reliability capability analysed jointly from three interacting perspectives (the query, the model's own knowledge, and human values), rather than as scattered task-specific refusal tricks.**

## Summary

Wen et al. (UW / AI2, 2024–2025; TACL) unify hallucination-avoidance, uncertainty, and safety-refusal as one capability — "know your limits". Failures come in two directions: answering when it should abstain (hallucination, unsafe compliance) and abstaining when it should answer (over-refusal). The survey organises methods by a three-perspective framework (query answerability / model knowledge-confidence / human values) crossed with lifecycle stage (pretraining → alignment → inference), and catalogues the benchmarks and 14+ metrics. Its most transferable value: each perspective maps to a distinct constitution-clause type, and the metric set (URUP, ARSP, Reliable Accuracy, Coverage@Acc) directly measures the over-refusal-vs-unsafe-compliance trade-off. Companion to [[sources/papers/abstention-bench]]'s empirical negative result.

## Why it matters here

The three perspectives translate directly into [[entities/constitution|constitution]] clauses — (a) refuse/clarify unanswerable or under-specified queries, (b) express honest uncertainty when confidence is low, (c) decline unsafe requests — grounding the honest-uncertainty pillar in a citable taxonomy. The metrics (ARSP = abstained on safe prompts; URUP = unsafe on unsafe prompts) are a ready scaffold for substance-based judging of *appropriate* abstention vs over-refusal. And LoRA-as-regulariser plus inference-time/prompting abstention are exactly the low-compute levers available on a sub-1B on-device model — the survey licenses these while warning that verbalised confidence is over-confident and safety refusals are fragile at any scale.

## Framework

- **Three perspectives:** query (ambiguity, false premise, unanswerable), model knowledge (calibration, uncertainty, knowledge boundaries), human values (toxicity, harmful requests, jailbreaks).
- **Lifecycle:** pretraining (essentially unexplored — an open direction), alignment (abstention-aware SFT + preference learning; LoRA as regulariser), inference-time (input/in/output processing — ambiguity detection, semantic entropy, verbalised confidence, self-consistency, self-evaluation).
- **Metrics (14+):** Abstention P/R/F1, Coverage, Reliable Accuracy, Effective Reliability, Coverage@Acc, AUROC, plus safety-specific URUP and ARSP.

## Key observations

- Over-abstention is a real cost; helpfulness vs appropriate abstention genuinely conflict.
- Neither probability-based nor verbalised confidence is reliably calibrated (verbalised is over-confident).
- Safety-driven abstention is fragile — persona, low-resource-language, and cipher attacks bypass it.
- Abstention learned via instruction tuning generalises poorly across domains/models, and can amplify demographic bias.

## Critical appraisal

The strongest contribution is conceptual — reframing three isolated problems as one capability gives a shared vocabulary and evaluation surface. It is a map, not a method: it offers no head-to-head recommendation on which abstention technique wins, the categories are non-exclusive (a false-premise query is both query- and knowledge-perspective), the pretraining row is essentially empty, and it under-covers multi-turn/agentic and tool-augmented settings — exactly where on-device assistants operate.

> ⚠ Future directions worth adopting: **partial abstention** (a weighted spectrum, not binary), **explainable abstention** (communicate the uncertainty), and **post-abstention information-seeking** (ask a clarifying question, then continue) — the last aligns directly with the project's clarify-before-assume stance.

## Related

- [[sources/papers/abstention-bench]] — the empirical companion (abstention scale-invariant; RLVR degrades it)
- [[sources/papers/hallucination-survey]] — the answer-when-you-shouldn't failure direction
- [[sources/papers/agent-cq]] — clarify-before-assume as post-abstention information-seeking
- [[sources/papers/nemo-guardrails]] — inference-time refusal (cheap but bypassable) and its ARSP/URUP cost
- [[entities/constitution]] — the three perspectives as clause types
- [[topics/explainability]] — honest, explained "I don't know"
- [[experiments/human-evaluation-rubric]] — URUP/ARSP as judged trade-off metrics

## Sources

- Wen, Yao, Feng, Xu, Tsvetkov, Howe, Wang (2024–2025) — arXiv:2407.18418 (TACL) — [arxiv.org/abs/2407.18418](https://arxiv.org/abs/2407.18418)
