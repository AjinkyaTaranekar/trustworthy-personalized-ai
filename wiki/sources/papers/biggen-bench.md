---
title: "The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models"
type: source
tags: [evaluation, llm-as-judge]
sources:
  - https://arxiv.org/abs/2406.05761
  - https://github.com/prometheus-eval/prometheus-eval
updated: 2026-07-22
status: current
---

# The BiGGen Bench: Principled Fine-grained Evaluation with Language Models

**Fine-grained, instance-specific rubrics — a bespoke 1–5 rubric written for each individual test instance, not a generic "helpfulness" scale — let LLM judges assess free-form generation across nine capabilities with statistically significant human agreement, and correlate with humans better than either coarse-grained or domain-level criteria.**

## Summary

Kim et al. (Prometheus-eval consortium, NAACL 2025 **Best Paper**) build a top-down generation benchmark — 9 capabilities × 77 tasks × 765 human-curated instances, each carrying its own instance-specific 5-point rubric — and grade 103 response LMs (1B–141B) with 5 evaluator LMs, validated against 3,236 human ratings. The load-bearing result: instance-specific criteria beat coarse (MT-Bench-style) and domain (FLASK-style) criteria, and an open **Prometheus-2-BGB 8×7B** judge matches Claude-3-Opus (0.577 vs 0.578 Pearson) and — with self-consistency — nears GPT-4-1106, so *rubric quality can outweigh judge-model strength*. Best single judge GPT-4-Turbo 0.623; 5-judge majority 0.627; verbosity bias r=0.05 (largely neutralised by detailed rubrics). This is the canonical evidence for the project's substance-based, instance-rubric harness — extending the [[sources/papers/prometheus]] line.

## Why it matters here

Direct support for the substance-based harness: instance-specific fine-grained rubrics produce more human-aligned judge scores than generic criteria — the empirical backbone for evaluating outputs on tailored rubrics rather than single-word/regex checks. The most hopeful finding for a sub-1B ambition is that an open 8×7B judge with a good rubric plus self-consistency rivals GPT-4-class judges. It composes cleanly with [[sources/papers/llm-as-judge-design|Yamauchi]] (both use BiGGenBench): BiGGen supplies the *rubric* half, Yamauchi the *decoding/CoT* half.

## Method

- **Hierarchy:** capabilities (Instruction Following, Grounding, Planning, Reasoning, Refinement, Safety, Theory of Mind, Tool Usage, Multilingualism) → 77 tasks → 765 instances, each `{system message, prompt, reference answer, instance rubric}`.
- **Judge protocol:** direct assessment (single response, 1–5 Likert, feedback-before-score, Prometheus template); 5 evaluator LMs; human-in-the-loop construction (hand-craft → GPT-4-augment → cross-validate → human-judge).

## Key results

- **Judge–human Pearson:** GPT-4-Turbo 0.623 (best single); 5-judge majority 0.627 (best overall); Prometheus-2-BGB 8×7B 0.577 ≈ Claude-3-Opus 0.578, → 0.607 with self-consistency.
- **Rubric ablation:** instance-specific > coarse > domain; an open 8×7B + instance rubric beat GPT-4-Turbo + coarse rubric.
- **Hardest to judge:** Theory of Mind (~0.48) and Tool Usage (~0.53) have the lowest judge–human correlation.
- Verbosity bias r=0.05 (detailed direct-assessment rubrics are largely length-robust).

## Critical appraisal

A landmark, carefully engineered benchmark; the strongest transferable result is the rubric-granularity ablation plus the demonstration that a small open judge + good rubric + self-consistency rivals GPT-4-class judges. Cautions: partial GPT-4 circularity (Step-2 augmentation and the trained judge both lean on GPT-4-1106, so "human agreement" is measured on a partly GPT-4-shaped set); Pearson-only reporting (no rank-level κ); ~85 instances per capability is modest; and instance-specific rubrics are labour-intensive to author — it defines the gold-standard rubric design without solving how to produce such rubrics cheaply at inference time.

> ⚠ 0.6B: the open judge is **8×7B (~13B active MoE), not sub-1B** — a hopeful ceiling, not proof a tiny model can judge well. Expect a sub-1B judge to trade accuracy; lean on self-consistency/juries to recover reliability. ToM/tool-use are where LLM-as-judge is least trustworthy.

## Related

- [[sources/papers/prometheus]] — the evaluator-LM line BiGGen extends (Prometheus-2-BGB is a derivative)
- [[sources/papers/llm-as-judge-design]] — the decoding/CoT half; both use BiGGenBench
- [[sources/papers/mt-bench]] — the coarse-grained baseline instance rubrics beat
- [[sources/papers/helm]] — holistic multi-capability evaluation
- [[experiments/human-evaluation-rubric]] — the project's instance-rubric judging
- [[topics/explainability]] — feedback-before-score, capability profiles

## Sources

- Kim, Suk, Cho, Longpre, et al. (2024/2025) — arXiv:2406.05761 (NAACL 2025 Best Paper) — [arxiv.org/abs/2406.05761](https://arxiv.org/abs/2406.05761)
