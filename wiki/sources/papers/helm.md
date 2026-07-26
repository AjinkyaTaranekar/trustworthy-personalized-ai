---
title: "Holistic Evaluation of Language Models (HELM)"
type: source
tags: [evaluation]
sources:
  - https://arxiv.org/abs/2211.09110
updated: 2026-07-21
status: current
---

# Holistic Evaluation of Language Models (HELM)

**Language models must be evaluated holistically — densely, across many scenarios and many metrics at once, not accuracy alone — so that trade-offs, harms, and coverage gaps become transparent rather than hidden.**

## Summary

Liang, Bommasani, Lee et al. (Stanford CRFM, 2022; TMLR 2023) argue that evaluation is the map of the field: if the map only measures accuracy on a handful of datasets, the community optimises a narrow proxy and goes blind to fairness, robustness, calibration, toxicity, and efficiency. HELM makes the taxonomy explicit — scenarios as (task, domain, language) triples × seven metrics — and measures densely (98/112 core scenario×metric pairs), lifting core-scenario coverage from 17.9% (before) to 96.0% (after) across 30 models in 4,939 runs. Its lasting value is conceptual: evaluation should be *designed*, its blind spots *documented*, and metrics measured densely rather than cherry-picked. It predates LLM-as-judge, so its scoring is reference-based (exact-match/F1) — the seam later filled by [[sources/papers/prometheus]] and open evaluator LLMs.

## Why it matters here

The natural framing citation for *why* the project's harness evaluates on multiple axes (trust, empathy, personalisation, safety, efficiency) rather than one accuracy number — it legitimises a holistic rubric and treating coverage as a first-class quantity. Its (task × domain × who/when/where) taxonomy maps cleanly onto the project's [[entities/5w-h|5W+H]] personalisation lens.

## Method

- **Scenarios:** (task, domain, language) triples; domain decomposes into *what* (genre), *when* (time period), *who* (demographics).
- **Seven metrics per scenario where feasible:** accuracy, calibration, robustness, fairness, bias, toxicity, efficiency.
- **Dense, standardised measurement:** identical 5-shot prompting across models; automatic reference-based scoring (no LLM-judge).

## Key results

- **Coverage 17.9% → 96.0%** of core scenarios; 98/112 (87.5%) scenario×metric pairs measured.
- **Robustness fragility:** TNLG-530B NarrativeQA 72.6% → 38.9% under perturbation.
- **Fairness disparity:** OPT-175B 1.506 → 2.114 bits/byte for White vs African-American English.
- **Scale threshold:** every head-to-head winner is ≥50B; code models reason better (code-davinci-002 GSM8K 52.1%).
- **Brittleness:** multiple-choice framing swings scores (OPT-175B HellaSwag 79.1% → 30.2% by presentation); perplexity poorly predicts downstream accuracy.

## Critical appraisal

The durable contribution is the taxonomy and the "coverage as a measured quantity" stance; the specific 2022 leaderboard is stale. Admitted limits: the standardisation paradox (uniform prompting may disadvantage models that shine under bespoke adaptation), English-centrism, shallow adaptation (no CoT), and no privacy/copyright/environment metrics.

> ⚠ For this project: HELM's automatic reference-based metrics buy breadth but are weakest on *open-ended* generation — the very axis where empathy and trust live. For substance-based judging of empathy/reasoning quality the harness needs the LLM-as-judge machinery of [[sources/papers/prometheus|Prometheus]]/BiGGen, not HELM's exact-match style. Cite the 2022 numbers as historical context.

## Related

- [[sources/papers/prometheus]] — open evaluator LLM that fills HELM's open-ended-scoring gap
- [[sources/papers/mt-bench]] — LLM-as-judge validation (the operational front-end)
- [[sources/papers/abstention-survey]] — the reliability metrics HELM's taxonomy complements
- [[experiments/human-evaluation-rubric]] — the project's multi-axis rubric HELM legitimises
- [[experiments/frontier-model-comparison]] — multi-metric model comparison design
- [[topics/explainability]] — documenting coverage gaps as a trust practice

## Sources

- Liang, Bommasani, Lee et al. (Stanford CRFM, 2022) — arXiv:2211.09110 (TMLR 2023) — [arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)
