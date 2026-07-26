---
title: "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models"
type: source
tags: [evaluation, llm-as-judge]
sources:
  - https://arxiv.org/abs/2310.08491
updated: 2026-07-21
status: current
---

# Prometheus: Inducing Fine-grained Evaluation Capability in Language Models

**An open-source 13B model, fine-tuned on a purpose-built dataset of fine-grained score rubrics and reference answers, can match GPT-4's ability to grade long-form responses — removing the need for a closed, costly, non-reproducible proprietary judge.**

## Summary

Kim et al. (KAIST / NAVER / UW / MIT, ICLR 2024) reframe judging as a *learnable skill* induced by the materials the judge is given: a customised 1–5 rubric, a reference answer, and the habit of writing feedback before scoring. They build the Feedback Collection (1K rubrics, 100K response+feedback records, GPT-4-generated, length- and score-balanced) and fine-tune Llama-2-Chat-13B to emit chain-of-thought feedback then a score after a `[RESULT]` token. Prometheus-13B reaches Pearson 0.897 with human judges (edging GPT-4's 0.882) in-distribution, and its ablations are the reusable asset: removing the reference answer collapses correlation 0.860→0.642. This is the concrete open alternative to a proprietary GPT-4 judge for the project's substance-based harness.

## Why it matters here

Directly load-bearing: an open evaluator LM the project can cite (and potentially run) to keep the pipeline transparent, reproducible, and free of API dependency — aligned with the on-device, trust-first ethos. The design lessons transfer wholesale to the project's judge: always give it (1) a fine-grained per-instance rubric, (2) a reference answer when one exists, and (3) required written feedback *before* the score — the ablations show each materially raises agreement, exactly the "credit clarify, never rig results" evaluation discipline.

## Method

- **Feedback Collection:** 50 seed rubrics → GPT-4-augmented to 1,000; 20 instructions each; 5 responses+feedback spanning scores 1–5; uniform length + balanced scores to suppress length/decision bias.
- **Model:** Llama-2-Chat-13B fine-tuned to write CoT feedback then score, `[RESULT]`-separated. Supports absolute grading and (empirically) pairwise/reward-model use.

## Key results

- **Human agreement (45 rubrics):** Prometheus 0.897 Pearson vs GPT-4 0.882, GPT-3.5 0.392; feedback preferred over GPT-4 in 58.62% of pairwise cases.
- **Generalises to unseen rubrics:** 0.860 Pearson.
- **OOD weakness (the honest soft spot):** Vicuna Bench 0.466, MT Bench 0.473 correlation with GPT-4.
- **Ablations:** reference answer is the single most load-bearing ingredient (0.860→0.642 without it); removing feedback distillation 0.860→0.668.

## Critical appraisal

Proof-of-concept that "open, cheap, reproducible judge" is achievable, and its ablations are a design spec for any builder. Cautions: the 0.897 headline is *in-distribution* (rubrics stylistically close to training) — the OOD ~0.47 figures are the more honest transfer guide; the hard dependence on a supplied reference answer is a deployment friction for open-ended empathy/trust judging where gold references rarely exist; and GPT-4 generated the training data, so Prometheus partly distils GPT-4's biases while claiming independence.

> ⚠ For a 0.6B on-device judge: a 13B evaluator is not itself an on-device judge — position Prometheus as the *teacher/oracle* whose judgements a smaller student judge could be distilled from. Plan for reference-free rubrics or synthesised references, and validate the judge on the project's *own* distribution (OOD drops to ~0.47).

## Related

- [[sources/papers/mt-bench]] — the GPT-4-judge validation Prometheus offers an open alternative to
- [[sources/papers/helm]] — the holistic-evaluation framing; Prometheus is the open-ended front-end
- [[sources/papers/constitutional-labeling-consistency]] — rubric detail drives inter-judge agreement
- [[experiments/human-evaluation-rubric]] — the project's rubric-based judging
- [[topics/explainability]] — feedback-before-score as an auditable judgement
- [[sources/code/training-and-benchmark]] — where an open judge would run

## Sources

- Kim, Shin, Cho, Jang, Longpre, Lee, Yun, Shin, Kim, Thorne, Seo (2023) — arXiv:2310.08491 (ICLR 2024) — [arxiv.org/abs/2310.08491](https://arxiv.org/abs/2310.08491)
