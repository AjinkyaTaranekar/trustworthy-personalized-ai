---
title: "Dual-Head Reasoning Distillation (DHRD)"
type: source
arxiv_id: 2509.21487v2
authors: Xu et al.
year: 2025
venue: NeurIPS 2025 Efficient Reasoning Workshop
tags: [reasoning, distillation, classifier, inference-efficiency]
sources:
  - docs/Assets/Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2).pdf
  - docs/Literature Notes/Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2).md
updated: 2026-04-19
status: current
---

# DHRD — Dual-Head Reasoning Distillation

**Adds a second "reasoning head" supervised by teacher rationales at
train-time only; at inference the reasoning head is disabled and a pooled
classifier head produces the answer — CoT-level accuracy at 96–142× the
throughput.**

## What it does
On seven SuperGLUE tasks, DHRD gains 0.65–5.47% over pooled baselines
while matching CoT-throughput of a simple classifier. Train-time reasoning
is treated as a *regulariser* of the latent features, not an inference-time
computation.

## Why it matters for this thesis
Compelling template for **empathy**: appraisal detection in
[[topics/empathy]] is a classification task, and runtime latency is a
usability gate. DHRD suggests the appraisal tagger can absorb CoT-level
reasoning *at training time* without paying the per-request cost that
[[sources/papers/token-hungry-deepseek-r1|Token-Hungry]] documents.

## Related

- [[topics/reasoning]] · [[topics/empathy]]
- [[sources/papers/token-hungry-deepseek-r1]]
- [[sources/papers/prompting-science-report-2]]

## Sources

- `docs/Assets/Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2).pdf`
- `docs/Literature Notes/Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2).md`
