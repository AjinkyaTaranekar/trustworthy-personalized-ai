---
title: "None of the Others: Reasoning vs Memorisation in MCQ Benchmarks"
type: source
arxiv_id: 2502.12896v5
authors: Sánchez Salido, Gonzalo, Marco (UNED)
year: 2025
tags: [evaluation, reasoning, memorisation]
sources:
  - docs/Assets/None of the Others a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks (2502.12896v5).pdf
  - docs/Literature Notes/None of the Others a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks (2502.12896v5).md
updated: 2026-04-19
status: current
---

# None of the Others

**A general multiple-choice variation that replaces every option with "none of the others" to dissociate the correct answer from previously seen tokens — forcing models to *understand*, not recall. All tested models drop 10–93% (avg 57% on MMLU).**

## What it does
Simple, general technique that works on any MCQ benchmark. Identifies that the "best" model by raw benchmarks (o3-mini) is not the most robust under this test (DeepSeek-R1-70B).

## Why it matters for this thesis
A **concrete operationalisation** of the "true reasoning vs pattern matching" question in the dissertation's §3.1. Directly usable in the repo's benchmark: apply the variation to the existing eval set to distinguish Conditions A/B/C/D by reasoning robustness, not just correctness. Also supports the thesis's framing that standard benchmarks over-reward memorisation.

## Related

- [[topics/reasoning]]
- [[sources/papers/prompting-science-report-2]]
- [[decisions/2025-11-10-ontology-focus-shift]] — verifiable reasoning is the shift

## Sources

- `docs/Assets/None of the Others a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks (2502.12896v5).pdf`
- `docs/Literature Notes/None of the Others a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks (2502.12896v5).md`
