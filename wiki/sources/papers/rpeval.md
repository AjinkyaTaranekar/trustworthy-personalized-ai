---
title: RPEval — Rational Preference Utilisation Benchmark
type: source
kind: paper
tags: [personalisation, over-personalisation, evaluation, benchmark, reasoning]
sources:
  - docs/Assets/How Does Personalized Memory Shape LLM Behavior Benchmarking Rational Preference Utilization in Personalized Assistants (2601.16621v1).pdf
  - docs/Literature Notes/How Does Personalized Memory Shape LLM Behavior Benchmarking Rational Preference Utilization in Personalized Assistants (2601.16621v1).md
arxiv: 2601.16621
updated: 2026-04-30
status: current
---

# RPEval — Rational Preference Utilisation Benchmark

**Introduces the Rational Personalisation (L2) framework — treating memory as pragmatic intent inference — and shows a 40–90% accuracy gap between LLMs and humans, with irrational personalisation worsening as model capability increases.**

## Summary

RPEval (Feng et al., Renmin University / Huawei, January 2026) proposes a three-level taxonomy of personalised assistants: L0 (non-personalised), L1 (literal: directly concatenates memory), L2 (pragmatic: infers intent, decides whether memory is applicable). L1 — the dominant design in commercial systems — is the source of over-personalisation: it mechanically injects memory regardless of relevance. L2 models memory use as posterior Bayesian inference (Rational Speech Acts theory). The benchmark RPEval contains 953 gold-standard samples with 91.86% human inter-annotator agreement, covering diverse preference-query pairs. Results: 40–90% accuracy gap between LLMs and humans on rational personalisation; the gap grows with model capability (inverse scaling). The proposed RP-Reasoner reduces error severity ~26% and resolves ~80% of bad cases in a large-scale commercial PA.

## Thesis Connections

- Provides the theoretical justification (RSA-grounded L2 design) for the thesis's selective memory injection approach.
- The inverse scaling finding — bigger models over-personalise more — reframes the problem: it is not a capability gap but a training-objective mismatch.
- Error taxonomy (Filter Bubble, RII, UPB, LF, VB) maps onto the LLNCS paper's failure mode framework.
- RP-Reasoner's pragmatic reasoning loop is the theoretical counterpart to the thesis's 5W+H intent-extraction query processing.

## Related

- [[topics/personalisation]] — over-personalisation section
- [[sources/papers/op-bench]] — companion OP-Bench paper
- [[sources/dissertation/overpersonalisation-paper]] — LLNCS paper citing RPEval
- [[entities/5w-h]] — the thesis's operationalisation of intent inference
