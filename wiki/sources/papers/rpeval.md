---
title: RPEval — Rational Preference Utilisation Benchmark
type: source
kind: paper
tags: [personalisation, over-personalisation, evaluation, benchmark, reasoning]
sources:
  - https://arxiv.org/abs/2601.16621
  - docs/Assets/How Does Personalized Memory Shape LLM Behavior Benchmarking Rational Preference Utilization in Personalized Assistants (2601.16621v1).pdf
  - docs/Literature Notes/How Does Personalized Memory Shape LLM Behavior Benchmarking Rational Preference Utilization in Personalized Assistants (2601.16621v1).md
arxiv: 2601.16621
updated: 2026-07-19
status: current
---

# RPEval — Rational Preference Utilisation Benchmark

**LLM assistants that mechanically inject recalled user preferences into context routinely over-personalise; RPEval measures whether a model *rationally* decides to Ignore/Support/Dominate each stored preference, and finds mainstream LLMs trail humans by 40–90% — with an inverse-scaling twist that stronger models over-personalise *more*.**

> Enriched 2026-07-19 from a full re-read of the arXiv PDF (the earlier version was summarised from the literature note). This is the Feng et al. paper "How Does Personalized Memory Shape LLM Behavior?"; the benchmark it introduces is named RPEval (repo: [github.com/XueyangFeng/RPEval](https://github.com/XueyangFeng/RPEval)). Not to be confused with PrefEval (Zhao et al., 2502.09597), a separate preference-following benchmark.

## Summary

Feng et al. (Renmin University / Huawei, Jan 2026) ground personalisation in Grice's pragmatics and Rational Speech Acts theory: because memories are sparse/fragmented and queries are open-ended, semantically-similar-but-irrelevant memories get injected and mislead the model away from true intent. They define three assistant levels — L0 non-personalised (under-personalises), L1 literal (appends high-similarity memory, over-personalises), L2 pragmatic (Bayesian posterior `P(i,r|m,q) ∝ P_user(q|i,m)·P(i|m)`, reasoning about *whether* to use each memory). RPEval judges responses against the user's *true intent*, not consistency with memory — its key departure from MemBench/LongMemEval/PrefEval. The proposed RP-Reasoner recovers ~+35% micro-accuracy, −26% error severity, and resolves ~80% of failures on a real Huawei commercial assistant. This is the most on-thesis paper for the over-personalisation problem and a ready-made evaluation frame.

## Thesis connections

- **Over-personalisation eval axis:** the Ignore/Support/Dominate labels and the FB/RII/UPB/LF/VG error taxonomy lift almost directly into the constitutional harness — does the 0.6B model know when *not* to apply a stored preference?
- **Rational preference utilisation:** the L0/L1/L2 hierarchy gives crisp language; the thesis target is an L2 assistant, and a constitution-gated small model can be scored for how often it reaches L2 vs collapses to L1 literal injection.
- **Scrutable profiles (5W+H):** the (preference, query, rationale, intent) quadruple maps onto a scrutable-profile design where each stored fact carries an applicability rationale — the counterpart to the thesis's 5W+H intent-extraction.
- **Inverse scaling is a pro-small-model argument:** if stronger contextual attention drives over-personalisation, a *constitution that gates memory* could matter more than sheer scale.

## Method

- **Three levels:** L0 (ignore memory), L1 (literal concatenation — the commercial default and the source of over-personalisation), L2 (pragmatic Bayesian intent inference).
- **Dataset (Personalized Intent Reasoning):** atomic unit `(preference, query, rationale, intent∈{Ignore,Support,Dominate})`; built by bootstrapping (20 base scenarios → 100 meta-scenarios, 12 categories), preference inversion (natural underspecified queries first), and iterative quality verification. 8,255-sample pool → 953 gold-standard test samples at 91.86% inter-annotator agreement. Expansions: Explicit2Implicit (5-turn dialogue), Single2Multi (Ignore-All, Leave-K-Out).
- **RP-Reasoner:** implements the L2 posterior via Query Likelihood Estimation (ABC-style counterfactual elimination — is there a better query for this candidate intent?) + Intent Prior Estimation (`P(i|m)`), aggregated by rank. GPT-4.1 as judge (QWK 0.87 vs humans).

## Key results

- **Discriminative (single-preference ALL accuracy):** Human 0.95; DeepSeek-V3 0.66; GPT-4.1 0.53; GPT-5 0.51; Qwen2.5-7B 0.38. On the hard **Ignore** case: Human 0.86 vs GPT-5 0.12, Qwen 0.06 — a ~55.8% human gap; multi-preference Macro gap up to 91.8%.
- **Inverse scaling:** more capable models are *worse* at ignoring irrelevant preferences (stronger contextual attention → more over-utilisation). Mechanism: attraction/echo bias amplifies memory tokens already in context during decoding.
- **RP-Reasoner (multi-preference generative, avg of 4 models):** +258% Macro-acc, +35% Micro-acc, −26% error severity vs best baseline; resolves ~80% of error cases in the real commercial deployment.

## Critical appraisal

The central claim — naive memory injection causes over-personalisation, and models should reason about *whether* to use a memory — is exactly the failure mode a scrutable, rational on-device assistant must avoid, and the Ignore/Support/Dominate taxonomy is a clean reusable label set. The inverse-scaling result is genuinely important and mildly encouraging for small-model work. Weaknesses: heavy reliance on synthetic data and on GPT-4.1 as both generator and judge (self-consistency risk despite QWK 0.87); English-only, discrete labels.

> ⚠ 0.6B tension: all four systems-under-test are ≥7B cloud-scale — small-model behaviour is untested (though inverse scaling hints small models may over-personalise *less*). RP-Reasoner's remedy is **inference-time multi-call Bayesian ranking**, too costly for a 0.6B at serve time — the thesis would need to *distil* the L2 behaviour into the model via SFT/constitution rather than run RP-Reasoner's full loop. It concerns *selective use* of already-retrieved memory, complementary to the *retention/forgetting* papers.

## Related

- [[topics/personalisation]] — over-personalisation section
- [[sources/papers/op-bench]] — companion over-personalisation benchmark
- [[sources/papers/avoiding-over-personalization]] — symbolic KG edits vs this paper's rational reasoning
- [[sources/papers/forgetful-but-faithful]] — rational *retention*; RPEval is rational *use* (retain vs apply)
- [[sources/papers/tears]] — scrutable profile; α ≈ how much a preference influences output
- [[sources/papers/mem0]] — the retrieval layer whose injected memories RPEval judges the *use* of
- [[sources/dissertation/overpersonalisation-paper]] — LLNCS paper citing RPEval
- [[entities/5w-h]] — the thesis's operationalisation of intent inference

## Sources

- Feng, Gan, Chen, Dai, Liu (2026) — arXiv:2601.16621 — [arxiv.org/abs/2601.16621](https://arxiv.org/abs/2601.16621)
- Raw layer: `docs/Assets/…(2601.16621v1).pdf`, `docs/Literature Notes/…(2601.16621v1).md`
