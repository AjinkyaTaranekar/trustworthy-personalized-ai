---
title: "What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests"
type: source
tags: [privacy, unlearning, memorisation, evaluation]
sources:
  - https://arxiv.org/abs/2507.11128
updated: 2026-07-18
status: current
---

# What Should LLMs Forget? Quantifying Personal Data for RTBF

**Before an LLM can honour a right-to-be-forgotten request you must first know what it has actually memorised about the person — so this paper introduces WikiMem (5,650+ natural-language canaries over 243 Wikidata properties) and a calibrated negative-log-likelihood ranking metric that decides, model-agnostically and even black-box, whether a specific subject–property–value fact is recoverable.**

## Summary

Staufer et al. (2025) reframe LLM right-to-erasure as first a *measurement* problem: GDPR's RTBF was designed for systems that store documents (de-index a URL), but LLMs encode facts in distributed weights, so "which document do we delete?" is ill-posed and every unlearning method assumes the forget-target is already identified. WikiMem probes whether a fact (person h, property p, value v) out-ranks 100 counterfactual distractors under a calibrated NLL score that neutralises the model's prior and name-bias. Across 15 open models, well-known subjects have ~37–39% of tested facts memorised in the strongest models (LLaMA-3.1-8B 38.84%), dropping ~half for lesser-known people; scale mainly increases *confidence* in memorised facts, not their count. This is the "what to forget" pillar bridging the leakage papers and the unlearning mechanism.

## Why it matters here

It gives the thesis a concrete, citable method to *measure* what personal data a model holds — the necessary precursor to any erasure claim. The metric is black-box and cheap enough to run as a **local, on-device compliance audit**, a practical mechanism the privacy chapter can propose for verifiable right-to-erasure instead of trusting a remote provider's opaque deletion. It also argues for keeping user facts in an external deletable store ([[entities/graph-rag]]), because facts in weights are both measurably recoverable and hard to verifiably erase.

## Method

- **"Personal data" formalised** as a factual association (h, p, v), across three regimes: eidetic (verbatim), approximate (reworded), and association (statistical inference).
- **WikiMem:** 5,650 canaries over 243 Wikidata properties; 100 counterfactual distractors per property; ground truth for 200 individuals (100 well-known / 100 lesser-known). Canary types: declarative, 10 paraphrases each, and context-prepended.
- **Calibrated NLL scoring:** rank the true value against counterfactuals; calibration removes the value's baseline prior (re-score with a generic subject) and name-bias (compare "Jane Doe" vs scrambled "Enaj Doe"). Black-box (needs only likelihoods).
- **Memorised** = ground-truth value is rank-1 against all counterfactuals across *all* paraphrases (a deliberately strict criterion). 15 models (Pythia/Qwen/LLaMA-3.1/Mistral), 4-bit quantised.

## Key results

- **Well-known subjects memorised (strict):** LLaMA-3.1-8B 38.84%, Qwen3-30B-A3B 37.41%, LLaMA-3.1-70B 38.33%, Pythia-410M 10.22%.
- **Lesser-known:** roughly half (LLaMA-3.1-8B 22.43%).
- **Scale sharpens certainty, not coverage:** LLaMA-8B (38.84%, strength 3.38) vs 70B (38.33%, strength 3.77) — near-identical coverage, higher confidence.
- **Definition-sensitive:** a lenient definition pushes some properties above 80%.

## Critical appraisal

Fills a genuine, previously-unaddressed gap; the calibrated, confound-neutralising metric is methodologically careful and descends cleanly from Carlini-style comparative scoring; broad 15-model sweep. Cautions: bounded to Wikidata-representable facts about *documented* people (under-samples exactly the private individuals RTBF most protects); all models 4-bit quantised; percentages swing 38%→80% with the definition. Absence of memorisation under WikiMem is not proof of erasure.

> ⚠ Scale nuance: memorisation and especially *confidence* grow with scale, and lesser-known subjects are memorised far less — supporting a small-on-device privacy narrative, tempered by the finding that even 8B models already memorise ~38% of tested facts about prominent people.

## Related

- [[sources/papers/extracting-training-data]] — the memorisation this quantifies; shared comparative-scoring lineage
- [[sources/papers/membership-inference]] — the membership-leakage anchor of the same arc
- [[sources/papers/federated-unlearning]] — the "how to forget" mechanism this measurement targets
- [[entities/graph-rag]] — external deletable memory as the RTBF-friendly alternative
- [[topics/security-and-privacy]] — GDPR right-to-erasure framing
- [[topics/personalisation]] — auditing whether a deletion request truly removed a user fact

## Sources

- Staufer et al. (2025) — arXiv:2507.11128 — [arxiv.org/abs/2507.11128](https://arxiv.org/abs/2507.11128)
