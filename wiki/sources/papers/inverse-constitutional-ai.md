---
title: "Decoding Human Preferences in Alignment: An Improved Approach to Inverse Constitutional AI"
type: source
tags: [constitutional-ai, constitution, alignment, evaluation]
sources:
  - https://arxiv.org/abs/2501.17112
updated: 2026-07-20
status: current
---

# Decoding Human Preferences: Improved Inverse Constitutional AI

**Improving the principle-generation, clustering, and embedding stages of Inverse Constitutional AI (ICAI) recovers more accurate, generalisable explicit principles from a preference dataset — making implicit human preferences legible as an auditable, editable written constitution.**

## Summary

Henneking and Beger (Cornell, 2025) refine ICAI, which reverse-engineers a natural-language "constitution" from pairwise preferences so that alignment becomes inspectable rather than an opaque weight update. Their Improved-2 pipeline embeds response pairs along content/style/sentiment axes, KMeans-clusters the differences, and folds in numeric rating scores. Preference-regeneration accuracy is strong on synthetic data (93–94%) but drops to 60.75% on real Anthropic HH — the honest headline that recovered constitutions remain lossy on subtle real preferences. Downstream in-context alignment (AlpacaEval 46.74%) is unconvincing. As a proof that constitutions are recoverable *in principle* it succeeds; as a deployable auditing tool it is early-stage — and it is the natural companion to the project's forward direction (write constitution → SFT).

## Why it matters here

The dissertation goes forward (write a constitution → SFT a small model); ICAI runs the same map *backward*, so running it on the SFT model's own preference/response data is a concrete "did the constitution take?" audit — checking whether the 0.6B student internalised the written principles or fell back on training priors. Its core lesson also informs *how to phrase* principles: vague or over-specific wording propagates error, so principles must be general yet discriminative. The rating-score finding argues for logging graded reward signal (not just binary picks) in the SFT/judge pipeline.

## Method

- **Improved-1:** more general principle-generation prompts + centroid-proximity subsampling.
- **Improved-2:** multi-axis (content/style/sentiment) embeddings → KMeans on paired-response differences → representative triplets jointly prompted to synthesise principles; incorporates numeric preference scores.
- **Eval:** preference-regeneration accuracy, constitution similarity to a ground-truth, AlpacaEval in-context alignment. Datasets: synthetic (150 pairs / 5 ground-truth principles), UltraFeedback, Anthropic HH-Harmlessness.

## Key results

- **Regeneration accuracy:** synthetic 93–94%, semi-synthetic 76.20%, **real HH 60.75%**.
- **Constitution similarity:** Improved-2 5.4 vs ground-truth 5.8 (opaque scale).
- **AlpacaEval:** 46.74% (flat; in-context gains unclear). **Rating scores:** lift extraction to 76.80% on scored data.

## Critical appraisal

A genuinely useful direction — turning opaque preference data into an inspectable constitution — with careful ablation of where ICAI breaks (the weak generation stage) and honesty about real-data limits. Weaknesses: strong numbers lean on synthetic scaffolding; the similarity metric is under-specified (5.4 vs 5.8, no rubric); downstream alignment gains are flat; scalability (multiple embeddings + sequential LLM prompting) is conceded not solved.

> ⚠ Small-model caution: the real-data ceiling (~60%) means a recovered constitution is lossy even at frontier scale; on a sub-1B student it would be lossier still — treat ICAI as an audit/diagnostic, not a guarantee.

## Related

- [[sources/papers/constitutional-ai-bai]] — the forward CAI this inverts
- [[sources/papers/c3ai]] — crafting/evaluating constitutions (the design companion)
- [[sources/papers/general-language-assistant]] — the HHH preference signal being decoded
- [[entities/constitution]] — the written principles this would audit
- [[topics/constitution-psychological-grounding]] — principle wording and grounding

## Sources

- Henneking, Beger (2025) — arXiv:2501.17112 — [arxiv.org/abs/2501.17112](https://arxiv.org/abs/2501.17112)
