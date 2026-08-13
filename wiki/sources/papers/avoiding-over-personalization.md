---
title: "Avoiding Over-Personalization with Rule-Guided Knowledge Graph Adaptation for LLM Recommendations"
type: source
tags: [over-personalisation, personalisation, graph-memory, small-model, privacy]
sources:
  - https://arxiv.org/abs/2509.07133
  - https://github.com/brains-group/KGAdaptation
updated: 2026-07-19
status: current
---

# Avoiding Over-Personalization with Rule-Guided KG Adaptation

**A lightweight, privacy-preserving neuro-symbolic framework mitigates over-personalisation in LLM recommenders by detecting filter-bubble-like feature-pair biases ("Personalized Information Environments") in a user's own knowledge graph and applying symbolic Soft/Hard/Removal edits at inference time — entirely client-side, driving a Qwen3-0.6B recommender, with no black-box retraining.**

## Summary

Spadea and Seneviratne (RPI, ISWC 2025) argue for shifting "from tuning black-box models to shaping the symbolic structures that guide them". They detect Personalized Information Environments (PIEs) — feature pairs a user rates far above/below the 2.5 midpoint (±0.5 threshold) — in a user-side Personalized Knowledge Graph, then apply Soft (invert ratings around the midpoint), Hard (extreme opposite), or Removal (delete triples) edits to steer a KTO-fine-tuned **Qwen3-0.6B** toward diverse Out-PIE recommendations. Personalised Soft tuning raised Out-PIE 25.2% → 32.4% and cut Invalid 49.0% → 46.0%; pure prompting to "be diverse" was worst (62.1% Invalid). It is a rare paper that explicitly uses a sub-1B model client-side — a direct precedent for the pipeline's own base — though absolute numbers are weak, so cite it as a design pattern, not a solved baseline.

## Why it matters here

Two hooks land squarely on the thesis. It uses the **same Qwen3-0.6B base**, client-side and privacy-preserving — direct precedent that a sub-1B model can carry personalised recommendation with symbolic scaffolding. And the PKG *is* an editable symbolic user profile: Soft/Hard/Removal edits are concrete scrutability operations a user could invoke ("show me less tomato"), mapping onto a [[entities/5w-h|5W+H]] scrutable profile the user can revise. Tension worth surfacing: it mechanically *over-rides* preferences rather than *reasoning* about when a preference should apply — contrast [[sources/papers/rpeval|rational preference utilisation]] (reason about whether to use a memory), so symbolic editing gives scrutable control but rational reasoning is still needed.

## Method

- **PIE detection:** feature-pair bias flagged when combined-feature ratings deviate ±0.5 from the 2.5 midpoint.
- **Outcome labels:** Out-PIE (has preferred feature, not the biased one — desired), In-PIE (both — reinforcing), Invalid (missing requested feature).
- **Three symbolic edits:** Soft (invert around midpoint), Hard (extreme opposite), Removal (delete PIE triples). A per-user `adaptProportion` (LR 0.05) vs a global one.
- **Recommender:** Qwen3-0.6B framed as KG-completion, fine-tuned with Kahneman–Tversky Optimization.

## Key results (Food.com, 20 users)

- **Soft personalised:** Out-PIE 0.324 / In-PIE 0.216 / Invalid 0.460 — best Out-PIE.
- **Prompt-based:** 0.193 / 0.186 / 0.621 — telling the LLM to diversify hurt validity badly.
- **Symbolic edits beat prompting**; Soft > Hard (aggressive flips raise Invalid); per-user tuning helps only single digits (global Removal 0.328 edges personalised Soft).

## Critical appraisal

Conceptually well-aligned and rare in explicitly using a sub-1B model with an on-device/privacy framing; the neuro-symbolic idea (edit the editable symbolic user model, not the weights) is exactly the scrutable-control mechanism the thesis cares about. But the empirical case is thin: even the best method leaves ~46% Invalid and Out-PIE tops ~32%; the ±0.5/2.5 PIE definition is simplistic and single-domain (recipes); no user study confirms Out-PIE recommendations are actually *wanted* (over-correction risk); 20 users only.

> ⚠ 0.6B caution: the weak absolute numbers (46% Invalid) warn a 0.6B doing KG-completion recommendation is fragile — cite as *motivation* for the constitution/harness to improve validity, and as a design pattern (symbolic editable user model on 0.6B), not as strong evidence the method works well.

## Related

- [[sources/papers/rpeval]] — rational *use* of preferences (reason about applicability) vs this paper's mechanical over-ride
- [[sources/papers/op-bench]] — over-personalisation benchmark; the failure mode quantified
- [[sources/papers/tears]] — scrutable/editable textual profiles (complementary substrate)
- [[entities/qwen3-0.6b]] — the shared sub-1B base
- [[entities/5w-h]] / [[entities/graph-rag]] — the editable symbolic user profile
- [[topics/personalisation]] — over-personalisation and scrutable control
- [[topics/security-and-privacy]] — client-side, privacy-preserving adaptation

## Sources

- Spadea, Seneviratne (2025) — arXiv:2509.07133 (ISWC 2025) — [arxiv.org/abs/2509.07133](https://arxiv.org/abs/2509.07133)
- Code — [github.com/brains-group/KGAdaptation](https://github.com/brains-group/KGAdaptation)
