---
title: "Transparent and Scrutable Recommendations Using Natural Language User Profiles"
type: source
tags: [personalisation, scrutability, explainability, retrieval]
sources:
  - https://arxiv.org/abs/2402.05810
updated: 2026-07-18
status: current
---

# Transparent and Scrutable Recommendations Using NL User Profiles

**Replace opaque latent user embeddings with human-readable natural-language user profiles automatically generated from the user's reviews, so recommendations become transparent and the user can directly read and edit their profile to steer results — while keeping accuracy competitive with matrix-factorisation and explainable-recommender baselines.**

## Summary

Ramos et al. (UCL / Sheffield, 2024) build UPR: mine salient preference features from a user's reviews, rank them by a utility score `U(f) = |r̄(f)| × cov(f) × sig(f)`, prompt a 7B LLM to write a ~200-token NL profile, then fine-tune **GPT-2** to predict ratings from `profile + item title` text alone (no user/item IDs). The plain-language profile is inspectable (transparent) and directly editable (scrutable) — edits change recommendations with no retraining. Accuracy lands within a few thousandths of the strongest baselines (Amazon-MT UPR-Mistral RMSE 0.941 vs MF 0.925), and a user study rates profiles fluent (92–95%) and relevant (87–90%), Fleiss' κ=0.82. This is the paper's strongest tie to the dissertation's scrutable 5W+H user model.

## Why it matters here

It operationalises exactly the project's scrutable natural-language user profile: a plain-language, user-readable, user-editable representation replacing an opaque latent vector. The feature-ranking → LLM-synthesis pipeline is a concrete template for building [[entities/5w-h|5W+H]]-style profiles from raw interaction, and the edit-then-observe scrutability experiment is a proof-of-concept for user-controllable personalisation. Encouragingly for on-device, the *scoring* model is only GPT-2-scale and still competitive.

## Method

1. **Preference identification/ranking:** phrase-level sentiment extracts feature words; rank by `U(f) = |r̄(f)| × cov(f) × sig(f)` (rating magnitude × coverage × significance).
2. **Profile transformation:** prompt Llama2-7B or Mistral-7B with top-5 features + 5 supporting reviews each → ≤200-token NL profile.
3. **Recommendation:** fine-tune GPT-2 on `profile + title` → 1–5 rating; **no IDs** (all text) is what makes it transparent.
- **Scrutability:** users edit the profile directly; recommendations update with no retraining (measured by Coverage@10).

## Key results

- **Amazon Movies & TV (RMSE):** UPR-Mistral 0.941 vs MF 0.925, PETER+ 0.924 (within 0.016); matches best MAP 0.870.
- **TripAdvisor (MAE):** UPR-Mistral 0.610 vs PEPLER-MLP 0.606 (within 0.004).
- **Profile quality (user study):** Fluency 92–95%, Relevance 87–90%, Fleiss' κ=0.82.
- **Scrutability:** adding a target preference reliably raises its Coverage@10; feature help saturates after ~3.

## Critical appraisal

A clean, well-motivated transparency + scrutability demonstration with honest (parity, not inflated) accuracy and a proper human quality study. Cautions: "competitive accuracy" is competitive-but-slightly-worse; Coverage@10 shows an edit *moved* recommendations, not that the result is *better* (scrutability is responsive but its value is untested, no live user study); warm-start only (≥5 reviews) sidesteps cold-start.

> ⚠ Caution for on-device: profile *generation* uses 7B LLMs, and the authors flag LLM latency and **hallucinated profile facts** as real problems — both central risks on-device, where a 7B profiler is unaffordable and a fabricated preference is a trust failure (see [[sources/papers/hallucination-survey]] and [[sources/papers/op-bench]]). A scrutable profile is also a human-readable dossier of the user's tastes — a legibility win but also a plain-text leak surface for privacy-on-mobile.

## Related

- [[entities/5w-h]] — the scrutable user-model schema this operationalises
- [[topics/explainability]] — scrutability and transparent recommendation
- [[topics/personalisation]] — user-controllable personalisation
- [[sources/papers/op-bench]] — over-personalisation as the failure mode of profile-driven systems
- [[sources/papers/hallucination-survey]] — fabricated profile facts as a hallucination risk
- [[sources/papers/rpeval]] — whether models actually follow stated preferences

## Sources

- Ramos, Rahmani, Wang, Fu, Lipani (2024) — arXiv:2402.05810 — [arxiv.org/abs/2402.05810](https://arxiv.org/abs/2402.05810)
