---
title: "Membership Inference Attacks against Machine Learning Models"
type: source
tags: [privacy, security, membership-inference]
sources:
  - https://arxiv.org/abs/1610.05820
updated: 2026-07-18
status: current
---

# Membership Inference Attacks against Machine Learning Models

**Given only black-box query access to a trained classifier, an adversary can reliably decide whether a specific record was in that model's training set — by training an "attack" model on the confidence-vector behaviour of many locally trained "shadow" models that imitate the target — turning membership itself into a privacy leak.**

## Summary

Shokri, Stronati, Song and Shmatikov (IEEE S&P 2017) prove that a model returning only predictions still leaks information about *who* was in its training data. Supervised models are measurably more confident on training members than on unseen inputs (a symptom of the generalisation gap), and that asymmetry is readable from the output probability vector alone. Their shadow-model methodology — the reusable engine every later privacy-audit paper builds on — reaches ~94% median accuracy against Google's Prediction API and ~74% against Amazon ML, with near-perfect precision on overfit many-class models (CIFAR-100) but near-chance on well-generalised ones (MNIST 0.517, Adult 0.503). This is the foundational "attack" anchor for the dissertation's privacy sub-theme.

## Why it matters here

It is the canonical proof that any model trained on user data can leak training-set membership from outputs alone — the natural motivation for local-first, on-device handling. Directly relevant to memory-augmented personalisation: if personalisation adapts a shared/cloud model, the adaptation set is inferable, so personalisation state should stay on-device and out of shared weights. It opens the leakage → quantify → erase arc completed by [[sources/papers/extracting-training-data]], [[sources/papers/what-should-llms-forget]] and [[sources/papers/federated-unlearning]].

## Method

- **Threat model:** strictly black-box — attacker submits a record with its label, receives the per-class probability vector, outputs member/non-member. No weights, gradients, or training data.
- **Shadow models:** train k models to imitate the target on auxiliary data where membership is known; running them on their own train vs held-out records yields labelled (prediction-vector, member?) examples.
- **Attack model:** a binary classifier (one per class) mapping a prediction vector to the membership decision.
- **Shadow-data synthesis (increasing realism):** model-based hill-climbing against the target (fully data-free), statistics-based marginal sampling, and noisy real data. Datasets span CIFAR-10/100, Purchase, Location, Texas hospital discharge, MNIST, Adult.

## Key results

- **Commercial MLaaS:** ~94% (Google Prediction API), ~74% (Amazon ML); ~90% even without prior data knowledge in favourable settings.
- **Overfit / many-class → leaks:** CIFAR-100 precision ≈ 1.0; Texas medical ≈ 0.68–0.70; Purchase-100 ≈ 0.93.
- **Well-generalised → safe:** MNIST 0.517, Adult 0.503 (≈ chance).
- **Drivers:** the generalisation gap, the number of output classes, and intra-class diversity — overfitting is sufficient but not the only cause.
- **Defences (top-k output, rounding, higher softmax temperature, L2 regularisation) reduce but never eliminate the leak** — even returning only the top label leaves residual signal.

## Critical appraisal

Landmark, highly reproducible, genuinely black-box against real systems, with a clean causal account of *why* leakage happens. Trust the mechanism completely; treat the high-percentage cells as regime-specific. Scope: classification models, not generative LLMs, and *membership* leakage, not *content* leakage (that is Carlini's domain).

> ⚠ Nuance for the thesis: well-generalised, low-capacity models leak little (MNIST/Adult ≈ chance). A lightly-adapted, well-regularised sub-1B on-device model may be *less* exposed than a large overfit cloud model — a defensible pro-privacy point, provided overfitting to a single user's data is controlled.

## Related

- [[sources/papers/extracting-training-data]] — the *content*-leakage counterpart (verbatim PII from LLMs)
- [[sources/papers/what-should-llms-forget]] — quantifying which memorised facts to erase
- [[sources/papers/federated-unlearning]] — efficiently removing a client's influence
- [[sources/papers/op-bench]] — over-personalisation from memory augmentation; the surface this attack targets
- [[topics/security-and-privacy]] — local-first privacy argument and threat taxonomy
- [[sources/dissertation/security-privacy-social-ethics]] — the project's own security analysis

## Sources

- Shokri, Stronati, Song, Shmatikov (2017) — arXiv:1610.05820 (IEEE S&P 2017) — [arxiv.org/abs/1610.05820](https://arxiv.org/abs/1610.05820)
