---
title: "Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression"
type: source
tags: [privacy, unlearning, on-device, training]
sources:
  - https://arxiv.org/abs/2603.13795
updated: 2026-07-18
status: current
---

# Federated Unlearning via On-server Gradient Conflict Mitigation (FOUL)

**Federated unlearning can be done entirely on the server — without clients re-uploading data and without full retraining — by disentangling each model's features into causal (keep) and non-causal (forgettable) parts, then computing an aggregated "unlearning gradient" that aligns with retain-client gradients while conflicting with forget-client gradients, matching a full retrain at roughly half the communication and computation.**

## Summary

Nguyen et al. (2026) target the *cost* barrier that makes right-to-be-forgotten impractical in federated learning: the naive fix (retrain from scratch on remaining clients) is prohibitive, and forget-clients often cannot re-engage or access the retain set. FOUL removes a whole client/domain's contribution server-side, with no server access to raw client data, by (1) a Learning-to-Unlearn split into a causal featurizer (domain-invariant, keep) and non-causal featurizer (domain-specific, forget), and (2) a closed-form optimal unlearning gradient expressed through a handful of per-client coefficients (U ≪ d). On PACS it matches the retrain oracle (forget-accuracy 70.51%→70.97%, retain +14.49%, MIA ~50%) at ~62% less communication and ~60% fewer FLOPs. This is the "how to forget, efficiently" pillar of the privacy arc.

## Why it matters here

It completes the leakage → quantify → erase arc: where [[sources/papers/membership-inference]]/[[sources/papers/extracting-training-data]] establish *that* models leak and [[sources/papers/what-should-llms-forget]] establishes *what* to forget, FOUL is a concrete mechanism to *efficiently remove* a participant's influence. Its obsession with communication and computation cost mirrors the dissertation's on-device, resource-constrained framing, and "forgetting need not be expensive" (optimise U ≪ d coefficients, not full weights) is a citable precedent for lightweight on-device correction. A privacy-preserving personalised assistant trained across users' devices *is* a federated system, so honouring one user's deletion without retraining the shared model is directly on-scenario.

## Method

- **Setting:** client-wise federated unlearning; 20 clients / 4 domains; forget-set = all clients of one domain; server has no data access.
- **Stage 1 — Learning-to-Unlearn:** split the feature extractor into causal (θK, domain-invariant, preserve) and non-causal (θV, domain-specific, forgettable) via a prototypical-network setup (cosine causal loss, hinge variance-maximising non-causal loss, plus reconstruction/classification).
- **Stage 2 — On-server gradient conflict:** aggregate a gradient that aligns with the retain direction and conflicts with the forget direction. Theorem 1: `∇FOUL = ∇FL + κ·(‖∇FL‖/‖∇ΓR−∇ΓF‖)·(∇ΓR−∇ΓF)`.
- **Expression:** optimise only U ≪ d per-client coefficients Γ instead of full model parameters — the source of the efficiency gains.
- **Datasets:** PACS/OfficeHome/VLCS (ResNet-18), TerraIncognita (ResNet-50).

## Key results

- **PACS vs retrain oracle:** forget-acc 70.51%→70.97%; retain-acc 82.84%→92.33% (+14.49%); test −1.02%; MIA 50.02%→51.93% (~ideal 50%).
- **Efficiency:** communication 42.73→16.02 MB (~62% less); computation 5.81e16→2.35e16 FLOPs (~60% less).
- **Time-to-Forget:** optimal forget-accuracy in <50 rounds (T2F >0.32/round vs retrain 0.13/round).
- Matches or beats 9 FUL baselines across four datasets; MIA ~50% is the evidence forgetting is *genuine*, not merely accuracy-degrading.

## Critical appraisal

Addresses the real deployment blocker (cost), and server-side, data-free-on-server is itself a strong privacy property. Trust the efficiency and oracle-matching claims *within the evaluated regime*. Caution on generality.

> ⚠ Conflict / caution: this is **vision classification with whole-domain forget-sets** (a clean, separable partition), IID clients, and assumption-dependent optimality — not language models, not erasing a specific fact from an LLM's weights, and not single-record forgetting. The dissertation should not overclaim direct LLM applicability; this is exactly where the thesis can argue that on-device personalisation via *external deletable memory* sidesteps weight-level unlearning altogether. Borrowable: the Time-to-Forget metric and MIA-≈50% indistinguishability framing.

## Related

- [[sources/papers/what-should-llms-forget]] — the measurement front-end this removal back-end targets
- [[sources/papers/membership-inference]] — MIA-≈50% is the forgetting-quality test, from this attack
- [[sources/papers/extracting-training-data]] — why weight-level influence (not stored documents) must be erased
- [[entities/graph-rag]] — external deletable memory as the alternative to weight unlearning
- [[topics/security-and-privacy]] — right-to-erasure and local-first framing
- [[topics/personalisation]] — deletion on a federated personalised assistant

## Sources

- Nguyen et al. (2026) — arXiv:2603.13795 — [arxiv.org/abs/2603.13795](https://arxiv.org/abs/2603.13795)
