---
title: "Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents"
type: source
tags: [privacy, memory, unlearning, personalisation]
sources:
  - https://arxiv.org/abs/2512.12856
updated: 2026-07-19
status: current
---

# Forgetful but Faithful: Privacy-Aware Cognitive Memory

**Forgetting should be a designed capability, not a failure: MaRS is a typed, provenance-aware cognitive memory architecture, paired with a family of forgetting policies (culminating in a Hybrid policy with differential-privacy guarantees at the eviction boundary) and the FiFA benchmark, showing that principled forgetting-by-design can simultaneously sustain narrative coherence, goal completion, social recall, privacy, and cost (composite ≈0.911).**

## Summary

Alqithami (Al-Baha University, 2025) frames unbounded agent memory as a three-way tension — retaining everything is expensive and incoherent, but naive forgetting destroys context and can leak sensitive information. He models the retained set as a constrained knapsack (maximise utility within a token budget), shows provenance-closed sets form antimatroids so a greedy algorithm is near-optimal, and attaches a per-node privacy-sensitivity score with a retention formula `score(i) = (Û_i − λ_priv·s_i)/w_i`. The Hybrid forgetting policy (temporal + reflection + importance + DP-privacy passes) reaches composite ≈0.911 on the FiFA benchmark. This is the dedicated privacy-aware-forgetting reference — strong on architecture and framing, weak on empirics (simulation-only, single author, unnamed backbone).

## Why it matters here

The primary hook is **privacy-aware forgetting on-device**. The per-node privacy-sensitivity score and retention formula give a concrete, adoptable mechanism for a privacy-on-mobile assistant to decide what to keep vs forget, with token-weight normalisation directly relevant to a memory-constrained 0.6B device. The typed categories (episodic/semantic/social/task) and provenance edges are a template for a scrutable, editable [[entities/5w-h|5W+H]] store (who/what/when provenance is exactly the metadata FiFA tracks), and the (ε,δ)-DP eviction boundary + GDPR right-to-be-forgotten framing give a formal privacy story to cite. It is the *retain rationally* complement to [[sources/papers/rpeval|rational preference utilisation's]] *apply rationally*.

## Method

- **MaRS:** typed memory nodes (episodic/semantic/social/task), each carrying content, type, creation time, **privacy-sensitivity score**, computational weight, and **provenance**; JSON-LD graph with temporal/semantic/causal/social edges.
- **Retention score:** `score(i) = (Û_i − λ_priv·s_i)/w_i` — utility penalised by privacy sensitivity, normalised by token weight.
- **Six forgetting policies:** FIFO, LRU, Priority Decay, Reflection-Summary, Random-Drop, and **Hybrid** (composed passes) with an **exponential mechanism giving (ε,δ)-DP at the eviction boundary**.
- **FiFA benchmark:** five dimensions (narrative coherence, goal completion, social recall, privacy preservation, cost efficiency), 300 runs across five budgets, rubricized LLM-judge.

## Key results

- **Hybrid policy composite ≈0.911** — best, maintaining high privacy while preserving coherence, beating temporal baselines on coherence/goal/social.
- **Honest boundary condition:** simple FIFO/LRU remain optimal when usefulness decays exponentially with staleness.
- Provenance/antimatroid structure enables greedy near-optimal retention (a clean theoretical link to matroid theory).

## Critical appraisal

The most theoretically developed forgetting paper (knapsack/antimatroid formalisation, DP guarantees, complexity table), with the valuable framing that forgetting is a *designed, auditable, privacy-motivated* operation. Weaknesses are empirical: **single-author simulation only** (300 synthetic runs, no real users, unnamed base LLM), LLM-judge privacy scoring is noisy, and the composite ≈0.911 is a weighting-sensitive aggregate.

> ⚠ Caveats for a 0.6B privacy-on-mobile thesis: (1) simulation-only with an unnamed (likely large) backbone and no device-resource profiling — supplies *architecture*, not small-model feasibility evidence; (2) the DP guarantee is at the *eviction boundary*, not end-to-end over generated text, so do not overclaim end-to-end privacy; (3) the finding that simple temporal forgetting is competitive when usefulness decays with staleness is a useful cost-conscious default for on-device — the full Hybrid machinery may be overkill unless privacy sensitivity varies sharply.

## Related

- [[sources/papers/rpeval]] — rational *use* of memory; this is rational *retention/forgetting* (retain vs apply)
- [[sources/papers/what-should-llms-forget]] — quantifying what to forget (RTBF measurement)
- [[sources/papers/federated-unlearning]] — efficient weight-level influence removal
- [[sources/papers/memmachine]] — memory retention vs the privacy liability of verbatim storage
- [[entities/5w-h]] — typed provenance memory as a scrutable user model
- [[topics/security-and-privacy]] — DP forgetting, GDPR right-to-erasure
- [[topics/personalisation]] — forgetting as a first-class memory operation

## Sources

- Alqithami (2025) — arXiv:2512.12856 — [arxiv.org/abs/2512.12856](https://arxiv.org/abs/2512.12856)
