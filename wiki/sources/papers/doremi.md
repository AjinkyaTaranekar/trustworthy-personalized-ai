---
title: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
type: source
tags: [foundations, sft]
sources:
  - https://arxiv.org/abs/2305.10429
updated: 2026-07-22
status: current
---

# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**Train a small (280M) proxy model with Group Distributionally Robust Optimization to learn the pretraining domain mixture weights that minimise worst-case excess loss, then reuse those weights to train a model 30× larger — buying a 2.6× pretraining speedup and +6.5 points average downstream accuracy without ever touching downstream tasks.**

## Summary

Xie et al. (Google/DeepMind + Stanford, NeurIPS 2023) reframe data mixing as a learnable optimisation: a small proxy upweights domains where the model has the most room to improve over a reference (high excess loss) and downweights domains already near their achievable loss. On The Pile, the learned weights heavily upweight general web text (Pile-CC 0.11 → 0.61) and sharply downweight specialist domains (arXiv, PubMed → near zero), yet *reduce perplexity on every domain*, and transfer from a 280M proxy to an 8B model at ~8% of the main-run compute. BACKGROUND/mechanism — it complements rather than proves the project's data-quality thesis (DoReMi is about *proportions* of sources; LIMA/phi-1 about *per-example quality*).

## Why it matters here

DoReMi supplies the "data composition is a first-class, learnable design choice" argument that motivates deliberately curating and *balancing* a small SFT corpus rather than scraping maximally. Hook: if the constitutional/SFT dataset spans identifiable categories (question types, tool-use vs no-tool, safety vs helpfulness), a DoReMi-style excess-loss reweighting could tune category proportions for the 0.6B student — a natural small ablation or future-work item, not a headline experiment.

## Method

1. **Reference model:** small (280M) model on default weights → per-domain reference losses.
2. **Group-DRO proxy:** minimax over domain weights on *excess loss* over the reference; online update `α ← α·exp(η·excess_loss)`, renormalise with uniform smoothing; return averaged weights.
3. **Resample + train:** use the learned weights to train the 8B main model.

## Key results

- **The Pile (8B):** +6.5 pts average one-shot downstream accuracy; reaches baseline 2.6× faster; perplexity improves on all 22 domains.
- **Learned weights:** Pile-CC 0.11 → 0.61; PubMed/arXiv/StackExchange → near zero.
- **Proxy sweet spot:** a 70M proxy already improves the 8B on all domains; 280M best; larger proxies show diminishing returns.
- **GLaM:** iterated DoReMi matches downstream-*tuned* oracle weights without seeing tasks.

*(Worst-case log-perplexity table values flagged lower-confidence transcriptions.)*

## Critical appraisal

A clean, influential result — data mixing as learnable optimisation with a small-proxy shortcut, and compelling transfer-across-scale evidence. Its main conceptual risk is **objective mismatch**: minimising worst-case *LM loss* over domains is not the same as maximising task usefulness, and the aggressive downweighting of specialist domains (arXiv, PubMed → ~0) warns the "right" mixture is target-dependent (a mixture optimal for broad QA may be poor for a code/math-heavy target). Domain granularity is fixed (you must pre-define domains), and the Pile-CC upweighting is corpus/dedup-specific — don't copy verbatim.

> ⚠ 0.6B caution: DoReMi's transfer story is proxy(70–280M) → main(8B) — but the project's student *is* 0.6B, i.e. the proxy scale itself, with no larger target, so the "small proxy transfers to a big model" argument does not directly apply. The usable idea is the *reweighting mechanism on excess loss*, not the transfer claim; adapting a pretraining LM-loss method to a few-thousand-example SFT set is non-trivial.

## Related

- [[sources/papers/the-pile]] — the corpus DoReMi reweights
- [[sources/papers/lima]] / [[sources/papers/phi1-textbooks]] — per-example quality (vs DoReMi's proportions)
- [[sources/papers/fineweb]] — ablation-driven data-decision discipline
- [[sources/papers/data-centric-training]] — the broader online-data-mixing family
- [[topics/llm-foundations]] — data composition as a design lever
- [[sources/code/sft-v2-pipeline]] — where category-balancing would apply

## Sources

- Xie, Pham, Dong, Du, Liu, Lu, Liang, Le, Ma, Yu (2023) — arXiv:2305.10429 (NeurIPS 2023) — [arxiv.org/abs/2305.10429](https://arxiv.org/abs/2305.10429)
