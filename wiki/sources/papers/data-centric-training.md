---
title: "Towards Next-Generation LLM Training: From the Data-Centric Perspective"
type: source
tags: [foundations, sft]
sources:
  - https://arxiv.org/abs/2603.14712
updated: 2026-07-22
status: current
---

# Towards Next-Generation LLM Training: From the Data-Centric Perspective

**The next leap in LLM training will come from treating data as an active, first-class component: an agent-based automatic data-preparation system to build corpora, plus a unified data-model interaction training system that dynamically selects, mixes, and reweights data during training.**

## Summary

Liang et al. (2026, position/vision paper — no experiments) argue data preparation, not modelling, is the dominant remaining bottleneck: it is done with ad-hoc scripts on static corpora that training consumes passively. They propose two systems — a three-layer agent-based data-preparation stack (operators → serving → data agent) and a three-module in-training data-model interaction interface (selection, mixture, reweighting driven by live loss/gradient signals) — and catalogue the algorithms each should unify (DoReMi, LESS, Aioli, REGMIX, AlpaGasus, Data Juicer). BACKGROUND/recent framing + reference net — a map and a manifesto, not a result, and oriented at frontier-scale trillion-token pipelines far from a sub-1B on-device setting.

## Why it matters here

Value is twofold: (1) a fresh 2026 citation that data-centric training is the recognised frontier, legitimising the project's data-quality emphasis; (2) a tidy taxonomy locating other data papers (it buckets [[sources/papers/doremi|DoReMi]] under "online data mixture" and lists quality-scoring methods — AlpaGasus, MoDS — and prep frameworks the project can cite for corpus curation). Its "Quality Evaluation & Statistics" phase (metric/LLM/human-based scoring) is directly analogous to how the project could *filter and score* its constitutional/SFT examples — a [[sources/papers/phi1-textbooks|phi-1]]-style educational-value filter fits here.

## Proposed systems

- **Agent-based data preparation (3 layers):** Data Operators (acquisition → processing → rewriting/augmentation → quality evaluation), Data Serving (storage/indexing), Data Agent (NL-driven orchestration of operators-as-skills, human-in-the-loop). Notes a "bidirectional funnel": volume falls while model involvement rises, shifting bottlenecks from I/O-bound to GPU-bound.
- **Unified data-model interaction (3 modules):** dynamic per-instance Selection (gradient/loss signals), domain-level Mixture, per-sample Reweighting — through interfaces abstracted from the trainer.

## Critical appraisal

Useful as a 2026 synthesis that ties the data-centric literature together and names the frontier. But it is a map and a manifesto, not a result — it offers no evidence the proposed unification beats today's siloed methods, the agentic-preparation-plus-live-interaction stack is heavy with unquantified cost/benefit, and second-order signals at trillion-token scale are acknowledged as hard.

> ⚠ 0.6B caution (heavy): this targets frontier-scale, trillion-token, multi-PB, agentic pipelines (NeMo Curator "100+ PB") — almost none of the infrastructure vision transfers to a 0.6B student on a small curated SFT set. Use only for high-level framing and as a citation hub; be explicit its systems are out of scope for an on-device small-model project. Not abstract-only, but effectively evidence-free (position paper) — cite as a perspective, never a measured finding.

## Related

- [[sources/papers/doremi]] — the online-data-mixture method it catalogues
- [[sources/papers/fineweb]] — ablation-driven data-decision discipline
- [[sources/papers/phi1-textbooks]] — quality-scoring/curation for small models
- [[sources/papers/instruction-tuning-survey]] — the SFT-data taxonomy companion
- [[sources/papers/deduplicating-training-data]] — a "Data Processing" operator
- [[topics/llm-foundations]] — data as a first-class design lever

## Sources

- Liang, Zhao, Han, Qiang, Ma, Zeng, Cai, Li, Tang, E, Zhang (2026) — arXiv:2603.14712 — [arxiv.org/abs/2603.14712](https://arxiv.org/abs/2603.14712)
