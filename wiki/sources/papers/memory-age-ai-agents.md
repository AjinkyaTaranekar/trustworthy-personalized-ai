---
title: "Memory in the Age of AI Agents"
type: source
tags: [memory, agents, personalisation, privacy]
sources:
  - https://arxiv.org/abs/2512.13564
  - https://github.com/Shichun-Liu/Agent-Memory-Paper-List
updated: 2026-07-19
status: draft
---

# Memory in the Age of AI Agents

**A large unifying survey that recasts fragmented agent-memory research into a single taxonomy along three axes — Forms (token-level / parametric / latent), Functions (factual / experiential / working), and Dynamics (formation / evolution / retrieval) — and sharply distinguishes "agent memory" from LLM memory, RAG, and context engineering.**

> ⚠ Access caveat: the arXiv HTML/ar5iv build for this paper failed ("Conversion to HTML had a Fatal error"), so this page is reconstructed from the official abstract plus two secondary reviews. The *taxonomy* (forms/functions/dynamics; F/E/R operators) is confirmed by the abstract, but specific system/benchmark attributions (MemOS, Memory-R1, LoCoMo, StreamBench, etc.) are second-hand — verify against the PDF before citing them. Marked `status: draft` until the full text is read.

## Summary

Hu, Liu, Yue et al. (47 authors, 2025/2026) provide the cleanest available conceptual vocabulary for agent memory. Its core definitional move: **agent memory is a persistent, externally managed, self-evolving store built from the agent's own experience across tasks** — distinct from LLM memory (attention/KV cache/context), RAG (retrieval from static knowledge within one episode), and context engineering (optimising the current window). It organises the field into Forms × Functions × Dynamics and a Formation/Evolution/Retrieval (F/E/R) lifecycle, and — notably for this project — names trustworthiness (privacy, poisoning-robustness, consent, auditability) as a first-class pillar rather than an afterthought.

## Why it matters here

The single best framing citation for the memory chapter's conceptual scaffolding. The F/E/R lifecycle maps directly onto a [[entities/5w-h|5W+H]] KG user-memory pipeline: Formation = extract 5W+H facts from turns; Evolution = consolidate/update/forget (retention + right-to-be-forgotten); Retrieval = query-time selection into the small model's context — reusable operator vocabulary verbatim. Its trustworthiness pillar (differential-privacy formation, memory-poisoning robustness, consent/access control, auditability) is a direct match for the privacy-on-mobile thesis and strong support for positioning trust + privacy as first-class.

## Taxonomy (three axes)

- **Forms:** token-level (human-readable external units — flat/planar/hierarchical), parametric (in weights; ROME/MEMIT edits or LoRA adapters — fast but poorly updatable/interpretable), latent (hidden activations/KV/embeddings — dense, machine-native, low interpretability, hard to persist).
- **Functions:** factual (declarative — preferences, entity graphs), experiential (procedural — case/strategy/skill-based), working (short-term scratchpad).
- **Dynamics (F/E/R):** Formation (summarise/structure/embed), Evolution (consolidation, updating, forgetting — the stability–plasticity dilemma, trending toward learned utility prediction), Retrieval (timing/intent → query construction → strategy → post-processing, trending toward generative retrieval).

## Critical appraisal

Strengths: the cleanest agent-memory-vs-RAG-vs-context-engineering distinctions and a very citable forms/functions/dynamics + F/E/R vocabulary. Cautions: **I could not access the full text**, so depth/rigour and any quantitative synthesis are unverified; as a 47-author survey it is likely broad but shallow per topic and quickly dated; it is cloud-agent-centric — small-model/on-device feasibility is not a first-order axis, and parametric/latent forms are weakly actionable for a token-level KG personalisation system.

> ⚠ Gap it leaves open: it treats parametric/latent and RL-driven memory as frontiers assuming capable models and compute, and does not validate small-model feasibility — reinforcing the novelty gap (no sub-1B constitutional/privacy-on-mobile memory paper). Cite it for the framework and the trust agenda; claim the on-device + privacy instantiation as the contribution.

## Related

- [[sources/papers/graph-agent-memory-survey]] — the graph-focused companion survey
- [[sources/papers/mem0]] — a token-level memory system it names
- [[sources/papers/memmachine]] — ground-truth-preserving memory
- [[sources/papers/what-should-llms-forget]] — the Evolution/forgetting + RTBF operator
- [[entities/5w-h]] — the schema the F/E/R lifecycle would populate
- [[topics/personalisation]] — memory as a first-class primitive
- [[topics/security-and-privacy]] — the named trustworthiness pillar

## Sources

- Hu, Liu, Yue, Zhang, Liu, Zhu, Lin, et al. (47 authors, 2025/2026) — arXiv:2512.13564 — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- Paper list — [github.com/Shichun-Liu/Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
