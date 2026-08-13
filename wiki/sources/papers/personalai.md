---
title: "PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM Agents"
type: source
tags: [memory, graph-memory, personalisation, small-model, retrieval]
sources:
  - https://arxiv.org/abs/2506.17001
updated: 2026-07-19
status: current
---

# PersonalAI: Comparing KG Storage & Retrieval for Personalised LLM Agents

**A systematic, controlled comparison of knowledge-graph storage schemas and graph-traversal retrieval algorithms for long-term personalised memory, showing the best configuration depends jointly on model scale and task type — small models need structure-aware, restricted traversal (BeamSearch over thesis/episodic vertices), while larger models benefit from unrestricted hybrid retrieval.**

## Summary

Menschikov et al. (Skoltech / Sber / AIRI, 2026) run the memory sweep most memory papers skip: a factorial comparison of a three-layer hybrid KG (object / thesis / episodic vertices) × six traversal algorithms (A* variants, WaterCircles BFS, BeamSearch, hybrids) × model scale (7B → GPT-4o-mini) × three QA benchmarks. The headline is scale-dependence: for 7B/8B models, BeamSearch with episodic restrictions dominates and *thesis* (atomic-thought) vertices are disproportionately valuable; for 14B+, unrestricted hybrid search wins. This directly supports the hypothesis that a sub-1B on-device model needs a carefully structured 5W+H user-memory graph to compensate for weak parametric reasoning — but the accuracy-winning small-model retriever (BeamSearch, ~6.6 min/query) is far too slow for interactive use, and best 7B accuracy is only 0.27 vs GPT-4o-mini's 0.77.

## Why it matters here

The closest paper to a privacy-on-mobile KG-memory thesis that *explicitly varies model scale*. Its object/thesis/episodic layering maps onto [[entities/5w-h|5W+H]]: objects ≈ who/what atoms, thesis hyper-edges ≈ complete user statements/preferences, episodic hyper-edges ≈ source turns (when/where). "Structure helps small models" is support; "BeamSearch costs 6.6 min/query and best-7B is only 0.27" is the cautionary counterweight — on-device KG retrieval must trade accuracy for a fast traversal to stay interactive.

## Method

- **Hybrid KG storage (three layers):** object vertices/edges (atomic concepts), thesis vertices (complete atomic thoughts as hyper-edges), episodic vertices (source passages as hyper-edges). Neo4j + Milvus.
- **Six traversal families:** A* (three heuristics), WaterCircles (BFS, fastest ~0.30 min/query), BeamSearch (parallel paths, slowest ~6.59 min/query), plus hybrids. Traversal restrictable by vertex type.
- **Eval:** LLM-as-judge (Qwen2.5-7B, binary) + Exact Match, on DiaASQ, HotpotQA, TriviaQA.

## Key results

- **By scale (mean LLM-judge):** GPT-4o-mini 0.77, DeepSeek-V3 0.70, Llama-3.1-8B 0.44, Qwen2.5-7B 0.27, DeepSeek-R1-7B 0.19.
- **Algorithm × scale:** 7B/8B → BeamSearch + episodic restriction; 14B+ → unrestricted hybrid BeamSearch+WaterCircles; thesis vertices critical for small models (excluding them hurts), tolerable-to-prunable for large.
- **Cost:** WaterCircles ~0.30 vs BeamSearch ~6.59 min/query. Beats GraphRAG on HotpotQA (60.0% EM, +14.1%) but underperforms a fine-tuned RAG on TriviaQA by 17.8%.

## Critical appraisal

A genuinely systematic ablation matrix that studies the small-vs-large axis most memory work ignores. Cautions: the LLM-judge is a weak Qwen2.5-7B (noisy small-model scores); "personalisation" is operationalised as long-history multi-hop/temporal QA over forum data, not true per-user preference modelling; and retrieval quality still depends on an under-reported upstream extraction step.

> ⚠ 0.6B caution: the accuracy-winning small-model retriever (BeamSearch) is wholly impractical for interactive/mobile use (~6.6 min/query), and best 7B mean accuracy (0.27) is far below cloud (0.77) — structure *narrows* but does not *close* the small-vs-large gap. Supports "small models need scaffolding"; partially contradicts "on-device can match cloud".

## Related

- [[sources/papers/mem0]] — memory extraction/update baseline (cloud)
- [[sources/papers/graph-agent-memory-survey]] — the taxonomy this instantiates
- [[entities/graph-rag]] — KG-backed user memory
- [[entities/5w-h]] — object/thesis/episodic ≈ who-what / preference / when-where
- [[entities/qwen3-0.6b]] — the sub-1B target below the 7B floor tested here
- [[topics/personalisation]] — structured user memory for small models

## Sources

- Menschikov, Evseev, Dochkina, Kostoev, Perepechkin, Anokhin, Semenov, Burnaev (2026) — arXiv:2506.17001 — [arxiv.org/abs/2506.17001](https://arxiv.org/abs/2506.17001)
