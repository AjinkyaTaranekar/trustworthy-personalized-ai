---
title: "Graph-based Agent Memory: Taxonomy, Techniques, and Applications"
type: source
tags: [memory, graph-memory, retrieval, personalisation]
sources:
  - https://arxiv.org/abs/2602.05665
  - https://github.com/DEEP-PolyU/Awesome-GraphMemory
updated: 2026-07-19
status: current
---

# Graph-based Agent Memory: Taxonomy, Techniques, and Applications

**A survey arguing that graph structure is the natural, unifying substrate for agent memory — nodes as memory units, edges as semantic/temporal/causal/logical relations — organised into a lifecycle taxonomy (extraction → storage → retrieval → evolution) that elevates memory from a passive log into an active structured model of experience.**

## Summary

Yang et al. (DEEP-PolyU et al., 2026) survey graph-based agent memory, arguing graphs generalise linear/vector/KV memory by making relationships first-class. They organise the field along four axes (temporal scope, functional role, structural organisation, cognitive component) and a four-stage lifecycle, catalogue three retrieval paradigms (semantic / structured / policy-based) and two evolution modes (internal self-evolving / external self-exploration), and provide a scenario-based benchmark taxonomy with a curated system inventory. Best used as the related-work backbone for a KG-user-memory chapter — it guides *design* more than it adjudicates methods (it reports almost no controlled numbers), and it is cloud-agent-centric with privacy and small-model feasibility under-served, which is precisely the gap this project occupies.

## Why it matters here

Its four axes and the extraction → storage → retrieval → evolution lifecycle give a ready framework to position a [[entities/5w-h|5W+H]] personalisation graph: episodic/temporal-graph structures (valid vs transaction time, monotonicity constraints) map onto when/where, KG triples onto who/what, hierarchical summarisation onto how, experience memory onto preference/interaction history. Cite it to justify structuring user memory as a *temporal graph* rather than a flat log — then differentiate on privacy + on-device compute, the dimensions it under-serves.

## Framework

- **Four axes:** temporal scope (short/long-term); functional role (knowledge/static vs experience/dynamic); structure (non-structural buffers/vectors/KV vs structural graphs); cognitive components (semantic, procedural, associative, working, episodic, sentiment).
- **Lifecycle:** extraction (NER/relation extraction/summarisation; trajectory segmentation; multimodal) → storage (KG triples, hierarchical trees/DAGs, temporal/bi-temporal graphs, hypergraphs, hybrid external+scratchpad) → retrieval (semantic / structured / policy-based RL-agentic) → evolution (consolidation, updating, forgetting; reorganisation via community detection/pruning/link prediction).
- **Representative systems:** AriGraph, Mem0, MemTree, Graphiti, HyperGraphRAG, Zep, G-Memory.

## Critical appraisal

An unusually comprehensive, well-organised map of the 2025–26 graph-memory landscape; the lifecycle and retrieval-paradigm taxonomies are genuinely useful scaffolding, and the companion repo aids literature (not results) reproducibility. Trustworthy as a taxonomy and reading list; do not cite it for empirical claims — it reports almost no controlled numbers, and the "graph generalises everything" framing is somewhat self-serving for a graph-learning group (it underweights graph construction/traversal cost).

> ⚠ Caution / gap: the survey implicitly assumes strong (often cloud) LLMs for extraction, graph reasoning, and RL/agentic retrieval — it does not demonstrate any of this on a sub-1B model, and despite naming Personalization as a scenario it gives privacy, access control, and forgetting only cursory treatment. It supports the *design* (graph memory for personalisation) but is silent on *small-scale feasibility and privacy* — the project's novelty gap.

## Related

- [[sources/papers/memory-age-ai-agents]] — the forms/functions/dynamics companion survey
- [[sources/papers/mem0]] — a KG-memory system it catalogues
- [[sources/papers/personalai]] — concrete scale-varying instantiation of these ideas
- [[entities/graph-rag]] — KG-backed user memory
- [[entities/5w-h]] — the user-model schema this framework can structure
- [[topics/personalisation]] — temporal graph over flat log
- [[topics/security-and-privacy]] — the privacy/retention gap to claim

## Sources

- Yang, Zhou, Xiao, Dong, Zhuang, Zhang, Wang, Hong, Yuan, Xiang, Chen, Zhou, Zhang, Liu, Su, Wang, Chang, Huang (2026) — arXiv:2602.05665 — [arxiv.org/abs/2602.05665](https://arxiv.org/abs/2602.05665)
- System list — [github.com/DEEP-PolyU/Awesome-GraphMemory](https://github.com/DEEP-PolyU/Awesome-GraphMemory)
