---
title: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph"
type: source
tags: [ontology, retrieval, reasoning, verification]
sources:
  - https://arxiv.org/abs/2307.07697
  - https://github.com/IDEA-FinAI/ToG
updated: 2026-07-20
status: current
---

# Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graph

**Think-on-Graph (ToG) treats a knowledge graph as an agent the LLM interactively explores via beam search over entities and relations, tightly coupling LLM reasoning with KG retrieval (LLM⊗KG) so the model performs deep, multi-hop, traceable, and correctable reasoning — achieving new SOTA on 6 of 9 QA/fact-checking datasets without any training.**

## Summary

Sun et al. (IDEA Research et al., ICLR 2024) address three LLM weaknesses — hallucination, stale/incomplete parametric knowledge, and unexplainable answers — by making the LLM steer a KG search at every hop rather than issuing one query ("LLM⊕KG"). ToG loops {search relations → LLM prunes top-N; search entities → LLM prunes top-N; LLM checks sufficiency}, keeping explicit, editable, auditable reasoning paths. The load-bearing result for a small-model thesis is that **ToG with Llama2-70B beats CoT with GPT-4** (CWQ +18.5→+23.5 over CoT), i.e. structured graph access substitutes for parameter scale — the clearest evidence a graph/ontology memory can compensate for a small model's limited parametric knowledge. The cost is many sequential LLM calls (2ND+D+1). This is the KG-grounded responsible-reasoning anchor for the ontology/graph-memory strand.

## Why it matters here

It supplies the mechanism (LLM as agent over a KG via beam search) behind the project's GraphRAG / knowledge-graph reasoning claims, and its explicit, auditable, correctable reasoning paths map onto the trustworthy-reasoning and non-hallucination constitution principles. Central 0.6B argument: ToG makes a smaller model beat GPT-4+CoT, motivating giving a 0.6B student a KG/ontology to reason over rather than relying on parametric recall — tempered by the multi-call cost, which is exactly the efficiency constraint an on-device privacy harness must respect. Use **ToG-R** (cheaper, `ND+D+1` calls, random entity sampling) and the depth-3 plateau to justify shallow, resource-conscious graph walks.

## Method

- **Phases:** initialise topic entities → iterative beam search (relation exploration + LLM prune; entity exploration + LLM prune) → LLM sufficiency check → answer or continue to `D_max` (default 3), width N=3.
- **ToG vs ToG-R:** ToG keeps (s,r,o) triples with LLM entity pruning (max info, higher cost); ToG-R keeps relation chains with random entity sampling (`ND+D+1` calls, robust when intermediate entities are unfamiliar). Both training-free, plug-and-play across LLMs (ChatGPT/GPT-4/Llama2-70B) and KGs (Freebase/Wikidata).

## Key results

- **SOTA on 6/9** (ToG w/GPT-4): WebQSP 82.6, GrailQA 81.4, QALD10 53.8, WebQ 57.9, Zero-Shot RE 88.3, Creak 95.6.
- **CoT → ToG gains:** Llama2-70B +18.5 (CWQ), GPT-4 +23.5; **ToG+Llama2-70B > CoT+GPT-4**.
- **Ablations:** LLM pruning ≫ BM25/SentenceBERT; Freebase > Wikidata; depth/width plateau beyond 3.
- **Weak spot:** single-hop (Simple Questions 66.7 vs 85.8 fine-tuned SOTA).

## Critical appraisal

The strongest available demonstration that grounding LLM reasoning in an explicit graph yields answers that are simultaneously more accurate, explainable, and correctable — "responsible reasoning" follows from the editable-path design, not marketing. Cautions: the `2ND+D+1` LLM-call cost is a real latency/small-model deployment tax; the single-hop weakness implies the machinery is overkill (and error-prone) when one lookup would do (a router before graph search is implied, not built); three of the six SOTA wins are on datasets where prior baselines were weak, so the multi-hop KBQA wins (WebQSP, GrailQA) are the load-bearing ones.

## Related

- [[topics/ontology-integration]] — ontology as KB / verifier; ToG is the flagship mechanism
- [[entities/graph-rag]] — KG-backed reasoning/memory
- [[sources/papers/search-r1]] — RL-trained search-tool use (contrast: ToG is training-free)
- [[sources/papers/rag-original]] — text RAG vs graph-structured retrieval
- [[sources/papers/thinker]] — the KAG structured retriever ToG-style reasoning enables
- [[topics/tool-use-and-verification]] — traceable, correctable reasoning paths
- [[topics/personalisation]] — a per-user preference graph explored the ToG way

## Sources

- Sun, Xu, Tang, Wang, Lin, Gong, Ni, Shum, Guo (2024) — arXiv:2307.07697 (ICLR 2024) — [arxiv.org/abs/2307.07697](https://arxiv.org/abs/2307.07697)
- Code — [github.com/IDEA-FinAI/ToG](https://github.com/IDEA-FinAI/ToG)
