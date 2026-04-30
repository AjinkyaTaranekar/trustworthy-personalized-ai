---
paper id: 2307.07697v6
title: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph"
authors: [Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, Jian Guo]
publication date: 2023-07-15T03:31
abstract: "LLMs struggle with hallucination in deep knowledge-reasoning tasks. ToG addresses this by introducing the LLM⊗KG paradigm: the LLM actively explores a knowledge graph via iterative beam search, ranks relation candidates, prunes paths, and reasons over retrieved triples until sufficient information is gathered. Training-free; SOTA on 6/9 KGQA datasets; small LLMs with ToG can exceed GPT-4 on certain tasks."
comments: Accepted by ICLR 2024
pdf: "[[Assets/Think-on-Graph Deep and Responsible Reasoning of Large Language Model on Knowledge Graph (2307.07697v6).pdf]]"
url: https://arxiv.org/abs/2307.07697v6
tags: [reasoning, tool-use, ontology, graph-rag, foundations]
---

## Key Claims

- **Think-on-Graph (ToG)**: LLM ⊗ KG tight-coupling paradigm — LLM *actively* explores KG iteratively (beam search: Search → Prune → Reason at each depth), contrasted with the weaker LLM ⊕ KG (retrieve once, augment prompt, answer).
- Three advantages: **deep reasoning** (multi-hop traversal), **knowledge traceability** (reasoning paths are explicit and user-editable), **flexibility** (plug-and-play for any LLM/KG, no training cost).
- SOTA on 6/9 KGQA datasets; ToG with LLaMA2-70B can exceed GPT-4 in certain knowledge-intensive scenarios — efficiency argument for on-device small models.
- Training-free at inference time; KG can be updated without retraining the LLM.

## Thesis Relevance

Primary literature reference for the [[entities/graph-rag]] entity and Experiment 6 (ontology as verifier). The LLM ⊗ KG paradigm maps to the thesis's Approach B (post-hoc ontology verification with iterative claim-checking). Knowledge traceability — explicit, inspectable reasoning paths — directly supports the thesis's scrutable reasoning goal. The training-free property is critical for the on-device deployment constraint.

## Questions / Open Issues

- ToG uses general KGs (Freebase, Wikidata); does it transfer to domain-specific ontologies (the thesis's personal-domain user knowledge)?
- Beam search requires multiple LLM calls per depth level; latency at 0.6B on-device may be prohibitive for real-time conversation.
- The thesis's user ontology must be updated as new user information arrives — ToG assumes a static KG; dynamic KG updating is an open problem.
