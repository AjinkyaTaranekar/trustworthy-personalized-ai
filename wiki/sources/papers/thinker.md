---
title: "Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction"
type: source
tags: [tool-use, sft, reasoning, retrieval, small-model]
sources:
  - https://arxiv.org/abs/2511.07943
updated: 2026-07-19
status: current
---

# Thinker: Hierarchical Thinking for Deep Search

**Rather than letting RL discover free-form search behaviour, supervise a small LLM to think in an explicit two-level hierarchy — decompose a hard question breadth-wise into atomic sub-problems, then solve each depth-wise with 0–M retrievals — so the trajectory is logically coherent, controllable and auditable; this SFT recipe matches or beats RL-based deep-search agents at a fraction of the cost, and its relative benefit grows as the model shrinks.**

## Summary

Xu et al. (2025) complain that RL-trained search agents optimise a final-answer reward and so learn *whatever* interaction maximises it — unstructured, hard to control, and able to succeed by trial-and-error rather than sound logic (unacceptable in medicine/law/finance). Thinker instead *imposes* the reasoning structure by SFT on 71K breadth-then-depth trajectories with dual natural-language + logical-function representation, plus a knowledge-boundary gate that skips retrieval when the model already knows a sub-answer. On Qwen2.5-7B it reaches 0.452 avg EM (vs ReSearch 0.411), trains in 24h (vs 71–90h), and hits near-SOTA with 1% of the data. Crucially the relative gain is *larger* at 3B (+7.9%) than 7B (+4.1%). This is close to a proof-of-concept for the project's SFT-first, thinker-executor stance, and a direct counterweight to RL framing.

## Why it matters here

This is the most methodologically aligned paper for the SFT-first thesis: it shows a small model can be *taught* an explicit thinker-style hierarchy purely by supervision, cheaply and sample-efficiently, and that structural scaffolding helps most exactly where the base is weakest (the 3B>7B relative gain). The knowledge-boundary gate is a borrowable mechanism for an on-device system that must avoid needless tool calls, and its logical-coherence metric suite is a ready way to *measure* trustworthy reasoning beyond EM.

## Method

- **Two-level hierarchy:** breadth decomposition into n atomic sub-problems; depth solving of each with 0–M retrievals (0 when known, gated by the knowledge boundary).
- **Dual representation:** each step in natural language *and* as a logical function, so one trajectory drives a dense or a structured/graph (KAG) retriever.
- **Knowledge-boundary gate:** cut retrieval queries 16.0% (TriviaQA) / 17.8% (Bamboogle) at 96.9%/97.8% skip-decision accuracy.
- **Training:** cross-entropy over assistant tokens only on 71K trajectories; optional GRPO on top (F1-shaped reward).

## Key results

- **Qwen2.5-7B:** avg EM 0.452 (vs ReSearch 0.411, +4.1); +GRPO → 0.479.
- **Qwen2.5-3B:** 0.430, +7.9% over ReSearch (larger relative gain than at 7B).
- **Coherence (HotpotQA):** Logical Hierarchy 0.975, Interleaved Solving 0.989 — measurable structure.
- **Sample efficiency:** 1% of data (~710 samples) ≈ 0.406 ≈ ReSearch. Med-Thinker (MedQA) 74.00% vs ReAct 39.22%.

## Critical appraisal

Argues with numbers that *supervising the reasoning structure* beats hoping RL discovers it, and introduces coherence metrics operationalising "trustworthy reasoning". Strong efficiency/sample-efficiency numbers. Caveats: the impressive coherence scores are partly baked in by the training format (a structured-looking trajectory is not proven *faithful* — no adversarial/human-faithfulness eval); the ceiling is the quality of the 71K distilled trajectories; MuSiQue stays weak (0.221) so deep 4-hop chains remain hard.

> ⚠ 0.6B caution: the floor tested is 3B; MuSiQue is weak even at 7B, so a 0.6B thinker attempting genuine multi-hop decomposition may hit a hard reasoning ceiling, and the method presupposes access to high-quality hierarchical trajectories to distil from.

## Related

- [[experiments/thinker-executor-experiment]] — the project's Thinker; this supervises the same hierarchy
- [[sources/papers/ragen]] — the RL counterweight; Thinker argues SFT structure beats RL discovery
- [[sources/papers/search-r1]] — the RL deep-search line Thinker critiques
- [[sources/papers/beyond-react]] — planner decomposition; stage-grouping schema
- [[entities/graph-rag]] — the KAG structured retriever the logical-function representation enables
- [[topics/reasoning]] — measurable trustworthy reasoning
- [[topics/tool-use-and-verification]] — the knowledge-boundary retrieval gate

## Sources

- Xu, Du, Ao, Zhao, Li, et al. (2025) — arXiv:2511.07943 — [arxiv.org/abs/2511.07943](https://arxiv.org/abs/2511.07943)
