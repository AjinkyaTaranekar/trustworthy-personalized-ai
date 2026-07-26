---
title: "OPERA: A Reinforcement Learning–Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval"
type: source
tags: [tool-use, agents, multi-agent, grpo, retrieval, reasoning]
sources:
  - https://arxiv.org/abs/2508.16438
  - https://github.com/Ameame1/OPERA
updated: 2026-07-19
status: current
---

# OPERA: Orchestrated Planner-Executor Architecture

**Multi-hop RAG works better when split into a planner that decomposes a query into dependency-linked sub-goals and a specialised executor pair (analyse-answer + rewrite), each trained separately with a progressive role-specific GRPO variant (MAPGRPO) so credit is assigned per-agent rather than to one monolithic policy — an ablation shows the planner carries almost all the accuracy.**

## Summary

Liu et al. (2026) argue monolithic RAG conflates planning, sufficiency judgement and query reformulation, giving muddy credit assignment and opaque behaviour. OPERA separates three cooperating agents — a Plan Agent (Goal Planning Module) that builds a placeholder-dependency plan graph, and an Analysis-Answer + Rewrite pair (Reason-Execute Module) — plus a Trajectory Memory Component for auditability, and trains each role in its own GRPO stage (MAPGRPO) with a bespoke reward. On Musique EM rises 21.2 (CoT) → 39.7 (full), and a 7B orchestrated system beats Llama-3.1-70B CoT (57.3 vs 54.9 on HotpotQA). The load-bearing result: removing the planner collapses Musique EM by 22.6, versus 5.2 for removing the rewrite executor. This is a direct architectural precedent for the thinker-supervising-thin-executor split.

## Why it matters here

OPERA's Plan Agent ≈ the project's [[experiments/thinker-executor-experiment|Thinker]], its executor pair ≈ the thin Executor, and the Trajectory Memory is exactly the auditable-trace mechanism a trust-focused system wants. The ablation (planner carries the accuracy; per-role rewards beat a monolithic reward) is strong support for investing capacity in the thinker and keeping executors thin.

## Method

- **Goal Planning Module:** Plan Agent decomposes a query into sub-goals with placeholder dependencies (later sub-goals reference earlier answers).
- **Reason-Execute Module:** Analysis-Answer Agent judges information sufficiency and extracts answers; when insufficient, the Rewrite Agent reformulates and retrieval retries.
- **MAPGRPO:** three sequential GRPO stages (Plan → Analysis-Answer → Rewrite), each with a hand-designed reward (plan logic/structure/exec; answer EM weight β=0.65; rewrite NDCG weight ω1=0.9), seeded by DeepSeek-R1-scored high-reward samples.
- **Models:** Plan + Analysis-Answer are Qwen2.5-7B; Rewrite is Qwen2.5-3B; retriever BGE-M3, top-5.

## Key results

- **HotpotQA EM 57.3** (+11.6), 2Wiki EM 60.2 (+15.9), Musique EM 39.7 (+15.4); gains scale with difficulty.
- **Ablation (Musique EM):** CoT 21.2 → SFT 24.3 → GRPO 34.8 → MAPGRPO 39.7; **−22.6 without the Plan Agent**, −5.2 without Rewrite.
- 7B OPERA beats 70B single-shot CoT; error analysis: retrieval errors dominate (47.8%).

## Critical appraisal

The transferable win is the ablation — an explicit planner drives almost all the gain and per-role rewards beat a monolithic reward. But this is a *heavyweight* instantiation (two 7B agents + a 3B agent + DeepSeek-R1 supervision), and the interpretability story (Trajectory Memory + placeholder plans) is asserted, not measured (no human-trust/faithfulness evaluation, only accuracy).

> ⚠ 0.6B caution: numbers come from Qwen2.5-7B/3B agents plus DeepSeek-R1-scored data — nothing here shows the split works at 0.6B, and it uses *separate* models per role rather than one small model wearing multiple hats. Use OPERA for the *shape* (heavy planner, thin executor, trajectory memory, per-role credit), not feasibility at sub-1B.

## Related

- [[experiments/thinker-executor-experiment]] — the project's planner(Thinker)/executor split this precedes
- [[sources/papers/reason-plan-react]] — same decoupling; enterprise focus
- [[sources/papers/small-agents-collaborate]] — corroborates the planner-carries-the-gain finding
- [[sources/papers/beyond-react]] — planner-centric DAG framework
- [[sources/papers/deepseekmath]] — the GRPO MAPGRPO extends per-role
- [[topics/tool-use-and-verification]] — tool delegation and auditable traces

## Sources

- Liu, Liu, Yuan, Cao, Sun, Peng, Chen, Li, Ma (2026) — arXiv:2508.16438 — [arxiv.org/abs/2508.16438](https://arxiv.org/abs/2508.16438)
- Code — [github.com/Ameame1/OPERA](https://github.com/Ameame1/OPERA)
