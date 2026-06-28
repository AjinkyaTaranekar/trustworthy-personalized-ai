---
title: "Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning"
type: source
tags: [reasoning, tool-use, architecture, agents, sft, rl, grpo, small-model, trade-off]
sources:
  - https://arxiv.org/html/2511.10037v1
  - https://github.com/weixiaolong94-hub/Beyond-React
updated: 2026-05-26
status: current
---

# Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning

**A dual-model framework that decouples a fine-tuned Planner (generating DAG-structured execution plans) from an Executor (GPT-4o), trained with SFT then GRPO; tested across Qwen3-0.6B through 8B, with GRPO found to be unstable at 0.6B.**

## Summary

Wei et al. (2025) argue that ReAct-style interleaved reasoning falls into "local optimisation traps" — each step is locally plausible but the sequence is globally suboptimal. Their fix is to front-load all planning into a dedicated Planner model that generates a complete Directed Acyclic Graph (DAG) before execution begins. Nodes in the DAG are tool calls; edges encode dependencies. Steps with no dependency between them can be parallelised by the Executor. The Planner is trained with SFT on ground-truth DAG plans, then refined with GRPO using a hierarchical reward that penalises structural errors (cycles, disconnected nodes) while rewarding plan fidelity to the ground truth.

Key results on StableToolBench with Qwen3-8B Planner + GPT-4o Executor: 59.8% end-to-end success rate vs 48.2% for GPT-4 ReAct baseline. Average 2.29 inference steps per task — fewer than all compared methods. Planner alone (planning evaluation, no execution): 80.3% exact-match on Easy, 31.9% on Hard.

## Model sizes tested

Planner trained at four scales: Qwen3-0.6B, 1.7B, 4B, 8B. All sizes trained with SFT successfully. **Qwen3-0.6B excluded from GRPO training due to instability** — RL phase could not be stabilised at this scale. The executor in all end-to-end evaluations is GPT-4o, not a small model.

## Dataset — ComplexTool-Plan

Auto-generated from 4,535 tool APIs via ModelScope using a three-stage pipeline:
1. `01_workflow.py` — generate DAG workflows from tool combinations
2. `02_reverse.py` — reverse-engineer user queries from DAGs
3. `03_replan.py` — validate and filter for high-fidelity plan–query pairs

Released: 3,000 SFT instances (three difficulty levels: Easy / Medium / Hard) + 787 RL instances (frontier difficulty — not trivial, not intractable). Available at [github.com/weixiaolong94-hub/Beyond-React](https://github.com/weixiaolong94-hub/Beyond-React).

## Relevance to Experiment 3

**Direct load-bearing findings:**

1. **GRPO instability at 0.6B is empirically confirmed.** The June 15–25 RL post-training slot for the Thinker must account for this. Three mitigation options documented in [[experiments/thinker-executor-experiment]] §9: skip Thinker RL entirely; use RAGEN/StarPO instead of GRPO; or upgrade Thinker to 1.7B if SFT-only fails.

2. **DAG insight → stage-grouping schema.** The full DAG is too complex for reliable generation at 0.6B after SFT. The `<stage>` grouping schema in [[experiments/thinker-executor-experiment]] §4.4 captures the same structural insight (independent steps can run in parallel; dependent steps wait) at a complexity a 0.6B model can learn from examples alone. Steps within a `<stage>` are parallel; stages are sequential.

3. **ComplexTool-Plan not used directly.** The dataset is calibrated for 8B models and uses a 4,535-API toolset far larger than the project's 4-tool Executor (`python_execute`, `web_search`, `read_url`, `get_datetime`). The generation scripts (`03_replan.py` especially) inform the synthesised Branch C (`executor_replan`) construction.

4. **Small Planner + small Executor gap.** Beyond ReAct pairs a small Planner with GPT-4o Executor. The present experiment is the first to test both at 0.6B under on-device and constitutional constraints — the novelty claim stands.

## Limitations noted in paper

- One-pass planning: no re-planning when Executor encounters unanticipated failures (the Branch C loop in Experiment 3 addresses this)
- No clarification mechanism: ambiguity always resolved by calling more tools, never by asking the human (the Branch B loop addresses this)
- No constitutional constraints
- RL instability at 0.6B left unresolved

## Related

- [[experiments/thinker-executor-experiment]] — primary consumer of this paper's findings
- [[sources/papers/reason-plan-react]] — also decouples planner from executor; enterprise task focus
- [[sources/papers/replacing-thinking-with-tool-usage]] — adjacent finding on tool use vs thinking at small scale
- [[entities/qwen3-0.6b]] — the base model tested here
- [[topics/tool-use-and-verification]] — tool delegation theory

## Sources

- Wei et al. (2025) — arXiv:2511.10037 — [arxiv.org/html/2511.10037v1](https://arxiv.org/html/2511.10037v1)
- Code + data — [github.com/weixiaolong94-hub/Beyond-React](https://github.com/weixiaolong94-hub/Beyond-React)
