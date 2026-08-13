---
title: "Reason-Plan-ReAct: A Reasoner-Planner Supervising a ReAct Executor for Complex Enterprise Tasks"
type: source
tags: [tool-use, agents, multi-agent, reasoning, architecture, on-device, privacy]
sources:
  - https://arxiv.org/abs/2512.03560
  - https://github.com/giargiapower/RP-ReAct
updated: 2026-07-18
status: current
---

# Reason-Plan-ReAct (RP-ReAct)

**A two-tier agent that decouples a strong-reasoning Reasoner-Planner (RPA) — which plans and continuously re-plans sub-steps — from thin ReAct Proxy-Execution Agents (PEA) that actually call tools, plus a context-offloading trick that keeps large tool outputs from overflowing the small context windows of privacy-driven local open-weight models; it wins on hard multi-step enterprise tasks and is far more stable across models.**

## Summary

Molinari and Ciravegna (2025; AAAI-2026 enterprise-tasks workshop) diagnose two structural failures of single-agent ReAct on complex enterprise work: a monolithic plan-execute loop lets one low-level execution error derail the whole plan ("trajectory deviation"), and the privacy requirement to run **local open-weight models** means small context windows that large tabular tool outputs (SQL/CSV) rapidly consume. RP-ReAct separates a high-level planner (a Large Reasoning Model) from one or more thin ReAct executors, and offloads all but the first `TT=100` tokens of any large tool output to external storage, fetched on demand. On ToolQA's hard split it beats ReAct (e.g. SciREX 0.26 vs 0.14) and is markedly more stable across six open-weight models; on easy tasks the lighter ReAct loop often wins. This is a clean external instantiation of the dissertation's thinker/planner-supervising-executor pattern with an explicitly privacy-first, on-device motivation.

## Why it matters here

The paper's motivation is nearly the dissertation's: *because* data privacy forces local open-weight deployment, context is scarce, so architecture must protect it. Hooks: RPA ≈ constitutional reasoner/Thinker, PEA ≈ thin tool [[experiments/thinker-executor-experiment|Executor]]; the `TT=100` external-offload pattern is a transferable engineering defence for keeping small-context on-device models from drowning in tool outputs; and the Combined Performance Score (CPS = Saturation × MaxAcc) is a template for reporting robustness across model scales in an evaluation chapter.

## Method

- **RPA (Reasoner-Planner Agent).** Holds the global strategy, plans each sub-step with a Large Reasoning Model, analyses returned results, and re-plans dynamically on error. Never touches tools directly. Emits queries wrapped in `<|begin_search_query|>` tags.
- **PEA (Proxy-Execution Agent), one or more.** Translates each sub-step into concrete tool calls via a ReAct Think-Act-Observe loop, and returns a compact result in `<|begin_search_result|>` tags.
- **Context-saving offload.** Tool outputs over the threshold inject only their first `TT=100` tokens; the remainder is stored externally and retrieved on demand — the key defence against context overflow, and what makes small-context local models viable.
- **Budgets.** ReAct N=20; Reflexion N=20 (≤3 reflections); RP-ReAct N=10 for the RPA and N=10 per PEA (~100 total). Temperature 0.6.

## Key results

- **Hard split (RP-ReAct superior):** SciREX 0.26 vs ReAct 0.14; Coffee 0.23 vs 0.11; Airbnb 0.38 vs 0.32. Structured planning prevents trajectory deviation on multi-step tasks.
- **Easy split (mixed):** ReAct's simpler loop often wins (e.g. Yelp 0.90 vs 0.53) — planning overhead can disrupt otherwise-short trajectories.
- **Stability:** lower cross-model variance and better CPS (Hard Coffee CPS 0.36 vs 0.12).
- **Planning beats brute force:** giving ReAct a 100-step budget (vs 20) improved hard-domain accuracy by only ~4.8% on average.
- **Capacity floor:** sub-10B models fail hard tasks (≤0.11) across every scaffold. Reflexion was the worst baseline overall.

## Critical appraisal

Strong structural diagnosis and a practical fix, with an informative 100-step ReAct control and public code. Treat the blanket "superior" framing cautiously: it wins clearly only on **hard** tasks, loses on easy ones, and the step budgets are asymmetric (ReAct 20 vs RP-ReAct ~100), so some hard-task gain may be extra compute rather than pure architecture. Absolute hard-task accuracies remain low (0.23–0.38).

> ⚠ Conflict / caution: RP-ReAct depends on a **Large Reasoning Model** as planner and collapses below ~10B — it validates the planner-executor *architecture* but is not evidence that a 0.6B planner works. It also reports [[sources/papers/reflexion|Reflexion]]-style verbal self-correction as the weakest baseline on tool-heavy tasks.

## Related

- [[experiments/thinker-executor-experiment]] — the project's planner(Thinker)/executor split; RP-ReAct is its closest enterprise analogue
- [[sources/papers/beyond-react]] — planner-centric DAG framework; same decoupling instinct
- [[sources/papers/small-agents-collaborate]] — "planner-limited, not executor-limited" corroborates putting reasoning in the planner
- [[sources/papers/reflexion]] — the self-correction baseline RP-ReAct finds weakest here
- [[topics/tool-use-and-verification]] — tool delegation and verification
- [[topics/security-and-privacy]] — local-first / on-device privacy argument the paper shares

## Sources

- Molinari, Ciravegna (2025) — arXiv:2512.03560 — [arxiv.org/abs/2512.03560](https://arxiv.org/abs/2512.03560)
- Code — [github.com/giargiapower/RP-ReAct](https://github.com/giargiapower/RP-ReAct)
