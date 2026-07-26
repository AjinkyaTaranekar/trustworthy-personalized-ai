---
title: "Can Small Agents Collaborate to Beat a Single Large Language Model?"
type: source
tags: [multi-agent, agents, small-model, tool-use, reasoning, memory, qwen]
sources:
  - https://arxiv.org/abs/2601.11327
updated: 2026-07-18
status: current
---

# Can Small Agents Collaborate to Beat a Single Large Language Model?

**A minimal multi-agent system of small models — one orchestrator plus a few restricted-communication specialised sub-agents sharing a structured memory — can match or occasionally beat a much larger single-agent LLM on tool-intensive benchmarks, and the gains come almost entirely from reasoning at the orchestrator (planning), not from sub-agent scale: agentic performance is planner-limited, not executor-limited.**

## Summary

Żywot, Chen, Yuan, Søgaard and de Rijke (Amsterdam / ETH / Copenhagen, 2026) build a deliberately minimal MAS — an orchestrator that decomposes tasks and dispatches to a Web Searcher, Coder and File Inspector, with sub-agents talking only to the orchestrator and a shared four-bucket structured memory — and wire Qwen3 (1.7B–32B) into it. Across GAIA, GPQA, AIME, MuSiQue and HLE, an 8B MAS with orchestrator thinking essentially matches a 32B single agent with direct tools (AIME 55 vs 45 win; GAIA 23.0 tie; GPQA/MuSiQue narrow losses) while using ~43% fewer tokens, ~32% fewer tool calls, and running 2.8–4.2× faster. The load-bearing finding is that **explicit reasoning pays off at the planner, not the executor** — which directly supports concentrating constitutional reasoning in a planner and keeping executors thin.

## Why it matters here

The strongest external argument for the project's thinker/executor division of labour: separate planning from execution, and put the "thinking" (and, by extension, the constitution) in the orchestrator. "Planner-limited, not executor-limited" justifies spending capacity/constitution budget on the reasoner rather than the tool-callers, and the efficiency numbers are concrete evidence that a small MAS beats a monolithic large model on cost and latency — relevant to an on-device argument. Structured memory as a load-bearing component maps onto the [[topics/personalisation|memory/personalisation]] strand.

## Method

- **Orchestrator** — sole locus of global reasoning: decomposes the task, selects tools/sub-agents, integrates results.
- **Sub-agents** — Web Searcher (retrieval), Coder (sandboxed Python), File Inspector (file parsing). **Restricted communication**: sub-agents never talk to each other, so all cross-task reasoning lives in the orchestrator.
- **Structured memory** — four buckets (query analysis, previous steps, tool results, sub-goals) that keep context compact.
- **Design of experiment** — "thinking" toggled independently at orchestrator and sub-agent levels (None / sub-agent-only / orchestrator-only / all); "Sub-agent" config vs a "Direct" single-model-holds-all-tools config; Qwen3 mixed across roles at 1.7B / 8B / 32B to separate their contributions.

## Key results

- **8B MAS vs 32B single-agent (direct tools):** AIME 55.0 vs 45.0 (MAS +10); GAIA 23.0 tie; GPQA 58.6 vs 60.1; MuSiQue 14.0 vs 15.0; HLE ~4 vs ~0 (both fail).
- **Orchestrator thinking gains:** GAIA +3.7, GPQA +5.1, **AIME +18.4**. Sub-agent-only thinking is marginal-to-negative and adds ~6.1 s latency (orchestrator thinking adds only +0.6 s).
- **Planner-limited:** with orchestrator thinking on, GAIA is ~flat in sub-agent size (23.0 / 23.0 / 23.6 for 1.7B / 8B / 32B sub-agents); without thinking, sub-agent size matters more (7.9 / 12.7 / 13.3).
- **Efficiency:** MAS holds 9.1–11.7K prompt tokens vs 10.1–19.5K for direct configs (~43% fewer); 476 vs 698 tool calls (~32% fewer); 7.9 s vs 21.8–32.9 s (2.8–4.2× faster).
- **Ablation:** removing structured memory collapses AIME by 18.3 and GPQA by 8.1 — memory is load-bearing for reasoning-heavy tasks.
- **Tools are double-edged:** they help retrieval/multi-hop tasks but can override correct parametric knowledge (GPQA −5.0).

## Critical appraisal

Unusually clean, ablation-rich design that isolates planner vs executor and thinking placement. Trust the "planner-limited" and orchestrator-only-thinking findings most. Treat the title cautiously: it is really "match, occasionally beat" — a clear AIME win but GPQA/MuSiQue losses and a GAIA tie. "Large" here is only 32B, so it may not hold against frontier-scale single agents, and only the Qwen family is tested.

> ⚠ Conflict / caution: the smallest model is **1.7B** and orchestrators are 8B — the paper does not demonstrate the effect at 0.6B, and sub-agent scale barely matters *only once an 8B orchestrator reasons*, so a very small planner may not reproduce the gains. The "tools/thinking can override correct knowledge or cause premature closure" finding is a reliability caution for tool-augmented small agents.

## Related

- [[experiments/thinker-executor-experiment]] — planner(Thinker)/executor split this paper supports empirically
- [[sources/papers/reason-plan-react]] — same planner-supervises-executor structure; enterprise focus
- [[sources/papers/beyond-react]] — planner-centric framework; small-model tool-use
- [[sources/papers/replacing-thinking-with-tool-usage]] — small-model tool-use vs thinking
- [[entities/qwen3-0.6b]] — the (smaller) sibling of the Qwen3 models tested here
- [[topics/tool-use-and-verification]] — tool delegation theory
- [[topics/personalisation]] — structured memory as a load-bearing agent component

## Sources

- Żywot, Chen, Yuan, Søgaard, de Rijke (2026) — arXiv:2601.11327 — [arxiv.org/abs/2601.11327](https://arxiv.org/abs/2601.11327)
