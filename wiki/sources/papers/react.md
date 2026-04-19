---
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
type: source
arxiv_id: 2210.03629v3
authors: Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao
year: 2022
venue: ICLR
tags: [tool-use, reasoning, agents, interleaved]
sources:
  - docs/Assets/ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3).pdf
  - docs/Literature Notes/ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3).md
updated: 2026-04-19
status: current
---

# ReAct

**Interleaves reasoning traces and actions in a single generation loop so the
model can think, act (call a tool or API), observe the result, and continue
thinking — recovering from errors mid-trajectory.**

## What it does
Generates reasoning and actions in the same stream. On HotpotQA and FEVER
reduces hallucination and error propagation by letting the model check facts
against a Wikipedia API. On ALFWorld and WebShop beats imitation/RL baselines
by 34% and 10% absolute.

## Why it matters for this thesis
ReAct is the **agentic** counterpart to [[sources/papers/pal]]: PAL delegates
computation once, ReAct delegates continuously. This is the template the
pipeline's tool-use traces follow — `<think>` / tool-call / observation /
`<think>` again. Crucial for [[topics/tool-use-and-verification]] because it
converts hallucinations from silent failures into observable ones: the tool
either returns or throws, and the reasoning can course-correct. Also the
progenitor of [[sources/papers/search-r1]]'s RL formulation.

## Related

- [[topics/reasoning]]
- [[topics/tool-use-and-verification]]
- [[sources/papers/pal]]
- [[sources/papers/search-r1]]
- [[sources/papers/mcp-multi-agent]]

## Sources

- `docs/Assets/ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3).pdf`
- `docs/Literature Notes/ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3).md`
