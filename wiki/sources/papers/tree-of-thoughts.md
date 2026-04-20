---
title: "Tree of Thoughts: Deliberate Problem Solving with LLMs"
type: source
arxiv_id: 2305.10601v2
authors: Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan
year: 2023
venue: NeurIPS
tags: [reasoning, search, deliberation, tot]
sources:
  - docs/Assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models (2305.10601v2).pdf
  - docs/Literature Notes/Tree of Thoughts Deliberate Problem Solving with Large Language Models (2305.10601v2).md
updated: 2026-04-19
status: current
---

# Tree of Thoughts (ToT)

**Generalises chain-of-thought into a search tree of partial "thoughts" with self-evaluation, lookahead, and backtracking.**

## What it does
Frames inference as search over coherent text chunks ("thoughts"), with the LLM acting as both proposer and evaluator. On Game-of-24, GPT-4 with CoT solves 4%; ToT solves 74%.

## Why it matters for this thesis
ToT is the canonical answer to the autoregressive-no-backtracking limitation from [[sources/papers/attention-is-all-you-need]]. It recovers deliberation **externally** — the Transformer itself is still feed-forward, but a controller wraps it in a search loop. For this thesis it anchors the "process matters" argument in [[topics/reasoning]]: good answers come from structured exploration, not from better next-token prediction. The trade-off — large inference cost per question — motivates thinking about when deliberation is worth paying for, and which tasks can instead be handled by [[topics/tool-use-and-verification|tool delegation]].

## Related

- [[topics/reasoning]]
- [[sources/papers/chain-of-thought-prompting]] — what ToT generalises
- [[sources/papers/react]] — action-interleaved alternative to pure search
- [[sources/papers/deepseek-r1]] — RL approach to the same problem

## Sources

- `docs/Assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models (2305.10601v2).pdf`
- `docs/Literature Notes/Tree of Thoughts Deliberate Problem Solving with Large Language Models (2305.10601v2).md`
