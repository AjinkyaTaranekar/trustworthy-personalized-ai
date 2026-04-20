---
title: "Search-R1: Training LLMs to Reason and Leverage Search Engines with RL"
type: source
arxiv_id: 2503.09516v5
authors: Jin, Zeng, Yue, Yoon, Arik, Wang, Zamani, Han
year: 2025
tags: [tool-use, rl, search, retrieval]
sources:
  - docs/Assets/Search-R1 Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2503.09516v5).pdf
  - docs/Literature Notes/Search-R1 Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2503.09516v5).md
updated: 2026-04-19
status: current
---

# Search-R1

**Trains the LLM itself — via RL with outcome-based rewards — to interleave search-engine calls inside step-by-step reasoning, rather than relying on prompting a pre-trained model to do so correctly.**

## What it does
Extends ReAct-style agentic reasoning to a trained setting. Uses retrieved- token masking to stabilise RL, outcome-only rewards. On seven QA datasets improves over RAG baselines by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B).

## Why it matters for this thesis
Search-R1 is the **direct template** for the `web_search` tool-integrity reward in this pipeline. It proves two useful things: (a) the model **learns** when to search — a prompted model often calls search too often or too rarely — and (b) Qwen 3B is already large enough for this to work, which is reassuring for [[entities/qwen3-0.6b|Qwen3-0.6B]]. The finding also informs the ablation design: Condition D (full behavioural rewards) is expected to beat Condition C (format+accuracy only) partly through better tool-trigger decisions, and Search-R1 quantifies how large that delta can be.

## Related

- [[topics/tool-use-and-verification]]
- [[topics/reasoning]]
- [[sources/papers/react]] — pre-RL agentic template
- [[sources/papers/deepseek-r1]] — sibling RL-for-reasoning work
- [[entities/constitution]]

## Sources

- `docs/Assets/Search-R1 Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2503.09516v5).pdf`
- `docs/Literature Notes/Search-R1 Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2503.09516v5).md`
