---
title: Trustworthy Reasoning
type: topic
tags: [reasoning, rl, sft, cot, explainability]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Assets/Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3).pdf
  - docs/Assets/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).pdf
  - docs/Assets/PAL Program-aided Language Models (2211.10435v2).pdf
updated: 2026-04-19
status: stub
---

# Trustworthy Reasoning

**How to make an LLM's reasoning honest, verifiable, and robust — not just fluent.**

## Summary
Standard LLMs generate "sociopath yapper" explanations: post-hoc rationalisations that look like reasoning but are just next-token prediction over an essay shape. This thesis frames *trustworthy* reasoning as three properties: (a) **process fidelity** — intermediate steps are what actually drove the answer, (b) **delegation honesty** — computation that belongs in a tool goes to a tool, and (c) **refusal honesty** — the model says "I don't know" when grounded evidence is missing.

## Key sub-ideas

- **Process-reward RL** — reward the thought process, not just the final answer (Seed1.5-Thinking, DeepSeek-R1). Trains "how to think" over "what to answer".
- **Tool-augmented reasoning** — PAL, ReAct: delegate computation to a Python interpreter / search tool. Enables the honest report "I used a calculator".
- **Interleaved thinking** — thinking and acting interleaved at the token level.
- **Constitution-driven SFT** — 19 principles covering capability honesty, tool discipline, refusal. See [[entities/constitution]].
- **Capability check** — every `<think>` block must include a `CAPABILITY_CHECK` step in the SFT v2 data filter.

## Open questions

- Does process-reward RL generalise to novel problem shapes, or does it overfit to rubric patterns?
- How do we score "process correctness" cheaply at scale without a human judge?
- Process vs. outcome rewards — when is each needed? (See user's rough notes.)

## Related

- [[topics/tool-use-and-verification]] · [[topics/empathy]] · [[topics/llm-foundations]] · [[topics/explainability]]
- [[entities/constitution]] · [[entities/grpo]] · [[entities/mcp]] · [[entities/qwen3-0.6b]]

## Sources (ingested)

**Prompted reasoning**
- [[sources/papers/chain-of-thought-prompting]] — CoT baseline
- [[sources/papers/tree-of-thoughts]] — search + backtracking
- [[sources/papers/auto-cot]] — automates exemplar creation
- [[sources/papers/prompting-science-report-2]] — diminishing returns on modern models

**RL for reasoning**
- [[sources/papers/deepseek-r1]] · [[sources/papers/seed15-thinking]]
- [[sources/papers/vapo]] — value-based alternative to GRPO
- [[sources/papers/understanding-r1-zero]] — critique: GRPO length bias, Dr. GRPO
- [[sources/papers/interleaved-reasoning]] — RL for interleaved thinking (cuts TTFT 80%)

**Latent / architectural reasoning**
- [[sources/papers/hierarchical-reasoning-model]] — slow-planner / fast-executor
- [[sources/papers/looped-transformers-reasoning]] — depth > params
- [[sources/papers/coconut-continuous-latent]] — reason in vector space
- [[sources/papers/ladir]] — latent diffusion of reasoning
- [[sources/papers/state-stream-transformer]] — persistent latent state
- [[sources/papers/diffusion-of-thoughts]] — diffusion LM reasoning

**Small-model / distillation**
- [[sources/papers/self-enhanced-reasoning]] — small models self-train
- [[sources/papers/hidden-reasoners]] — LaTRO self-reward
- [[sources/papers/dual-head-reasoning-distillation]] — train-time-only reasoning

**Tool-grounded reasoning**
- [[sources/papers/pal]] · [[sources/papers/react]] · [[sources/papers/search-r1]]

**Evaluation**
- [[sources/papers/none-of-the-others]] — reasoning-vs-memorisation variation
- [[sources/papers/token-hungry-deepseek-r1]] — accuracy-vs-efficiency trade-off

**Multimodal RL siblings**
- [[sources/papers/ui-r1]] · [[sources/papers/vlm-r1]]

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §3.1, §4
- `pipeline/constitution.md` (via [[entities/constitution]])
