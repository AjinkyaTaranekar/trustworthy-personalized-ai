---
title: "Road Towards Trustworthy and Empathetic AI (dissertation draft)"
type: source
kind: dissertation-draft
author: the user
tags: [thesis, synthesis, literature-review]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
updated: 2026-04-19
status: current
---

# Road Towards Trustworthy and Empathetic AI

**The user's primary thesis document — a long-form synthesis that poses the
central research question, lays out the motivating problems, surveys the
literature, and sketches the proposed architecture.**

## Structure of the source

1. **Central research question + secondary "what is reasoning?" question**
2. **What makes humans and AI different** — consciousness, embodiment,
   causal understanding, meta-cognition
3. **Foundational mechanics** — tokenisation, attention, BERT vs GPT,
   contextualised embeddings, RAG, tool use, MCP
4. **Why current architecture fails** — the "sociopath yapper" problem,
   process-vs-outcome RL, emergence debate, catastrophic forgetting,
   computation vs pattern matching
5. **Personalisation gaps** — late-to-class example, Frappuccino scenario,
   cold-start, privacy paradox, 5W+H, SHAP limits
6. **Explainability** — mechanistic interpretability, attention is not
   explanation, calibration
7. **Reasoning in LLMs** — CoT, System 1/2, few-shot/in-context, ToT,
   ReAct, SERT, Auto-CoT, hidden reasoners (LaTRO), HRM, Coconut, LaDiR,
   DoT, neuro-symbolic, dual-head, interleaved thinking (MiniMax, Kimi 2),
   prompting diminishing returns
8. **Core research themes** — scrutability, user modelling, appraisal
   theory

## Key claims that anchor the wiki

- Reasoning is not solved by scale alone; architecture + process reward
  matters (supports [[topics/reasoning]], see
  [[sources/papers/seed15-thinking]], [[sources/papers/deepseek-r1]]).
- Catastrophic forgetting forces personalisation out of weights into
  retrieval ([[topics/personalisation]], [[entities/rag]]).
- "Attention is not explanation" — mechanistic interpretability is a
  separate track from prose rationalisation ([[topics/explainability]]).
- Appraisal theory is the structured handle for empathy
  ([[topics/empathy]], [[entities/appraisal-theory]]).
- Scrutability is achieved through citations (RAG), honest tool reports
  (PAL), and translating internal reasoning state — not self-explanation.

## Related

- [[overview]]
- [[topics/reasoning]] · [[topics/personalisation]] · [[topics/empathy]] · [[topics/explainability]] · [[topics/llm-foundations]] · [[topics/tool-use-and-verification]]
- [[entities/5w-h]] · [[entities/appraisal-theory]] · [[entities/mcp]] · [[entities/rag]]
- [[sources/dissertation/experimental-planning-document]]

## Raw

- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md`
