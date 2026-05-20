---
title: Meeting Notes — January 2026
type: source
tags: [advisor-meeting, tool-use, reasoning, planning]
sources:
  - docs/meetings-notes/january2026.md
updated: 2026-05-02
status: current
---

# Meeting Notes — January 2026

**The January meeting drew concrete lessons from the Boolean/math GPT ML assignment: LLMs are poor at arithmetic and should delegate numerical tasks to specialised tools rather than attempting them directly.**

## Summary

Ajinkya reported on the ML assignment — a GPT-2 model trained to do Boolean logic and arithmetic. Accuracy collapsed for two- and three-digit numbers despite step-by-step reasoning in the training data, due to the small model size and limited training. Owen's response was direct: this is a poor use of GPT; the right approach is a *hybrid AI system* where the LLM recognises its own limitations and hands off to specialist systems. They discussed the importance of ensuring the LLM does not re-interpret results returned by expert systems — the delegation must be complete, with the LLM accepting the specialist output rather than second-guessing it. Owen suggested the "interleaved thinking" approach as a bridge: the LLM provides structural breakdown of problems (semantic understanding) while delegating actual computation to external tools. Next meeting: Wednesday January 4 at noon.

## Key concepts

**Hybrid AI / explicit delegation.** The core architectural pattern that emerged from the Boolean GPT failure: the LLM should identify the class of problem (arithmetic, structured query, weather lookup) and delegate to a specialist system, then synthesise the results into a natural-language response. The LLM should not attempt the specialist's task itself, and must not re-interpret the specialist's output. This maps directly to the Tool Integration Layer in [[decisions/2025-10-01-four-module-architecture]].

**LLM-recognised limitation.** Owen emphasised the importance of training or prompting the LLM to know *when* it should not attempt a task. This is a non-trivial capability: the model must have calibrated self-awareness about its failure modes. This connects to the constitution's honest-refusal principle (P5: acknowledge limitations) and the broader scrutability goal.

**Context forgetting as a compounding failure.** Owen noted that LLMs tend to forget earlier context in multi-turn interactions, leading to inconsistent answers. When combined with arithmetic failures, the result is a system that is unreliable and hard for users to trust. The architectural response is explicit tool delegation with logged outputs that remain anchored in the conversation context.

**Interleaved thinking for semantic + numeric understanding.** Owen revisited the interleaved thinking concept from the November meeting, now framing it as the mechanism for combining language understanding (what the problem is asking) with numerical precision (what the answer actually is). The LLM structures the problem; the tool solves it.

## Action items

- Ajinkya: explore using LLM for structural breakdown while delegating calculations to external tools.
- Ajinkya: investigate structured output formats the LLM can produce that lock in tool results in context.
- Ajinkya: research methods to make LLM recognise when to hand off rather than attempt.
- Ajinkya: decide between fine-tuning a pre-trained model vs creating a specialised dataset.
- Ajinkya: address compute challenges for training iterations.
- Next meeting: Wednesday January 4 at noon.

## Related

- [[decisions/2025-10-01-four-module-architecture]] — the Tool Integration Layer implements the delegation pattern described here
- [[topics/tool-use-and-verification]] — the broader context for delegation and logged tool calls
- [[sources/papers/pal]] — PAL (code-as-reasoning) formalises the delegation pattern
- [[sources/papers/react]] — ReAct interleaves reasoning and action in multi-turn tool use
- [[sources/meetings/november2025]] — interleaved thinking concept introduced here, revisited in January
- [[sources/meetings/february2026]] — follow-up on hybrid AI and behaviorist framing

## Sources

- `docs/meetings-notes/january2026.md`
