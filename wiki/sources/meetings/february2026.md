---
title: Meeting Notes — February 2026
type: source
tags: [advisor-meeting, reasoning, scrutability, planning]
sources:
  - docs/meetings-notes/february2026.md
updated: 2026-05-02
status: current
---

# Meeting Notes — February 2026

**The February meeting introduced behaviorism as a psychological lens for analysing LLM behaviour, and pressed Ajinkya to contract the research focus and articulate a clear framing before the next meeting.**

## Summary

Owen discussed his module group naming confusion, then turned to Ajinkya's progress: a chat assistant with mathematical capabilities. Ajinkya reported that the agent sometimes produced incorrect calculation results but could be corrected by injecting a Python script — an informal demonstration of the delegation pattern from the January meeting. Owen responded by suggesting *behaviorism* as the analytical framework: study the agent's observable outputs without inspecting internals, as a psychologist would. They explored post-hoc solutions (constraints applied after model generation rather than inside it), context retention mechanisms, and a recursive language model paper Ajinkya had been studying. Owen concluded by pressing Ajinkya to clearly define what the work is trying to achieve, what its bounds are, and whether the innovation is *on the LLM model* (architectural change) or *around the model* (constraint post-generation). They agreed to reconvene only once Ajinkya had a clearer research direction.

## Key concepts

**Behaviorism as LLM analysis lens.** Owen's suggestion: rather than trying to inspect the internal state of the LLM (which is largely opaque), characterise its behaviour through systematic input-output experiments, as behaviourist psychologists characterised human responses. This is practically sound — the key measurements are observable (accuracy, refusal rate, tool delegation rate, format compliance) and can be run systematically without access to weights. The adversarial probe suite in [[sources/code/training-and-benchmark]] is a concrete implementation of this approach.

**Post-hoc constraint vs in-model change.** Owen's key question: is the innovation happening *inside* the model (new architecture, new training objective) or *around* the model (post-generation constitutional verification, tool delegation, structured prompting)? The thesis ultimately answers: both — GRPO trains the reasoning module to be constitutionally reliable, while the constitutional verifier (Blocker 2) is an out-of-band post-hoc check. Owen's framing here clarified that these are complementary, not competing, approaches.

**Recursive language model.** Ajinkya was studying a paper on recursive language models as a potential approach to context retention. This connects to the broader concern about context forgetting discussed in the January meeting.

**Research focus contraction.** Owen issued a direct challenge: the research is too broad. Ajinkya needs to decide what the core contribution is before the next meeting. This pressure ultimately led to the cleaner framing of the thesis around the four-module architecture with GRPO as Experiment 1 and ontology-LLM as the flagship Experiment 6.

## Action items

- Ajinkya: do reading on behaviourism as a psychological construct.
- Ajinkya: capture the chat-assistant experiment in a way that will feed into the thesis.
- Ajinkya: research why LLMs appear to be good at writing code.
- Ajinkya: study the recursive language model paper and code.
- Ajinkya: articulate the framing of the work — what it is trying to achieve.
- Ajinkya: design and think through what the experimental setup would look like.
- Ajinkya: determine whether innovation is on the LLM model or around constraint post-model.
- Ajinkya: contract the research focus to make it clearer.
- Ajinkya: drop an email or catch Owen at the end of a lecture when ready for the next meeting.

## Related

- [[decisions/2025-10-01-four-module-architecture]] — the answer to Owen's "on-model vs post-model" question
- [[topics/reasoning]] — the behaviourist lens applies directly to how reasoning is evaluated
- [[sources/code/training-and-benchmark]] — the adversarial probe suite embodies the behaviourist measurement approach
- [[sources/meetings/january2026]] — the delegation pattern and context forgetting discussed here build on January's findings
- [[sources/meetings/april2026]] — the next meeting where a clearer direction had been established

## Sources

- `docs/meetings-notes/february2026.md`
