---
title: Meeting Notes — October 2025
type: source
tags: [advisor-meeting, rl, personalisation, planning]
sources:
  - docs/meetings-notes/october2025.md
updated: 2026-05-02
status: current
---

# Meeting Notes — October 2025

**The second meeting exploring reinforcement learning applied to thought processes rather than token prediction, and the concept of a values-interpreter intermediary sitting between users and LLMs.**

## Summary

Ajinkya raised the question of rewarding *reasoning processes* rather than output tokens — giving RL signals to the thought chain that determines which data or API to invoke, rather than to the surface response. Owen reframed this as building a "tin veneer" of a human-like reward system and proposed a two-LLM experimental setup: one LLM acts as a human, the other as a values interpreter that evaluates the first's outputs against established principles. The meeting also highlighted the ethical risks of AI companions that learn to maximise engagement rather than user wellbeing. Action items focused on sketching the architecture without building it, and gathering relevant literature before the next meeting (November 22).

## Key concepts raised

**RL for thought processes.** Ajinkya proposed rewarding the reasoning chain (which tool to call, which data to retrieve) rather than the token output. Owen connected this to creating a layer of trust that remains consistent despite backend changes — paralleling how human relationships build trust through consistent values over time.

**Values interpreter architecture.** Owen sketched a two-LLM experimental setup: LLM-A acts as the human user; LLM-B is the values interpreter that sits between the user and other AI systems, learning to evaluate outputs against the user's declared values and principles. This is an early ancestor of the four-module architecture's Reasoning + User Modelling split.

**Honest refusal over fabrication.** Owen emphasised that the reward structure should prioritise honesty — the system should say "I don't know" rather than produce plausible-sounding wrong answers. This became a core principle in the constitution (P5/P7 in the 19-principle document).

**Ethical risks of AI companions.** Owen warned against creating AI that learns to maximise engagement at the expense of user wellbeing, reinforcing harmful behaviours or cultural biases. This motivates the constitution's autonomy-preservation principle (P17).

**Longitudinal companions research.** Owen asked Ajinkya to investigate biologically influenced algorithms and existing research on longitudinal AI companions to ground the architecture in prior work.

## Action items (from meeting)

- Ajinkya: sketch the values interpreter / intermediary AI system architecture.
- Ajinkya: research longitudinal AI companions and biologically influenced algorithms.
- Ajinkya: gather papers without deep reading — map the space.
- Next meeting: Wednesday, November 22nd at 1:30 PM.

## Related

- [[decisions/2025-10-01-four-module-architecture]] — the four-module architecture that formalised the values interpreter concept
- [[entities/constitution]] — the 19-principle document that operationalised "honest refusal over fabrication"
- [[topics/personalisation]] — user values as the substrate for the intermediary system
- [[topics/empathy]] — ethical AI companion risks discussed here
- [[entities/grpo]] — the RL algorithm that eventually implements process rewards

## Sources

- `docs/meetings-notes/october2025.md`
