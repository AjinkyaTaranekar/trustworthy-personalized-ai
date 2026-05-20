---
title: Meeting Notes — April 2026
type: source
tags: [advisor-meeting, constitution, planning, thesis, evaluation]
sources:
  - docs/meetings-notes/april2026.md
updated: 2026-05-02
status: current
---

# Meeting Notes — April 2026

**The April meeting confirmed Ajinkya's exceptional semester-2 performance, formalised the constitution-drift problem as a key research concern, and introduced the distinction between probes (internal observation) and tests (direct interaction) as the mitigation strategy. Ajinkya also accepted an Apple internship for June–September 2026.**

## Summary

Owen praised Ajinkya's adaptive applications project as "superb and standout quality," noting it mirrored their joint research by combining LLMs with knowledge graphs. Ajinkya had connected his security/privacy module work to the dissertation, applying Anthropic's constitutional AI principles. The main technical discussion focused on *constitution drift*: as context grows during a conversation, adherence to constitutional principles tends to degrade. Owen validated this concern and proposed two diagnostic mechanisms — *probes* (internal registers that observe model behaviour without direct interaction) and *tests* (direct prompts to the model designed to elicit constitution-governed responses). He cautioned about circular logic when using an LLM to test another LLM's constitutional adherence. They discussed dissertation timeline planning around Ajinkya's Apple internship (June–September 2026 in Dublin) and scheduled the next meeting for May 15 at noon.

## Key concepts

**Constitution drift.** As conversation context grows, the constitutional principles established in training become harder to maintain — the model's attention is split across a longer context window, and user-provided content can nudge the generation away from constitutional behaviour. This is distinct from adversarial jailbreak; it can happen through natural multi-turn interaction. The concern motivates the dependency monitor in [[sources/code/training-and-benchmark]] and the adversarial regression probes (REG1–REG6).

**Probes vs tests.** Owen's distinction:
- *Probes* are internal registers or lightweight activations that observe whether constitutional principles are being respected during generation, without interrupting or redirecting the conversation.
- *Tests* are direct interaction prompts — well-designed scenarios that should elicit constitution-governed behaviour (honest refusal, calibrated confidence, format compliance).
Neither is sufficient alone: probes are passive (may miss emerging drift), tests risk circular logic if the grading is done by the same or a similar LLM.

**Circular logic in LLM-graded LLM testing.** Owen identified a key methodological trap: if you use LLM-B to evaluate whether LLM-A is following its constitution, LLM-B's own biases and training objectives contaminate the measurement. The architectural response in the pipeline is the independent constitutional verifier (Blocker 2) — a separate, small, rule-based + LLM critic that grades outputs on explicit principles rather than holistic impression.

**Apple internship timeline constraint.** Ajinkya accepted an Apple internship in Dublin, June–September 2026. Owen's guidance: outline dissertation phases and predict how much time the internship will leave for writing. The experimentation phase must be completed before the internship begins (or at least gated so that only the writing phase overlaps). Owen noted Apple and Anthropic as companies with explicit ethical focus in AI — the constitutional work is directly relevant to the internship domain.

**Dissertation structure advice.** Owen recommended: (1) outline all phases/experiments/gates explicitly; (2) predict time requirements for each; (3) determine the last safe date to finish experimentation; (4) ensure the research question is clearly articulable before the next meeting. Owen emphasised that without a sharp research question, the experiments risk becoming "rambly."

## Action items

- Ajinkya: outline phases/experiments/gates for dissertation work.
- Ajinkya: predict time needed to write the dissertation.
- Ajinkya: determine date to finish experimentation (to leave runway for writing).
- Ajinkya: articulate the research question for the thesis.
- Ajinkya: develop a mechanism to prevent AI drift from constitutional principles as context grows.
- Ajinkya: design tests and probes to detect drift and when it occurs.
- Owen: next meeting online, May 15 at 12 PM.

## Related

- [[entities/constitution]] — the 19-principle document whose drift is the concern
- [[sources/code/training-and-benchmark]] — adversarial probe suite (REG1–REG6) implements the tests Owen described
- [[topics/security-and-privacy]] — constitution drift and alignment regression are security concerns
- [[topics/reasoning]] — constitutional trustworthiness as a reasoning property
- [[sources/meetings/february2026]] — the previous meeting where Owen pressed for a clearer research direction

## Sources

- `docs/meetings-notes/april2026.md`
