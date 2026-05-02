---
title: Meeting Notes — December 2025
type: source
tags: [advisor-meeting, planning, thesis, scrutability]
sources:
  - docs/meetings-notes/december2025.md
updated: 2026-05-02
status: current
---

# Meeting Notes — December 2025

**Two meetings in December: the first (December 11) focused on research rigour and citation practices; the second (December 16) refined the formal research proposal before its December 17 submission deadline.**

## Meeting 1 — December 11 (online)

### Summary

Owen shared updates about a successful EU project review, then turned to Ajinkya's upcoming dissertation deadlines. Ajinkya is completing two module assignments: a research plan (due December 17) and an ML project (creating Boolean GPT and math GPT using GPT-2 architecture, due January). Owen emphasised rigour and citation — any claim about LLM components must be backed by literature at tutorial level, written for non-experts. He recommended a YouTube playlist covering tokenisation, encoding/decoding, and transformer architecture as foundational grounding. He also warned against "rabbit holes" and encouraged mapping ideas into a coherent research outline.

### Key concepts

**Citation for non-experts.** Owen's recurring guidance: the literature review must be grounded enough that someone without ML background can follow why the four-component system is necessary. This shaped the dissertation's decision to include [[topics/llm-foundations]] as Pillar 0 — explaining tokenisation and causal attention before the research contributions.

**Boolean GPT / math GPT as baseline.** Ajinkya's ML assignment (GPT-2 with step-by-step reasoning) became an early experiment demonstrating LLM limitations in arithmetic — a direct precursor to the hybrid delegation approach developed in [[sources/meetings/january2026]].

### Action items

- Ajinkya: work on research plan assignment this week (deadline Dec 17).
- Ajinkya: complete ML assignment by January deadline.
- Ajinkya: review YouTube playlist on deep learning (tokenisation, architecture).
- Owen: review research plan document on December 16 at noon.
- Next meeting: December 16, noon.

---

## Meeting 2 — December 16 (in-person review)

### Summary

Owen and Ajinkya reviewed the research proposal document "Building Trustworthy and Empathetic Conversational Layer." Owen gave detailed structural feedback: narrow the research question to be specific and labelled "initial/guiding"; add literature review as the first objective; reduce the total number of objectives; group the 10 literature items logically; add a soft skills / personal management section alongside technical and research skills. Owen's key framing for the project's scope: this is a *prototype* that explores potential benefits — it is not competing with ChatGPT or frontier models. This prototype framing became an anchor for the small-model constraint that runs through the thesis.

### Key concepts

**Prototype scope, not frontier competition.** Owen explicitly scoped the work: Ajinkya should report on initial experiments and findings rather than claiming a complete solution. The value is demonstrating that the right modular architecture produces more trustworthy, scrutable, and empathetic outputs than a larger monolithic model with no structure — not benchmarking against GPT-4 on capability tasks.

**Ethical considerations for user data.** Owen flagged consent and data privacy as future concerns — particularly around emotional experience modelling and persona profiles. This anticipates the local-first privacy architecture (User Modelling Module on local MCP server) and the GDPR constraints in [[topics/security-and-privacy]].

**Four research areas to justify.** Owen challenged Ajinkya to justify why all four aspects (reasoning, personalisation, empathy, tool use) are being explored together, rather than picking one. The answer — that they are inseparable in a trustworthy system — became the thesis's central argument for the modular architecture.

**Pivoting as legitimate research.** Owen recommended adding explicit text about the role of pivoting and decision points in the research plan, normalising the expectation that the focus will evolve. The two pivots (monolithic → modular; RL-primary → ontology-primary) are now documented in [[decisions/2025-10-01-four-module-architecture]] and [[decisions/2025-11-10-ontology-focus-shift]].

### Action items

- Ajinkya: apply all structural feedback and submit proposal by December 17.
- Ajinkya: explore candidate technologies over the Christmas break.
- Touch base in early January (teaching resumes January 19).

## Related

- [[decisions/2025-10-01-four-module-architecture]] — the binding modular design the proposal describes
- [[sources/dissertation/research-plan]] — the formal plan submitted after this meeting
- [[topics/security-and-privacy]] — ethical/GDPR concerns flagged here
- [[topics/llm-foundations]] — foundational layer added partly in response to Owen's "non-expert reader" guidance
- [[sources/meetings/january2026]] — the Boolean/math GPT experiment from the ML assignment surfaces here

## Sources

- `docs/meetings-notes/december2025.md`
