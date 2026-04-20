---
title: Appraisal Theory
type: entity
tags: [empathy, appraisal-theory]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
updated: 2026-04-19
status: current
---

# Appraisal Theory

**A psychology-of-emotion framework: emotions arise from a structured
evaluation ("appraisal") of an event along several dimensions, rather
than from the event itself. The thesis proposes this as the structured
substrate for AI empathy.**

## Core appraisal dimensions (examples)

| Dimension | Question |
| --------- | -------- |
| Novelty | Is this event new or unexpected? |
| Valence | Pleasant or unpleasant? |
| Goal conduciveness | Does it help or hinder my goals? |
| Coping potential | Can I handle this? |
| Agency | Who caused this — self, others, circumstances? |
| Fairness | Is this just? |

## Operational use in the thesis

- **Dataset** — Crowd-event dataset, 6,600 events × 21 appraisal
  dimensions. Intended training corpus for an AppraisePLM-style tagger.
- **Two-phase empathy pipeline**
  1. Tag an incoming user message with its appraisal profile.
  2. Generate a response conditioned on those tags.
  This gives empathy an auditable signal instead of a generic sympathy
  habit.
- **Critical questions** raised in the dissertation drafts:
  - How were the 21 dimensions set? Do users understand them?
  - What was the user's pre-event mood?
  - Cultural variation — appraisal of anger or shame is not universal.
  - Temporal gap between event and recording — consolation effects.

## Design tension

The 21-dimension vocabulary may be too fine-grained for end users.
Experiment 2 (see [[experiments/experiment-catalog]]) is expected to
answer whether the dimensions collapse to a smaller usable set.

## Related

- [[topics/empathy]]
- [[entities/5w-h]] — structured user-context counterpart on the
  cognitive axis
- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent affective
  classification template
- [[sources/papers/dual-head-reasoning-distillation]] — latency-efficient
  classifier training template
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §5.3

## Sources

- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md` §5.3
- `docs/Dissertation/Experimental Planning Document.md` — Experiment 2
