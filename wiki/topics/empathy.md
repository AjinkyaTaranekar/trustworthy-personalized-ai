---
title: Empathy
type: topic
tags: [empathy, appraisal-theory]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
updated: 2026-04-19
status: stub
---

# Empathy

**How to condition responses on *detected* user appraisals rather than generating generic sympathy tokens.**

## Summary
Current LLMs produce "simulated empathy" — surface-level validation that has no structured handle on what the user actually feels. The thesis proposes **appraisal-theoretic conditioning**: (1) detect the appraisal dimensions present in the user's message using an AppraisePLM-style tagger, (2) generate a response conditioned on those tags. This gives empathy an explicit, testable signal rather than a stylistic habit.

## Key sub-ideas

- **Appraisal theory** — emotions arise from a structured evaluation of events; 21 appraisal dimensions (crowd-event dataset).
- **AppraisePLM** — a PLM trained to tag user utterances with appraisals.
- **Two-phase generation** — detect → condition → generate. Evaluation can score each phase separately.
- **Qualia gap** — even perfect appraisal simulation lacks subjective experience; thesis takes the position that this matters less for utility than for philosophy.

## Open questions

- Are the 21 appraisal dimensions tractable for users / annotators, or do they need collapsing?
- Temporal drift: does appraisal detection degrade as conversations lengthen?
- Evaluation: can human raters distinguish appraisal-conditioned responses from strong baseline responses?

## Conversation design (from the draft)

- **Turn-taking** — LLMs currently wait passively; thesis argues they should know when to interject with a clarifying question.
- **Grounding** — "uh-huh, I see" acknowledgements matter for perceived empathy. LLMs rarely do this unmanaged.
- **Gricean implicature** — "Can you pass the salt?" is not a yes/no question. Understanding subtext is a prerequisite for emotional support.
- **Simple over open questions** — when clarity is needed, prefer yes/no.

## Related

- [[topics/personalisation]] — empathy conditions on user state
- [[topics/reasoning]] — "refusal honesty" is the non-empathy axis of honesty
- [[topics/explainability]] — transparent reasoning traces increase perceived empathy
- [[entities/appraisal-theory]] · [[entities/5w-h]]
- [[experiments/experiment-catalog]] — Experiments 2 + 3

## Sources (ingested)

- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent reasoning for affective classification
- [[sources/papers/dual-head-reasoning-distillation]] — cheap-inference template for the appraisal tagger
- [[sources/papers/interleaved-reasoning]] — TTFT win matters for perceived empathy

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §3.2 "Personalisation" & §5.2–5.3
- [[sources/dissertation/experimental-planning-document]] — Experiment 2
