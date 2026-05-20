---
title: Empathy
type: topic
tags: [empathy, appraisal-theory, small-model, graph-memory]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
updated: 2026-05-01
status: draft
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

## Ethical Boundary: Dependency and Deskilling

A highly personalised and empathetic model creates two risks identified in the [[sources/dissertation/security-privacy-social-ethics|security analysis]]. First, **emotional dependency**: the model is always available, always patient, and optimised for engagement — unlike human relationships, which are non-linear and reciprocal. Research on parasocial bonds (Lipin 2025, unacquired) shows emotionally reactive systems produce attachment behaviours functionally similar to human relationships, but with no reciprocal stake in the user's wellbeing. Disclosing AI identity is not enough; the system should actively monitor for and interrupt dependency formation. Second, **deskilling**: users who habitually delegate uncertainty to a confident model may lose independent evaluative capacity — documented empirically in clinical AI settings (endoscopist adenoma detection decline, Budzyń et al. 2025). The empathy design must therefore include an autonomy-preserving constraint: the model should redirect users to better consultants and surface uncertainty rather than provide the most reassuring answer.

## Related

- [[topics/personalisation]] — empathy conditions on user state
- [[topics/reasoning]] — "refusal honesty" is the non-empathy axis of honesty
- [[topics/explainability]] — transparent reasoning traces increase perceived empathy
- [[topics/security-and-privacy]] — emotional dependency and deskilling are social-ethical risks
- [[entities/appraisal-theory]] · [[entities/5w-h]]
- [[experiments/experiment-catalog]] — Experiments 2 + 3

## Industry reference: Hume AI EVI

Hume AI's Empathic Voice Interface (EVI) uses 48 emotion dimensions fused from voice prosody, intonation, pacing, and linguistic context — validated across 50+ languages and grounded in 53+ peer-reviewed publications. Hume partnered with Anthropic in 2024 to add emotionally intelligent voice interactions to Claude. The architectural pattern is identical to this thesis's text-only approach: a separate emotion-detection module (their prosody tagger, this thesis's AppraisePLM) gates generation style before the language model produces output. The key validation: emotion detection as a modular component that conditions generation is the production-validated approach, not an experimental hypothesis.

## Sources (ingested)

- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent reasoning for affective classification
- [[sources/papers/dual-head-reasoning-distillation]] — cheap-inference template for the appraisal tagger
- [[sources/papers/interleaved-reasoning]] — TTFT win matters for perceived empathy
- [[sources/dissertation/security-privacy-social-ethics]] — §5 social-ethical concerns (dependency + deskilling)
- [[sources/papers/appraise-plm]] — AppraisePLM: 21-dim appraisal regression; supervisor Conlan co-author; Experiment 2 unblocked; code at https://github.com/alokdebnath/appraise-PLM

## Sources (to acquire — see [[questions/2026-04-30-asset-acquisition-todo]])

- Simulating Emotions with Appraisal + RL (CHI 2024) — integrates OCC appraisal dimensions with RL; validates appraisal-conditioned generation approach; directly citable for Layer 4
- Graph-based Agent Memory survey (arXiv:2602.05665) — taxonomy covering empathic agent memory architectures

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §3.2 "Personalisation" & §5.2–5.3
- [[sources/dissertation/experimental-planning-document]] — Experiment 2
