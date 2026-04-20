---
title: 5W+H User-Modelling Framework
type: entity
tags: [personalisation, 5w-h]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - researchplan.tex
updated: 2026-04-19
status: current
---

# 5W+H

**A structured user-modelling schema the thesis uses to gather context
systematically: Who, What, When, Where, Why, How. Designed as an
alternative to "assume defaults" personalisation.**

## The six slots

| Slot | What it captures | Example question |
| ---- | ---------------- | ---------------- |
| **Who** | Identity, relationships, role | "Who are you doing this for?" |
| **What** | Goal, object of action | "What are you trying to achieve?" |
| **When** | Time pressure, timing | "Is this urgent or longer-term?" |
| **Where** | Location, context, platform | "Where will this happen?" |
| **Why** | Motivation, purpose | "Why this, not something else?" |
| **How** | Constraints, method | "How much time / budget / skill do you have?" |

## Why the thesis uses it

- **Cold-start solution** — systematic first-session context collection
  beats demographic-proxy stereotyping.
- **Per-turn adaptivity** — the dissertation's "late-to-class" and
  "Frappuccino" examples show that LLMs assume when they should be asking.
  5W+H turns assumption into enquiry.
- **Structured slot → structured memory** — each answered slot is a clean
  target for a [[entities/graph-rag|GraphRAG]]-style user graph node.
- **Privacy-respecting** — each slot is explicit so users see exactly
  what the system remembers.

## Related

- [[topics/personalisation]] · [[topics/empathy]]
- [[entities/rag]] · [[entities/graph-rag]]
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §5.2 "User Modeling & Empathy"
- [[sources/papers/rag-original]]

## Sources

- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md`
- `researchplan.tex`
