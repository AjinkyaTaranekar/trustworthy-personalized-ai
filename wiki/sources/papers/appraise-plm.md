---
title: AppraisePLM — Appraisal Theory for Affect Flow in Conversation (Debnath, Graham, Conlan)
type: source
kind: paper
tags: [empathy, appraisal-theory, evaluation, foundations]
sources:
  - docs/Assets/2025.conll-1.16.pdf
  - docs/Literature Notes/AppraisePLM Debnath 2025.md
arxiv: ""
acl: 2025.conll-1.16
updated: 2026-05-01
status: current
---

# AppraisePLM — Appraisal Theory for Affect Flow in Conversation

**A multitask DeBERTa model jointly performing 21-dimension appraisal regression and emotion classification on conversational corpora, trained on crowd-EnVent — directly implements the thesis's empathy chapter, co-authored by supervisor Owen Conlan (CoNLL 2025).**

## Summary

Debnath, Graham, and Conlan (ADAPT Centre, Trinity College Dublin, CoNLL 2025) introduce AppraisePLM, a computational operationalisation of Appraisal Theory for conversation analysis. The model jointly performs appraisal regression (predicting continuous values on 21 appraisal dimensions from crowd-EnVent: pleasantness, self-control, suddenness, alignment with social norms, etc.) and emotion classification. DeBERTa yields the best architecture. Trained on crowd-EnVent (6,600 event descriptions with 21 appraisal annotations + emotion labels), the model is then applied cross-corpus to EmoWOZ, EmpatheticDialogues, DailyDialog, and EPITOME — demonstrating cross-domain extrapolation and capturing affect flow (change in emotional appraisal across conversation turns). Empathetic conversations show improved pleasantness scores over turns.

## Significance for the Thesis

This paper is the direct technical implementation of the thesis's empathy chapter (Experiment 2). Three critical points:

1. **Supervisor co-authorship**: Owen Conlan (supervisor) is a co-author alongside Debnath and Graham. The AppraisePLM work is the supervisor's own group's contribution to the appraisal-conditioned AI problem that the thesis also addresses.
2. **Unblocks Experiment 2**: The thesis's two-phase generation pipeline (detect appraisals → condition generation) requires an appraisal tagger. AppraisePLM is that tagger, available at https://github.com/alokdebnath/appraise-PLM.
3. **Affect flow as novel contribution**: The focus on turn-wise change in appraisals — not just single-turn classification — is the open research gap Experiment 2 addresses. The thesis can build on AppraisePLM's affect-flow analysis and evaluate whether conditioning on appraisal trajectories improves empathetic response quality.

## Architecture Details

AppraisePLM uses a two-task training objective:
- **Appraisal estimation** (f_app): regression from PLM embedding to 21-dimensional appraisal vector (MSE loss)
- **Emotion classification** (f_emo): classification using combined PLM embedding + appraisal estimates (cross-entropy loss)

The crowd-EnVent training corpus: 6,600 event descriptions, 550 per 13 emotion categories, each annotated with 21 appraisal variables on a 5-point scale (skewed: >33% of values are 1 or 5).

## Related

- [[entities/appraisal-theory]] — the theoretical substrate
- [[topics/empathy]] — empathy chapter and Experiment 2
- [[experiments/experiment-catalog]] — Experiment 2: appraisal-conditioned generation
- [[sources/papers/xai-sentiment-deepseek-r1]] — complementary affective reasoning paper
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — §3.2 "Personalisation" and §5 empathy design
