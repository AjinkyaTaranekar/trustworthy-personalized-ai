---
paper id: 2025.conll-1.16
title: "An Appraisal Theoretic Approach to Modelling Affect Flow in Conversation Corpora"
authors: [Alok Debnath, Yvette Graham, Owen Conlan]
publication date: 2025-07-31
abstract: "This paper presents AppraisePLM, a model of affect in conversations leveraging Appraisal Theory as a generalizable framework. AppraisePLM is a regression and classification model trained on the crowd-EnVent corpus that outperforms existing models in predicting 21 appraisal dimensions including pleasantness, self-control, and alignment with social norms. Applied to four benchmark conversation corpora spanning task-oriented dialogue, general chit-chat, affect-specific conversation, and domain-specific affect analysis, the model successfully extrapolates emotion labels across datasets while capturing domain-specific affect flow patterns."
comments: "CoNLL 2025 — ACL Anthology 2025.conll-1.16, pages 233–250. Supervisor Owen Conlan is co-author."
pdf: "[[Assets/2025.conll-1.16.pdf]]"
url: https://aclanthology.org/2025.conll-1.16
tags: [empathy, appraisal-theory, evaluation, foundations]
---

## Key Claims

- **AppraisePLM**: a multitask DeBERTa-based model jointly performing (1) appraisal regression — predicting continuous values on 21 appraisal dimensions from Appraisal Theory, and (2) emotion classification — predicting categorical emotion labels.
- Trained on the **crowd-EnVent corpus** (6,600 event descriptions annotated with 21 appraisal variables + emotion labels + author demographics); DeBERTa yields best performance.
- Successfully extrapolates across four conversation corpora: EmoWOZ (task-oriented), EmpatheticDialogues, DailyDialog, EPITOME (mental health) — capturing domain-specific affect flow patterns without retraining.
- Appraisal theory reveals distinct patterns per domain: e.g. empathetic conversations improve pleasantness appraisal scores over turns.
- **Positions affect flow** (change in emotional appraisal over a conversation) as a promising model for holistic emotion analysis in conversational agents.

## Thesis Relevance

This is the direct technical implementation of the thesis's empathy chapter. AppraisePLM provides the appraisal tagger needed for the two-phase generation pipeline (detect appraisals → condition generation on them). Critically, **Owen Conlan (supervisor) is a co-author** alongside Debnath and Graham at the ADAPT Centre, Trinity College Dublin. This paper unblocks Experiment 2 (appraisal-conditioned empathy evaluation). The crowd-EnVent corpus with 21 appraisal dimensions is the training data source. The demonstration on EmpatheticDialogues directly maps to the thesis's target dialogue domain.

## Questions / Open Issues

- AppraisePLM uses DeBERTa (110M+ parameters) as the backbone — this runs as a separate tagger module, not within the 0.6B Qwen3 model itself. How is it integrated at inference time without prohibitive latency?
- The 21 appraisal dimensions may be too fine-grained for practical conditioning — are there collapsed/clustered variants tested?
- Affect flow (turn-wise change in appraisals) is identified as promising but not yet used for generation conditioning — this is the open research gap the thesis's Experiment 2 addresses.
- Code available at: https://github.com/alokdebnath/appraise-PLM
