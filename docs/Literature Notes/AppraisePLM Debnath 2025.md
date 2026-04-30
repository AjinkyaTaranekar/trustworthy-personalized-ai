---
paper id: Debnath-2025-AppraisePLM
title: "AppraisePLM (working title)"
authors: [Debnath et al.]
publication date: 2025
abstract: "A pre-trained language model trained to tag user utterances with appraisal theory dimensions, enabling structured emotion/affect detection for downstream conditioning."
comments: "No arXiv found — may be a workshop or conference paper. Acquisition method unknown."
pdf: ""
url: ""
tags: [empathy, evaluation]
---

## Status

PDF not yet acquired. No arXiv ID known. This paper **blocks Experiment 2** (empathy evaluation). Priority acquisition via Google Scholar search for "AppraisePLM" or "Debnath 2025 appraisal".

## What is known (from citations in wiki)

- Trains a PLM to tag user utterances with the 21 appraisal theory dimensions (valence, control, novelty, etc.) from the crowd-event dataset.
- Enables the two-phase appraisal-conditioned generation proposed in the thesis's empathy chapter: detect → condition → generate.
- Without this tagger (or a substitute), Experiment 2 cannot be properly evaluated.

## Thesis Relevance

Critical dependency: the thesis's appraisal-theoretic empathy approach (entities/appraisal-theory) relies on detecting appraisal dimensions in user utterances. If AppraisePLM cannot be acquired or reproduced, an alternative appraisal tagger must be found or trained from the crowd-event dataset directly.
