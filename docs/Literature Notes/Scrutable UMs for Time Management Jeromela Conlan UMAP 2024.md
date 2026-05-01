---
paper id: UMAP-2024-3631700.3665182
title: "Devising Scrutable User Models for Time Management Assistants"
authors: [Jovan Jeromela, Owen Conlan]
publication date: 2024-07-01
abstract: "This paper contemplates how scrutability — the ability of the user to study their assistant and its underlying user model — fits within the vision of more complex Intelligent Personal Assistants (IPAs) for time management. It describes an ongoing project investigating user interest in and expectations of the scrutability of a proactive calendaring assistant, and outlines potential avenues for further research."
comments: "UMAP Adjunct '24, July 01–04, 2024, Cagliari, Italy. 6 pages. DOI: 10.1145/3631700.3665182. Supervisor Owen Conlan is co-author."
pdf: "[[Assets/3631700.3665182.pdf]]"
url: https://doi.org/10.1145/3631700.3665182
tags: [personalisation, scrutability, xai, empathy]
---

## Key Claims

- Applies Kay & Kummerfeld's scrutability framework to **proactive IPA for time management** — extending from reactive/web systems to proactive, multimodal, delegating agents.
- **Six domain-specific scrutability challenges**: (1) *Proactivity* — when/why IPA acts unsolicited; (2) *Delegation* — IPA acts autonomously (books appointments, allocates tasks); (3) *Modality Selection* — GUI vs CUI vs voice choice; (4) *Humanness and Expediency* — balancing human-like responses with efficiency; (5) *Minimal Interpretability* — what is the minimum the IPA must expose?; (6) *Data Stewardship* — explicit data licensing via "Stages" levels.
- **Stages concept**: a novel scrutability principle — users define data licences (what data the IPA may use and for which functions), giving transparency about data stewardship and enabling control before use rather than after.
- Extends Kay & Kummerfeld's 4 principles to the proactive IPA context; finds they need revision for proactive and delegative capabilities.
- **Two planned user studies**: task-based scrutability study (browser prototype of proactive calendaring assistant) and longitudinal study with students at TCD across an academic semester.

## Thesis Relevance

Directly from the supervisor's group at ADAPT/TCD. The six challenges are a design checklist for the thesis's proactive empathetic assistant. The Stages concept is an elegant operationalisation of the GDPR Article 7 (specific, informed consent) in a conversational AI context — directly applicable to the thesis's local-first privacy argument. The planned longitudinal study using TCD students is likely ongoing research that the user's dissertation could build on or reference. Must-read before any supervisor meeting.

## Questions / Open Issues

- The Stages concept maps data usage to licensed purposes — how does this interact with the thesis's on-device model, where "data" (fine-tuning gradients) is harder to describe in user-legible terms?
- The planned user studies may have completed or be ongoing — check with supervisor about results.
- Modality Selection challenge (§3.1) is the thesis's "empathy design" question: when should the model use voice tone markers, acknowledgements, or clarifying questions?
