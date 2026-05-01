---
paper id: ACM-2395123.2395129
title: "Creating Personalized Systems that People Can Scrutinize and Control: Drivers, Principles and Experience"
authors: [Judy Kay, Bob Kummerfeld]
publication date: 2012-12-01
abstract: "This article argues the importance of scrutable user modeling and personalization, illustrating key elements in case studies from our work. It identifies the broad roles for scrutable user models, describes how to tackle the technical and interface challenges of designing and building scrutable user modeling systems, presents design principles, and shows how they were established over twenty years of work on the Personis software framework."
comments: "ACM Trans. Interact. Intell. Syst., Vol. 2, No. 4, Article 24, December 2012. DOI: 10.1145/2395123.2395129."
pdf: "[[Assets/2395123.2395129.pdf]]"
url: https://doi.org/10.1145/2395123.2395129
tags: [personalisation, scrutability, xai, foundations]
---

## Key Claims

- Identifies **five core problems of personalisation**: Privacy (personal data used invisibly), Invisibility (users cannot see what is being personalised), Errors in user models (noisy inference from behaviour), Wasted user models (data collected but not used beneficially), Control (users cannot adjust personalisation they receive).
- **Scrutable user model**: a user model the user can actively study — they can see what information is held, how it was captured, and how it drives personalisation. Related to but stronger than "open", "transparent", "inspectable", "intelligible" models.
- **Four principles for scrutable personalisation**: (1) *Parallel design* — interface and user model designed together; (2) *First-class citizen* — unified control interface across all personalised applications; (3) *Context-based interpretation* — model stored in a way that users can understand contextually; (4) *Client-side* — user model stored where the user has full access and control.
- Implementation: **Personis framework** (University of Sydney) — 20+ years demonstrating these principles in learning, health, and productivity applications.
- Evidence that people willingly share more personal data when they feel in control of their user model.

## Thesis Relevance

Foundational paper for the entire scrutability design tradition. Kay and Kummerfeld's five problems map directly onto the thesis's over-personalisation failure modes (Privacy → local-first, Invisibility → opacity failure mode, Errors → sycophancy mechanism, Wasted → context inflation, Control → scrutability requirement). The four principles are the design specification for the thesis's 5W+H user model: it must be client-side, first-class, context-interpretable, and parallel-designed with the interaction interface. Supervisor Conlan's group (Jeromela & Conlan 2024, Akbar & Conlan 2024) explicitly builds on this paper.

## Questions / Open Issues

- The Personis framework predates LLMs — does the client-side principle survive in an era where the personalisation model IS the LLM (not a separate database)?
- The five problems are structural: how many are actually solved by on-device LLM deployment (local-first)? Privacy: yes. Wasted: partially. Control: not automatically.
- "Scrutable" requires active effort from the user — how to make scrutiny low-effort and low-friction without removing its epistemic benefit?
