---
paper id: UMAP-2024-3631700.3664903
title: "Towards Integrating Human-in-the-loop Control in Proactive Intelligent Personalised Agents"
authors: [Awais Akbar, Owen Conlan]
publication date: 2024-07-01
abstract: "This research explores the integration of Human-in-the-loop (HITL) control within Proactive Intelligent Personalised Agents (PIPAs) that possess the capability to proactively anticipate users' needs and perform tasks on their behalf. It investigates the conditions that trigger HITL control, its mechanisms, and the challenges associated with these triggers."
comments: "UMAP Adjunct '24, July 01–04, 2024, Cagliari, Italy. 5 pages. DOI: 10.1145/3631700.3664903. Supervisor Owen Conlan is co-author."
pdf: "[[Assets/3631700.3664903.pdf]]"
url: https://doi.org/10.1145/3631700.3664903
tags: [personalisation, scrutability, empathy]
---

## Key Claims

- **HITL in PIPAs**: proactive personalised agents must balance autonomous action with user control — the when/how of triggering human involvement is the core research question.
- **Three HITL activation factors**: (1) User preferences for agent autonomy (derived from user preference data — e.g. "always confirm travel bookings >€30"); (2) Cost implications — cost of autonomous decisions, erroneous actions, user handoff, and miscoordination in multi-agent settings; (3) Uncertainty in user intent — confidence score from probability difference between top two travel mode options triggers HITL when low.
- **Simulation-based approach** using London TFL (Transport for London) survey data: synthetic users created from real travel behaviour clusters enable controlled experiments without privacy concerns.
- Applied to travel assistance use case: agent proactively books transport; HITL triggered when agent is uncertain or action cost is high.
- Adjustable autonomy concept: user preferences define decision rules, not a fixed on/off switch.

## Thesis Relevance

Both the autonomy-gradient concept and the cost-of-user-handoff framing are directly applicable to the thesis's empathy and personalisation design. The HITL trigger framework maps onto the thesis's autonomy-preserving constraint: the model should trigger HITL (ask the user to decide) when confidence is low or stakes are high, rather than always acting on stored preferences. Owen Conlan (supervisor) is co-author — familiarity with this work is essential for supervisor meetings.

## Questions / Open Issues

- The travel assistance use case involves discrete bookable actions — how does HITL generalise to conversational advice-giving where actions are soft (recommendations not bookings)?
- The cost model is well-specified for travel but needs domain-specific re-instantiation for the thesis's empathetic conversation setting.
- Multi-agent miscoordination costs (§3.3.3) are relevant to the thesis's MCP multi-tool pipeline: if multiple MCP tools are in flight, HITL delays can cascade.
