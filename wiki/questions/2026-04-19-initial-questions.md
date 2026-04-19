---
title: Initial Open Questions (seeded 2026-04-19)
type: question
tags: [questions, todos, bootstrap]
sources:
  - docs/Dissertation/Experiment.md
  - docs/Dissertation/Rough Notes.md
  - docs/Dissertation/Experimental Planning Document.md
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
updated: 2026-04-19
status: current
---

# Initial Open Questions

**Consolidated from user TODOs, post-meeting question lists, and
literature-surfaced conflicts. Grouped by topic. File new questions as
they arise; do not delete — mark `resolved` inline.**

## Exploration TODOs (from `Experiment.md` + `Rough Notes.md`)

- [ ] Build a Boolean-and-math GPT as an ML assignment; note failure
      modes from tokenisation (ties to
      [[sources/papers/bpe-subword-units]]).
- [ ] Walk through DeepSeek-R1 architecture; replicate via a dummy model.
- [ ] Review MiniMax-M2 code on HF for interleaved-thinking techniques
      (pairs with [[sources/papers/interleaved-reasoning]]).
- [ ] Read vLLM interleaved thinking docs.
- [ ] Evaluate open-source graph tools for reasoning plots: **Cognee**,
      **FalkorDB**, **Neo4J** — decide which backs
      [[entities/graph-rag]].
- [ ] Explore: "How can user modelling be done with interleaved
      thinking?" — asking right questions, showing users *why* the model
      thought of this for them.

## Ontology-LLM integration (Experiment 6 — advisor meeting prep)

- [ ] Pick an ontology: DBpedia, Wikidata, or domain-specific (political,
      medical, legal)?
- [ ] Commit to Approach A (ontology as KB) or Approach B (post-hoc
      verifier) first — or both in smaller scope.
- [ ] Query language: SPARQL, Cypher, custom?
- [ ] Claim-extraction pipeline design for Approach B.
- [ ] Test dataset with verifiable ground truth — who defines "correct"
      for political questions?
- [ ] Baselines: pure LLM, RAG, tool-augmented, other neuro-symbolic.
- [ ] Latency budget for ontology reasoning in an interactive system.
- [ ] Ethical: every ontology encodes a worldview; how to disclose?

## Reasoning / RL design

- [ ] GRPO length bias — adopt Dr. GRPO
      ([[sources/papers/understanding-r1-zero]]) for the C/D conditions?
- [ ] Process-reward scoring at scale without a human judge — is LaTRO
      ([[sources/papers/hidden-reasoners]]) self-reward viable on
      [[entities/qwen3-0.6b|Qwen3-0.6B]]?
- [ ] Confound: how much of Condition C/D gains is
      [[sources/papers/understanding-r1-zero|pretraining bias in Qwen]]
      rather than our RL?
- [ ] Apply [[sources/papers/none-of-the-others|"None of the Others"]]
      variation to the benchmark to separate reasoning from
      memorisation.

## Personalisation / empathy

- [ ] Are the 21 appraisal dimensions tractable for annotators / users,
      or should they be collapsed?
- [ ] Crowd-event dataset demographics — is there cultural bias that
      breaks Dublin-user appraisal tagging?
- [ ] Cold-start via 5W+H: how many slots before first useful response?
- [ ] Privacy architecture: local-only MCP store vs federated learning
      vs differential privacy — which first?
- [ ] Can SHAP be made to work on LLMs given token dependence?

## Infrastructure

- [x] ~~`2c_rl_trainer.py` referenced in memory is missing from `pipeline/`~~
      — **resolved 2026-04-19**: lives on a separate branch, not `main`.
- [ ] Update repo `project_state` memory after next `git pull` / pipeline
      audit (still points at paths + file that live on the RL branch).
- [x] ~~Reconcile planning-doc priority ranking (Experiment 1 = "Lower")
      with the fact that the repo implements Experiment 1.~~
      — **resolved 2026-04-19**: Experiment 1 is active supporting
      infrastructure for Experiment 6 comparisons; "Lower Priority" was a
      post-meeting relative ranking, not a deprecation.

## Tension points from the literature

- **Scrutability vs. performance.**
  [[sources/papers/coconut-continuous-latent|Coconut]] /
  [[sources/papers/ladir|LaDiR]] / [[sources/papers/hierarchical-reasoning-model|HRM]]
  show that latent reasoning is faster and stronger — but kills the
  ontology-verification story. How far can the thesis push
  ontology-verification before it becomes a performance ceiling?
- **CoT diminishing returns.**
  [[sources/papers/prompting-science-report-2]] suggests CoT helps
  less on modern reasoning-tuned bases. Does the v2 constitution's
  `<think>` format still add value on Qwen3-0.6B, or does Condition B
  already get most of it?

## Related

- [[overview]]
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]]
- [[sources/dissertation/experimental-planning-document]]
- [[experiments/experiment-catalog]]
