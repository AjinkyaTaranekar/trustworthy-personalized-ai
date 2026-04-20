---
title: "PAL: Program-Aided Language Models"
type: source
arxiv_id: 2211.10435v2
authors: Gao, Madaan, Zhou, Alon, Liu, Yang, Callan, Neubig
year: 2022
tags: [tool-use, reasoning, code, delegation]
sources:
  - docs/Assets/PAL Program-aided Language Models (2211.10435v2).pdf
  - docs/Literature Notes/PAL Program-aided Language Models (2211.10435v2).md
updated: 2026-04-19
status: current
---

# PAL — Program-Aided Language Models

**The LLM reads the problem and writes a Python program as its reasoning trace; a Python interpreter — not the LLM — runs the program and produces the answer.**

## What it does
Reframes CoT: decomposition stays with the LLM, execution moves to the interpreter. On GSM8K, PAL + Codex beats PaLM-540B with CoT by 15% top-1 despite being far smaller.

## Why it matters for this thesis
PAL is the **foundational paper for honest delegation**. It operationalises the thesis claim that an LLM is better as a *reasoner-and-interface* than as a *solver*. This justifies the `python_execute` tool on the [[entities/constitution|constitution]] surface, and the tool-integrity reward in the RL pipeline. Also, PAL-style outputs are inherently more auditable: the code **is** the reasoning trace, and anyone can run it to verify — a concrete win for [[topics/tool-use-and-verification]] over prose-CoT.

## Related

- [[topics/reasoning]]
- [[topics/tool-use-and-verification]]
- [[sources/papers/chain-of-thought-prompting]] — what PAL replaces for arithmetic
- [[sources/papers/react]] — generalises tool delegation beyond code
- [[entities/constitution]]

## Sources

- `docs/Assets/PAL Program-aided Language Models (2211.10435v2).pdf`
- `docs/Literature Notes/PAL Program-aided Language Models (2211.10435v2).md`
