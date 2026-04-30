---
title: Tags Index
type: meta
updated: 2026-04-30
---

# Tags Index

**Canonical tag vocabulary. Before adding a new tag to a page's frontmatter, check this file.** If a near-match already exists, reuse it. If nothing fits, add the tag here *with a one-line meaning* in the same commit.

Counts below were last audited on **2026-04-20** (post-normalisation). Expect them to drift; re-audit during lint.

---

## Themes
Tags that correspond 1:1 with a page in `topics/`.

| Tag | Count | Meaning |
| --- | ----- | ------- |
| `reasoning` | 21 | Trustworthy reasoning — anything that informs [[topics/reasoning]] |
| `empathy` | 4 | Empathy via appraisal theory — informs [[topics/empathy]] |
| `personalisation` | 3 | User modelling / privacy — informs [[topics/personalisation]] |
| `explainability` | 3 | Scrutability + XAI — informs [[topics/explainability]] |
| `tool-use` | 6 | Tool delegation (PAL/ReAct/MCP) — informs [[topics/tool-use-and-verification]] |
| `ontology` | 4 | Ontology-LLM integration — informs [[topics/ontology-integration]] |
| `foundations` | 8 | Tokenisation / attention / embeddings — informs [[topics/llm-foundations]] |
| `verification` | 3 | Post-hoc claim verification (incl. ontology) |
| `security` | 5 | Security threats to LLMs (prompt injection, alignment regression, OWASP taxonomy) — informs [[topics/security-and-privacy]] |
| `privacy` | 4 | Data privacy, user-data protection, GDPR — informs [[topics/security-and-privacy]] |

## Techniques
Named methods or paradigms.

| Tag | Count | Meaning |
| --- | ----- | ------- |
| `rl` | 14 | Reinforcement learning (any algorithm) |
| `sft` | 3 | Supervised fine-tuning |
| `cot` | 6 | Chain-of-Thought prompting |
| `rag` | 3 | Retrieval-Augmented Generation (pattern) |
| `retrieval` | 4 | Retrieval mechanisms generally (supersets `rag`) |
| `lora` | 1 | Low-Rank Adaptation fine-tuning |
| `tokenisation` | 2 | BPE / subword tokenisation (British spelling) |
| `prompting` | 3 | Prompt-engineering methods (no weight updates) |
| `distillation` | 3 | Teacher → student knowledge transfer |
| `latent` | 3 | Latent-space / continuous reasoning |
| `diffusion` | 2 | Diffusion-language-model family |
| `architecture` | 3 | New architecture or architectural modification |
| `interleaved` | 2 | Interleaved thinking / reasoning with action |
| `search` | 3 | Search-based reasoning (tree, beam, deliberation) |
| `attention` | 2 | Attention mechanism |
| `embeddings` | 3 | Word / token / contextual embeddings |
| `agents` | 2 | Agentic (multi-turn, tool-using) LLMs |
| `process-rewards` | 1 | RL reward on reasoning process, not outcome |
| `sycophancy` | 3 | Model over-agreement with user preference; alignment failure — informs [[topics/personalisation]] and [[topics/security-and-privacy]] |
| `over-personalisation` | 2 | Applying personalisation when it degrades task quality or overrides explicit intent |
| `scrutability` | 3 | User's ability to inspect, contest, and correct the model's beliefs — informs [[topics/explainability]] and [[topics/personalisation]] |

## Entities & algorithms
Concrete artefacts with their own pages in `entities/`.

| Tag | Count | Meaning |
| --- | ----- | ------- |
| `grpo` | 5 | Group Relative Policy Optimization |
| `ppo` | 1 | Proximal Policy Optimization (value-based RL baseline) |
| `mcp` | 2 | Model Context Protocol |
| `constitution` | 3 | The 19-principle constitution |
| `principles` | 2 | The individual constitution principles |
| `graph-rag` | 1 | KG-backed RAG for user memory |
| `5w-h` | 1 | Who/What/When/Where/Why/How user-modelling schema |
| `appraisal-theory` | 3 | Appraisal-theoretic emotion framework |
| `transformers` | 2 | Transformer architecture (as a named family) |
| `bpe` | 1 | Byte-Pair Encoding |
| `deepseek` | 2 | DeepSeek model family |
| `qwen` | 1 | Qwen model family |

## Modalities
| Tag | Count | Meaning |
| --- | ----- | ------- |
| `multimodal` | 2 | Vision + language or GUI |
| `vision-language` | 1 | VLMs specifically |
| `gui` | 1 | GUI-agent tasks |
| `small-model` | 3 | Sub-7B model work |

## Evaluation & caveats
| Tag | Count | Meaning |
| --- | ----- | ------- |
| `evaluation` | 3 | Benchmark / evaluation methodology |
| `memorisation` | 1 | Memorisation-vs-reasoning distinction |
| `caveat` | 1 | Paper documenting a limitation of prevailing technique |
| `trade-off` | 1 | Explicit accuracy-vs-efficiency analysis |
| `latency` | 1 | Latency / TTFT concerns |
| `ttft` | 1 | Time-to-first-token (sub-case of latency) |

## Document types
| Tag | Count | Meaning |
| --- | ----- | ------- |
| `thesis` | 4 | Core dissertation document |
| `synthesis` | 2 | Cross-cutting synthesis writing |
| `literature-review` | 1 | Lit-review section of dissertation |
| `plan` | 1 | Formal research plan |
| `notes` | 1 | Personal / informal notes |
| `todos` | 2 | Actionable item lists |
| `exploration` | 1 | Open exploration items (pre-decision) |
| `questions` | 1 | Open-question files |
| `bootstrap` | 1 | Generated at wiki bootstrap — may need later revision |

## Workflow & infrastructure
| Tag | Count | Meaning |
| --- | ----- | ------- |
| `code` | 3 | Code-scoped source pages |
| `pipeline` | 1 | The SFT/RL pipeline as a whole |
| `training` | 2 | Training-time concerns |
| `benchmark` | 1 | Benchmark harness / script |
| `context-degradation` | 1 | Multi-turn context-length eval |
| `experiments` | 2 | Experiment catalog / design |
| `planning` | 2 | Planning documents |
| `ablation` | 1 | Ablation study |
| `direction` | 1 | Research-direction decision |
| `protocol` | 2 | A named protocol (e.g. MCP) |
| `multi-agent` | 1 | Multi-agent system design |

## Narrow / specialised tags (kept, but low-frequency)
These are single-use tags retained because they name a specific technique that could recur. Review during lint — fold in if a second paper uses them.

`critique`, `dr-grpo`, `value-based`, `hierarchical`, `metacognition`, `state`, `continuous`, `refinement`, `self-reward`, `self-training`, `self-correction`, `variational`, `classifier`, `inference-efficiency`, `delegation`, `deliberation`, `automation`, `depth`, `looped`, `moe`, `bidirectional`, `significance`, `sentiment`, `graph`, `memory`, `knowledge`, `anthropic`, `code` (as in source-code), `formal`, `objectives`, `timeline`.

---

## Deprecated — do NOT use

These appeared once or twice in early files and have been normalised away.

| Deprecated | Use instead | Reason |
| ---------- | ----------- | ------ |
| `foundation` | `foundations` | Plural matches topic name |
| `tokenization` | `tokenisation` | British spelling (matches user's prose) |
| `tools` | `tool-use` | Canonical theme name |
| `vectors` | `embeddings` | Redundant with `embeddings` |
| `model` | _drop_ | Too broad; use a specific family tag (`qwen`, `deepseek`) |
| `base` | _drop_ | Covered by `small-model` or family tag |
| `framework` | _drop_ | Too broad; use the specific framework tag (`5w-h`, `appraisal-theory`) |
| `affect` | `empathy` | Redundant |
| `emotion` | `empathy` | Redundant |
| `psychology` | `empathy` | Redundant |
| `experiment-6` | `ontology` | Don't tag by experiment number — experiments get renumbered |
| `benchmarks` | `evaluation` | Keep evaluation as the umbrella |

---

## Rules for adding new tags

1. **Read this file first.** If a canonical tag or near-match covers the concept, reuse it.
2. **Prefer the shortest meaningful form.** `cot` over `chain-of-thought`.
3. **Kebab-case.** `tool-use`, not `tool_use` or `toolUse`.
4. **Avoid experiment numbers.** They drift. Tag by theme instead.
5. **Avoid "framework" / "method" / "approach".** These say nothing. Use a specific name.
6. **Match the user's spelling** — British (`tokenisation`, `personalisation`, `memorisation`).
7. **When you add a new tag to a page, add it to this file in the same edit.** Include a one-line meaning.
8. **During lint, promote high-frequency narrow tags** (≥3 uses) into the main sections; **demote low-use duplicates** to this file's Deprecated section.

## Related

- `../CLAUDE.md` §3 "File Conventions" — frontmatter + tag rules
- [[index]] — full content catalog
- [[log]] — change history
