---
title: GRPO + Empathetic Personalisation Master Plan
type: query
tags: [planning, grpo, rl, personalisation, empathy, graph-rag, small-model, dapo, gemma, graph-memory]
sources:
  - wiki/overview.md
  - wiki/experiments/experiment-catalog.md
  - pipeline/2_model_trainer.py
  - pipeline/4_benchmark.py
updated: 2026-05-01
status: current
---

# GRPO + Empathetic Personalisation Master Plan

**Implementation roadmap for the thesis's two primary missing components: (1) the GRPO RL training loop for the Reasoning Module and (2) the 5W+H graph-memory + appraisal-conditioned empathy stack for the User Modelling and Generator Modules — both grounded in the official research question and the four-module architecture decision.**

## Official Research Question (anchor for all decisions)

> **"How can we architect a modular conversational AI system that prioritises transparency and is capable of genuine contextual empathy through systematic User Modelling and appropriate tool delegation, rather than relying solely on end-to-end neural generation?"**

Every design decision in this plan must trace back to one of: **modular** (not monolithic), **transparent** (audit trail, not post-hoc rationalisation), **systematic User Modelling** (5W+H + KG, scrutable), or **tool delegation** (MCP + logged).

## Summary

The thesis has a complete SFT pipeline and a well-designed RL architecture on paper, but zero GRPO code, zero personalisation/empathy code, four open security blockers that must be resolved before GRPO, and a missing reasoning paradigm comparison experiment (Phase 3 of the formal plan, researchplan.tex). Small models (Qwen3-0.6B, Gemma 4) are used because they enable **local on-device deployment** — the architectural basis for the privacy guarantee, not as an efficiency goal. This plan is grounded in the four-module architecture (see [[decisions/2025-10-01-four-module-architecture]]): Reasoning Module (SFT + GRPO), User Modelling Module (5W+H graph), Tool Integration Layer (MCP + logging), Generator Module (prompting + RAG). The plan is opinionated on implementation: DAPO over vanilla GRPO (entropy collapse fix for sub-1B), FalkorDB + Cognee over Neo4j (inference latency), Mem0g write-pipeline pattern (conflict detection before write, not silent overwrite). Layer 5 (scrutability) is a genuine novel contribution — no current production system (Mem0, ChatGPT, Gemini) provides user-facing inspect/contest/correct/deprecate/audit over their stored user model.

## Honest Audit — What Exists vs What Is Missing

| Component | Status | Gap |
|---|---|---|
| SFT data pipeline (`sft_*.py`) | Done — 13 categories, 23 principles, TOOL_PROFILES, multi-turn scaffolds, interleaved_tool_reasoning | Ready to run |
| SFT trainer (`2_model_trainer.py`) | Done — Unsloth + LoRA + SFTTrainer, 3 epochs, eval split | Ready to run |
| Constitutional drift detection (`4_benchmark.py`) | Done — 12 probes, regex scoring, SFT baseline + GRPO diff workflow | Run after SFT |
| **GRPO trainer** | **Missing** — not one line of GRPO code on `main` branch | Must build |
| **5W+H graph schema + backend** | **Missing** — conceptual only, no schema, no DB chosen | Must build |
| **User state update pipeline** | **Missing** — no entity extractor, no conflict detector | Must build |
| **Retrieval gating module** | **Missing** — no per-query relevance check | Must build |
| **Appraisal tagger** | **Missing** — AppraisePLM referenced, not integrated | Must build |
| **Appraisal-conditioned generation** | **Missing** — no conditioning mechanism | Must build |
| **Scrutability layer** | **Missing** — no user-facing graph inspection | Must build |

## Industry Patterns Worth Stealing

### On GRPO

**DeepSeek (R1/R1-Zero, arXiv:2501.12948)** introduced GRPO: drop the critic, compute baseline from mean reward across G=16 sampled completions per prompt, reward = accuracy (rule-based verifiable) + format (structural regex), KL coefficient β=0.001. Reference policy is the SFT checkpoint, not the base model — this anchors the constitution. The key lesson: start from `checkpoint_sft`, not from `unsloth/Qwen3-0.6B`.

**ByteDance (DAPO, arXiv:2503.14476)** published the most important GRPO improvement: three fixes that are especially critical for sub-1B models. (1) Clip-Higher: asymmetric clipping (ε_low=0.2, ε_high=0.28) lets high-reward completions update more freely, preventing entropy collapse where all G completions become identical; (2) Dynamic sampling: skip prompts where all G completions score identically — zero-gradient batches waste compute and reward signal; (3) Token-level policy gradient loss (= Dr.GRPO): normalise loss by actual completion token count, not sequence count, eliminating the bias that rewards verbosity. DAPO scored 50 points on AIME 2024 with Qwen-32B. For a 0.6B model with extremely limited capacity for entropy, implementing DAPO rather than vanilla GRPO is not optional.

**OpenAI (o1/o3/o4-mini)** use outcome-only RL with verifiable rewards — no process supervision, no reward model. Final answer correctness is checked by a rule-based verifier. The lesson: if you can check the answer automatically (Python tool execution, format regex, constitutional probe), you do not need a reward model. Your composite reward — format + accuracy (tool-executed) + tool_integrity (regex) + constitution_score (pre-computed in SFT data) — is entirely verifiable.

**Anthropic (Constitutional AI, arXiv:2212.08073)** use DPO + AI-feedback, not GRPO. Their generate–critique–revise loop (already in your `sft_gold_response_generator.py` via the `--critic_model` flag) is their SFT stage. The RL stage uses AI-generated preferences evaluated against constitutional rules. The key lesson already captured in your design: the frozen critic (`claude-opus-4-7`) grading every draft is the Anthropic SFT pattern; the constitutional probe baseline is the Anthropic RL monitoring pattern.

**Qwen (Qwen2.5-Math, arXiv:2409.12122)** use iterative generate → reward-model grade → SFT on accepted → GRPO, with a separate mathematical reward model. For your thesis, verifiable rewards replace the reward model — safer and more directly measurable.

### On Graph-Based Personalisation + Empathy

**ChatGPT Memory** stores lightweight conversation summaries as flat text, pre-computed and injected at context start. No graph, no typed relations. Multi-hop questions ("why does the user prefer X given their stated goal Y?") fail because there are no explicit edges. This is the architecture your thesis improves on.

**Mem0g (arXiv:2504.19413)** is the most directly relevant production system. It builds a directed, labelled knowledge graph alongside a vector store. The write pipeline is: entity extractor → relation generator → conflict detector → conditional write. It achieves 91% lower p95 latency vs full-context injection and beats vector-only Mem0 on multi-hop questions. The conflict detector (flag contradictions before overwriting) is the mechanism that keeps the graph scrutable — users can see what was flagged. This four-stage pipeline is the blueprint for Track 2 Layer 2.

**Hume AI (EVI)** uses 48 emotion dimensions, fused from voice prosody + linguistic + contextual signals. Partnered with Anthropic for emotionally intelligent Claude interactions. The key validation: emotion detection as a separate module that gates generation style (their prosody tagger → conditioned TTS) is the right architecture. Your AppraisePLM tagger → appraisal-conditioned generation is the text-only analogue. The 2024 paper (Simulating Emotions with Appraisal + RL, CHI 2024) validates integrating OCC appraisal dimensions with RL — directly citable.

**FalkorDB vs Neo4j vs Cognee**: FalkorDB wins on inference-time latency (500× faster p99 on neighbourhood expansion, 11/12 benchmark queries) using Sparse Matrix Algebra. Cognee wins as an orchestration layer: it abstracts over FalkorDB, Neo4j, NetworkX, and Kuzu without requiring a code rewrite when switching backends. The recommended stack is **Cognee + FalkorDB**: Cognee for LLM-native KG construction and unified memory abstraction, FalkorDB as the backend for low-latency retrieval. This replaces the three-way undecided choice currently in [[entities/graph-rag]].

## Four Security Blockers That Must Be Resolved BEFORE GRPO

**This section is drawn directly from `docs/security-analysis/security-review.tex` §5 "Mitigations and Open Problems". The author (Ajinkya) explicitly states: "The following problems remain open and needs to be addressed before the GRPO stage and any public deployment." Skipping these is out of scope.**

### Blocker 1 — Prompt Injection Hardening (OWASP LLM01) — ✅ DONE (2026-05-01)

**Implemented:** AST-based code validator (`_validate_code`) blocks all non-math imports and dangerous builtins before any `subprocess.run` call in `3_infererence.py`, `sft_rejection_sampler.py`, and `sft_math_question_generator.py`. Tool-output sanitiser (`_sanitise_tool_output`) strips prompt-injection patterns (`<tool>`, `<think>`, `ignore previous instructions`, etc.) from web/URL content before it enters the model context, truncates to 3,000 chars, and wraps in a structured `[TOOL_RESULT]` envelope. **Remaining gap:** process-level isolation (seccomp/containers) for a production deployment — current fix is training/research minimum viable hardening. See log `[2026-05-01] refactor | Security Blocker 1`.

### Blocker 2 — Independent Constitutional Verification (OWASP LLM04) — ✅ DONE (2026-05-02)

**Implemented:** `rule_check_response()` in `sft_gold_response_generator.py` provides deterministic out-of-band checks for P1 (CAPABILITY_CHECK present), P3 (no hallucinated or unavailable tools), P4 (math without code when code is available), P14 (adversarial capitulation in turn_2), and P18 (structural `<answer>` block). `_merge_violations()` ensures rule violations cannot be suppressed by an LLM `NO_VIOLATIONS` response. Loud warning printed when `--critic_model` not set. Smoke tested across 7 cases. **Remaining gap:** semantic violations still depend on the LLM critic — use `--critic_model claude-opus-4-7` for full compliance. See log `[2026-05-02] refactor | Security Blocker 2`.

### Blocker 3 — Adversarial Benchmark Suite (OWASP LLM01, LLM04) — ✅ DONE (2026-05-02)

**Implemented:** `ADVERSARIAL_PROBES` (14 probes) + `run_adversarial_probes()` in `4_benchmark.py`. Three categories: jailbreak (5 probes — direct override, DAN persona, system-prompt fishing, harmful roleplay, false authority), injection (3 probes — embedded instruction in structured content, false constitution update, numeric payload), regression (6 probes — sycophancy under expert pressure, format under pressure, tool hallucination, calibrated confidence on known/unknown facts, honesty under social pressure). CLI: `--adversarial`, `--adversarial_only`, `--attack_types`. Usage before GRPO: `python 4_benchmark.py --adversarial_only`. See log `[2026-05-02] refactor | Security Blockers 3 + 4`.

### Blocker 4 — Dependency Detection Protocol (OWASP LLM09) — ✅ DONE (2026-05-02)

**Implemented:** `DependencyMonitor` class in `3_infererence.py`. Tracks per-session interaction frequency (≥10/hour) and short-interval bursts (≥5 gaps < 30s). When either trigger fires, appends an autonomy-preserving disclosure to the model's answer — non-blocking, redirects to human support, never blocks the conversation. Cooldown prevents re-triggering within 1 hour. `session_id` added to `CompletionRequest`. Two new endpoints: `GET /dependency/status/{id}`, `POST /dependency/reset/{id}`. In-memory only — no persistence across restarts (privacy by design). See log `[2026-05-02] refactor | Security Blockers 3 + 4`.

---

## Track 1: GRPO Training — Six-Step Loop

Everything lives in `pipeline/`. The SFT steps are done; the GRPO step needs to be built.

```
Step 1  Generate SFT data          sft_question_generator + sft_gold_response_generator + sft_dataset_assembler → data/train_interleaved.jsonl
Step 2  Run SFT                    2_model_trainer.py → models/checkpoint_sft/
Step 3  Record SFT baseline        4_benchmark.py --probe_only --save_as_baseline → reports/constitution_baseline.json
Step 4  Implement GRPO trainer     2_model_trainer.py — add train_grpo() using trl.GRPOTrainer with DAPO fixes
Step 5  Run GRPO                   2_model_trainer.py --mode grpo --checkpoint checkpoint_sft → models/checkpoint_grpo/
Step 6  Drift monitoring           4_benchmark.py --probe_only --baseline reports/constitution_baseline.json  (after every checkpoint)
```

### Step 4 in detail — GRPO trainer design

The TRL library ships `GRPOTrainer` as a near drop-in for `SFTTrainer`. Critical differences:

**Composite reward function** (already designed in log entries, now needs code):
```python
def compute_reward(prompts, completions, metadata) -> list[float]:
    format_score   = check_think_tags(completions)           # verifiable: regex
    accuracy_score = check_tool_output_or_math(completions)  # verifiable: execute tool
    tool_integrity = check_no_hallucinated_tools(completions) # verifiable: regex
    constitution   = metadata["constitution_score"]           # pre-computed in SFT data
    return 0.30*format_score + 0.40*accuracy_score + 0.15*tool_integrity + 0.15*constitution
```

**DAPO improvements (mandatory for 0.6B)**:
- Token-level loss normalisation: divide by completion length, not group count
- Dynamic sampling: skip prompts where variance across G completions is zero
- Clip-Higher: ε_low=0.2, ε_high=0.28 asymmetric clipping

**Starting hyperparameters for Qwen3-0.6B**:
- Group size G=8 (memory-constrained; NVIDIA used G=16 on 1.5B)
- KL coefficient β=0.001 (same as R1 stage 1)
- Learning rate 1e-6 (lower than SFT — fine-tuning a fine-tuned model)
- Max sequence length 4096 (already set in MODEL_CONFIG)
- Rollout temperature 1.0
- Reference policy = `checkpoint_sft` (not base model)

### Ablation sequencing

| Run | Config | What it tests |
|---|---|---|
| A | Base Qwen3-0.6B, no training | Zero-shot floor |
| B | SFT only | Does constitutional formatting transfer? |
| C | SFT → GRPO, format+accuracy only | Does RL improve correctness? |
| D | SFT → GRPO, full composite reward | Full thesis contribution |

Run the constitutional probe after each checkpoint in C and D. If `constitution_score` drops ≥5 points from B's baseline, increase β and add SFT replay mixing (as documented in the log `[2026-05-01]` entries).

## Track 2: Empathetic Personalisation Stack — Five Layers

Build bottom-up. Each layer is a new pipeline script or module.

```
Layer 5  Scrutability UI           MCP /memory/inspect endpoint; NL summaries of graph; :USER_CORRECTED edge on correction
Layer 4  Empathy conditioning      AppraisePLM tagger → 21-dim appraisal vector → conditioned system prompt injection
Layer 3  Retrieval gating          Per-query 5W+H slot relevance classifier; only retrieve if slot is relevant
Layer 2  Graph update pipeline     Entity extractor → relation generator → conflict detector → conditional write (Mem0g pattern)
Layer 1  5W+H Graph + MCP server   Cognee + FalkorDB; 5W+H node schema; local MCP server (privacy-by-locality)
```

### Layer 1 — 5W+H Graph Schema

Node types following the 5W+H schema:
```
(User)   — who: expertise level, domain, role
(Goal)   — why: what the user is trying to achieve long-term
(Task)   — what: current task or question type
(Context)— where: situation, environment, constraints
(Time)   — when: session timestamp, recency metadata
(Style)  — how: communication preference, verbosity, formality
(Skill)  — who/how: known skills and gaps
```

Relations:
```
(User)-[:PURSUES]->(Goal)
(User)-[:HAS_SKILL]->(Skill)
(User)-[:PREFERS]->(Style)
(User)-[:IS_WORKING_ON]->(Task)
(Goal)-[:CONFLICTS_WITH]->(Goal)          — contradiction tracking
(Preference)-[:DEPRECATED_BY]->(Preference)  — user correction history
```

Backend: Cognee as the LLM-native KG construction layer; FalkorDB as the graph backend for neighbourhood expansion at inference time.

### Layer 2 — User State Update (Mem0g Pattern)

Trigger: after every user turn (not after every assistant response — update on what the user reveals, not what the assistant assumes).

Four stages: (1) entity extractor — LLM call extracting WHO/WHAT/WHEN/WHERE/WHY/HOW typed entities from the turn; (2) relation generator — LLM call inferring typed edges given the entities and existing graph context; (3) conflict detector — rule-based: does any new relation contradict an existing edge? Flag if yes, do not silently overwrite; (4) conditional write — only write after conflict resolution; mark deprecated facts with a `:DEPRECATED_BY` edge, never delete (deletions destroy the scrutability audit trail).

### Layer 3 — Retrieval Gating

The anti-over-personalisation gate. Do not inject the user graph into every prompt. Per-query: classify which 5W+H slots are relevant to this query; if none, return empty (a factual maths question needs no user context); if relevant slots found, fetch the subgraph at depth=2 from those nodes only. This is the architectural response to the 13.9–85% context-inflation degradation documented by Du et al. 2025 ([[sources/papers/context-length-hurts]]) and the 26–61% task degradation documented by OP-Bench ([[sources/papers/op-bench]]).

### Layer 4 — Appraisal-Conditioned Empathy

Two phases. Phase A: run AppraisePLM (code at https://github.com/alokdebnath/appraise-PLM) over the user's current turn to produce a 21-dimensional appraisal vector (novelty, valence, goal-relevance, coping-potential, etc.). Phase B: inject the top-scoring appraisal dimensions as structured context into the system prompt before generation — "User appraisal: high novelty, high goal-relevance, low coping potential, negative valence → user is stressed about something new they cannot control." The GRPO-trained model (Track 1, Condition D) then generates a response that is constitutionally grounded and empathetically conditioned. Evaluation: Experiment 2 — human raters blind-compare (a) base model, (b) appraisal-conditioned, (c) human gold standard.

### Layer 5 — Scrutability (the named contribution)

**No current production system has this.** Mem0's conflict resolution is entirely silent — the LLM Update Resolver runs ADD/UPDATE/DELETE/NOOP with zero user notification. ChatGPT's "Manage Memories" covers only explicitly saved memories; auto-learned beliefs from chat history are behind an all-or-nothing toggle. Letta is the most transparent but is a developer tool. The research literature has no formal definition of user-centric AI memory scrutability as a distinct concept — only a documented "transparency asymmetry" and a 20-year UMAP tradition that the field has not yet applied to conversational memory. This layer is therefore a genuine novel contribution.

The thesis defines scrutability for conversational agent memory as five constraints: (1) inspect — the user can read all beliefs the system holds about them, as NL summaries not raw data; (2) contest — the user can flag a belief as wrong before the system acts on it; (3) correct — the user supplies the accurate belief; (4) deprecate (not delete) — the corrected belief is marked `:USER_CORRECTED` and de-prioritised but kept, preserving the audit trail; (5) audit — the full history of what changed, when, and why is always readable. Implementation: the MCP server exposes `/memory/inspect` returning the 5W+H subgraph as NL; before any high-stakes response (emotional support, recommendation) the system cites which graph nodes influenced it; user correction triggers the `:USER_CORRECTED` edge and re-runs the relevance gate with the updated graph.

## Small-Model Constraint

All experiments use Qwen3-0.6B as the primary model and Gemma 4 (any available small variant) as the secondary comparison model. The central claim is not that a 0.6B model beats GPT-5 — it is that a 0.6B model with the right modular architecture (GRPO-trained reasoning + graph-gated personalisation + appraisal-conditioned empathy) is more trustworthy and empathetic than a larger monolithic model with no structure. Gemma 4 is included because Google's on-device inference story for Gemma directly mirrors the local-first privacy argument in [[sources/dissertation/security-privacy-social-ethics]]. Results from both model families strengthen the generalisability claim.

## Sequencing (respects the formal researchplan.tex phase ordering)

**Pre-GRPO prerequisites must be complete before any GRPO run starts.** Phase 3 (reasoning comparison) is also formally prior to GRPO.

| Week | Pre-GRPO + Reasoning Track | Personalisation Track |
|---|---|---|
| 1 | **Security Blocker 1:** Implement tool-output extraction layer (structured data conversion before model sees web content) | Read Mem0 (2504.19413) + PersonalAI (2506.17001) |
| 2 | **Security Blocker 2+3:** Add independent constitutional verifier + adversarial benchmark probes to `4_benchmark.py`; **Experiment 0:** run CoT / ToT / interleaved / latent reasoning comparison on GSM8K subset | Design FalkorDB + Cognee 5W+H schema; select node/edge types |
| 3 | **Security Blocker 4:** Add dependency detection monitor; read DAPO + VAPO; implement `train_grpo()` | Build Layer 1: MCP server + FalkorDB + 5W+H schema |
| 4 | Generate SFT data → run SFT → record constitutional baseline → Ablation C (GRPO format+accuracy only) | Build Layer 2: Mem0g-pattern update pipeline |
| 5 | Ablation D (full composite reward) + drift monitoring; Gemma 4 replication | Build Layer 3: retrieval gating + read Avoiding Over-Personalisation (2509.07133) |
| 6 | Ablation A/B/C/D full results → Experiment 1 chapter | Build Layers 4+5: AppraisePLM integration + scrutability endpoint; Experiment 2 pilot |

## Papers to Acquire Now

| Paper | ArXiv ID | Track | Why |
|---|---|---|---|
| DAPO | 2503.14476 | 1 | Implement this, not vanilla GRPO; entropy collapse fix for small models |
| DeepSeekMath (GRPO origin) | 2402.03300 | 1 | Foundational maths; group size + KL hyperparameters |
| LUSPO (length bias) | 2602.05261 | 1 | Explains why token-level normalisation fixes length bias |
| Mem0 | 2504.19413 | 2 | Production blueprint for Layer 2 write pipeline |
| PersonalAI (KG for agents) | 2506.17001 | 2 | KG schema design; hyper-edge pattern for 5W+H |
| Simulating Emotions (Appraisal + RL) | CHI 2024 | 2 | Validates Layer 4; OCC + RL integration; directly citable |
| Graph-based Agent Memory survey | 2602.05665 | 2 | Places your architecture in the literature taxonomy |
| Avoiding Over-Personalisation (rule-guided KG) | 2509.07133 | 2 | Validates Layer 3 gating logic; citable for the relevance gate |

## The Thesis Argument These Two Tracks Prove

Once both tracks are running, the ablation argument is: a constitutionally-grounded RL-trained small model (Track 1, Condition D) paired with a scrutably-gated, appraisal-conditioned user model (Track 2) outperforms both a generic LLM (Condition A) and an over-personalised model (no gating) on empathy scale, constitutional adherence, task accuracy, and user trust — while avoiding the context-inflation degradation documented by OP-Bench and the sycophancy incentivisation documented by Sharma et al. (2024). That argument is publishable on a 0.6B model. No other open-source sub-1B pipeline has this combination.

## Related

- [[experiments/experiment-catalog]] — Experiments 1, 2, 3, 5; ablation A/B/C/D
- [[entities/grpo]] — GRPO entity; DAPO implementation notes now added
- [[topics/personalisation]] — 5W+H + GraphRAG; FalkorDB/Cognee decision now recorded
- [[topics/empathy]] — Appraisal conditioning; CHI 2024 paper now added to sources
- [[entities/graph-rag]] — Backend candidates; decision now made: Cognee + FalkorDB
- [[entities/qwen3-0.6b]] — Primary training model
- [[sources/papers/deepseek-r1]] — GRPO origin
- [[sources/papers/vapo]] — Value-based alternative; read for trade-offs
- [[sources/papers/op-bench]] — Over-personalisation evidence; motivates Layer 3
- [[sources/papers/appraise-plm]] — Empathy tagger; Experiment 2 unblocked
- [[sources/dissertation/security-privacy-social-ethics]] — Local-first privacy argument; motivates MCP-wrapped FalkorDB

## Sources

- This query was generated by cross-referencing the repo's current pipeline against industry practice reports (2024–2025) covering DeepSeek R1, ByteDance DAPO, Qwen2.5-Math, Anthropic Constitutional AI, Mem0g, Hume AI EVI, FalkorDB/Neo4j benchmarks, and PersonalAI arXiv:2506.17001.
- All raw pipeline state drawn from: `pipeline/2_model_trainer.py`, `pipeline/4_benchmark.py`, `wiki/log.md` (2026-05-01 entries).
