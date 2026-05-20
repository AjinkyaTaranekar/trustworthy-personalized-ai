---
title: Full Pipeline Implementation Plan
type: query
tags: [planning, pipeline, grpo, personalisation, empathy, ontology, tool-use, training]
sources:
  - wiki/queries/grpo-and-personalisation-master-plan.md
  - wiki/experiments/experiment-catalog.md
  - wiki/decisions/2025-10-01-four-module-architecture.md
updated: 2026-05-02
status: current
---

# Full Pipeline Implementation Plan

**Phase-by-phase build-out of the complete four-module pipeline: GRPO training, User Modelling (FalkorDB + 5W+H + Mem0g write pattern + scrutability), Empathy (AppraisePLM-labelled Qwen fine-tuning), Ontology Verifier, and Retrieval Gating — all gated behind feature flags so every component is independently toggleable and the pipeline runs end-to-end in stub mode before any GPU work.**

---

## Current State of Branch (`feat/grpo-and-personalisation-stack`)

| File | Status |
|---|---|
| `sft_*.py` + `1_dataset_generator.py` | ✅ Complete |
| `2_model_trainer.py` (SFT + GRPO/DAPO modes) | ✅ Complete |
| `3_infererence.py` (FastAPI server, tool registry, dependency monitor, security blockers) | ✅ Complete |
| `4_benchmark.py` (constitutional probes + adversarial suite) | ✅ Complete |
| `5_context_degradation.py` | ✅ Complete |
| `experiment0_reasoning_comparison.py` | ✅ Complete |
| `run_all.sh` + `preflight_check.sh` | ✅ Complete |
| Feature flag system | ❌ Missing |
| FalkorDB / Docker setup | ❌ Missing |
| User Modelling module | ❌ Missing |
| Empathy module | ❌ Missing |
| Ontology Verifier module | ❌ Missing |

---

## Feature Flags (`pipeline/config.py`)

Single `PipelineConfig` dataclass imported by every script. All flags default to `False` except `ENABLE_SFT`, which is the baseline. When a flag is `False` the module returns a neutral stub (empty user profile, no appraisal block, no verification pass) so the full pipeline always executes end-to-end without crashing.

| Flag | Default | Controls |
|---|---|---|
| `ENABLE_SFT` | `True` | Phase 1 supervised fine-tuning — always on, the constitutional baseline |
| `ENABLE_GRPO` | `False` | Phase 2 GRPO/DAPO RL training loop; requires `checkpoint_sft` to exist first |
| `ENABLE_USER_MODELLING` | `False` | FalkorDB 5W+H graph, Mem0g write pipeline, retrieval gating, scrutability endpoints |
| `ENABLE_EMPATHY` | `False` | Appraisal-conditioned generation; requires `data/appraisal_labels.jsonl` |
| `ENABLE_PERSONALISATION` | `False` | Per-query 5W+H slot relevance gate; depends on `ENABLE_USER_MODELLING` |
| `ENABLE_ONTOLOGY_VERIF` | `False` | Post-hoc SPARQL claim scorer against a chosen OWL ontology |

**Dependency rule:** `ENABLE_GRPO` requires `ENABLE_SFT` to have run first (reference policy = `checkpoint_sft`). `ENABLE_PERSONALISATION` requires `ENABLE_USER_MODELLING`. All other flags are independent.

---

## Dependency Graph

```
Phase 0  (config.py + docker-compose.yml)
    │
    ├──────────────────────────────────────────────────────┐
    │                         │                            │
Phase 1                  Phase 2                      Phase 3
(User Modelling)         (Empathy)                    (Ontology Verifier)
    │                         │                            │
    │        [PARALLEL — all three independent]            │
    └──────────────────┬───────────────────────────────────┘
                       │
                  Phase 4 (Integration — wire all modules into 3_infererence.py + run_all.sh)
                       │
                  Phase 5 (Smoke test — every flag combination, dry run passes end-to-end)
                       │
                  Phase 6 (GPU work — Monday onwards, data generation + training runs)
```

**Sequential constraints:**
- Phase 0 must complete before anything else starts.
- Phases 1, 2, 3 are fully independent of each other and can run in parallel.
- Phase 4 requires Phases 1, 2, and 3 to be complete.
- Phase 5 requires Phase 4 to be complete.
- Phase 6 requires Phase 5 to pass cleanly.

**Within Phase 2, two internal tracks run in parallel:**
- Track A (offline labelling: P2.1–P2.2) can start on day one alongside Phase 1.
- Track B (SFT pipeline updates: P2.3–P2.5) needs the appraisal format spec from P2.2 but not the full labels file.

---

## Phase 0 — Foundation
**Sequential. Blocks everything. Estimated ~1 hour.**

### P0.1 — `pipeline/config.py`

New file. `PipelineConfig` dataclass with all six flags and their defaults. Every pipeline script imports this at the top and checks `config.ENABLE_X` before invoking a module. The config can also be loaded from a YAML file at runtime so flags can be set per experiment run without editing source code.

### P0.2 — `docker-compose.yml` (repo root)

Single service: `falkordb/falkordb:latest` on port 6379 with a named volume for graph persistence. One `docker compose up -d` gives the User Modelling module its backend. No other service dependencies.

### P0.3 — Baseline verification

Run `./pipeline/preflight_check.sh` and `./pipeline/run_all.sh --dry_run` on the current branch state before any new code is added. Every existing check must pass. This is the contract: the baseline is clean before we extend it.

---

## Phase 1 — User Modelling
**Parallel with Phases 2 and 3. Estimated ~5 hours. One new file: `pipeline/user_modelling.py`.**

The write pipeline follows the Mem0g pattern (arXiv:2504.19413) but diverges at conflict resolution: Mem0's resolver is silent (the LLM decides ADD/UPDATE/DELETE/NOOP with no user notification); this implementation surfaces conflicts to the user before writing, and never deletes — deprecated facts get a `:DEPRECATED_BY` edge, preserving the scrutability audit trail.

### P1.1 — FalkorDB client + 5W+H schema

`GraphClient` class wraps the `falkordb` Python package. All Cypher queries live in this class — no raw Cypher scattered across other files. Node types and edge types are defined as constants at the top of the file.

**Node types:**

| Node | 5W+H slot | Captures |
|---|---|---|
| `User` | Who | Expertise level, domain, role |
| `Goal` | Why | Long-term objectives |
| `Task` | What | Current question type or task |
| `Context` | Where | Situation, environment, constraints |
| `Time` | When | Session timestamp, recency metadata |
| `Style` | How | Communication preference, verbosity, formality |
| `Skill` | Who/How | Known skills and knowledge gaps |

**Edge types:**

```
(User)-[:PURSUES]->(Goal)
(User)-[:HAS_SKILL]->(Skill)
(User)-[:PREFERS]->(Style)
(User)-[:IS_WORKING_ON]->(Task)
(Goal)-[:CONFLICTS_WITH]->(Goal)
(Preference)-[:DEPRECATED_BY]->(Preference)
(Belief)-[:USER_CORRECTED]->(Belief)
```

`GraphClient` methods: `create_node()`, `create_edge()`, `query_subgraph(node_id, depth)`, `get_all_nodes(session_id)`, `deprecate(edge_id, reason)`, `mark_user_corrected(node_id)`.

### P1.2 — Mem0g write pipeline (4 stages)

Triggered after every user turn when `ENABLE_USER_MODELLING = True`.

1. **`entity_extractor(turn_text) → list[TypedEntity]`** — LLM call (Qwen via the inference server) extracts typed 5W+H entities from the turn as structured JSON. Prompt instructs the model to tag each entity with its 5W+H slot type.
2. **`relation_generator(entities, existing_graph) → list[TypedEdge]`** — LLM call infers typed edges between the new entities and nodes already in the graph. Provides the current graph as context so it can detect natural relations.
3. **`conflict_detector(new_edges, graph_client) → list[Conflict]`** — pure Cypher: for each new edge, queries whether a contradicting edge already exists. Returns a list of conflicts with the conflicting edge ID and content. Does not write anything.
4. **`conditional_write(edges, conflicts, graph_client) → WriteResult`** — if conflicts exist: append a conflict flag to the response metadata (surfaced to the user at the API layer); write the new edge with `:DEPRECATED_BY` pointing to the old contradicting edge. If no conflicts: write directly. Never delete any node or edge.

### P1.3 — Retrieval gating

**`retrieve_for_query(query_text, session_id, graph_client) → dict`**

The anti-over-personalisation gate. Per query: classify which 5W+H slots are relevant to this specific query (lightweight LLM call or keyword classifier). If no slots are relevant (e.g., a factual maths question), return `{}` — no user context injected. If relevant slots are identified, fetch the depth-2 subgraph from those nodes only and return as a structured context dict. This is the architectural response to the 13.9–85% context-inflation degradation (Du et al. 2025) and the 26–61% task degradation from unconditional memory injection (OP-Bench, Hu et al. 2026).

### P1.4 — Scrutability endpoint handlers

Three FastAPI route handler functions written in `user_modelling.py`, ready to be registered on the server in Phase 4:

- `GET /memory/inspect/{session_id}` — returns NL summary of the user's full 5W+H graph (all nodes + active edges), formatted for a non-technical user.
- `POST /memory/contest` — user flags a specific belief node as wrong. Marks `:CONTESTED` on that node; the retrieval gate will not inject contested nodes until resolved.
- `POST /memory/correct` — user provides the accurate belief. Triggers `:USER_CORRECTED` edge from old node to new node; re-runs retrieval with updated graph.

---

## Phase 2 — Empathy
**Parallel with Phases 1 and 3. Estimated ~5 hours. Two internal tracks.**

AppraisePLM (Debnath, Graham, Conlan — CoNLL 2025, supervisor co-authored) is used **offline as a labeller only** — it generates ground-truth appraisal vectors for the training data. At inference time Qwen produces its own appraisal analysis from its fine-tuning, with no external model dependency. This approach cites and builds on AppraisePLM while making the thesis contribution the adaptation to generative conditioning, which is a different task from classification.

### Track A — Offline labelling (P2.1–P2.2)

Can start immediately on day one, in parallel with all other phases. Runs on CPU.

**P2.1 — `pipeline/appraisal_labeller.py`**

Standalone offline script. Steps:
1. Download `empathetic_dialogues` dataset via HuggingFace `datasets`.
2. Clone and load `alokdebnath/appraise-PLM` locally (DeBERTa checkpoint).
3. For each conversation turn, run the two-head forward pass: appraisal regression (21-dim float vector) + emotion classification.
4. Save `(utterance, speaker, emotion_label, appraisal_21dim, top3_dimensions)` as `data/appraisal_labels.jsonl`.

CLI: `python appraisal_labeller.py --output data/appraisal_labels.jsonl --limit 5000`

**P2.2 — Appraisal format specification**

Defines how appraisal reasoning appears inside Qwen's `<think>` block. Standardised XML element so the constitution checker can validate it:

```xml
<appraisal>
  novelty: 0.82 | valence: -0.41 | coping_potential: 0.23 | goal_relevance: 0.91
  reading: high novelty event, negative valence, low coping capacity, highly goal-relevant
  → user is stressed about something new they cannot control and it matters to them
</appraisal>
```

This element lives between `<think>` and `</think>`, before the main reasoning. The `<answer>` block is then conditioned on the reading. When `ENABLE_EMPATHY = False`, Qwen generates without the `<appraisal>` element and the format checker ignores its absence.

### Track B — SFT pipeline update (P2.3–P2.5)

Needs format spec from P2.2 before starting.

**P2.3 — `sft_question_generator.py` — add `appraisal_empathy` category**

New question category draws from `data/appraisal_labels.jsonl`. Each example is a real EmpatheticDialogues turn where AppraisePLM detected a non-neutral appraisal profile (valence ≠ 0.5, coping_potential < 0.4, or goal_relevance > 0.7 — thresholds to be tuned). The question is the user's utterance; the expected output format includes an `<appraisal>` block in `<think>` and an empathetically conditioned `<answer>`.

**P2.4 — `sft_gold_response_generator.py` — handle appraisal category**

When category is `appraisal_empathy`: inject the ground-truth appraisal vector (from P2.1 labels) into the critic prompt so the frozen critic (`claude-opus-4-7`) grades whether (a) the `<appraisal>` block correctly identifies the dominant dimensions, and (b) the `<answer>` is empathetically appropriate given the stated reading. Both criteria must pass for the example to be accepted by the rejection sampler.

**P2.5 — `pipeline/empathy.py`**

Runtime empathy module used by Phase 4. One public function:

`analyse_appraisal(turn_text) → AppraisalContext`

When `ENABLE_EMPATHY = True`: returns the top-3 appraisal dimensions and a one-line reading as a structured dict, to be injected into the system prompt prefix before generation. The Qwen model (trained on appraisal examples from P2.3–P2.4) generates its own `<appraisal>` block in the think chain — the system prompt injection is a nudge, not a constraint. When `ENABLE_EMPATHY = False`: returns an empty dict, no prompt injection.

---

## Phase 3 — Ontology Verifier
**Parallel with Phases 1 and 2. Estimated ~2 hours. One new file: `pipeline/ontology_verifier.py`.**

Implements Experiment 6 Approach B (post-hoc verifier). The ontology is loaded once at server startup; SPARQL queries run against it per response when the flag is on.

### P3.1 — Ontology selection

A config constant `ONTOLOGY_PATH` points to the OWL file used. The choice is a research decision logged separately. Candidates: schema.org subset (broad factual claims), a mathematics/CS ontology, or a domain-specific one relevant to the test queries. The code is ontology-agnostic as long as the file is valid OWL and supports SPARQL via `rdflib`.

### P3.2 — `pipeline/ontology_verifier.py`

Three public functions:

- **`extract_claims(response_text) → list[str]`** — regex + short LLM call to pull atomic factual assertions from an assistant response (e.g., "The capital of France is Paris", "Python uses dynamic typing").
- **`verify_claim(claim, graph) → VerificationResult`** — constructs a SPARQL query from the claim, fires it against the loaded `rdflib` graph, returns `(verified: bool, confidence: float, evidence: str | None)`.
- **`score_response(response_text, graph) → OntologyScore`** — calls both functions, returns `ontology_score` (mean confidence across all extracted claims), per-claim breakdown, and a list of unverified claims. When `ENABLE_ONTOLOGY_VERIF = False`, returns `score=1.0` and empty evidence — no cost.

---

## Phase 4 — Integration
**Sequential after Phases 0, 1, 2, 3. Estimated ~4 hours. Three files modified.**

### P4.1 — `pipeline/3_infererence.py`

Add flag-gated module hooks to the request/response lifecycle:

```
[User turn received]
  if ENABLE_USER_MODELLING:
      user_modelling.write_pipeline(turn, session_id)

  if ENABLE_PERSONALISATION:
      user_context = user_modelling.retrieve_for_query(turn, session_id)
  else:
      user_context = {}

  if ENABLE_EMPATHY:
      appraisal_ctx = empathy.analyse_appraisal(turn)
  else:
      appraisal_ctx = {}

  system_prompt = build_system_prompt(user_context, appraisal_ctx)

  response = model.generate(system_prompt, turn)

  if ENABLE_ONTOLOGY_VERIF:
      ontology_score = ontology_verifier.score_response(response)
  else:
      ontology_score = None

  return response + metadata(ontology_score, appraisal_ctx, conflict_flags)
```

Also register the three scrutability routes from P1.4: `/memory/inspect`, `/memory/contest`, `/memory/correct`.

### P4.2 — `pipeline/run_all.sh`

Add two conditional stages before the SFT stage:

- **Stage 0**: `docker compose up -d` (start FalkorDB) — only executes when `ENABLE_USER_MODELLING=true` in the config.
- **Stage 0.5**: `python appraisal_labeller.py` — only executes when `ENABLE_EMPATHY=true` and `data/appraisal_labels.jsonl` does not already exist.

### P4.3 — `pipeline/preflight_check.sh`

Add three new checks:

1. If `ENABLE_USER_MODELLING`: verify FalkorDB is reachable on port 6379.
2. If `ENABLE_EMPATHY`: verify `data/appraisal_labels.jsonl` exists and is non-empty.
3. If `ENABLE_ONTOLOGY_VERIF`: verify `ONTOLOGY_PATH` file exists and is readable.

---

## Phase 5 — Smoke Test + Docs
**Sequential after Phase 4. Estimated ~2 hours.**

### P5.1 — End-to-end dry run

All six flags `True`, `--dry_run` mode. Every code path executes; all modules return stubs or minimal real outputs. Target: zero crashes, zero missing imports, zero unresolved config references.

### P5.2 — Flag toggle matrix

For each of the six flags, test independently: (a) flag `True` → module executes correctly, (b) flag `False` → stub fallback returns neutral output, (c) downstream response is unaffected by the stub. This is the minimum confidence bar before GPU day — if any combination crashes, it gets fixed here.

### P5.3 — `README.md`

New section: module architecture diagram showing the four-module design and which flags control which code paths; Docker prerequisite for User Modelling; note that AppraisePLM labelling is a one-time offline step.

### P5.4 — Wiki sync

Update `wiki/sources/code/training-and-benchmark.md` to reflect all new modules and flag system. Append to `wiki/log.md`.

---

## Phase 6 — GPU Work (Monday onwards)
**Sequential after Phase 5. Steps run strictly in order.**

| Step | Command | Output | Prerequisite |
|---|---|---|---|
| 6.1 | `python appraisal_labeller.py` | `data/appraisal_labels.jsonl` | AppraisePLM repo cloned |
| 6.2 | `python 1_dataset_generator.py` | `data/train_interleaved.jsonl` (incl. appraisal category) | 6.1 done |
| 6.3 | `python 2_model_trainer.py --mode sft` | `models/checkpoint_sft/` | 6.2 done |
| 6.4 | `python 4_benchmark.py --probe_only --save_as_baseline` | `reports/constitution_baseline.json` | 6.3 done |
| 6.5 | `python experiment0_reasoning_comparison.py --benchmark all` | `reports/experiment0_*.json` | 6.3 done (parallel with 6.4) |
| 6.6 | `python 2_model_trainer.py --mode grpo --reward_type c` | `models/checkpoint_grpo_c/` | 6.3 done |
| 6.7 | `python 2_model_trainer.py --mode grpo --reward_type d` | `models/checkpoint_grpo_d/` | 6.6 done |
| 6.8 | `python 4_benchmark.py --adversarial_only` on A/B/C/D | Full ablation comparison | 6.3, 6.6, 6.7 done |

Steps 6.4 and 6.5 can run in parallel (both only need `checkpoint_sft`). All GRPO steps are strictly sequential.

---

## New Files

| File | Phase | Purpose |
|---|---|---|
| `pipeline/config.py` | 0 | Single source of truth for all feature flags |
| `docker-compose.yml` | 0 | FalkorDB local instance (Redis + FalkorDB module) |
| `pipeline/user_modelling.py` | 1 | FalkorDB client, 5W+H schema, 4-stage Mem0g write pipeline, retrieval gate, scrutability handlers |
| `pipeline/appraisal_labeller.py` | 2A | Offline: AppraisePLM → EmpatheticDialogues appraisal labels JSONL |
| `pipeline/empathy.py` | 2B | Runtime appraisal analysis + system-prompt injection |
| `pipeline/ontology_verifier.py` | 3 | SPARQL claim extractor + verifier + response scorer |

## Modified Files

| File | Phase | What changes |
|---|---|---|
| `pipeline/3_infererence.py` | 4 | Flag-gated lifecycle hooks for all four new modules; scrutability routes registered |
| `pipeline/sft_question_generator.py` | 2B | `appraisal_empathy` question category added |
| `pipeline/sft_gold_response_generator.py` | 2B | Appraisal category critic prompt; validates `<appraisal>` block |
| `pipeline/sft_dataset_assembler.py` | 2B | `appraisal_meta` key added to output JSONL |
| `pipeline/run_all.sh` | 4 | FalkorDB start stage + appraisal labelling stage |
| `pipeline/preflight_check.sh` | 4 | FalkorDB reachability + appraisal labels + ontology file checks |
| `README.md` | 5 | Module architecture diagram + Docker prerequisite + new flags documented |

---

## Fastest Path to GPU-Ready

```
Day 1 (now):
  → Phase 0 (~1 hr): config.py + docker-compose.yml + baseline dry-run
  → Then start Phase 1 + Phase 2 Track A + Phase 3 in parallel

Day 2:
  → Finish Phase 1 (user_modelling.py complete)
  → Finish Phase 2 Track B (SFT pipeline updates)
  → Finish Phase 3 (ontology_verifier.py complete)

Day 3:
  → Phase 4: wire everything into 3_infererence.py + run_all.sh + preflight
  → Phase 5: smoke tests + docs

Monday:
  → Phase 6: GPU work starts
```

---

## Related

- [[queries/grpo-and-personalisation-master-plan]] — detailed design rationale for each module; industry benchmarks; sequencing rationale
- [[decisions/2025-10-01-four-module-architecture]] — the binding architectural decision this plan implements
- [[experiments/experiment-catalog]] — the experiments this pipeline enables
- [[sources/papers/appraise-plm]] — AppraisePLM; the offline labeller for Phase 2
- [[topics/personalisation]] — 5W+H schema design; Mem0g pattern rationale; scrutability gap
- [[topics/empathy]] — appraisal-conditioned generation design
- [[topics/ontology-integration]] — Experiment 6 design; Approach A vs B
- [[entities/grpo]] — GRPO/DAPO training design; composite reward function
