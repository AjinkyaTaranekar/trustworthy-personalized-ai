---
title: "Experiment 3 — Thinker–Executor Dual-SFT Architecture"
type: experiment
tags: [sft, reasoning, tool-use, small-model, architecture, agents, multi-agent, on-device, trade-off, distillation, curriculum-learning]
sources:
  - pipeline/4_benchmark.py
  - wiki/experiments/sft-benchmark-analysis-20260525.md
updated: 2026-05-29
status: draft
---

# Experiment 3 — Thinker–Executor Dual-SFT Architecture

**The hypothesis is that a 0.6B model cannot simultaneously hold constitutional reasoning and reliable tool execution within a single fine-tuned weight set, and that splitting these responsibilities across two specialised SFT models — a Thinker and an Executor — recovers both capabilities without sacrificing either.**

## 1. Motivation

The SFT benchmarking campaign (2026-05-20 through 2026-05-25, documented in [[experiments/sft-benchmark-analysis-20260525]]) produced a finding that directly motivates this experiment. The vanilla base model (unsloth/Qwen3-0.6B) and the SFT fine-tuned model (ajinkyataranekar/trustworthy-ai-sft) exhibit **perfectly complementary but mutually exclusive failure modes**:

| Capability | Vanilla (base) | SFT fine-tuned |
|---|---|---|
| `<think>` block generation | Present (avg 906 chars, 0% empty) | Collapsed (avg 40 chars, 95% empty) |
| Answer production | Catastrophic failure (59/63 empty `answer_content`) | Restored |
| Constitutional compliance | Mixed (P1, P5, P6, P8 etc.) | Degraded on thinking-dependent principles |
| Adversarial robustness | Critically unsafe (2/14, 14.3%) | Substantially safer (9/14, 64.3%) |
| Tool call rate | Near zero (1 call across 21 probes) | Explosion (56 calls across 21 probes) |

The last row is the key diagnostic. **Tool-calling has replaced thinking.** The SFT training regime taught the model to emit tool calls instead of reasoning traces — a capacity displacement rather than a capacity addition. This is not a data-quality artefact; it is a model-scale constraint. At 0.6B parameters, the model does not have the representational headroom to maintain two competing output formats (long reasoning chains and structured JSON tool calls) simultaneously.

This observation is corroborated by the independent literature finding in [[sources/papers/replacing-thinking-with-tool-usage]] (Rainone et al., 2025, arXiv:2507.05065): *"for sizes up to 3B, the tool-calling approach most successfully elicits reasoning behaviour, while the text-based CoT approach fails to induce such improvement."* The paper treats this as a positive result (tool use as a reasoning proxy); our benchmarking reveals the inverse: when tool calling is the training target, constitutional reasoning collapses.

## 2. Related Work

### 2.1 The Thinking–Tool Calling Trade-off at Small Scale

**Rainone et al. (2025) — "Replacing thinking with tool usage enables reasoning in small language models"** ([arXiv:2507.05065](https://arxiv.org/abs/2507.05065)) is the closest published treatment of the phenomenon we observed empirically. The authors propose externalising reasoning into tool states via a Chain-of-Editing loop, showing that for models up to 3B parameters this outperforms internal CoT. Their framing is optimistic (tool use *replaces* thinking successfully); our finding is the dual risk: when a constitutional system needs *both* internal deliberation and tool execution, the substitution is destructive.

**Dualformer (2024) — "Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces"** ([arXiv:2410.09918](https://arxiv.org/pdf/2410.09918)) trains a single model to switch between fast (no trace) and slow (full CoT) modes at inference time via randomised trace masking during SFT. This is the single-model alternative to the proposal here. However, it requires the model to have sufficient capacity to hold both modes — a constraint that does not hold at 0.6B as our benchmarks demonstrate.

**Qwen3-0.6B Technical Report** ([entities/qwen3-0.6b]) explicitly supports a thinking/non-thinking mode toggle (`enable_thinking=True/False`) at the base model level. The design philosophy is the same: separate the two competencies. The SFT fine-tuning process, however, collapsed this separation by overwriting the thinking pathway with tool-call patterns.

### 2.2 Thinker–Executor as an Architectural Pattern

**Reason-Plan-ReAct (2025) — "A Reasoner-Planner Supervising a ReAct Executor for Complex Enterprise Tasks"** ([arXiv:2512.03560](https://arxiv.org/abs/2512.03560)) is the most direct architectural antecedent. The paper decouples a **Reasoning-Planning Agent (RPA)** — a large reasoning model that generates no tool calls, only plans — from a **ReAct Executor** — a smaller tool-calling model that acts on the plan and reports results back to the RPA for re-planning. Crucially: the two models share full conversation context but **never share the same weights**. This is the architecture the present experiment proposes to instantiate at 0.6B scale under constitutional constraints.

**Planner and Executor: Collaboration between Discrete Diffusion and Autoregressive Models** ([arXiv:2510.15244](https://arxiv.org/html/2510.15244v1)) demonstrates that a textual plan generated by a planner model, when appended to the executor's input, functions as an external chain-of-thought — matching the CoT paradigm across a model boundary rather than within a single model. This supports the communication protocol proposed in §4.3 below.

**OPERA: Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval** ([arXiv:2508.16438](https://arxiv.org/abs/2508.16438)) applies the same split for retrieval: a Goal Planning Module decomposes questions into sub-goals; a Reason-Execute Module acts on them. Reinforcement learning then tightens the interface between the two. The RL post-training approach is out of scope for the initial experiment run but marks a clear future-work direction if time permits before June 30.

**Can Small Agents Collaborate to Beat a Single Large Language Model?** ([arXiv:2601.11327](https://arxiv.org/abs/2601.11327), Żywot et al., January 2026) provides the most recent empirical evidence. Using Qwen3 models at 4B–32B scale, the study finds that *a small orchestrator with specialised sub-agents can outperform a single large monolithic model on tool-intensive benchmarks*. The specific finding most relevant here: **planner-only thinking improves decomposition quality, but unrestricted full thinking in the planner degrades tool orchestration**. This is exactly the regime we are in — the Thinker model must reason with full `<think>` blocks, while the Executor must never emit them.

**Wei et al. (2025) — "Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning"** ([arXiv:2511.10037](https://arxiv.org/html/2511.10037v1)) is the most architecturally precise antecedent found. It explicitly decouples a fine-tuned Planner from an Executor (GPT-4o in their experiments), trains the Planner with SFT then GRPO with hierarchical rewards, and represents the plan as a DAG. Key results with Qwen3 models: the 8B Planner achieves 80.3% on Easy / 31.9% on Hard tasks; average 2.29 inference steps per task. Code and training data (ComplexTool-Plan: 3K SFT + 787 RL instances) are publicly released at [github.com/weixiaolong94-hub/Beyond-React](https://github.com/weixiaolong94-hub/Beyond-React).

Two findings from this paper are directly load-bearing for the present experiment. First, they tested Qwen3-0.6B as a Planner and found SFT successful but **GRPO training unstable at 0.6B** — the RL phase had to be excluded for the smallest model. This is one reason the optional RL pass is cut (see §8/§9). Second, their finding that upfront DAG planning is unreliable at small scale directly motivated abandoning the batch-`<delegation>` design in favour of the **step-by-step loop** (§4.4): rather than asking the 0.6B Thinker to emit a structured multi-step plan at all, it decides one natural-language `<act>` at a time and reacts to each result — no graph topology, no stage grouping, and no second structured output format competing with reasoning. The ComplexTool-Plan dataset is not used directly (it is calibrated for 8B and uses 4,535-API toolsets far larger than the project's 4-tool Executor set).

### 2.3 Catastrophic Forgetting in SFT

**Liu et al. (2025) — "Improved SFT for LLMs to Mitigate Catastrophic Forgetting"** ([arXiv:2506.09428](https://arxiv.org/abs/2506.09428)) proposes Self-Distillation Fine-Tuning (SDFT), where the model teaches itself via in-context demonstrations. The technique substantially reduces forgetting on tasks not in the training distribution. This is relevant as a training procedure improvement for both the Thinker and Executor models independently, but does not address the structural capacity-displacement problem that motivated this experiment.

**Entropy-Adaptive Fine-Tuning (EAFT)** ([arXiv:2601.02151](https://arxiv.org/html/2601.02151v1)) addresses forgetting at the gradient level by selectively updating parameters based on prediction entropy. Domain-agnostic, including agent tool-use domains. Again applicable within each model in the dual architecture, not a substitute for the split itself.

### 2.4 Industry Precedent — Thinking Machines Lab

On 11 May 2026, Thinking Machines Lab (founded by Mira Murati, former CTO of OpenAI) released a research preview of **TML-Interaction-Small** — a 276B parameter Mixture-of-Experts model with 12B active parameters that operationalises the same thinker–executor split at frontier scale ([blog post](https://thinkingmachines.ai/blog/interaction-models/), [SiliconANGLE coverage](https://siliconangle.com/2026/05/11/thinking-machines-drops-new-highly-responsive-model-designed-humanlike-interactions-real-time/)).

The TML architecture uses **two models in parallel** at all times:

- **Interaction Model** — always on; processes audio, video, and text in 200ms *micro-turns*; interleaves input processing and output generation on the same clock cycle; handles real-time human dialogue without waiting for turn boundaries.
- **Background Model** — runs asynchronously; handles deeper reasoning, planning, and tool execution; shares full conversation context with the interaction model via continuous synchronisation.

The interaction model achieved 0.40s latency on FD-bench v1, compared to GPT-realtime-2.0 at 1.18s and Gemini at 0.57s ([Unite.AI](https://www.unite.ai/thinking-machines-lab-ships-first-model-with-200ms-real-time-interaction/)). The defining design principle is identical to what this experiment proposes: **real-time interactivity and deep reasoning are fundamentally different computational regimes that should not compete for the same weights**.

The key difference from the present experiment is scale and motivation. TML operates at 276B/12B active parameters on server hardware; the present proposal targets two 0.6B models on consumer or mobile hardware. The privacy and on-device deployment constraint (see [[experiments/experiment-catalog]] binding note and [[topics/security-and-privacy]]) means the TML approach validates the *architecture*, not the deployment envelope. This is a genuine novelty contribution: the dual-model thinker–executor pattern applied at sub-1B scale under constitutional constraints has no published precedent (see [[reference/harness-engineering-research-papers]] gap: no sub-1B constitutional harness paper exists).

Also relevant: **Pangu Embedded (2025) — "An Efficient Dual-system LLM Reasoner with Metacognition"** ([arXiv:2505.22375](https://arxiv.org/abs/2505.22375)) implements fast/slow thinking within a **single** 7B model, using a metacognition module to route requests. The Pangu approach is a single-model analogue. Our benchmarks indicate that single-model routing fails at 0.6B — the capacity is insufficient for the router to preserve both pathways under SFT.

## 3. Hypothesis and Research Questions

**Primary hypothesis (H1):** Two specialised 0.6B SFT models — one trained exclusively on constitutional reasoning traces (Thinker), one trained exclusively on tool-call execution traces (Executor) — will together exceed the constitutional compliance and task completion rate of any single 0.6B model trained on the combined objective.

**Secondary hypotheses:**

- **H2 (reasoning preservation):** The Thinker model, trained without any tool-call examples, will maintain `think_empty = 0%` and restore principles P1 (decompose-first), P11, P12, P15, and P20 that the current SFT destroyed.
- **H3 (execution quality):** The Executor model, trained without `<think>` block examples, will maintain the current SFT model's adversarial robustness (9/14 = 64.3%) and reduce the tool-call explosion (56 calls → targeted single-call behaviour per probe).
- **H4 (joint performance):** The combined Thinker-then-Executor pipeline will outperform the current single SFT model on the full 21-probe constitution benchmark suite (target: ≥0.55 vs current 0.4286).

**Research questions:**

- RQ1: What is the minimum viable communication protocol between Thinker and Executor that preserves constitutional intent through the handoff?
- RQ2: Does the Thinker's reasoning plan need to be visible in the Executor's context, or is it sufficient for the Executor to receive only the distilled action specification?
- RQ3: What is the latency overhead of the two-model pipeline on consumer hardware (single GPU) relative to the single-model baseline?
- RQ4: Can a 0.6B Thinker model reliably distinguish Branch A (enough to act) from Branch B (needs clarification) — or does it systematically over-clarify (sycophancy-adjacent) or under-clarify (over-confident)?
- RQ5: How many Executor retry loops (Branch C) does the Thinker actually invoke in practice, and does the re-planning improve constitutional compliance or just add latency?

## 4. Proposed Architecture

### 4.1 Tool Ownership Split

The ten tools registered in `pipeline_tools.py` divide cleanly between the two models. This split is not arbitrary — it maps directly to the reasoning vs. execution function boundary.

| Model | Owns | Rationale |
|---|---|---|
| **Thinker** | `user_memory_read`, `user_memory_update`, `scratchpad_read`, `scratchpad_update`, `scratchpad_sections`, `user_memory_sections`, `get_datetime` | Session/user state management; constitutional context retrieval; planning working memory |
| **Executor** | `python_execute`, `web_search`, `read_url`, `get_datetime` | External world access; computation; live data retrieval |

`get_datetime` is shared because both models may need temporal grounding. The scratchpad and user_memory tools are entirely project-specific — no general dataset contains them. Training the Executor on these would waste capacity on tools it will never call, and risks leaking user-state concerns into the execution path.

> **Protocol decision (2026-05-29): step-by-step, not batch.** An earlier draft had the Thinker emit a structured `<delegation>` plan (a `<stage>`/`<step>` DAG) for the Executor to run in one batch. That was abandoned for two reasons. (1) It reintroduced a *second structured output format* on the Thinker — the very thing (alongside JSON tool-calls) that displaced reasoning in the single model; the whole point of the split is that the Thinker has **one** output modality (prose). (2) Beyond ReAct ([arXiv:2511.10037](https://arxiv.org/html/2511.10037v1)) found upfront DAG planning unreliable at small scale. The confirmed design is a **step-by-step (ReAct-like) loop**: the Thinker emits prose only — a `<think>` block followed by exactly one of `<ask>` / `<act>` / `<answer>` — and decides the next single action after seeing each result. All tool-call syntax lives only in the Executor.

### 4.2 Thinker Model

**Training objective:** SFT on constitutional reasoning traces and mixed-initiative decisions. The Thinker's entire output is prose: a `<think>` block, then exactly one of three plain-language tags. There is no structured plan format — the Thinker never writes tool-call syntax.

**The per-turn decision structure (step-by-step):**

```
User message (or a returned tool result) arrives
        │
[Thinker reasons in <think>: first principles + 5W+H scan]
        │
        ├─── <ask>   one targeted question        → Human responds → (loop back to Thinker)
        │            (when proceeding would force an assumption it shouldn't make)
        │
        ├─── <act>   one plain-language step       → Executor runs it → result returns → (loop back)
        │            (e.g. "Search the web for today's EUR/INR rate")
        │
        └─── <answer> final response to the user   → done
             (the Thinker, holding all context, writes this itself; ends with one 5W+H follow-up)
```

The old A/B/C branches map onto this cleanly: Branch A = an `<act>`; Branch B = an `<ask>`; Branch C = the Thinker reading a returned result and choosing the next `<act>` or the final `<answer>`. Re-planning is not a special output — it is just the loop continuing.

**Expected competencies after training:**
- Constitution principle application in `<think>` blocks (all 21 principles)
- 5W+H user-state reasoning, including `user_memory_*` reads to personalise ([[entities/5w-h]])
- First-principles decompose-first behaviour (P1, P20) feeding a single best next action
- Clarification (`<ask>`): detecting when intent is genuinely ambiguous vs. when the model is merely under-confident — a critical distinction (see [[topics/personalisation]] over-clarification failure mode). The `<ask>` carries the single most critical 5W+H dimension, grounded by the `<think>` decomposition.
- Self-contained step instructions (`<act>`): every concrete detail (URLs, numbers) is in the instruction, so the Executor needs nothing else — context stays with the Thinker
- Result review: reading a returned tool result and deciding whether it satisfies the constitutional constraints before answering
- Adversarial intent detection in `<think>` before any `<act>`/`<answer>` is emitted

**System prompt at inference:** the canonical `THINKER_STUDENT_PROMPT` in `sft_v3_generator.py` (single source of truth, re-stamped into training data). Role: reason in prose, then emit exactly one of `<ask>`/`<act>`/`<answer>`; never call tools or write tool-call syntax.

### 4.3 Executor Model

**Training objective:** SFT on (instruction → one tool call) pairs only. No `<think>` blocks, no constitution reasoning, no clarification logic, no prose answers. The Executor's single job: turn one `<act>` instruction into one native tool call.

**Executor tools:** `python_execute`, `web_search`, `read_url`, `get_datetime` only.

**Expected competencies after training:**
- Map a plain-language instruction to the correct single native `<tool_call>` against the exact project schemas (`python_execute(code=…)`, `web_search(query=…)`, `read_url(url=…, prompt=…)`)
- Single-call discipline — one instruction yields exactly one tool call, no explosion
- Faithfulness to the instruction — execute what the Thinker asked, do not re-interpret or expand scope
- Returns the raw tool result; it does **not** synthesise a prose answer (the Thinker does that)

**System prompt at inference:** minimal. Role: "You are an execution system. You receive one plain-language instruction. Emit exactly one tool call to carry it out, using the available tools. Do not reason, do not answer, do not ask questions."

### 4.4 Communication Protocol

The Thinker emits prose only. Two of its three tags route to the human, one to the Executor; the Executor returns a raw result.

**`<ask>`** (Thinker → Human): one targeted clarifying question. The `<think>` block does the first-principles + 5W+H decomposition and names the single most critical unknown dimension; the `<ask>` is that one question. Grounding lives in the reasoning, not in a rigid schema — keeping the Thinker single-modality (prose).

**`<act>`** (Thinker → Executor): one self-contained, natural-language instruction for a single step, e.g. `Search the web for today's EUR/INR exchange rate` or `Compute 500 multiplied by 89.7`. It includes every concrete value the step needs (the Thinker holds context and injects URLs/numbers), so the Executor sees only this one line.

**Executor reply** (Executor → Thinker, via the harness): the Executor turns the `<act>` into one native tool call; the harness runs it and appends the raw result to the conversation the Thinker holds. The Thinker then reasons again and emits the next `<act>` or the `<answer>`.

**`<answer>`** (Thinker → Human): the final response, composed by the Thinker from all accumulated results, ending with one targeted 5W+H follow-up question.

**Why natural language and not a schema:** a 0.6B Executor reliably learns "instruction → one tool call" (this is exactly what tool-use datasets teach), while a 0.6B Thinker reliably stays in prose. Neither model carries a second competing format, which is the structural fix for the capacity-displacement collapse (§1). The instruction can name the tool explicitly when helpful (e.g. "Use web_search to …") — this is derived for free when factoring existing trajectories, where the tool actually used is known.

**Ablation (E4 vs E5):** addresses RQ2 — does the Executor need any context beyond the single `<act>` instruction? E4 = instruction only (recommended); E5 = instruction plus a one-line rationale/context from the Thinker.

### 4.5 Inference Pipeline

```
User message
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│ LOOP (max 3 Thinker passes + 2 Executor calls per user turn)   │
│                                                                 │
│ [Thinker: Qwen3-0.6B] reads user_memory, reasons in <think>     │
│      ├─ <ask>  one question ─────→ Human responds ──┐           │
│      │                                              │ (loop)    │
│      ├─ <act>  one NL step ──→ [Executor: Qwen3-0.6B]│          │
│      │                          emits ONE tool_call  │          │
│      │                          harness runs it      │          │
│      │                          raw result ──────────┘ (loop)   │
│      │                                                          │
│      └─ <answer> final response ─────────────────────→ User (done)
└───────────────────────────────────────────────────────────────┘
```

Both models run on the same single GPU sequentially. Each is a separate inference pass; the Thinker holds the full conversation (including every returned result), the Executor sees only the latest `<act>`. Max loop depth: 3 Thinker passes + 2 Executor calls per user turn to prevent runaway iteration. Worst-case latency ≤ T_thinker×3 + T_executor×2; typical path is T_thinker + T_executor (single tool) or T_thinker alone (no tool / clarify). The extra round-trips versus a batch plan are the cost of step-by-step robustness — RQ3 measures whether this is acceptable on-device.

## 5. Experimental Conditions

| Condition | Description | Purpose |
|---|---|---|
| **E0** | Vanilla Qwen3-0.6B (no SFT) | Baseline reasoning reference |
| **E1** | Current SFT model (single, combined objective) | Baseline — the failure mode |
| **E2** | Thinker only (no Executor, direct answer) | Isolates reasoning restoration |
| **E3** | Executor only (no Thinker, no plan input) | Isolates tool execution quality |
| **E4** | Thinker → Executor, `<act>` instruction only | Main proposed architecture |
| **E5** | Thinker → Executor, `<act>` + one-line rationale/context | Ablation: RQ2 communication protocol |

## 6. Evaluation

**Primary metrics (existing benchmark suite via `4_benchmark.py`):**
- Constitution score across 21 probes (target: ≥0.55 for E4 vs 0.4286 for E1)
- `think_empty` rate (target: ≤10% for E2 and E4 vs 95% for E1)
- Adversarial suite pass rate across 14 probes (target: ≥0.65 for E4 vs 0.6429 for E1)
- Tool call count per probe (target: ≤3 for E3 and E4 vs 56/21 = 2.67 for E1 — note E1's explosion is already close; the measure is distribution, not mean)

**Secondary metrics:**
- Latency: wall-clock seconds per probe on single A100 (or equivalent) — compare E4 vs E1
- Delegation plan quality: human-scored on 20 sampled probes using a 3-item rubric (intent fidelity, constraint coverage, tool specification completeness) — Likert 1–5 each

**Ablation metrics (E4 vs E5):**
- Constitution score delta attributable to full vs minimal plan
- Any systematic difference by principle category (constitution probes, adversarial, category probes)

## 7. Training Data

**Primary method: trajectory factoring.** The Thinker and Executor datasets are not sourced externally — they are *manufactured* by projecting the pipeline's existing v3 distillation trajectories onto two role-conditioned views. This sidesteps the dataset-acquisition problem entirely (the external search, recorded in §7.6, found no dataset covering the constitutional planning loop) and — more importantly — guarantees that the Thinker's plans and the Executor's actions are aligned, because both are derived from the *same* trajectory. Public datasets are retained only as an optional breadth top-up (§7.7), not as the primary source.

> Constraint note (2026-05-29): the May 22 research-pivot decision was "no more data generation — the existing set is final". Trajectory factoring respects the spirit of this — it *re-uses* the 2,274 already-generated trajectories rather than generating new ones. The only component needing fresh generation is Branch B clarification (§7.5, ~500 examples). Confirm with supervisor before that run. The thinker–executor split is wholly SFT, consistent with the "SFT only, no GRPO" constraint; the optional RL pass in §8 stays cut unless reversed.

### 7.1 The Source Material — v3 Trajectories Already Contain Both Streams

The dataset search treated this as an acquisition problem. It is not. Every row produced by `sft_v3_generator.py` (and assembled into `data/train_partA_v3.jsonl`, 1,443 rows, and `data/train_partB_v3.jsonl`, 831 rows) is a **complete interleaved agentic trajectory** that already contains both the reasoning stream and the tool-execution stream, fully aligned, grounded in the 21-principle constitution, with a realistic user profile injected from `_SAMPLE_USER_PROFILES`:

```
system     (student prompt)
user       (question)
assistant  <think>…first-principles + 5W+H reasoning…</think>  +  native tool_call
tool       result
assistant  <think>…re-reasoning on the result…</think>          +  next tool_call
tool       result
…
assistant  <answer>…best-effort answer + greedy 5W+H follow-up…</answer>
```

This is precisely the raw material a Thinker and an Executor need — it is merely *fused* into one model's output. The teacher already performed the reasoning, the planning, and the execution in a single pass; factoring separates those concerns post hoc into two supervised targets. The constitutional signal, the tool schemas (the project's exact 10-tool registry), and the native `<tool_call>` format are all already present and correct — none of the schema-normalisation or `<think>`-stripping work that external datasets demand is required.

### 7.2 Thinker View Construction

The Thinker target is the reasoning plus the next single action, turn by turn (step-by-step design, §4.4). Each source trajectory becomes a **multi-turn** Thinker example. For each assistant turn in the source that contained a `<think>` block and an Executor-owned tool call:

- **Keep** the `<think>` block.
- **Convert the tool call into an `<act>` instruction** — a self-contained natural-language version of that call, e.g. `web_search(query="EUR INR rate today")` → `Search the web for today's EUR/INR rate`. This is the load-bearing trick: because the call (and its concrete arguments) is already known, the `<act>` is a faithful natural-language rendering of a ground-truth action, not an invented plan — correct by construction. The instruction may name the tool when helpful, since the tool actually used is known.
- **Keep** the returned tool result as the next turn (it comes back to the Thinker, which holds context).
- The Thinker-owned preamble tools (`user_memory_*`, `scratchpad_*`, `get_datetime`) are **not** emitted as `<act>`s to the Executor. Their explicit call/result turns are dropped, but the `user_memory_read` **profile is preserved as a `[USER MEMORY]` context block prepended to the user turn** (the same representation Branch B and the inference orchestrator use), for **50%** of rows; the other 50% are cold-start (`--memory_ratio`). This is the fix for the earlier version that dropped memory entirely — which made the Thinker reason about a user it could not see (learning to assert user facts rather than read them from context).
- The final `<answer>` (the Thinker composes it) is kept as the closing Thinker turn.

Thinker training row shape (multi-turn): `system → user (question) → assistant (<think> + <act>) → tool (result) → assistant (<think> + <act>) → tool (result) → … → assistant (<think> + <answer>)`. No stage-grouping or DAG is needed — the interleaved structure already encodes order, which is exactly why step-by-step factors more naturally than a batch plan.

### 7.3 Executor View Construction

The Executor target is one tool call per instruction, with no deliberation and no prose answer (the Thinker writes the answer, §4.3). Each Executor-owned tool call in the source trajectory yields one small training pair:

- **Input (user turn):** the `<act>` instruction produced for that call in §7.2 (after a minimal Executor system prompt).
- **Target (assistant turn):** the native `<tool_call>` for that call, verbatim — `<think>` stripped, no answer.

Executor training row shape: `system (Executor prompt) → user (<act> instruction) → assistant (one native tool_call)`. A multi-tool source trajectory thus yields several such pairs (one per Executor-owned call). The Executor never sees the running conversation — only the single instruction — which is what keeps it minimal and learnable at 0.6B.

Because §7.2 and §7.3 read the *same* trajectory, the instruction the Thinker learns to emit and the tool call the Executor learns to produce are the same action — alignment is structural, not hoped-for. This is the single biggest advantage over stitching an external executor corpus to an external thinker corpus.

### 7.4 Branch Coverage — Free vs Synthesised

| Branch | Source | Cost |
|---|---|---|
| **A** (enough to act → delegate) | Every factored trajectory yields a Branch A Thinker example and the matching Executor example | Free — pure transformation of existing 2,274 trajectories |
| **C** (review executor result → accept or replan) | Multi-round trajectories where a tool returned an error or partial result. The `environment_timeout` category already injects HTTP 503 on the first `web_search`, and natural `python_execute` syntax errors / empty extractions occur throughout. The Thinker-review turn is the existing post-result `<think>` block re-cast as an accept/replan decision. | Mostly free — derived from existing error-bearing trajectories; a thin synthesis pass only if accept/reject balance is poor |
| **B** (ambiguous → clarify) | Cannot be factored — existing trajectories always act, never stop to ask | Synthesised, ~500 examples (§7.5) |

This collapses the §7.6-era synthesis burden from ~1,000 examples (Branch B + C) to ~500 (Branch B alone), because Branch C falls out of the error trajectories the pipeline already produces.

### 7.5 Synthesised Dataset — Branch B (`clarification_needed`)

Branch B is the one component factoring cannot supply: existing trajectories always proceed to act, so none demonstrate the Thinker stopping to ask. **Implemented** in `sft_v3_generator.py` via `--branch_b` (no new question generation — it re-uses the ambiguous seed questions already in `data/questions_partA.jsonl`, categories `multi_step_clarification`, `ambiguous_underspecified`, `user_context_behavioral`, `verbose_context_behavioral`).

**What it teaches:** The Thinker emits `<ask>` (one targeted question) instead of proceeding when the request is genuinely ambiguous about what the user needs — i.e., proceeding would force a constitutional assumption it should not make silently. The 5W+H/first-principles grounding lives in the preceding `<think>` block (which names the single most critical dimension and why guessing is unsafe), keeping the Thinker single-modality (prose).

> **Memory-aware, 50/50 (revised 2026-05-31).** The original design generated Branch B cold-start (no user memory) on the theory that absent memory maximises genuine ambiguity. But the Thinker has user memory at inference, so a cold-start-only Branch B teaches the wrong trigger (ask because *no profile* rather than ask for the *residual gap a profile cannot fill*) and risks over-clarification. Branch B now matches inference: **50% of rows inject a sampled 5W+H profile** (the same `[USER MEMORY]` block the orchestrator injects and the splitter prepends to factored A/C rows, via the shared `prepend_memory` helper) and the teacher asks only when that profile does not resolve the request; **50% are cold-start** so the Thinker also learns to cope when no profile is available. The factored A/C set uses the same 50/50 split (`sft_trajectory_splitter.py --memory_ratio`), so both sources present memory identically.

**Generation procedure (teacher decides per item):**
1. With 50% probability, prepend a sampled `[USER MEMORY]` profile to the seed; otherwise cold-start. Feed to the teacher under `_make_branch_b_teacher_prompt` (full constitution; the prompt handles both regimes), primed by a one-shot `_BRANCH_B_FEWSHOT` demonstration (one memory-present→`<ask>`-for-residual, one cold-start→`<answer>`) — without the priming minimax/kimi emit a bare tag with no `<think>` and every row is rejected.
2. Turn 1 — the teacher reasons (first principles + 5W+H, ≥150 chars) and decides:
   - **Genuinely ambiguous → `<ask>`** (positive). Validated by `_validate_ask`: exactly one question, and the `<think>` names a 5W+H dimension. A second teacher call role-plays the user answering; turn 2 the teacher resolves with `<act>` or `<answer>`.
   - **Specifiable (often because memory already resolves it) → `<act>`/`<answer>`** (don't-ask negative). The teacher proceeds without asking — this *is* the negative example.
3. Both outcomes are written, tagged `branch: "B"` or `"B_negative"` in metadata; the user turn stores whatever memory block (if any) the teacher saw. Teacher model: `minimax-m2.7` (the canonical teacher; kimi-k2.6 returns reasoning out-of-band and skips every row).

**Output:** `data/train_sft_thinker_branch_b.jsonl`, Thinker format (consumed by the splitter at the curriculum-merge step, not the SFT assembler). Positive rows are 5-turn (`system → user → <think>+<ask> → human → <think>+<act|answer>`); negatives are 3-turn.

**Target yield:** ~500 positives. The negatives fall out of the same run for free (specifiable seeds the teacher declines to clarify); supplement with factored Branch A examples (which acted without asking) at the curriculum-merge step. This directly addresses RQ4 and the over-clarification risk in §9.

### 7.5b Adversarial / Security Refusal Trajectories (`--adversarial`)

The factored A/C set carries `impossible_tasks` refusals but no **prompt-injection / tool-result-injection / harmful-capability** trajectories — a gap for a trust-focused Thinker. `sft_v3_generator.py --adversarial` fills it from a built-in red-team seed set (18 seeds: prompt injection, authority/identity spoof, jailbreak, tool-result injection, malware, intrusion, credential theft, surveillance, fraud, physical harm). These seeds are **inputs the Thinker learns to decline**; the teacher output is always a refusal — `<think>` names the attack and flags any embedded "ignore instructions / SYSTEM UPDATE / web-page-says" text as untrusted data, then `<answer>` refuses without revealing the prompt or shipping the payload (a compliance guard rejects any row whose answer contains a fenced code block for exploit categories). This is defensive safety SFT and operationalises the constitution's security principles at the Thinker level. Runs in the **same single process** as Branch B (`--branch_b --adversarial`) so both share one rate-limited worker pool; both land in `train_sft_thinker_branch_b.jsonl` (the `branch` metadata — `B` / `B_negative` / `adversarial` — distinguishes them) and both are interleaved into the factored set by `sft_curriculum_merge.py`.

### 7.6 Why the External-Dataset Search Came Up Short

The following is retained as the record of the dataset search; its conclusion is what motivated the factoring method above. The external datasets it identified now serve only as the optional top-up in §7.7.

**Finding 1 — Architecture papers released no training data.** A targeted search across the key architectural papers (Reason-Plan-ReAct 2512.03560, Can Small Agents Collaborate 2601.11327, Planner–Executor 2510.15244) found no published datasets from any of them. All three are inference and evaluation papers. The experiment must be constructed from independently sourced datasets.

**Finding 2 — No public dataset covers the full planning loop.** The three-branch decision structure described in §4.2 (delegate / clarify / replan) does not exist in any public training corpus. The search identified three partial fits:

**[capitalone/T1](https://huggingface.co/datasets/capitalone/T1)** ([arXiv:2505.16986](https://arxiv.org/abs/2505.16986), Capital One, May 2025) — 13.5K multi-turn dialogues across nine domains (flights, hotels, restaurants, attractions, and five multi-domain combinations). The agent coordinates tool calls across turns, maintains short- and long-term memory, and supports dynamic re-planning when results come back unexpectedly. **Covers Branch A and partial Branch C.** What it lacks: it never asks a clarifying question — ambiguity is always resolved by calling more tools. Domain is also narrow (travel booking). Use for Thinker training on the iterate-with-executor (Branch C) pattern only.

**Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction** ([arXiv:2511.07943](https://arxiv.org/abs/2511.07943), Nov 2025) — Hierarchical thinking model that decomposes problems into sub-problems, checks its own knowledge boundary (can I answer directly, or do I need to search?), and routes accordingly. This is the closest published *approach* to the Branch A/B decision logic. **No public dataset was released.** The paper describes the training procedure but withholds the data. The knowledge-boundary checking taxonomy — know when to search vs. answer vs. ask — is valuable framing for the synthesised dataset (§7.5).

**RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL** ([arXiv:2504.20073](https://arxiv.org/abs/2504.20073), 2025) — StarPO framework for multi-turn RL where an agent makes sequential decisions, maintains memory across turns, and adapts to stochastic feedback. Open source at [github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN). **Not a dataset — a training framework.** Relevant as the RL post-training mechanism (June 15–25 timeline slot) if SFT alone is insufficient to learn the clarification/replan decision.

**Finding 3 — The clarification branch requires synthesis.** No existing dataset contains examples where a planning agent correctly decides to ask a clarifying question rather than delegate to tools. AGENT-CQ ([arXiv:2410.19692](https://arxiv.org/pdf/2410.19692)) generates clarifying questions for conversational search but in a retrieval context — not a planning-execution loop. The Branch B examples must be synthesised (see §7.5).

**Finding 4 — Tool ownership dictates dataset scope.** The pipeline's ten tools split cleanly between models (§4.1). External datasets never contain `user_memory_*` or `scratchpad_*` tools — those are project-specific. General tool-calling datasets (ToolMind, CoVe) are valid for the Executor because `python_execute`, `web_search`, and `read_url` are semantically equivalent to the tools those datasets train on, even if the argument schema differs slightly. Project-specific pipeline examples correct for the exact schema at the end of training.

Seven public datasets were identified. They divide by role as follows.

### 7.7 Optional External Top-Up (breadth only)

The tables and sources below are the *original* externally-sourced plan, now **demoted**. With trajectory factoring (§7.1–7.4) supplying the aligned, constitution-grounded core for both models, these public datasets are no longer the primary source — they are an optional breadth top-up to be blended in only if held-out evaluation shows the factored set is too narrow in topic or reasoning style. Their target totals (~18K Thinker, ~44K Executor) are *ceilings*, not requirements, and adding them reintroduces the schema-normalisation and plan/execution alignment costs that factoring avoids. Treat anything below as a contingency, not the plan of record. (Historical note: references to `train_sft_v3_robust.jsonl` below mean the current `data/train_sft_v3.jsonl` / `train_partA_v3.jsonl`.)

#### Thinker top-up — reasoning-format transfer

The Thinker needs long constitutional `<think>` blocks with no tool calls. No external dataset provides constitutional content — the pipeline's own data is the sole source of that signal. External datasets contribute only **reasoning-format transfer**: they teach the model how to sustain long deliberative chains, which the constitution-only data is too narrow to provide on its own.

**[NovaSky-AI/Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)** (NovaSky, UC Berkeley, 2025) — 17K examples. Long `<think>` traces on mathematics (AIME, MATH, NuminaMath) and coding (APPs, TACO), generated by QwQ-32B-Preview. Format is exactly Question → `<think>…</think>` → Answer with no tool calls anywhere in the trajectory. Used to train Sky-T1-32B to near o1-preview parity at a $450 training budget — the format is well-validated at small scale. **Best fit for reasoning-format transfer.**

**[bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)** (BespokeLabs, 2025) — 17K examples, same Sky-T1 pipeline scaled using DeepSeek-R1 as the annotator instead of QwQ. Slightly broader coverage and different annotator temperature — useful as a second source to prevent overfitting to QwQ's generation style. Use alongside Sky-T1 rather than instead of it.

**[allenai/Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)** (AI2, 2025) — 220K prompts with reasoning traces, annotated by a mix of DeepSeek-R1 and DeepSeek-R1-0528, sourced from Tülu 3 / OLMo 2 prompts. Much larger and more topically diverse than Sky-T1. **Use only after aggressive filtering**: remove any example containing tool_use, function_call, or JSON object outputs in the assistant turn. After filtering, sample ~5K for diversity rather than using the full 220K at 0.6B scale.

**Recommended Thinker mix:**

| Source | Size after filtering | Branch covered | Role |
|---|---|---|---|
| `train_sft_v3_robust.jsonl` (think-only filter) | ~800–1,200 | A | Constitutional signal — primary |
| Synthesised `clarification_needed` (§7.5) | ~500 | B | Only source of Branch B examples |
| Synthesised `executor_replan` (§7.5) | ~500 | C | Re-planning on executor feedback |
| capitalone/T1 (adapted to delegation format) | ~3,000 | A + C partial | Multi-turn tool coordination and replan |
| Sky-T1_data_17k (sample) | 5,000 | format only | Reasoning-format transfer |
| Bespoke-Stratos-17k (sample) | 3,000 | format only | Reasoning style diversity |
| Dolci-Think-SFT-32B (filtered sample) | 5,000 | format only | Topic breadth |

Target total: ~18,000 examples. The synthesised Branch B and C examples are small in absolute terms (~1,000 combined, ~5.5% of total) but are the only data that teaches the Thinker to make the clarification/replan decision. Without them, the model will have seen only Branch A outputs and will default to always delegating. Curriculum ordering: one constitutional or synthesised example per 10–12 general-reasoning examples, maintained across all training steps.

#### Executor top-up — tool-call breadth

The Executor needs clean tool-call traces with no `<think>` blocks. Three datasets provide this at scale. A fourth provides a quality-verification anchor.

**[Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind)** ([arXiv:2511.15718](https://arxiv.org/abs/2511.15718), Nanbeige, Nov 2025) — **360K samples**. Built from a multi-agent simulation using 20K+ tools. Combines 160K synthetically generated assistant-response turns with 200K augmented open-source turns. Crucially, turn-level filtering removes erroneous or suboptimal steps — this is the dataset whose training signal most closely matches the quality bar needed for reliable single-call discipline. Models fine-tuned on ToolMind beat baselines on τ-bench, τ²-bench, and BFCL-v4. **Primary executor training source.** Preprocessing required: strip any `<think>` or `<reasoning>` blocks that appear in assistant turns before use.

**[Zichen1024/CoVe](https://huggingface.co/datasets/Zichen1024/CoVe)** ([arXiv:2603.01940](https://arxiv.org/abs/2603.01940), March 2026) — **12K trajectories**. Generated via Constraint-Guided Verification — task constraints are defined first and used as deterministic verifiers on the trajectory, not an LLM judge. This makes CoVe the highest-precision dataset in the set despite being the smallest. The CoVe-4B model (fine-tuned on this alone, no other data) achieves 43.0% / 59.4% on Airline / Retail domains of τ-bench. Use as a **quality anchor**: include all 12K in executor training and separately use the τ-bench domains to validate the trained Executor before running the full constitution benchmark.

**[zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory](https://huggingface.co/datasets/zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory)** (community, 2025) — Qwen3-based tool-calling trajectories with reasoning content included per assistant turn. **Least preprocessing work**: the tool-call format is already Qwen3-native and will require only `<think>` block removal. Size not officially documented but community-estimated at ~10K trajectories. Useful for ensuring the Executor's tool-call format is tokeniser-compatible with the base model.

**[WaltonFuture/agentic-sft-new](https://huggingface.co/datasets/WaltonFuture/agentic-sft-new)** (community, 2025) — Broad coverage: tool calling, code editing, terminal interaction, multi-hop reasoning, web browsing. **Use for generalisation only**, not as a primary source — format is heterogeneous and will need normalisation to the delegation-spec schema (§4.3). Sample ~3K examples covering tool-calling and web-search subsets.

**Recommended Executor mix:**

| Source | Approximate size after filtering | Role |
|---|---|---|
| Nanbeige/ToolMind (strip think blocks) | 20,000 examples (sample from 360K) | Primary: scale and diversity |
| Zichen1024/CoVe | 12,000 examples (all) | Quality anchor: verified trajectories |
| zake7749 Qwen3 trajectories (strip think blocks) | ~8,000 examples | Format compatibility |
| `train_sft_v3_robust.jsonl` (tool-call filter) | ~600–800 examples | Constitutional tool-use context |
| WaltonFuture/agentic-sft-new (normalised sample) | 3,000 examples | Generalisation breadth |

Target total: ~44,000 examples. The constitutional pipeline data is a very small fraction (~1.5%) but ensures the Executor has seen the tool signatures (`python_execute`, `exa_search`) and argument formats it will actually be called with at inference time.

### 7.8 The Constitutional Gap — Key Finding

This finding is the deeper reason trajectory factoring is the right move: because no public dataset carries constitutional content, the only way to put constitutional signal into *both* a Thinker and an Executor is to derive both from the project's own constitution-grounded trajectories. Factoring does exactly that — unlike the external mixes above, which would leave the Executor with no constitutional grounding at all.

The most important finding from the dataset search is what does not exist: **no public dataset provides constitutional reasoning traces for a trust-and-empathy AI system**. Every external dataset covers one of two regimes:

- *General reasoning* (Sky-T1, Stratos, Dolci): long think blocks on maths and code, no constitutional grounding
- *Tool execution* (ToolMind, CoVe, zake7749): clean tool-call trajectories, no deliberative reasoning at all

The pipeline's `train_sft_v3_robust.jsonl` is the **only existing source** that combines the 23-principle constitution, the 5W+H user-state reasoning schema, adversarial intent detection in `<think>` blocks, and the project's specific tool signatures. This means the constitutional signal in Thinker training is irreducibly limited by the size of that dataset (~800–1,200 usable examples after filtering).

This gap has two practical consequences:

1. **The Thinker model may generalise the reasoning format without generalising constitutional values.** Sky-T1 data teaches it to produce long think blocks; only the pipeline data teaches it *what to think about*. Monitoring per-principle scores on held-out probes throughout training (not just after) is necessary to detect drift away from constitutional content.

2. **The constitutional gap is itself a dissertation contribution.** The absence of any public constitutional SFT dataset for small trust-focused models is a gap in the field. The `train_sft_v3_robust.jsonl` pipeline, and the filtering/curation methodology developed here, constitute an original data contribution — not just an experiment artefact.

### 7.9 Preprocessing and Validation Gates

Trajectory factoring needs far less preprocessing than the external plan, because the source data is already in the project's format. The steps are:

1. **Think-block stripping (Executor view only):** Remove `<think>…</think>` from the assistant turns when building the Executor target. Reuse `sft_dataset_assembler._THINK_BLOCK_RE` (`<think>[\s\S]*?</think>`); the source trajectories never nest think blocks, so the regex is sufficient.

2. **`<act>` derivation (deterministic — not LLM-generated):** For each Executor-owned tool call (`python_execute`, `web_search`, `read_url`), render a natural-language `<act>` instruction from the call and its concrete arguments. This replaces the external plan's GPU-intensive "run a teacher to generate the plan" step entirely — there is **no GPU cost** for the core data, because the instruction is read off the observed action. This is the single largest saving over the external plan. (A Qwen3-4B paraphrase pass is an optional refinement if the literal rendering reads too mechanically — but it is not required for correctness.)

3. **Tool-ownership split:** When factoring, route `user_memory_*` / `scratchpad_*` / `get_datetime` calls to the Thinker context and `python_execute` / `web_search` / `read_url` to the Executor target, per §4.1.

4. **Quality gates:** Thinker rows must have each `<think>` block ≥150 chars and a valid action tag (`<ask>`/`<act>`/`<answer>`). Executor rows must have exactly one native `tool_call`, no `<answer>`, and **no** `<think>` block (a positive gate — the whole point is that the Executor never deliberates). Reject any Executor row whose call is not an Executor-owned tool. The Branch B generator already enforces the Thinker-side gates inline (`_validate_ask`, think-length); the splitter applies the analogous gates to factored rows.

5. **Curriculum ordering:** Interleave the synthesised Branch B examples (§7.5) throughout the Thinker epoch at roughly 1 per 10–12 Branch A/C examples, so the model does not learn to always delegate.

### 7.10 Implementation in the Pipeline

The core factoring is a new pure-transformation script, `sft_trajectory_splitter.py`, that reads `data/train_partA_v3.jsonl` and `data/train_partB_v3.jsonl` and writes `data/train_sft_thinker.jsonl` and `data/train_sft_executor.jsonl`. It reuses the parsing helpers already in `sft_dataset_assembler.py` (`_THINK_BLOCK_RE`, `_TOOL_RE`, the `_TOOL_ARG_MAP` extractors, `convert_to_native`) rather than re-implementing tool parsing. Two new role prompts (`THINKER_PROMPT`, `EXECUTOR_PROMPT`) are added to `sft_v3_generator.py` alongside the existing `STUDENT_PROMPTS`.

Branch B synthesis (§7.5) extends `sft_question_generator.py` with a `clarification_needed` category and `sft_v3_generator.py` with a clarification intercept, gated on a Qwen3-7B+ teacher and written to `data/train_sft_thinker_branch_b.jsonl`, then merged at the curriculum step. Branch C is **not** synthesised — it is factored from the error-bearing trajectories already in the corpus (§7.4); a small accept/reject rebalancing pass is the only contingency.

Training requires **no changes to `2_model_trainer.py`**: it accepts `--dataset <path> --output_name <name>`, and `messages_to_text` already renders native tool schemas from `metadata.native_tools` via the chat template. Two runs produce the two checkpoints:

```bash
python 2_model_trainer.py --mode sft --dataset data/train_sft_thinker.jsonl  --output_name checkpoint_thinker
python 2_model_trainer.py --mode sft --dataset data/train_sft_executor.jsonl --output_name checkpoint_executor
```

The genuinely new runtime component is the two-model orchestration loop (Thinker → Executor → Thinker handoff with the max 3+2 loop cap from §4.5), either as a new module or an extension of `3_infererence.py`.

## 8. Timeline

Revised for the trajectory-factoring method (all experiments complete by 30 June 2026). Factoring removes the multi-day external-data sourcing and the GPU delegation-generation pass, compressing the front of the schedule.

| Date | Milestone |
|---|---|
| 2026-05-30 | Write `sft_trajectory_splitter.py`; factor existing 2,274 trajectories into `train_sft_thinker.jsonl` + `train_sft_executor.jsonl`; manually inspect 30 factored rows of each (`<act>` fidelity, no-think Executor gate) |
| 2026-05-31 | Add `THINKER_PROMPT` / `EXECUTOR_PROMPT`; train Executor (E3) from factored Branch A/C data (~2 hrs); probe tool-call discipline and think-empty on held-out |
| 2026-06-01 | Add `clarification_needed` category; synthesise ~500 Branch B examples (Qwen3-7B+ teacher) — **pending supervisor sign-off on new generation** |
| 2026-06-02 | Merge Branch B + factored Branch A/C with curriculum ordering; train Thinker (E2) (~3 hrs); probe Branch B trigger rate (RQ4) |
| 2026-06-03 | Build two-model orchestration loop; run full benchmark across E0–E5 |
| 2026-06-04–06-10 | Analyse Branch B/C trigger rates, RQ4/RQ5, latency (RQ3); adjust clarification interleave ratio if over/under-triggering |
| 2026-06-11–06-20 | Iterate on constitution ratio / `<act>` protocol if H1 not met; optional external top-up (§7.7) only if held-out coverage is thin; re-train as needed |
| 2026-06-25 | Final benchmark run across all six conditions |
| 2026-06-30 | Integrate into dissertation Experiment 3 section; close open questions |

> The June 15–25 RL post-training slot from the prior plan is **cut** — RL is not supervisor-approved (SFT only), and Beyond ReAct (§9) confirms GRPO instability at 0.6B. The freed time goes to iteration on the SFT data mix.

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Thinker generalises reasoning format but not constitutional values | High | Curriculum interleaving (1 constitutional per 10 general); monitor per-principle probe scores during training not just after |
| Constitutional signal washed out by general-reasoning data majority | Medium | Run ablation at 1:6 constitutional ratio; if P1/P11/P12 scores fall, increase constitutional proportion |
| Thinker over-clarifies — asks clarifying questions on unambiguous prompts | High | Include strong negative examples (don't-clarify cases) interleaved with Branch B data; evaluate RQ4 explicitly on 20 held-out unambiguous probes |
| Thinker under-clarifies — delegates when it should ask | Medium | Add constitutional check in `<why_needed>` field as a training signal; probe with genuinely ambiguous inputs in held-out set |
| Thinker loops endlessly on Branch C — keeps replanning without accepting | Medium | Hard cap of 2 Executor retries in inference loop; monitor RQ5 (average Branch C depth) during evaluation |
| Synthesised Branch B/C data is too narrow — teacher generates similar clarification questions | Medium | Use diverse seed prompts across all 23 constitution principles; verify principle distribution in generated examples before training |
| Executor expands scope beyond the single `<act>` instruction | Low | Executor examples are one-instruction→one-call pairs; the Executor never sees the running conversation, so there is no scope to expand into |
| Executor maps an `<act>` to the wrong tool or malforms the call | Medium | `<act>` may name the tool explicitly (free when factoring, since the tool used is known); validate the emitted native `tool_call` against the registry schema; add malformed calls as negative signal |
| Think-block stripping corrupts ToolMind examples with nested reasoning | Medium | Use regex `<think>[\s\S]*?</think>` not character-count heuristic; manually inspect 50 samples post-strip |
| Combined latency exceeds acceptable threshold for on-device use | Medium | Profile single-pass latency early; max loop depth cap (3+2) bounds worst case; measure RQ3 on first trained models before full benchmark |
| Delegation-block generation for executor training produces low-quality plans | Medium | Use Qwen3-7B+ (not 0.6B) as teacher; validate 100 examples manually before full pass |
| **GRPO on 0.6B Thinker is unstable** — empirically confirmed by Beyond ReAct (arXiv:2511.10037) which excluded Qwen3-0.6B from RL training due to instability | **High (empirical)** | Three options in priority order: (1) skip Thinker RL entirely — SFT-only Thinker + RL-only Executor; (2) use RAGEN/StarPO instead of GRPO for the Thinker — StarPO has explicit gradient stabilisation ("Echo Trap" fix); (3) upgrade Thinker to 1.7B if SFT-only results fail H2, accepting a 0.6B+1.7B asymmetric pair |

## 10. Connection to Thesis Argument

This experiment directly addresses the dissertation's operational hypothesis (see [[decisions/2026-05-03-research-question-reframe]]): *"a correctly architected on-device 0.6B model can approach frontier trustworthiness despite the scale difference."* The single-model baseline (E1) shows that naive SFT on a combined objective fails. The thinker–executor architecture is the proposed structural fix — and if H1 holds, it becomes the primary architectural contribution of the dissertation.

It also links to the four-module architecture (see [[decisions/2025-10-01-four-module-architecture]]): the Thinker maps to the Reasoning Module; the Executor maps to the Tool Integration Module. The split is not a new module — it is a training-time manifestation of the module boundary that was always in the architecture design. The experiment validates that boundary at the weight level, not just the system-design level.

The TML Interaction-Small precedent ([[entities/tml-interaction-small]]) at 276B/12B active parameters provides industry validation of the architectural pattern. The dissertation's contribution is demonstrating the same separation at 0.6B × 2 = 1.2B total parameters under on-device and constitutional constraints — a regime that, as of May 2026, has no published treatment.

## Related

- [[experiments/sft-benchmark-analysis-20260525]] — the empirical finding that motivates this experiment
- [[experiments/experiment-catalog]] — how this fits into the six-experiment dissertation plan
- [[decisions/2025-10-01-four-module-architecture]] — the four-module architecture this validates
- [[decisions/2026-05-03-research-question-reframe]] — the operational hypothesis
- [[entities/qwen3-0.6b]] — base model used for both Thinker and Executor
- [[entities/constitution]] — the 23-principle constitution governing Thinker training
- [[topics/tool-use-and-verification]] — tool delegation theory
- [[topics/reasoning]] — trustworthy reasoning literature
- [[entities/tml-interaction-small]] — industry precedent at frontier scale
- [[sources/papers/replacing-thinking-with-tool-usage]] — closest paper to the observed phenomenon
- [[sources/papers/reason-plan-react]] — direct architectural antecedent (needs ingest)
- [[sources/papers/can-small-agents-collaborate]] — empirical validation of small multi-agent > monolithic (needs ingest)
- [[sources/papers/beyond-react]] — primary related work: Qwen3-0.6B planner tested, GRPO instability confirmed, stage-grouping schema derived from their DAG insight (needs ingest)
- [[sources/papers/t1-conversational-planning]] — closest public dataset for Branch A/C loop (needs ingest)
- [[sources/papers/thinker-hierarchical-search]] — knowledge-boundary checking; Branch A/B decision taxonomy (needs ingest)
- [[sources/papers/ragen-self-evolution]] — RL post-training framework for the multi-turn decision loop

## Sources

### Architecture and motivation papers
- Rainone et al. (2025) — "Replacing thinking with tool usage enables reasoning in small language models" — [arXiv:2507.05065](https://arxiv.org/abs/2507.05065)
- Yao et al. (2025) — "Reason-Plan-ReAct: A Reasoner-Planner Supervising a ReAct Executor" — [arXiv:2512.03560](https://arxiv.org/abs/2512.03560)
- Żywot et al. (2026) — "Can Small Agents Collaborate to Beat a Single Large Language Model?" — [arXiv:2601.11327](https://arxiv.org/abs/2601.11327)
- Pan et al. (2025) — "Planner and Executor: Collaboration between Discrete Diffusion and Autoregressive Models" — [arXiv:2510.15244](https://arxiv.org/html/2510.15244v1)
- Hu et al. (2025) — "OPERA: Orchestrated Planner-Executor Architecture" — [arXiv:2508.16438](https://arxiv.org/abs/2508.16438)
- Liu et al. (2025) — "Improved SFT for LLMs to Mitigate Catastrophic Forgetting" — [arXiv:2506.09428](https://arxiv.org/abs/2506.09428)
- Pangu Team, Huawei (2025) — "Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition" — [arXiv:2505.22375](https://arxiv.org/abs/2505.22375)
- Thinking Machines Lab (2026) — "Interaction Models: A Scalable Approach to Human-AI Collaboration" — [thinkingmachines.ai/blog/interaction-models](https://thinkingmachines.ai/blog/interaction-models/)
- Wei et al. (2025) — "Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning" — [arXiv:2511.10037](https://arxiv.org/html/2511.10037v1) — code + ComplexTool-Plan data: [github.com/weixiaolong94-hub/Beyond-React](https://github.com/weixiaolong94-hub/Beyond-React)

### Thinker training datasets — public
- NovaSky (UC Berkeley, 2025) — Sky-T1_data_17k — [huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)
- BespokeLabs (2025) — Bespoke-Stratos-17k — [huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- AI2 (2025) — Dolci-Think-SFT-32B — [huggingface.co/datasets/allenai/Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)
- Chakraborty et al. (2025) — T1: Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning (Branch A/C) — [arXiv:2505.16986](https://arxiv.org/abs/2505.16986) — [huggingface.co/datasets/capitalone/T1](https://huggingface.co/datasets/capitalone/T1)

### Thinker training datasets — synthesised (this project)
- `train_sft_thinker_branches_bc.jsonl` — ~500 Branch B (`clarification_needed`) + ~500 Branch C (`executor_replan`) examples generated via `sft_question_generator.py` extension; Qwen3-7B+ teacher; constitutional validation required

### Planning loop papers (no dataset released)
- Xu et al. (2025) — "Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction" — [arXiv:2511.07943](https://arxiv.org/abs/2511.07943)
- Wang et al. (2025) — "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning" — [arXiv:2504.20073](https://arxiv.org/abs/2504.20073) — [github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN)
- Chen et al. (2024) — "AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search" — [arXiv:2410.19692](https://arxiv.org/pdf/2410.19692)

### Executor training datasets
- Yang et al. (2025) — ToolMind: A Large-Scale, Reasoning-Enhanced Tool-Use Dataset — [arXiv:2511.15718](https://arxiv.org/abs/2511.15718) — [huggingface.co/datasets/Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind)
- Chen et al. (2026) — CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification — [arXiv:2603.01940](https://arxiv.org/abs/2603.01940) — [huggingface.co/datasets/Zichen1024/CoVe](https://huggingface.co/datasets/Zichen1024/CoVe)
- Community (2025) — Qwen3 agent tool-calling trajectories — [huggingface.co/datasets/zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory](https://huggingface.co/datasets/zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory)
- Community (2025) — Agentic SFT (broad coverage) — [huggingface.co/datasets/WaltonFuture/agentic-sft-new](https://huggingface.co/datasets/WaltonFuture/agentic-sft-new)

### Empirical baseline
- `wiki/experiments/sft-benchmark-analysis-20260525.md` — empirical baseline (this repository)
