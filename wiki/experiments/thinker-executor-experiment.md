---
title: "Experiment 3 — Thinker–Executor Dual-SFT Architecture"
type: experiment
tags: [sft, reasoning, tool-use, small-model, architecture, agents, multi-agent, on-device, trade-off, distillation, curriculum-learning]
sources:
  - pipeline/4_benchmark.py
  - wiki/experiments/sft-benchmark-analysis-20260525.md
updated: 2026-05-26
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

Two findings from this paper are directly load-bearing for the present experiment. First, they tested Qwen3-0.6B as a Planner and found SFT successful but **GRPO training unstable at 0.6B** — the RL phase had to be excluded for the smallest model. This elevates the June 15–25 RL pass from a medium risk to a confirmed empirical failure mode at this scale (see §9). Second, the DAG plan structure is adopted here in simplified form: rather than requiring a full graph with explicit dependency edges (which is beyond reliable generation at 0.6B), the `<stage>` grouping schema in §4.4 captures the same structural insight — steps within a stage are independent, stages are sequential — at a complexity the 0.6B Thinker can learn from SFT alone. The ComplexTool-Plan dataset is not used directly (it is calibrated for 8B and uses 4,535-API toolsets far larger than the project's 4-tool Executor set), but the generation scripts (`01_workflow.py`, `02_reverse.py`, `03_replan.py`) inform the synthesised Branch A data construction.

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

### 4.2 Thinker Model

**Training objective:** SFT on constitutional reasoning traces and mixed-initiative planning decisions. The Thinker produces one of three output types per turn — not just delegation plans.

**The three-branch decision structure:**

```
User message arrives
        │
[Thinker reasons in <think> block]
        │
        ├─── Branch A — Enough information to act
        │         → emit <delegation> → Executor acts
        │
        ├─── Branch B — Ambiguous or missing information
        │         → emit <clarification_request> → Human responds → Thinker re-evaluates
        │
        └─── Branch C — Executor returned result → Thinker reviews
                  ├─ Satisfactory → emit <final_answer>
                  └─ Insufficient/wrong → emit revised <delegation> → Executor retries
```

**Expected competencies after training:**
- Constitution principle application in `<think>` blocks (all 23 principles)
- 5W+H user-state read/write via `user_memory_*` tools ([[entities/5w-h]])
- Decompose-first behaviour (P1) in Branch A
- Clarification generation (Branch B): detecting when user intent is genuinely ambiguous vs. when the model is just under-confident — a critical distinction (see [[topics/personalisation]] over-clarification failure mode)
- Re-planning on executor feedback (Branch C): reading the Executor's result and deciding whether it satisfies the constitutional constraints before accepting it
- Adversarial intent detection in `<think>` blocks before any delegation is emitted

**System prompt at inference:** Full 23-principle constitution. Role: "You are a reasoning system. Read the user's request and the conversation history. Think through what is needed. Then either: (1) produce a structured delegation plan for the execution system, (2) ask the user a clarifying question, or (3) if you have just received execution results, decide if they are satisfactory or revise the plan. Never call execution tools yourself."

### 4.3 Executor Model

**Training objective:** SFT on tool-call execution traces only. No `<think>` blocks. No constitution reasoning. No clarification logic — the Executor never decides whether to ask the human; that decision belongs entirely to the Thinker.

**Executor tools:** `python_execute`, `web_search`, `read_url`, `get_datetime` only.

**Expected competencies after training:**
- Clean tool call emission against the exact project argument schemas (`python_execute(code='...')`, `web_search(query='...')`, `read_url(url='...', prompt='...')`)
- Single-call-per-need discipline (no tool-call explosion)
- Result integration and concise answer production
- Graceful degradation on missing tool / timeout (negative trajectory categories: `inventory_constraint`, `environment_timeout`)
- Faithfulness to the delegation spec — the Executor executes what the Thinker planned, and does not re-interpret or expand the scope

**System prompt at inference:** Minimal. Role: "You are an execution system. You receive a plan from a reasoning system. Execute it using the available tools: python_execute, web_search, read_url, get_datetime. Report results exactly. Do not reason about the plan. Do not ask clarifying questions."

### 4.4 Communication Protocol

The Thinker produces three possible output types. Two are passed to the Executor; one goes directly to the human.

**Branch A — Delegation spec** (Thinker → Executor, following [arXiv:2510.15244](https://arxiv.org/html/2510.15244v1)):

The schema uses `<stage>` grouping to encode execution order without requiring full DAG reasoning. Steps within a `<stage>` are independent and the Executor may run them in parallel; stages are sequential and each stage may use results from the previous one. This captures the structural insight from Beyond ReAct ([arXiv:2511.10037](https://arxiv.org/html/2511.10037v1)) — that independent steps should not be forced to wait — at a complexity level a 0.6B model can reliably generate from SFT alone.

```xml
<delegation>
  <intent>What the user actually needs (constitutional framing)</intent>
  <constraints>Constitution principles that apply (e.g., P4: math=code, P10: correct tool)</constraints>
  <stage>
    <!-- Steps here are independent — Executor may run them in parallel -->
    <step tool="web_search" reason="…">natural-language search query</step>
    <step tool="get_datetime" reason="…"/>
  </stage>
  <stage>
    <!-- Steps here depend on the previous stage's results -->
    <step tool="read_url" reason="…" url="placeholder — fill from stage 1 result">what to extract</step>
  </stage>
  <stage>
    <step tool="python_execute" reason="…">calculation using retrieved data</step>
  </stage>
  <fallback>What to say if all tools fail</fallback>
</delegation>
```

**Stage grouping rules the Thinker must learn:**
- `web_search` and `get_datetime` are always independent — put in the same stage
- `read_url` almost always depends on a `web_search` result (needs the URL) — new stage after search
- `python_execute` depends on whatever data it processes — new stage after retrieval
- A single-step delegation is a single `<stage>` with one `<step>` — valid and common

The Thinker only needs to learn a binary decision per step: *does this step depend on any prior result?* If no, same stage. If yes, new stage. This is tractable at 0.6B after SFT on staged examples; it does not require graph topology reasoning.

**Branch B — Clarification request** (Thinker → Human):

```xml
<clarification_request>
  <ambiguity>What specific information is missing or unclear</ambiguity>
  <question>The single most important clarifying question to ask</question>
  <why_needed>Which constitution principle or delegation step cannot proceed without this</why_needed>
</clarification_request>
```

**Branch C — Re-plan** (Thinker reviews Executor result, emits revised delegation or final answer):

```xml
<!-- If result is satisfactory -->
<final_answer>…</final_answer>

<!-- If result is insufficient -->
<delegation>
  <intent>Revised intent after reviewing executor output</intent>
  <revision_reason>Why the previous result was insufficient</revision_reason>
  <steps>…revised steps…</steps>
  <fallback>…</fallback>
</delegation>
```

**Ablation (E4 vs E5):** Full plan includes `<intent>` and `<constraints>`; minimal plan includes only `<steps>` and `<fallback>`. Addresses RQ2.

### 4.5 Inference Pipeline

```
User message
      │
      ▼
[Thinker: Qwen3-0.6B, SFT-Think]
  Reads user_memory, updates scratchpad
  Constitutional reasoning in <think> block
      │
      ├─ Branch B ──→ <clarification_request> ──→ Human responds ──→ (loop back to Thinker)
      │
      └─ Branch A ──→ <delegation>
                           │
                           ▼
              [Executor: Qwen3-0.6B, SFT-Execute]
                Calls python_execute / web_search / read_url
                Returns tool results
                           │
                           ▼
              [Thinker reviews result — Branch C]
                  ├─ Satisfactory ──→ <final_answer> ──→ User
                  └─ Insufficient ──→ revised <delegation> ──→ Executor (max 2 retries)
```

Both models run on the same single GPU sequentially. The Thinker's KV cache is not reused by the Executor (separate inference passes); the Executor's results are appended to the conversation context passed back to the Thinker for Branch C evaluation. Max loop depth: 3 Thinker passes + 2 Executor passes per user turn, to prevent runaway iteration. Total latency ≤ T_thinker×3 + T_executor×2 in the worst case; typical path is T_thinker + T_executor.

## 5. Experimental Conditions

| Condition | Description | Purpose |
|---|---|---|
| **E0** | Vanilla Qwen3-0.6B (no SFT) | Baseline reasoning reference |
| **E1** | Current SFT model (single, combined objective) | Baseline — the failure mode |
| **E2** | Thinker only (no Executor, direct answer) | Isolates reasoning restoration |
| **E3** | Executor only (no Thinker, no plan input) | Isolates tool execution quality |
| **E4** | Thinker → Executor, full delegation plan | Main proposed architecture |
| **E5** | Thinker → Executor, minimal delegation plan | Ablation: RQ2 communication protocol |

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

### 7.1 Dataset Search Findings

**Finding 1 — Architecture papers released no training data.** A targeted search across the key architectural papers (Reason-Plan-ReAct 2512.03560, Can Small Agents Collaborate 2601.11327, Planner–Executor 2510.15244) found no published datasets from any of them. All three are inference and evaluation papers. The experiment must be constructed from independently sourced datasets.

**Finding 2 — No public dataset covers the full planning loop.** The three-branch decision structure described in §4.2 (delegate / clarify / replan) does not exist in any public training corpus. The search identified three partial fits:

**[capitalone/T1](https://huggingface.co/datasets/capitalone/T1)** ([arXiv:2505.16986](https://arxiv.org/abs/2505.16986), Capital One, May 2025) — 13.5K multi-turn dialogues across nine domains (flights, hotels, restaurants, attractions, and five multi-domain combinations). The agent coordinates tool calls across turns, maintains short- and long-term memory, and supports dynamic re-planning when results come back unexpectedly. **Covers Branch A and partial Branch C.** What it lacks: it never asks a clarifying question — ambiguity is always resolved by calling more tools. Domain is also narrow (travel booking). Use for Thinker training on the iterate-with-executor (Branch C) pattern only.

**Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction** ([arXiv:2511.07943](https://arxiv.org/abs/2511.07943), Nov 2025) — Hierarchical thinking model that decomposes problems into sub-problems, checks its own knowledge boundary (can I answer directly, or do I need to search?), and routes accordingly. This is the closest published *approach* to the Branch A/B decision logic. **No public dataset was released.** The paper describes the training procedure but withholds the data. The knowledge-boundary checking taxonomy — know when to search vs. answer vs. ask — is valuable framing for the synthesised dataset (§7.6).

**RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL** ([arXiv:2504.20073](https://arxiv.org/abs/2504.20073), 2025) — StarPO framework for multi-turn RL where an agent makes sequential decisions, maintains memory across turns, and adapts to stochastic feedback. Open source at [github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN). **Not a dataset — a training framework.** Relevant as the RL post-training mechanism (June 15–25 timeline slot) if SFT alone is insufficient to learn the clarification/replan decision.

**Finding 3 — The clarification branch requires synthesis.** No existing dataset contains examples where a planning agent correctly decides to ask a clarifying question rather than delegate to tools. AGENT-CQ ([arXiv:2410.19692](https://arxiv.org/pdf/2410.19692)) generates clarifying questions for conversational search but in a retrieval context — not a planning-execution loop. The Branch B examples must be synthesised (see §7.6).

**Finding 4 — Tool ownership dictates dataset scope.** The pipeline's ten tools split cleanly between models (§4.1). External datasets never contain `user_memory_*` or `scratchpad_*` tools — those are project-specific. General tool-calling datasets (ToolMind, CoVe) are valid for the Executor because `python_execute`, `web_search`, and `read_url` are semantically equivalent to the tools those datasets train on, even if the argument schema differs slightly. Project-specific pipeline examples correct for the exact schema at the end of training.

Seven public datasets were identified. They divide by role as follows.

### 7.2 Thinker Training Data

The Thinker needs long constitutional `<think>` blocks with no tool calls. No external dataset provides constitutional content — the pipeline's own data is the sole source of that signal. External datasets contribute only **reasoning-format transfer**: they teach the model how to sustain long deliberative chains, which the constitution-only data is too narrow to provide on its own.

**[NovaSky-AI/Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)** (NovaSky, UC Berkeley, 2025) — 17K examples. Long `<think>` traces on mathematics (AIME, MATH, NuminaMath) and coding (APPs, TACO), generated by QwQ-32B-Preview. Format is exactly Question → `<think>…</think>` → Answer with no tool calls anywhere in the trajectory. Used to train Sky-T1-32B to near o1-preview parity at a $450 training budget — the format is well-validated at small scale. **Best fit for reasoning-format transfer.**

**[bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)** (BespokeLabs, 2025) — 17K examples, same Sky-T1 pipeline scaled using DeepSeek-R1 as the annotator instead of QwQ. Slightly broader coverage and different annotator temperature — useful as a second source to prevent overfitting to QwQ's generation style. Use alongside Sky-T1 rather than instead of it.

**[allenai/Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)** (AI2, 2025) — 220K prompts with reasoning traces, annotated by a mix of DeepSeek-R1 and DeepSeek-R1-0528, sourced from Tülu 3 / OLMo 2 prompts. Much larger and more topically diverse than Sky-T1. **Use only after aggressive filtering**: remove any example containing tool_use, function_call, or JSON object outputs in the assistant turn. After filtering, sample ~5K for diversity rather than using the full 220K at 0.6B scale.

**Recommended Thinker mix:**

| Source | Size after filtering | Branch covered | Role |
|---|---|---|---|
| `train_sft_v3_robust.jsonl` (think-only filter) | ~800–1,200 | A | Constitutional signal — primary |
| Synthesised `clarification_needed` (§7.6) | ~500 | B | Only source of Branch B examples |
| Synthesised `executor_replan` (§7.6) | ~500 | C | Re-planning on executor feedback |
| capitalone/T1 (adapted to delegation format) | ~3,000 | A + C partial | Multi-turn tool coordination and replan |
| Sky-T1_data_17k (sample) | 5,000 | format only | Reasoning-format transfer |
| Bespoke-Stratos-17k (sample) | 3,000 | format only | Reasoning style diversity |
| Dolci-Think-SFT-32B (filtered sample) | 5,000 | format only | Topic breadth |

Target total: ~18,000 examples. The synthesised Branch B and C examples are small in absolute terms (~1,000 combined, ~5.5% of total) but are the only data that teaches the Thinker to make the clarification/replan decision. Without them, the model will have seen only Branch A outputs and will default to always delegating. Curriculum ordering: one constitutional or synthesised example per 10–12 general-reasoning examples, maintained across all training steps.

### 7.3 Executor Training Data

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

### 7.4 The Constitutional Gap — Key Finding

The most important finding from the dataset search is what does not exist: **no public dataset provides constitutional reasoning traces for a trust-and-empathy AI system**. Every external dataset covers one of two regimes:

- *General reasoning* (Sky-T1, Stratos, Dolci): long think blocks on maths and code, no constitutional grounding
- *Tool execution* (ToolMind, CoVe, zake7749): clean tool-call trajectories, no deliberative reasoning at all

The pipeline's `train_sft_v3_robust.jsonl` is the **only existing source** that combines the 23-principle constitution, the 5W+H user-state reasoning schema, adversarial intent detection in `<think>` blocks, and the project's specific tool signatures. This means the constitutional signal in Thinker training is irreducibly limited by the size of that dataset (~800–1,200 usable examples after filtering).

This gap has two practical consequences:

1. **The Thinker model may generalise the reasoning format without generalising constitutional values.** Sky-T1 data teaches it to produce long think blocks; only the pipeline data teaches it *what to think about*. Monitoring per-principle scores on held-out probes throughout training (not just after) is necessary to detect drift away from constitutional content.

2. **The constitutional gap is itself a dissertation contribution.** The absence of any public constitutional SFT dataset for small trust-focused models is a gap in the field. The `train_sft_v3_robust.jsonl` pipeline, and the filtering/curation methodology developed here, constitute an original data contribution — not just an experiment artefact.

### 7.5 Preprocessing Pipeline

Four preprocessing steps are needed before any external data enters training:

1. **Think-block stripping (Executor data):** Remove `<think>…</think>` from all assistant turns in ToolMind, zake7749, and WaltonFuture datasets. Use regex `<think>[\s\S]*?</think>` rather than a heuristic character-count filter — ToolMind contains multi-paragraph think blocks that a naive truncation would corrupt.

2. **Delegation-spec prepending (Executor data):** Each executor training example must begin with a `<delegation>` block matching the schema in §4.3. For pipeline examples, run the vanilla Qwen3-0.6B (or a Qwen3-4B teacher) over the original user message to generate the delegation spec. For external datasets, generate synthetically from the conversation context.

3. **Constitutional seed filtering (Thinker data):** From `train_sft_v3_robust.jsonl`, retain only examples where `<think>` block length > 200 chars AND no `tool_use`/`tool_result` messages appear. Expected yield: 800–1,200 examples from the full dataset.

4. **Curriculum ordering:** Interleave constitutional seed examples throughout each epoch rather than batching them at the start or end. Effective ratio: 1 constitutional example per 10–12 general examples, maintained consistently across all training steps.

The delegation-block generation pass (step 2) is the only GPU-intensive preprocessing step — budget approximately 3 hours on a single A100 for ~44,000 executor examples at batch size 16.

### 7.6 Synthesised Dataset Plan — Branch B and Branch C

The two missing branches cannot be sourced from any public dataset and must be generated using an asymmetric distillation approach consistent with the existing `sft_v3_generator.py` intercept loop. This requires adding two new question categories to `sft_question_generator.py`.

#### Branch B — `clarification_needed` category

**What it teaches:** The Thinker must emit `<clarification_request>` rather than `<delegation>` when the user's intent is genuinely ambiguous at the level of constitutional principle — i.e., proceeding without clarification would force a constitutional assumption the model should not make silently.

**Generation procedure:**
1. Write ~60 seed prompts where a naive model would proceed to delegate but where a constitutionally careful model should stop and ask. Examples: a user asks "help me write the message" without saying what message or to whom; a user asks "calculate the cost" without specifying units or context; a user asks for a recommendation without revealing a relevant constraint (allergy, budget, legal jurisdiction).
2. Run teacher (Qwen3-7B or 14B with full constitution system prompt) over each seed. Intercept the output at the point where it would emit a `<delegation>` and instead prompt it to produce a `<clarification_request>` block with `<ambiguity>`, `<question>`, and `<why_needed>` fields.
3. Follow the clarification with a synthetic human response that resolves the ambiguity, then let the teacher complete the trajectory with a proper `<delegation>`.
4. Validate: the clarification question must reference a specific constitution principle in `<why_needed>`; the question must be singular (not a list); the human response must make the subsequent delegation unambiguous.

**Target yield:** 500 high-quality trajectories. Each trajectory is a 3-turn sequence: user prompt → `<clarification_request>` → synthetic human answer → `<delegation>`. This directly mirrors the Branch B → Branch A flow the Thinker will execute at inference time.

**Critical distinction to encode in training data:** The Thinker must learn to distinguish *genuinely ambiguous* from *under-confident*. A probe where the user says "what is 2+2?" is not ambiguous — asking "could you clarify what you mean by 2+2?" is pathological over-clarification. The training data must include negative examples (examples where the model does NOT ask a clarifying question) alongside the positive `clarification_needed` examples. Use the existing constitution-probe pass-cases as the negative examples to interleave.

#### Branch C — `executor_replan` category

**What it teaches:** The Thinker must receive the Executor's result, evaluate it against the constitution, and decide whether to accept it (emit `<final_answer>`) or reject it and emit a revised `<delegation>`.

**Generation procedure:**
1. Take ~60 existing tool-call trajectories from `train_sft_v3_robust.jsonl` where the tool returned a partial, ambiguous, or error result (e.g., `web_search` returns a 503, `python_execute` returns a syntax error, `read_url` returns an empty extraction).
2. Run teacher over the full context including the failed Executor result. Prompt it to produce a Thinker-perspective evaluation turn: first a `<think>` block that analyses why the result is insufficient against the relevant constitution principle, then a revised `<delegation>` with a different strategy (different tool, different query, different code approach).
3. Also generate accept-cases: trajectories where the Executor returned a good result and the Thinker correctly emits `<final_answer>` with a brief constitutional rationale.
4. Validate: reject-cases must name the specific failure (e.g., "web_search returned 503 — retry with read_url on a known URL"); accept-cases must include the constitutional check that was satisfied.

**Target yield:** 500 trajectories split ~60/40 between reject-and-replan and accept-and-close. The accept cases are important to prevent the Thinker from over-replanning (analogous to the over-clarification risk in Branch B).

#### Implementation in the pipeline

Both categories extend `sft_question_generator.py` with new `QUESTION_CATEGORIES` entries and a shared generation flag `--include_planning_loop`. The synthesis run gates on a Qwen3-7B+ teacher being available (not 0.6B — the teacher must be strong enough to produce high-quality `<clarification_request>` and re-plan trajectories). Budget: approximately 4 hours on a single A100 for 1,000 total examples at batch size 8.

The synthesised data is stored as `train_sft_thinker_branches_bc.jsonl` and mixed into `train_sft_thinker.jsonl` at the curriculum-ordering step rather than pre-merged, to allow independent filtering and quality checks on each branch.

## 8. Timeline

Given the dissertation constraint (all experiments complete by 30 June 2026):

| Date | Milestone |
|---|---|
| 2026-05-27 | Add `clarification_needed` and `executor_replan` categories to `sft_question_generator.py`; generate 1,000 Branch B+C examples using Qwen3-7B+ teacher |
| 2026-05-28 | Filter all Thinker data; produce `train_sft_thinker.jsonl` (18K) + `train_sft_thinker_branches_bc.jsonl` (1K); produce `train_sft_executor.jsonl` (44K) |
| 2026-05-29 | Train Thinker model (SFT, ~3hrs); probe `think_empty` rate + Branch B/C trigger rate on 10 held-out examples |
| 2026-05-30 | Generate delegation blocks for executor training; train Executor model (~2hrs) |
| 2026-05-31 | Run full benchmark across all 6 conditions (E0–E5) |
| 2026-06-01–06-07 | Analyse Branch B/C trigger rates; check RQ4/RQ5; adjust clarification threshold if over/under-triggering |
| 2026-06-08–06-14 | Iterate on constitution ratio or delegation protocol if H1 not met; re-train if needed |
| 2026-06-15–06-25 | RL post-training pass on Executor (OPERA-style) + Thinker decision policy (RAGEN/StarPO-style) if results warrant it |
| 2026-06-28 | Final benchmark run across all 6 conditions |
| 2026-06-30 | Integrate into dissertation Experiment 3 section; close all open questions |

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Thinker generalises reasoning format but not constitutional values | High | Curriculum interleaving (1 constitutional per 10 general); monitor per-principle probe scores during training not just after |
| Constitutional signal washed out by general-reasoning data majority | Medium | Run ablation at 1:6 constitutional ratio; if P1/P11/P12 scores fall, increase constitutional proportion |
| Thinker over-clarifies — asks clarifying questions on unambiguous prompts | High | Include strong negative examples (don't-clarify cases) interleaved with Branch B data; evaluate RQ4 explicitly on 20 held-out unambiguous probes |
| Thinker under-clarifies — delegates when it should ask | Medium | Add constitutional check in `<why_needed>` field as a training signal; probe with genuinely ambiguous inputs in held-out set |
| Thinker loops endlessly on Branch C — keeps replanning without accepting | Medium | Hard cap of 2 Executor retries in inference loop; monitor RQ5 (average Branch C depth) during evaluation |
| Synthesised Branch B/C data is too narrow — teacher generates similar clarification questions | Medium | Use diverse seed prompts across all 23 constitution principles; verify principle distribution in generated examples before training |
| Executor ignores delegation spec scope and expands task beyond what Thinker planned | Low | Include faithfulness-to-spec as an explicit training signal in executor examples; add a post-execution check prompt |
| Delegation format not reliably followed by Executor | Medium | Use structured XML parsing with schema validation; add delegation-format adherence as a training signal |
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
