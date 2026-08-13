---
title: Wiki Index
type: meta
updated: 2026-07-22
---

# Index

Catalog of everything in the wiki. Regenerated on every ingest. One line per entry. For chronological activity, see [[log]].

## Meta

- [[overview]] — thesis synthesis and research-question anchor
- [[log]] — append-only chronological journal
- [[tags]] — canonical tag vocabulary (check before adding new tags)
- `../CLAUDE.md` — schema and rules governing this wiki

## Topics

- [[topics/constitution-psychological-grounding]] — all 19 constitution principles mapped to peer-reviewed psychology/HCI theory (Mayer 1995, Kahneman 2011, Clark & Brennan 1991, etc.)
- [[topics/llm-foundations]] — tokenisation, attention, embeddings; why monolithic LLMs fail
- [[topics/reasoning]] — trustworthy reasoning across SFT, RL, architecture, evaluation
- [[topics/personalisation]] — 5W+H, GraphRAG, cold start; over-personalisation failure modes; scrutability
- [[topics/empathy]] — appraisal theory, Gricean grounding, conversation design; dependency/deskilling risks
- [[topics/tool-use-and-verification]] — PAL/ReAct delegation, MCP, ontology verification; prompt injection risk
- [[topics/explainability]] — citations, honest tool reports, translated latent state
- [[topics/ontology-integration]] — flagship: ontology as KB (A) or post-hoc verifier (B)
- [[topics/security-and-privacy]] — local-first privacy argument, OWASP threat taxonomy, Log-To-Leak, alignment regression

## Entities

- [[entities/constitution]] — 23-principle SFT v2 constitution
- [[entities/grpo]] — group relative policy optimisation (the repo's RL algorithm)
- [[entities/mcp]] — Model Context Protocol — "USB for AI"
- [[entities/rag]] — retrieval-augmented generation pattern
- [[entities/qwen3-0.6b]] — the pipeline's base model
- [[entities/graph-rag]] — KG-backed RAG for user-state memory
- [[entities/5w-h]] — who/what/when/where/why/how user-modelling schema
- [[entities/appraisal-theory]] — structured empathy substrate
- [[entities/tml-interaction-small]] — Thinking Machines Lab 276B/12B MoE; 0.40s real-time multimodal; frontier scale/privacy contrast

## Sources

### Papers — Foundations
- [[sources/papers/attention-is-all-you-need]] — the Transformer
- [[sources/papers/bert]] — bidirectional pre-training
- [[sources/papers/word2vec]] — static word embeddings
- [[sources/papers/bpe-subword-units]] — subword tokenisation (root of arithmetic failure)
- [[sources/papers/measuring-word-significance]] — vector length as importance
- [[sources/papers/gpt3-few-shot]] — in-context few-shot learning; scaling laws; the pre-alignment capability baseline
- [[sources/papers/scaling-laws]] — loss as a power law in params/data/compute; the "capability needs scale" backdrop (+ Chinchilla caveat)
- [[sources/papers/t5]] — text-to-text framing + C4; controlled-ablation discipline; encoder-decoder + denoising (scale-first, cuts against sub-1B)
- [[sources/papers/roformer]] — RoPE rotary position embedding; the base model's (Qwen/LLaMA) positional provenance; length flexibility
- [[sources/papers/talking-about-llms]] — Shanahan: anti-anthropomorphism; "a model that answers is not a person who answers" (background pillar 1)
- [[sources/papers/theoretical-impediments-ml]] — Pearl: Ladder of Causation; correlation ≠ causation; CoT sits at Level-1 association (background pillar 2)

### Papers — Prompted reasoning
- [[sources/papers/chain-of-thought-prompting]] — CoT as prompted reasoning
- [[sources/papers/auto-cot]] — automates CoT exemplar creation
- [[sources/papers/prompting-science-report-2]] — diminishing returns on modern models
- [[sources/papers/tree-of-thoughts]] — search-based deliberation

### Papers — RL for reasoning
- [[sources/papers/deepseekmath]] — **origin of GRPO**; critic-free group-baseline RL; 120B-token maths corpus
- [[sources/papers/dapo]] — four practical fixes over naive GRPO (clip-higher, dynamic sampling, token-loss, overlong shaping); removes KL
- [[sources/papers/luspo]] — RLVR length bias in GRPO/GSPO gradient normalisation; drop 1/|y| to cure length collapse; RLVR-fragility evidence (7B–30B)
- [[sources/papers/gsm8k]] — the GSM8K benchmark + origin of outcome-reward verification (6B-verify ≈ finetuned-175B); trust caveat: rewards correct-but-unfaithful reasoning
- [[sources/papers/deepseek-r1]] — R1-Zero + multi-stage R1
- [[sources/papers/seed15-thinking]] — process-reward RL exemplar
- [[sources/papers/vapo]] — value-based PPO for long-CoT
- [[sources/papers/understanding-r1-zero]] — GRPO length-bias critique, Dr. GRPO
- [[sources/papers/interleaved-reasoning]] — RL-trained interleaved thinking
- [[sources/papers/hidden-reasoners]] — LaTRO self-rewarding latent reasoning

### Papers — Architectural / latent reasoning
- [[sources/papers/hierarchical-reasoning-model]] — HRM: slow + fast modules
- [[sources/papers/looped-transformers-reasoning]] — depth > parameters
- [[sources/papers/coconut-continuous-latent]] — reason in vector space
- [[sources/papers/ladir]] — latent diffusion reasoning
- [[sources/papers/state-stream-transformer]] — persistent latent state
- [[sources/papers/diffusion-of-thoughts]] — diffusion-LM CoT
- [[sources/papers/llada]] — masked-diffusion LLM rivals LLaMA3 8B, beats GPT-4o on reversal; non-AR contrast (KV-cache-unfriendly on-device)
- [[sources/papers/pangu-embedded]] — 7B dual-system (fast/slow) reasoner with metacognitive mode-switching
- [[sources/papers/planner-executor-diffusion]] — diffusion planner + AR executor; latent 64-token plan beats R1 at ~44× fewer tokens (trust-vs-latent tension)

### Papers — Small-model / distillation
- [[sources/papers/phi1-textbooks]] — phi-1: 1.3B on ~7B "textbook-quality" tokens rivals 10×-larger code models; emergent 29→50.6% jump (flagship quality>scale, teacher-distilled)
- [[sources/papers/self-enhanced-reasoning]] — SERT small-model self-training
- [[sources/papers/simple-self-distillation]] — SSD: sample → SFT on own outputs; no verifier/teacher/RL; foundation for Constitutional SSD adaptation
- [[sources/papers/dual-head-reasoning-distillation]] — train-time-only reasoning
- [[sources/papers/token-hungry-deepseek-r1]] — accuracy-vs-efficiency trade-off
- [[sources/papers/qwen25-math]] — self-improvement (data synthesis + RM curation + GRPO); 7B matches prior 72B; RM@N > majority vote
- [[sources/papers/structured-templates]] — SST: template scaffolding lifts a 1.5B model +6.2 GSM8K; "scaling law by difficulty" (too much easy data hurts)

### Papers — Tool use
- [[sources/papers/pal]] — code-as-reasoning delegation
- [[sources/papers/react]] — interleaved reason-act loop
- [[sources/papers/mcp-multi-agent]] — MCP coordination protocol
- [[sources/papers/search-r1]] — RL-trained search-tool use
- [[sources/papers/rag-original]] — retrieval-augmented generation
- [[sources/papers/beyond-react]] — planner-centric DAG framework; Qwen3-0.6B GRPO instability confirmed; stage-grouping schema derived from their DAG insight
- [[sources/papers/replacing-thinking-with-tool-usage]] — Chain-of-Edits: structured tool use replaces language CoT for 1B–3B models; advantage reverses at 8B
- [[sources/papers/t1]] — inter-tool-dependency multi-turn dataset + cache memory; 8B-SFT beats 70B on tool selection
- [[sources/papers/toolmind]] — 360k tool-use dataset; turn-level filtering is decisive (τ-bench +11 vanishes without it)
- [[sources/papers/cove]] — constraint-guided deterministic verification; CoVe-4B ≈ 70B; SFT+RL regresses vs SFT (weak simulator)

### Papers — Agents / planner-executor
- [[sources/papers/reason-plan-react]] — RPA planner supervising ReAct executors; context-offload for local small-context models; wins hard enterprise tasks
- [[sources/papers/small-agents-collaborate]] — planner-limited not executor-limited; small MAS matches a 32B single agent; ~43% fewer tokens
- [[sources/papers/reflexion]] — verbal RL: self-reflection in episodic memory, no weight updates; cautionary at small scale
- [[sources/papers/opera]] — orchestrated planner-executor RAG; per-role GRPO (MAPGRPO); planner carries the accuracy (−22.6 EM without it)
- [[sources/papers/thinker]] — SFT-taught breadth→depth hierarchy beats RL deep-search; relative gain larger at 3B than 7B
- [[sources/papers/ragen]] — multi-turn agent RL "Echo Trap" collapse (on Qwen-0.5B); unrewarded think-traces decorative; supports the no-GRPO pivot

### Papers — Multimodal / agent RL
- [[sources/papers/ui-r1]] — RL for GUI-agent action prediction
- [[sources/papers/vlm-r1]] — R1-style RL for vision-language

### Papers — Evaluation
- [[sources/papers/none-of-the-others]] — reasoning-vs-memorisation variation
- [[sources/papers/mt-bench]] — LLM-as-judge validated (~85% human agreement); position/verbosity/self-enhancement/math-grading biases + mitigations
- [[sources/papers/hallucination-survey]] — canonical NLG hallucination taxonomy; intrinsic vs extrinsic; faithfulness vs factuality
- [[sources/papers/abstention-bench]] — abstention is scale-invariant and reasoning-tuning *degrades* it (−24%); a place a small model can compete on honesty
- [[sources/papers/agent-cq]] — clarify-before-assume: generate + LLM-judge clarifying questions; usefulness/clarity drive quality, complexity irrelevant; teachable to small models
- [[sources/papers/helm]] — holistic evaluation: dense multi-metric × multi-scenario; coverage 17.9%→96%; the "why multi-axis" framing (reference-based, pre-judge)
- [[sources/papers/prometheus]] — open 13B evaluator LLM rivals GPT-4 judge (0.897 vs 0.882); rubric+reference+feedback-before-score; the open-judge alternative
- [[sources/papers/abstention-survey]] — abstention taxonomy (query/knowledge/values × lifecycle); URUP/ARSP metrics; over-refusal vs unsafe-compliance
- [[sources/papers/biggen-bench]] — instance-specific rubrics beat coarse/domain; open 8×7B judge rivals GPT-4-class; ToM/tool-use hardest to judge (NAACL 2025 Best Paper)
- [[sources/papers/llm-as-judge-design]] — judge-design ablation: criteria > reference > mean-sampling > CoT; endpoint-anchored rubrics; consistency ≠ correctness

### Papers — Empathy / affect
- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent sentiment classification
- [[sources/papers/appraise-plm]] — AppraisePLM: 21-dim appraisal regression + emotion classification; CoNLL 2025; Debnath, Graham, Conlan (TCD); unblocks Experiment 2
- [[sources/papers/computational-empathy]] — EPITOME: 3 empathy mechanisms × 3 levels; ~251M-param encoder (sub-1B); the empathy-chapter framework/rubric

### Papers — Personalisation / over-personalisation
- [[sources/papers/op-bench]] — first benchmark for over-personalisation; 26–61% degradation from memory augmentation
- [[sources/papers/rpeval]] — Feng RPEval: rational preference utilisation (Ignore/Support/Dominate); 40–90% human-LLM gap; inverse scaling; RP-Reasoner
- [[sources/papers/avoiding-over-personalization]] — rule-guided KG edits (Soft/Hard/Removal) on Qwen3-0.6B, client-side; symbolic control vs rational reasoning
- [[sources/papers/sycophancy-sharma]] — RLHF structurally incentivises sycophancy; foundational mechanism paper (ICLR 2024)
- [[sources/papers/syc-eval]] — 58.19% sycophancy rate; 78.5% persistence; cross-model measurement (AIES 2025)

### Papers — Memory / personalisation systems
- [[sources/papers/mem0]] — extract-then-update memory (ADD/UPDATE/DELETE/NOOP); Mem0g graph variant; ~90% token savings (cloud GPT-4o-mini)
- [[sources/papers/memmachine]] — ground-truth-preserving 3-layer memory; LoCoMo 0.917, ~80% fewer tokens vs Mem0; cloud/frontier-model contrast
- [[sources/papers/personalai]] — KG storage/retrieval sweep across model scale; small models need structure-aware traversal (BeamSearch ~6.6 min/query)
- [[sources/papers/graph-agent-memory-survey]] — graph-memory taxonomy + extraction→storage→retrieval→evolution lifecycle (related-work backbone)
- [[sources/papers/memory-age-ai-agents]] — forms/functions/dynamics (F/E/R) taxonomy; trustworthiness a named pillar (abstract-only, draft)
- [[sources/papers/forgetful-but-faithful]] — MaRS typed provenance memory + DP-scored forgetting; FiFA benchmark; privacy-aware retention

### Papers — Security / alignment
- [[sources/papers/constitutional-ai-bai]] — original CAI paper; generate–critique–revise loop; RLAIF (Anthropic 2022)
- [[sources/papers/constitution-or-collapse]] — CAI at 8B: 40.8% ASR reduction, 9.8% helpfulness cost, model collapse
- [[sources/papers/effective-cai-small-llms]] — CAI at 7–9B is architecture-dependent; critique-step harm detection fails on Gemma/Qwen; R1-distill benefits
- [[sources/papers/reducing-safety-tax]] — OPSA on-policy self-distillation reverses the safety tax on Qwen3-0.6B (+5.49pp safety, over-refusal 24→8%)
- [[sources/papers/general-language-assistant]] — the HHH (helpful/honest/harmless) foundation; prompting + context distillation; small models hurt by the HHH prompt (Anthropic 2021)
- [[sources/papers/c3ai]] — how to craft/prune a constitution; positive framing +27% with humans but models adhere best to negative prohibitions; 15 ≈ 58 principles on safety
- [[sources/papers/safety-tax]] — reasoning-training raises harm +43.7; reasoned refusals (SafeChain) beat hardcoded (DirectRefusal); defines the safety tax
- [[sources/papers/inverse-constitutional-ai]] — recover an auditable constitution from preference data (ICAI); real-data ceiling ~60%; "did the constitution take?" audit
- [[sources/papers/generative-value-conflicts]] — ConflictScope: value priorities flip MCQ→open-ended; system-prompt steering only ~14% (argues for SFT)
- [[sources/papers/constitutional-labeling-consistency]] — detailed per-category constitutions cut cross-model disagreement up to 57×; nano-class an order worse (bake in via SFT)
- [[sources/papers/hierarchical-safety-adherence]] — MiniGrid probe; "illusion of compliance" (adherence can be incompetence); cost of compliance 80%→14%
- [[sources/papers/trustllm]] — eight-dimension trustworthiness taxonomy; over-alignment (Llama2-7B refuses 57% of benign prompts); capability ≈ trustworthiness
- [[sources/papers/nemo-guardrails]] — programmable runtime rails (Colang); post-hoc external control vs in-model constitution; ~3× latency; over-refusal cost

### Papers — Context / preference tracking
- [[sources/papers/context-length-hurts]] — 13.9–85% degradation from context length alone despite perfect retrieval; small models worst; "Retrieve Then Solve"
- [[sources/papers/prefeval]] — Zhao ICLR 2025 oral; preference following <10% at 10 turns, ~0 at long context; SFT fix generalises (distinct from RPEval)
- [[sources/papers/transparent-scrutable-recs]] — NL user profiles: scrutable + editable + competitive accuracy; GPT-2 scorer; template for the 5W+H user model (UCL/Sheffield 2024)
- [[sources/papers/tears]] — TEARS: textual editable profiles + optimal-transport alignment; scrutability without accuracy cost (text-only collapses to 0.031 R@20)

### Papers — Knowledge graphs / tool use
- [[sources/papers/think-on-graph]] — LLM ⊗ KG tight coupling; beam search on KG; SOTA on 6/9 datasets (ICLR 2024)
- [[sources/papers/llm-guided-tot]] (Literature Note only) — ToT software system with checker module and backtracking controller

### Papers — Adaptation / training
- [[sources/papers/gpt3-few-shot]] — see Foundations; the scale-hungry baseline this lineage reacts to
- [[sources/papers/flan]] — instruction tuning; teachable zero-shot but degrades below ~8B (the 0.6B challenge)
- [[sources/papers/instructgpt]] — RLHF (SFT→RM→PPO); alignment beats scale (1.3B preferred over 175B); the alignment tax
- [[sources/papers/lora]] — low-rank PEFT; ~10,000× fewer trainable params; enables single-GPU 0.6B SFT
- [[sources/papers/qlora]] — 4-bit NF4 + LoRA; 65B finetune <48 GB; data quality > quantity (training enabler)
- [[sources/papers/lima]] — Superficial Alignment Hypothesis; 1,000 curated examples rival RLHF; quality/diversity > quantity (65B base)
- [[sources/papers/instruction-tuning-survey]] — SFT-pillar map: human/distilled/self-improved dataset taxonomy; quality>quantity; efficient tuning (survey)
- [[sources/papers/improved-sft-forgetting]] — mitigate forgetting via self-reconstructed distribution-aligned rehearsal (data-side); mix with domain/constitution data
- [[sources/papers/entropy-adaptive-ft]] — EAFT: entropy-gate the SFT loss to spare "confident conflicts" (optimisation-side); caveat — may resist installing new constitutional behaviour
- [[sources/papers/seal]] — SEAL: LLMs generate natural-language self-edits for their own weight updates via an RL outer loop; forgetting + cost caveats (MIT 2025)

### Papers — On-device / edge
- [[sources/papers/mobillama]] — 0.5B shared-FFN SLM; fully transparent (open data); runs on Snapdragon-685 at ~7 tok/s in <1 GB
- [[sources/papers/llm-inference-edge]] — sustained-load edge benchmark; phones lose 15–41.5% to thermal throttling; NPU flat at <2 W
- [[sources/papers/phi3-tr]] — 3.8B data-quality SLM; 4-bit ≈1.8 GB, >12 tok/s on iPhone 14; weak factual recall by design
- [[sources/papers/sustainable-edge-inference]] — Raspberry Pi real-Joules benchmark; up to ~79% energy cut; 3-bit not reliably cheaper than 4-bit (Pareto)
- [[sources/papers/on-device-llm-eval]] — ~3.5-bit quality floor, 4-bit sweet spot; small models collapse below 4-bit; compute-bound <0.5B, bandwidth-bound >1B

### Papers — Data (corpora & curation)
- [[sources/papers/the-pile]] — 825 GiB 22-domain diverse corpus; the "curate a mixture" pole; bias/consent/PII admissions
- [[sources/papers/refinedweb]] — web-only + aggressive dedup rivals curated corpora (Falcon); curation may be unnecessary
- [[sources/papers/fineweb]] — ablation-per-decision curation; FineWeb-Edu matches a ~10× larger corpus (score-for-quality template)
- [[sources/papers/deduplicating-training-data]] — dedup cuts memorised-output ~10× at flat perplexity; the data-hygiene→memorisation→privacy link
- [[sources/papers/doremi]] — learn domain mixture weights with a 280M proxy → +6.5 pts / 2.6× faster at 8B (mixture optimisation)
- [[sources/papers/data-centric-training]] — 2026 vision: agentic data prep + live data-model interaction; taxonomy/citation hub (frontier-scale, no experiments)

### Papers — Security threats
- [[sources/papers/membership-inference]] — shadow-model attack; ~94% on Google ML; re-identification risk; leaks track overfitting + class count
- [[sources/papers/extracting-training-data]] — 604 verbatim examples from GPT-2 incl. real PII; memorisation grows with size + duplication
- [[sources/papers/ignore-previous-prompt]] — PromptInject: goal hijacking 58.6%, prompt leaking 23.6%; inverse scaling (bigger = more injectable); constitution-leak risk
- [[sources/papers/adversarial-attacks-zou]] — GCG suffix; transfers to GPT-3.5 86.6% / GPT-4 46.9% but Claude-2 ~2.1%; alignment as a breakable outer layer

### Papers — Privacy / unlearning
- [[sources/papers/what-should-llms-forget]] — WikiMem: quantify what personal data a model memorised before RTBF; black-box local audit
- [[sources/papers/federated-unlearning]] — FOUL: server-side client unlearning; matches retrain oracle at ~60% less compute/comms
- [[sources/papers/unlearning-at-scale]] — exact RTBF via deterministic filtered replay (32-byte WAL); cohort-scoped LoRA adapter deletion fits on-device (CPU-toy validation only)

### Papers — Frontier model references
- [[sources/papers/qwen3-tr]] — Qwen3 family 0.6B–235B; unified thinking/non-thinking + budget; the base model report; 0.6B capability floor
- [[sources/papers/phi4-tr]] — Phi-4 14B; 40% synthetic data; beats teacher GPT-4o on GPQA/MATH; Pivotal Token Search DPO; weak factual recall
- [[sources/papers/gpt5-system-card]] — GPT-5 safe-completions; sycophancy 0.04 vs GPT-4o 0.145, prompt-injection defence 0.97–0.99; but hallucination/deception residuals persist (self-reported, relative)

### Advisor meetings (supervisor–student)
- [[sources/meetings/september2025]] — First meeting: scrutability framing, Inside Out multi-agent concept, AI as "sociopath"
- [[sources/meetings/october2025]] — RL for thought processes, values-interpreter architecture, ethical AI companion risks
- [[sources/meetings/november2025]] — Ontology-LLM pivot (Nov 11); interleaved thinking + scrutability (late Nov)
- [[sources/meetings/december2025]] — Research plan refinement; prototype scope defined; ethical/GDPR flags
- [[sources/meetings/january2026]] — Boolean/math GPT failure → hybrid delegation architecture confirmed
- [[sources/meetings/february2026]] — Behaviourism lens; post-hoc constraint vs in-model change; focus contraction
- [[sources/meetings/april2026]] — Constitution drift + probes vs tests; Apple internship June–Sept 2026; dissertation timeline

### Dissertation drafts (user-authored raw)
- [[sources/dissertation/research-plan]] — formal CS7CS6 plan: title, 5 objectives, 7 phases, 2 pivots
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — main thesis argument + literature review
- [[sources/dissertation/experimental-planning-document]] — 6 experiments, 2025-11-10 meeting, timeline
- [[sources/dissertation/personal-notes]] — Experiment.md + Rough Notes.md + Research Plan Edits.md
- [[sources/dissertation/overpersonalisation-paper]] — LLNCS paper: three failure modes, sycophancy mechanism, UMAP scrutability tradition, commercial memory architecture comparison
- [[sources/dissertation/security-privacy-social-ethics]] — security analysis: local-first privacy argument, Log-To-Leak, alignment regression, critique-loop SPOF, dependency/deskilling ethics

### Code (pipeline summaries)
- [[sources/code/pipeline-overview]] — professor-facing end-to-end walkthrough: question generation → constitutional distillation (MiniMax teacher) → LoRA → inference → benchmark → GLM-5.1 judge; rubrics, situations, personas
- [[sources/code/sft-v2-pipeline]] — SFT v2/v3 data generation pipeline: Part A + Part B → transform → robustness variants → native tool examples → train_sft_v3_robust.jsonl
- [[sources/code/sft-v3-pipeline]] — v3 asymmetric distillation: intercept loop, negative trajectories, curriculum training
- [[sources/code/constitution-document]] — full 23-principle source
- [[sources/code/training-and-benchmark]] — SFT + GRPO (DAPO) + dual tool-call modes (xml/native) + Experiment 0 + adversarial suite + run_all.sh + preflight

## Experiments

- [[experiments/experiment-catalog]] — all six experiments + ablation A/B/C/D
- [[experiments/frontier-model-comparison]] — study design comparing Qwen3-0.6B (base + fine-tuned) vs Claude Sonnet 4.6, Minimax M2.7, Kimi K2.6 across 50 prompts; two evaluation tracks (automated + human)
- [[experiments/human-evaluation-rubric]] — 12-item Likert rubric from Mayer et al. trust model + Davis empathy index; the external human-judgment ground truth
- [[experiments/sft-benchmark-analysis-20260525]] — full 5-run benchmark analysis: per-principle trend, probe failure anatomy, latency profile, cross-cutting diagnoses (memory overuse, empty think blocks)
- [[experiments/thinker-executor-experiment]] — Experiment 3 design: dual-SFT 0.6B architecture splitting constitutional reasoning (Thinker) from tool execution (Executor); motivated by capacity-displacement finding in SFT benchmarking

## Decisions

- [[decisions/2025-10-01-four-module-architecture]] — **binding** Pivot 1: Reasoning / User Modelling / Tool Integration / Generator modules; Professor Conlan feedback; anchors entire thesis design
- [[decisions/2025-11-10-ontology-focus-shift]] — primary focus moves to ontology-LLM integration
- [[decisions/2026-05-14-scratchpad-tool]] — scratchpad working memory + P24/P25 principles + partial-capability honesty training (spec + plan in docs/superpowers/)
- [[decisions/2026-05-03-research-question-reframe]] — operational hypothesis added: on-device 0.6B model vs frontier models; psychological grounding for constitution; human evaluation rubric introduced
- [[decisions/2026-06-28-reasoning-tier-weighting]] — P1 decompose + P20 first-principles moved to the Tier-2 substrate in the purpose weighting; reasoning family made coherent; score-audited (ranking unchanged)
- [[decisions/2026-07-04-tier-map-completion]] — H2b (tier 2, personalisation) + P22_scratchpad (tier 3, tool) added to the canonical map; 25-defined/23-scored/21-covered accounting fixed; score-audited (deltas ≤ 0.008, ordering unchanged); 5-tier scheme rejected in favour of a weight sensitivity analysis

## Questions

- [[questions/2026-04-19-initial-questions]] — consolidated TODOs, advisor-prep questions, literature tensions
- [[questions/2026-04-30-asset-acquisition-todo]] — 27-paper acquisition checklist from overpersonalisation + security papers (+ 5 carry-over from prior lint)

## Queries

- [[queries/grpo-and-personalisation-master-plan]] — two-track implementation roadmap: GRPO trainer (Track 1) + 5W+H graph-memory empathy stack (Track 2); industry benchmarks; 8-paper acquisition list; 6-week sequencing plan
- [[queries/full-pipeline-implementation-plan]] — phase-by-phase build plan for all six modules (SFT, GRPO, User Modelling, Empathy, Ontology Verifier, Retrieval Gating) with feature flags; parallel/sequential dependency map; GPU-day checklist

---

## Not yet ingested

- `IMPROVEMENT_ROADMAP.md` — 54KB roadmap at repo root. Ingest if still authoritative.
- Per-file deep-dives of individual `pipeline/sft_*.py` scripts — summarised together in [[sources/code/sft-v2-pipeline]] for now.

## Cited but not yet ingested as wiki stubs

- Thinking Machines Lab (2026) — "Interaction Models" blog post / TML-Interaction-Small release (has entity page [[entities/tml-interaction-small]]; no source page)

## Cited but not in `docs/Assets/`

Papers with stub Literature Notes but no PDF. Full checklist: [[questions/2026-04-30-asset-acquisition-todo]].

- Budzyń et al. 2025 — endoscopist deskilling (Lancet, paywalled; institution access needed)
- Google DeepMind 2026 — Gemma 4 Technical Report (no arXiv; blog only)
