---
title: Frontier Model Comparison Study
type: experiment
tags: [evaluation, trust, constitution, sft, rl, on-device, psychology]
sources:
  - pipeline/constitution.md
  - wiki/experiments/human-evaluation-rubric.md
  - wiki/topics/constitution-psychological-grounding.md
updated: 2026-05-03
status: draft
---

# Frontier Model Comparison Study

**Can a 0.6B model, fine-tuned with constitution-guided SFT and GRPO, achieve trust and empathy ratings comparable to frontier models on targeted trust-relevant behaviours — while running entirely on-device with local user memory and no persistent internet access?**

This is the central empirical question of the thesis. The comparison study is the primary evaluation mechanism that validates or falsifies the thesis claim. It is the answer to the critic's objection that LLM-generated training data is circular: even if the training data is synthetic, the evaluation is not.

---

## Motivation

The privacy argument in this thesis is architectural: a model that runs on the user's device physically cannot exfiltrate user data, because network access is gated. Frontier models (Claude, Minimax, Kimi) are API-only — they necessarily receive every user message on a remote server. The comparison is therefore not just about capability; it is about whether the on-device privacy guarantee can be achieved at negligible trust cost. See [[decisions/2026-05-03-research-question-reframe]] for the full framing.

---

## Models

| Model | Type | Deployment | Notes |
|---|---|---|---|
| Qwen3-0.6B (base) | Untuned | Local | Baseline — raw capability before SFT/GRPO |
| Qwen3-0.6B (constitution SFT) | Fine-tuned | Local | After SFT pipeline only |
| Qwen3-0.6B (SFT + GRPO) | Fine-tuned | Local | Full pipeline — primary candidate |
| Claude Sonnet 4.6 | Frontier | API | Anthropic; current production model |
| Minimax M2.7 | Frontier | API | ⚠ Verify exact model ID against Minimax API docs — identifier unconfirmed post my knowledge cutoff |
| Kimi K2.6 | Frontier | API | ⚠ Verify exact model ID against Moonshot AI API docs — identifier unconfirmed post my knowledge cutoff |

> The frontier models serve as an upper-bound reference. The thesis claim is not "we beat Claude" — it is "the fine-tuned small model is significantly better than its untuned baseline and closes a meaningful portion of the gap to frontier models on constitution-specific trust dimensions, while being deployable on consumer hardware."

---

## Prompt set

**50 prompts** drawn proportionally from the 11 SFT question categories:

| Category | # prompts | Constitution principles exercised |
|---|---|---|
| user_context_behavioral | 8 | P1, P6, P17 |
| real_time_dependent | 6 | P5, P2, P16 |
| mathematical_precision | 5 | P4, P10 |
| adversarial_pressure | 6 | P14, P7, P18 |
| tool_unavailable_graceful | 5 | P3, P12, P18 |
| knowledge_boundary | 5 | P7, P8, P18 |
| entity_facts_web_search | 5 | P11, P19, P16 |
| multi_turn_conversation | 4 | P17, P6, P15 |
| impossible_task | 3 | P8, P18 |
| tradeoff_question | 3 | P9 |
| appraisal_empathy | 5 | P6, P17 — empathy dimension |

Prompts are drawn from the hand-crafted evaluation set where available; novel prompts for categories without hand-crafted examples. Each prompt is fixed across all models (identical wording, no system-prompt manipulation).

---

## Evaluation tracks

### Track 1 — Automated (constitution compliance)

Each model response is scored by the constitution compliance critic (Groq/Gemma-2 9B or equivalent) against the 19 principles. Output: compliance score per principle (0–1) and aggregate score.

This provides a fast, reproducible measurement but has a known limitation: the critic is itself an LLM and can be wrong. Track 2 is the ground truth.

### Track 2 — Human evaluation

All 50 responses per model are rated by 5 evaluators using the [[experiments/human-evaluation-rubric]]. Evaluators are blind to model identity. Responses are anonymised and presented in randomised order.

Output: per-dimension scores (Ability, Integrity, Benevolence, Empathy, Overall) per model, with inter-rater reliability (Krippendorff's alpha).

---

## Hypotheses

**H1 (ability):** The fine-tuned Qwen3-0.6B (SFT + GRPO) scores significantly higher on Ability than the untuned baseline (paired t-test, p < 0.05).

**H2 (trust parity):** The fine-tuned model achieves a composite trust score within 0.5 points (on the 1–5 scale) of the frontier models on at least the Integrity and Ability dimensions.

**H3 (empathy gap):** The empathy dimension shows the largest gap between the fine-tuned small model and frontier models — this is the expected hard ceiling for a 0.6B model without the full four-module stack.

**H4 (modular recovery):** Adding the User Modelling module and Appraisal Empathy module to the fine-tuned model closes the empathy gap by ≥ 30% on the appraisal_empathy prompt category. This validates the modular architecture claim.

---

## On-device capability profile

In addition to trust ratings, record the following for each local model run:

| Metric | Target |
|---|---|
| First-token latency (ms) on MacBook M-series / equivalent | < 1000ms |
| Memory footprint (MB) of model weights | < 1500MB (INT4 quantised) |
| Tokens per second (generation speed) | > 15 tok/s |
| Internet access required | None — all generation and user memory local |

Frontier API models are excluded from this profile (they require internet by definition). The profile demonstrates the deployment gap, motivating why a 0.5-point trust score difference may be acceptable to users who prioritise privacy.

---

## Statistical analysis plan

- **Primary comparison**: one-way ANOVA across models per dimension, with post-hoc Tukey HSD for pairwise comparisons.
- **Effect size**: Cohen's d between fine-tuned Qwen3-0.6B and untuned baseline.
- **Equivalence test**: TOST procedure to test whether fine-tuned model is practically equivalent (within 0.5 points) to frontier models on Integrity + Ability dimensions.
- **Reliability**: Krippendorff's alpha per rubric dimension. Target α ≥ 0.67.

---

## Timeline

| Step | When |
|---|---|
| Prompt set finalised (50 prompts) | Before GPU training run |
| All models run on prompt set | After SFT + GRPO training complete |
| Human evaluation data collected | Within 2 weeks of training completion |
| Analysis and write-up | Before Apple internship start (June 2026) |

---

## Related

- [[experiments/human-evaluation-rubric]]
- [[topics/constitution-psychological-grounding]]
- [[decisions/2026-05-03-research-question-reframe]]
- [[entities/qwen3-0.6b]]
- [[entities/constitution]]
- [[entities/grpo]]
- [[topics/security-and-privacy]]

## Sources

- Mayer, R. C., Davis, J. H., & Schoorman, F. D. (1995). An integrative model of organizational trust. *Academy of Management Review*, 20(3), 709–734.
- Davis, M. H. (1983). Measuring individual differences in empathy. *Journal of Personality and Social Psychology*, 44(1), 113–126.
