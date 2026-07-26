---
title: "Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation"
type: source
tags: [safety-tax, distillation, small-model, security, qwen]
sources:
  - https://arxiv.org/abs/2605.15239
updated: 2026-07-18
status: current
---

# Reducing the Safety Tax with On-Policy Self-Distillation (OPSA)

**Safety alignment usually taxes reasoning ("safety tax"); On-Policy Self-Distillation removes most of that tax by applying dense per-token KL supervision to the model's own rollouts, guided by a frozen teacher conditioned on a "privileged safety context", concentrating updates on the few early compliance-decision tokens — so safety and over-refusal and reasoning all improve, including on Qwen3-0.6B.**

## Summary

Fu et al. (2026) argue the safety tax comes from two mismatches: safety SFT trains on fixed *external* demonstrations (off the model's own policy) and applies supervision across the *whole* sequence. Their token-level analysis shows safety corrections concentrate in the first few tokens — the model decides to comply or refuse early — so blanket sequence supervision is both unnecessary and damaging to reasoning. OPSA instead distils the model toward its own on-policy generations under dense per-token KL from a frozen teacher given a privileged refusal/helpfulness context, localising updates to the compliance decision. Across Qwen3 (0.6B/1.7B/8B) and DeepSeek-R1-Distill (1.5B/8B) it beats off-policy self-distillation by +4.00 pp safety on average with the **largest gains on the smallest models** — and, uniquely, nudges reasoning *up* (+3.04) rather than down. This is arguably the single most on-target paper for the dissertation's core tension.

## Why it matters here

It tackles the exact safety-versus-capability trade-off the thesis studies and shows it can be reduced **on Qwen3-0.6B itself** (+5.49 pp safety; over-refusal cut 24.44% → 7.62%). It offers a distillation recipe where a teacher under a "privileged safety context" plays a role analogous to a [[entities/constitution|constitution]]/critic — a natural pairing with the project's teacher/critic scaffold — and supplies reusable evaluation machinery (Llama-Guard scoring over HarmBench/StrongReject/WildJailbreak/XSTest/WildBenign).

## Method

- **On-policy dense supervision:** per-token KL to the model's *own* sampled rollouts, concentrated where safety decisions occur (early tokens); full-parameter fine-tuning.
- **Type-conditional privileged context:** a frozen teacher (copy of the base) sees a refusal context `I_h` for harmful queries or a helpfulness context `I_b` for benign ones — activating latent safety reasoning rather than teaching new capability; the student is distilled toward it.
- **Teacher Flip Rate (TFR):** a training-free criterion measuring how often a privileged context converts unsafe→safe responses, used to search contexts across strength/length/framing/specificity/style.
- **Baselines:** Initial, SafeChain (external 70B-teacher distillation), ThinkSafe (off-policy SFT on self-generated traces). Key contrast is OPSA vs ThinkSafe (both self-generated) to isolate the on-policy + localised effect.

## Key results

- **Safety gain vs ThinkSafe:** Qwen3-0.6B +5.49, Qwen3-1.7B +3.05, Qwen3-8B +0.30, R1-Distill-1.5B +8.85, R1-Distill-8B +2.32 (avg **+4.00 pp**).
- **Reasoning preserved:** avg +3.04 across GSM8K/MATH500/GPQA/HumanEval/MBPP — the tax is reversed, not merely paid.
- **Over-refusal:** Qwen3-0.6B WildBenign 24.44% → 7.62%.
- **Adaptive robustness:** Prefilling attack ASR drops sharply (Qwen3-1.7B 3.8%→0%), but on the iterative **PAIR** attack OPSA *regresses* on three configs — a real limitation.
- **Absolute floor:** the smallest R1-Distill-1.5B stays leaky (StrongReject 47.28%) despite the largest relative gain.

## Critical appraisal

Crisp mechanistic hypothesis (early-token safety decisions) validated by token analysis and the Prefilling results, with a genuinely simultaneous safety/over-refusal/reasoning improvement and sensible ablations (advantage persists at 10% data). Cautions: absolute safety is Llama-Guard-dependent and imperfect on the tiniest models; gains are measured *relative to* ThinkSafe under matched data, so this is about supervision efficiency, not beating a strong external teacher on absolute safety.

> ⚠ Conflict / caution: "reduced tax" is not "solved safety" — the smallest models stay leaky and OPSA is not robust to adaptive PAIR-style jailbreaks. Temper any claim that a 0.6B model can be made *robustly* safe.

## Related

- [[sources/papers/effective-cai-small-llms]] — the recognition-vs-application gap OPSA sidesteps by activating latent safety
- [[sources/papers/constitution-or-collapse]] — the collapse/helpfulness-cost this method aims to avoid
- [[sources/papers/qwen3-tr]] — the base whose capability floor sets the "tax" baseline
- [[sources/papers/simple-self-distillation]] — self-distillation family; contrast on-policy vs sample-then-SFT
- [[entities/grpo]] — RL alignment alternative; OPSA is a distillation route instead
- [[topics/security-and-privacy]] — safety alignment and its costs
- [[experiments/thinker-executor-experiment]] — teacher/critic scaffolding analogue

## Sources

- Fu, Yu, Shahgir, Wei, Liu, Erichson, Dong (2026) — arXiv:2605.15239 — [arxiv.org/abs/2605.15239](https://arxiv.org/abs/2605.15239)
