---
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
type: source
tags: [agents, tool-use, reasoning, memory, self-correction, evaluation]
sources:
  - https://arxiv.org/abs/2303.11366
  - https://github.com/noahshinn024/reflexion
updated: 2026-07-18
status: current
---

# Reflexion: Language Agents with Verbal Reinforcement Learning

**Language agents can "learn" from failure without any weight updates by turning sparse task feedback into natural-language self-reflections stored in a small episodic memory that conditions the next attempt — a lightweight "verbal reinforcement learning" that reaches 91% pass@1 on HumanEval (vs GPT-4's 80%), but whose gains hinge on a usable evaluator and a base model capable of useful reflection.**

## Summary

Shinn, Yao and colleagues (Northeastern / MIT / Princeton, NeurIPS 2023) reframe policy optimisation as memory updating in language. Three modules iterate: an Actor produces a trajectory, an Evaluator scores it, and a Self-Reflection model writes a verbal lesson that is appended to episodic memory (bounded to ~1–3 entries) and prepended to the next trial. No LLM weights change — the "policy" is the (LLM, memory) pair. It delivers large gains across decision-making (ALFWorld 130/134, +22%), reasoning (HotPotQA 61%→75%), and coding (91% HumanEval pass@1), with clean ablations showing the reflection step — not mere retrying — drives the improvement. For this dissertation it is the canonical self-correction/verbal-RL reference and a useful cautionary baseline for whether such loops transfer down to sub-1B models.

## Why it matters here

Reflexion is the reference point for any retry/self-correction loop in a constitutional harness, and its Actor/Evaluator/Self-Reflection decomposition prefigures the Thinker/Executor and judge roles the project uses — its LLM-as-evaluator design also supports substance-based (not regex) judging. Its "improvement in memory, not weights" ethos both motivates a lightweight on-device harness and contrasts cleanly with the dissertation's constitution-in-weights SFT route (a compare-and-contrast worth drawing out).

## Method

- **Actor** M_a — an LLM (ReAct- or CoT-prompted) sampling actions conditioned on observations and memory.
- **Evaluator** M_e — task-specific scoring: exact-match for reasoning, hand-designed heuristics for decision-making, LLM-classification or unit-test execution for code.
- **Self-Reflection** M_sr — reads the (trajectory, reward) pair and writes verbal experience feedback.
- **Memory** — short-term = current trajectory; long-term = stored reflections bounded by capacity Ω (typically 1–3). Loop until solved or trial cap. For coding it self-generates unit tests via CoT (up to ~6 per suite) and reflects on failures — an internal test-driven loop.

## Key results

- **ALFWorld:** ReAct+Reflexion completes 130/134 tasks (+22% over ReAct), eliminating almost all hallucination-driven failures.
- **HotPotQA:** 61% → 75% (+14). Ablation: episodic memory alone +6%, adding self-reflection a further +8%. Baselines show no gain across trials — the lift is reflection, not resampling.
- **Coding (pass@1):** HumanEval Python 91.0% (GPT-4 80.1%); HumanEval Rust 68.0%; MBPP Rust 75.4%; Leetcode Hard Python 15.0% (GPT-4 7.5%). On MBPP Python it slightly trails GPT-4 (77.1% vs 80.1%).
- **Oracle dependence:** the coding benefit tracks self-generated test reliability — HumanEval Python false-positive rate 1.4% vs MBPP Python 16.3%, explaining where Reflexion helps most.

## Critical appraisal

An influential, simple idea with ablations that cleanly separate reflection from resampling, plus released code and a benchmark (LeetcodeHardGym). Trust the qualitative claim — verbal self-reflection over episodic memory beats blind retrying **when a usable evaluator exists** — more than the period- and oracle-dependent coding SOTA. It is a prompting/memory scaffold, not a training method: its ceiling is the base model's own reflective ability.

> ⚠ Conflict / caution: gains depend on a strong (GPT-3.5/4-class) actor. Two sibling papers here find this does not transfer down-scale — [[sources/papers/reason-plan-react]] reports Reflexion as the *worst* baseline on enterprise tool tasks, and [[sources/papers/small-agents-collaborate]] finds sub-agent reasoning marginal — so a 0.6B model may not produce useful reflections. It also failed on WebShop, where success needs broad exploration reflection cannot shortcut.

## Related

- [[sources/papers/react]] — the Actor loop Reflexion wraps; Yao is a co-author of both
- [[sources/papers/reason-plan-react]] — finds Reflexion the weakest tool-use baseline
- [[sources/papers/small-agents-collaborate]] — corroborates that self-reflection at small scale is marginal
- [[experiments/human-evaluation-rubric]] — LLM-as-evaluator design relevant to substance-based judging
- [[topics/tool-use-and-verification]] — self-correction and verification loops
- [[topics/personalisation]] — memory-as-policy framing for user modelling

## Sources

- Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao (2023) — arXiv:2303.11366 (NeurIPS 2023) — [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- Code — [github.com/noahshinn024/reflexion](https://github.com/noahshinn024/reflexion)
