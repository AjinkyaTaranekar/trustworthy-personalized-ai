---
title: Constitution (SFT v2)
type: entity
tags: [sft, constitution, principles, security, sycophancy]
sources:
  - pipeline/constitution.md
  - README.md
  - docs/security-analysis/security-review.tex
updated: 2026-05-07
status: current
---

# Constitution (SFT v2)

**The 23-principle document that governs every gold response generated in the SFT v2 pipeline. Covers capability honesty, tool discipline, honest refusal, first-principles reasoning, 5W+H framing, consequence accountability, and interleaved tool chaining.**

## Summary
The v2 supervised-fine-tuning pipeline replaces a 42-template scenario approach with a **constitution-driven** generator: the teacher model drafts a response, critiques it against the 23 principles in `pipeline/constitution.md`, and revises on violations. Only the revised response enters the training set. Every `<think>` block in the assembled dataset must contain a `CAPABILITY_CHECK` — asserted by the dataset assembler quality filter.

## Where it lives in the code

- `pipeline/constitution.md` — the document itself.
- `pipeline/sft_gold_response_generator.py` — consumes it for draft / critique / revise.
- `pipeline/sft_dataset_assembler.py` — enforces `CAPABILITY_CHECK` presence.
- See `README.md` §"SFT v2 Pipeline (Constitution-Based, Domain-Unbounded)".

## Training tool surface (per README)

| Tool             | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| `python_execute` | Precision arithmetic and computation                     |
| `web_search`     | Real-time data, current events, proper nouns             |
| `read_url`       | Follow up on a specific search result                    |
| `get_datetime`   | Current date/time for time-aware responses               |

## P20–P23: Advanced Reasoning Principles (added 2026-05-06)

Four principles were appended to the constitution to encode structured epistemic reasoning behaviours that earlier principles did not cover.

| # | Principle | One-line rule |
|---|-----------|---------------|
| 20 | FIRST PRINCIPLES | Before non-trivial answers, identify irreducible truths and name unverified assumptions |
| 21 | 5W+H QUESTIONING | Every CAPABILITY_CHECK must address Who/What/When/Where/Why/How |
| 22 | CONSEQUENCE_CHECK | Assess stakes, failure mode, user action, and accountability in every response; surface high-stakes caveats in `<answer>` |
| 23 | INTERLEAVED TOOL CHAINING | When data retrieval AND computation are both needed, chain the tool calls — never stop at one tool when a second would verify or precise the answer |

These principles are now checked deterministically by `rule_check_response()` in `sft_gold_response_generator.py` alongside the original structural checks. The `interleaved_tool_reasoning` question category in `sft_question_generator.py` specifically trains the model on P23 patterns. See [[topics/constitution-psychological-grounding]] for the expanded mapping.

---

## Security Risks in the Constitution Design

Three security issues surface from the [[sources/dissertation/security-privacy-social-ethics|security analysis paper]]:

**Principle 10 amplifies prompt injection risk.** Principle 10 (use tools correctly when available) makes the model structurally disposed to follow tool-returned content — which is the correct behaviour in normal operation, but amplifies the attack surface when tool output is adversarially crafted. The Log-To-Leak attack pattern exploits exactly this: a malicious MCP server provides crafted content that the model, following Principle 10, processes as trusted instruction. A runtime extraction layer (not yet implemented) is needed to sanitise tool outputs before they reach the main model.

**Alignment regression risk from GRPO.** If the GRPO reward signal favours user satisfaction without an explicit anti-sycophancy penalty, the model may learn to agree with incorrect premises from users perceived as authoritative — selectively weakening Principle 7 (uncertainty quantification). Principle 14 (hold under pressure) is the training-time mitigation; an adversarial benchmark suite is required before GRPO begins.

**Critique loop as single point of failure.** The generate–critique–revise loop in `sft_gold_response_generator.py` uses the same model as generator and critic. Shared distributional biases can cause the loop to degenerate: outputs satisfy the critique format without correcting the underlying problem. Rule-based `rule_check_response()` now covers P1, P3, P4, P14, P18, P20, P21, P22, P23 deterministically, but semantic violations on the remaining principles still depend on the LLM critic. Use `--critic_model` with a separate model (e.g. `nvidia_nim/minimaxai/minimax-m2.7`) for full Blocker 2 compliance.

## Related

- [[topics/reasoning]] — constitution encodes reasoning-honesty rules
- [[topics/tool-use-and-verification]] — constitution encodes tool discipline
- [[topics/security-and-privacy]] — security risks in the constitution design
- [[entities/mcp]] — Principle 10 interacts with MCP tool execution
- [[sources/code/sft-v2-pipeline]]
- [[sources/dissertation/security-privacy-social-ethics]] — source of the security analysis

## Sources

- `pipeline/constitution.md`
- `README.md`
- `docs/security-analysis/security-review.tex`
