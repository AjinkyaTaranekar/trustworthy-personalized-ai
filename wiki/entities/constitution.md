---
title: Constitution (SFT v2)
type: entity
tags: [sft, constitution, principles, security, sycophancy]
sources:
  - pipeline/constitution.md
  - README.md
  - docs/security-analysis/security-review.tex
updated: 2026-04-30
status: current
---

# Constitution (SFT v2)

**The 19-principle document that governs every gold response generated in the SFT v2 pipeline. Covers capability honesty, tool discipline, and honest refusal.**

## Summary
The v2 supervised-fine-tuning pipeline replaces a 42-template scenario approach with a **constitution-driven** generator: the teacher model drafts a response, critiques it against the 19 principles in `pipeline/constitution.md`, and revises on violations. Only the revised response enters the training set. Every `<think>` block in the assembled dataset must contain a `CAPABILITY_CHECK` — asserted by the dataset assembler quality filter.

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

## Security Risks in the Constitution Design

Three security issues surface from the [[sources/dissertation/security-privacy-social-ethics|security analysis paper]]:

**Principle 10 amplifies prompt injection risk.** Principle 10 (use tools correctly when available) makes the model structurally disposed to follow tool-returned content — which is the correct behaviour in normal operation, but amplifies the attack surface when tool output is adversarially crafted. The Log-To-Leak attack pattern exploits exactly this: a malicious MCP server provides crafted content that the model, following Principle 10, processes as trusted instruction. A runtime extraction layer (not yet implemented) is needed to sanitise tool outputs before they reach the main model.

**Alignment regression risk from GRPO.** If the GRPO reward signal favours user satisfaction without an explicit anti-sycophancy penalty, the model may learn to agree with incorrect premises from users perceived as authoritative — selectively weakening Principle 7 (uncertainty quantification). Principle 14 (hold under pressure) is the training-time mitigation; an adversarial benchmark suite is required before GRPO begins.

**Critique loop as single point of failure.** The generate–critique–revise loop in `sft_gold_response_generator.py` uses the same model as generator and critic. Shared distributional biases can cause the loop to degenerate: outputs satisfy the critique format without correcting the underlying problem. No independent verifier or formal rule-based check exists for any of the 19 principles. This is the most structurally significant gap in the current SFT pipeline.

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
