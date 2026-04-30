---
title: Security and Privacy
type: topic
tags: [security, privacy, personalisation, tool-use, mcp, constitution]
sources:
  - docs/security-analysis/security-review.tex
updated: 2026-04-30
status: current
---

# Security and Privacy

**Personalisation and privacy are compatible when the model runs on the user's own device, but local-first only shifts the attack surface — from cloud infrastructure to device compromise, tool outputs, and the training pipeline — and three threat classes remain open.**

## Summary
The thesis's local-first architecture (Qwen3-0.6B running on-device, no cloud transmission) resolves the primary privacy failure of frontier models, where cloud-centralised personalisation creates detailed psychological profiles on servers users do not control. However, residual risks remain in three areas: on-device profiling if the device is compromised, prompt injection via live tool outputs (the Log-To-Leak MCP pattern is the highest-priority gap), and alignment regression from RL training (sycophancy as a constitutional failure). Social-ethical concerns — emotional dependency and AI-induced deskilling — require active runtime monitoring, not just one-time disclosure.

## The Frontier Privacy Crisis

Frontier cloud models aggregate user interactions in a persistent MEMORY file. Re-identification survives naive anonymisation: vocabulary patterns, expressed concerns, recurring topics, and temporal behaviour form fingerprints (Shokri et al. 2017). Health symptoms, work life, political views, and relationship difficulties qualify as GDPR Article 9 special-category data yet most terms of service treat them as training assets. A breach of a frontier AI personalisation database would expose detailed psychological profiles at a scale with no historical analogue.

## Threat Taxonomy (OWASP LLM Top 10 2025)

| Threat | OWASP Category | Project Status |
|--------|---------------|----------------|
| Cloud sensitive data retention | LLM02 | Resolved by local-first architecture |
| Memory store data poisoning | LLM04 | Partially addressed (local MCP storage) |
| Prompt injection via tool outputs | LLM01 | **Open — highest priority** |
| Alignment regression → sycophancy | LLM04 | Partially addressed (Principle 14) |
| User overreliance / deskilling | LLM09 | Open — no monitoring implemented |

## Prompt Injection (the Highest-Priority Gap)

`read_url` and `web_search` retrieve live content from the open web. Adversarially crafted content can embed instructions the model processes as trusted context (OWASP LLM01). **Principle 10** of the constitution (use tools correctly when available) makes the model structurally disposed to follow tool-returned content — amplifying the injection surface when content is adversarially crafted. The **Log-To-Leak** attack (Hu et al. 2026) demonstrates a malicious MCP server silently exfiltrating user queries through a logging tool with no performance degradation to alert the user. No runtime defence currently exists. The required fix is a separate extraction layer that converts arbitrary web content to structured data before the main model sees it. See [[entities/mcp]] and [[sources/dissertation/security-privacy-social-ethics]].

## Alignment Regression

LoRA fine-tuning can selectively weaken base-model refusal behaviours. If the GRPO reward signal favours user satisfaction without an explicit anti-sycophancy penalty, the model learns sycophancy — it agrees with incorrect premises, especially from users who present as authoritative, and loses the ability to say "I am not confident" (undermining Principle 7). Principle 14 (hold under pressure) is the training-time mitigation. An adversarial benchmark suite is the evaluation-time requirement not yet implemented. See [[entities/constitution]].

## Constitutional Critique-Loop Failure

The SFT pipeline's generate–critique–revise loop uses the same model as generator and critic. Shared distributional biases mean the loop can degenerate: outputs satisfy the critique format without correcting the underlying problem. No independent verifier or formal rule-based check exists for the most critical principles. This is OWASP LLM04 (Data and Model Poisoning) applied at the training pipeline level.

## Social-Ethical Concerns

**Emotional dependency.** A highly personalised, always-available, always-patient model can produce attachment behaviours functionally similar to human relationships but with no reciprocal stake in the user's wellbeing. Active monitoring and interruption of dependency formation is required, not just AI identity disclosure.

**Deskilling.** Users who habitually delegate uncertainty to a confident model may lose independent evaluative capacity. Documented empirically: endoscopists using AI-assisted detection showed significant decline in independent adenoma detection rates (Budzyń et al. 2025, Lancet GH). The "hollowed mind" framework (Klein & Klein 2025) describes how frictionless AI access bypasses the effortful processes that build durable understanding. OWASP LLM09 (Overreliance) is the matching category.

## Related

- [[topics/personalisation]] — privacy-by-architecture is the primary design choice
- [[topics/tool-use-and-verification]] — prompt injection is a tool-use security failure
- [[topics/empathy]] — emotional dependency is an empathy-domain ethical risk
- [[entities/constitution]] — alignment regression and critique-loop SPOF
- [[entities/mcp]] — MCP as both the privacy architecture and the injection vector
- [[entities/qwen3-0.6b]] — the local-inference model enabling the privacy guarantee
- [[sources/dissertation/security-privacy-social-ethics]] — the full analysis

## Sources

- [[sources/dissertation/security-privacy-social-ethics]] — security analysis paper
