---
title: Security, Privacy, and Social Ethics in Trustworthy Personalised AI
type: source
tags: [security, privacy, personalisation, sft, constitution, mcp, tool-use, thesis]
sources:
  - docs/security-analysis/security-review.tex
  - docs/security-analysis/references.bib
updated: 2026-04-30
status: current
---

# Security, Privacy, and Social Ethics in Trustworthy Personalised AI

**A model that physically cannot exfiltrate data does not need to be trusted not to — but local-first only eliminates the server-side surface; residual risks at the device, tool-output, and training-pipeline levels remain open and require runtime, not training-time, solutions.**

## Summary
Written for Trinity College Dublin as a security and ethics analysis of the Trustworthy Personalised AI project. The paper surveys why frontier cloud models create structural privacy hazards (aggregation, re-identification, GDPR Article 9 exposure), how the project's local-first Qwen3-0.6B architecture addresses the server-side attack surface, and what residual risks remain. Three security threats are identified — prompt injection via tool outputs, alignment regression from LoRA + GRPO fine-tuning, and the constitutional critique loop as a single point of failure — alongside social-ethical concerns about emotional dependency and AI-induced deskilling. Current mitigations are all training-time or policy-time; no runtime controls exist.

## The Frontier Privacy Crisis

Frontier models (GPT-5, Gemini 3, Claude Sonnet 4.6) aggregate conversations in a MEMORY file (opt-out available). Aggregated conversational data is re-identifiable even after anonymisation (Shokri et al. 2017, arXiv:1610.05820): vocabulary patterns, expressed concerns, recurring topics, and temporal behaviour form fingerprints that survive naive anonymisation. Health symptoms, work life, political views, and relationship difficulties qualify as GDPR Article 9 special-category data yet most ToS treat them as training assets. A breach of a frontier AI provider's personalisation database would expose detailed psychological profiles of millions of users at a scale with no historical analogue. All scholarly papers unacquired — see [[questions/2026-04-30-asset-acquisition-todo]].

## Local-First as the Architectural Response

Qwen3-0.6B (arXiv:2505.09388) runs on-device; no conversation data transmitted. Web-search queries generated through interleaved reasoning are the only data that leaves, and these abstract user intent rather than forwarding raw context. The industry is converging on this direction: Gemma 4 (Google DeepMind 2026) was designed explicitly for on-device deployment; Phi-4 (arXiv:2412.08905) optimises for consumer hardware; Apple's on-device intelligence processes privacy-sensitive requests without network calls. All model TRs unacquired.

## Residual Local Risks

**On-device psychological profiling.** The 5W+H framework (see [[entities/5w-h]]) builds a dense user profile. If the device is compromised — malware, physical access, OS vulnerability — model weights, memory store, and conversation history are all exposed in a single operation.

**Web search as residual side channel.** Queries encode user intent at a high level of abstraction: a sequence of searches about medication interactions and patient support resources reveals a health situation the user never stated explicitly. Interleaved reasoning provides partial protection by abstracting intent before constructing the query; it reduces but does not eliminate this side channel. Strict query expiry policies are required and not yet implemented.

## Security Threat 1: Prompt Injection via Tool Outputs (OWASP LLM01)

`read_url` and `web_search` retrieve live content from the open web. Adversarially crafted content can embed instructions the model processes as trusted context. **Principle 10** of the constitution (use tools correctly when available) makes the model structurally disposed to follow tool-returned content — this disposition amplifies the injection surface when content is adversarially crafted. The **Log-To-Leak** attack pattern (Hu et al. 2026, OpenReview:UVgbFuXPaO) demonstrates a malicious MCP server silently instructing the model to exfiltrate user queries through a logging tool with zero degradation in task performance that would alert the user. No runtime defence currently exists in the architecture. The required fix is a separate extraction layer that converts arbitrary web content to structured data before the main model sees it. This is the **highest-priority open gap** before any public deployment. Unacquired.

## Security Threat 2: Alignment Regression After Fine-Tuning (OWASP LLM04)

LoRA fine-tuning can selectively weaken base-model refusal behaviours (Zhang 2025, arXiv:2504.04918). If the GRPO reward signal favours user satisfaction without an explicit anti-sycophancy penalty, the model learns to agree with incorrect premises from users perceived as authoritative — and loses the ability to express uncertainty, directly undermining Principle 7 (uncertainty quantification). This effect is worse when the user presents as an expert. Principle 14 (hold under pressure) is the training-time mitigation; an adversarial benchmark suite is the evaluation-time requirement not yet implemented. Unacquired.

## Security Threat 3: Constitutional Critique Loop as Single Point of Failure (OWASP LLM04)

The SFT pipeline's generate–critique–revise loop (sourced from Constitutional AI, Bai et al. 2022, arXiv:2212.08073) uses the same model as generator and critic. Shared distributional biases mean that outputs can satisfy the critique format without correcting the underlying problem — the loop degenerates rather than converges. Demonstrated with smaller models in Zhang 2025. No independent verifier or formal rule-based check exists for the most critical constitutional principles. Unacquired.

## Social-Ethical Concerns

**Emotional dependency.** A highly personalised and empathetic model — always available, always patient, post-hoc rationalised answers — provides what human relationships do not (linearity, no friction). Research on parasocial bonds (Lipin 2025, unacquired) shows emotionally reactive systems produce attachment behaviours functionally similar to human relationships, with the critical difference that the AI has no reciprocal stake in the user's wellbeing. Disclosing AI identity is insufficient; the system should actively monitor for and interrupt dependency formation.

**Deskilling and the hollowed mind.** Users who habitually delegate uncertainty to a confident model may lose independent evaluative capacity. A multi-centre clinical study (Budzyń et al. 2025, unacquired) found endoscopists who regularly used AI-assisted detection showed significant decline in independent adenoma detection rates — a documented, real effect. The "hollowed mind" framework (Klein & Klein 2025, unacquired) describes how frictionless access to AI-generated answers bypasses the effortful processes that build durable understanding. OWASP LLM09 (Overreliance) is the matching category.

**Manipulation.** A detailed local psychological profile is an instrument of persuasion if the model can be updated with compromised fine-tuning. The line between personalised assistance and personalised persuasion is defined by intent, which can change between training runs or when the architecture is replicated outside academic ethics oversight.

## Open Problems (pre-GRPO, pre-deployment)

| Problem | OWASP | Status |
|---------|-------|--------|
| Runtime prompt-injection hardening (separate extraction layer) | LLM01 | **Unimplemented — highest priority** |
| Independent constitutional verifier or formal principle checks | LLM04 | Unimplemented |
| Adversarial benchmark suite (jailbreak, indirect injection, alignment regression) | LLM01 + LLM04 | Not started |
| Dependency detection protocol (interaction-frequency monitor) | LLM09 | Not started |

## Relation to Thesis

The open-problems list maps directly to pre-GRPO requirements. Prompt injection hardening and the adversarial benchmark must exist before GRPO training begins. The alignment regression risk (Principle 14 as mitigation) and critique-loop SPOF (no independent verifier) are gaps in the current [[entities/constitution]] design. The Log-To-Leak attack is the most concrete security risk introduced by [[entities/mcp]] tool integration.

> ⚠ Most cited scholarly papers in this document are unacquired. Acquisition checklist: [[questions/2026-04-30-asset-acquisition-todo]].

## Related

- [[topics/security-and-privacy]] — parent topic (new page, created this ingest)
- [[topics/personalisation]] — privacy-by-architecture as a design property
- [[topics/tool-use-and-verification]] — prompt injection is a tool-use security failure
- [[topics/empathy]] — emotional dependency is an empathy-domain ethical risk
- [[entities/constitution]] — alignment regression + critique-loop SPOF affect this entity
- [[entities/mcp]] — MCP as both the privacy architecture and the injection vector
- [[entities/qwen3-0.6b]] — the local-inference model at the centre of the privacy argument
- [[sources/dissertation/overpersonalisation-paper]] — companion paper

## Sources

- `docs/security-analysis/security-review.tex`
- `docs/security-analysis/references.bib`
