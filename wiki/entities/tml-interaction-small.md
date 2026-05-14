---
title: TML-Interaction-Small
type: entity
tags: [multimodal, latency, empathy, evaluation]
sources:
  - https://www.trendingtopics.eu/mira-muratis-thinking-machines-lab-challenges-openai-with-real-time-response-model/
updated: 2026-05-13
status: current
---

# TML-Interaction-Small

**The first publicly benchmarked real-time multimodal model from Thinking Machines Lab (Mira Murati, former OpenAI CTO) — 276B total / 12B active parameters (MoE), 400ms end-to-end response latency, simultaneous audio/video/text processing without external control components.**

## Specs

| Property | Value |
|---|---|
| Architecture | Mixture-of-Experts, encoder-free early fusion |
| Total parameters | 276B |
| Active parameters | 12B |
| Modalities | Audio (dMel), Video (40×40 hMLP), Text |
| Response latency | 0.40s (vs GPT-Realtime-2.0 at 1.18s) |
| Processing granularity | 200ms micro-turn blocks |
| Tool use | Parallel — web search, data retrieval, UI generation |

Benchmark (FD-bench V1.5 interaction quality, Audio MultiChallenge, May 2026):

| Model | Latency | FD-bench V1.5 | Audio MultiChallenge |
|---|---|---|---|
| TML-Interaction-Small | **0.40s** | **77.8** | **43.4** |
| GPT-Realtime-2.0 (minimal) | 1.18s | 46.8 | 37.6 |
| Gemini-3.1-Flash-Live | 0.57s | 54.3 | 26.8 |

## Key design choices

**Multi-Stream Micro-Turn Design**: Rather than processing a complete utterance as a unit, the model processes in 200ms blocks. This enables simultaneous listen-and-speak, pause/interruption recognition, and visual cue detection — behaviours associated with attentive human conversation. The model delegates complex reasoning to an asynchronous background model, keeping the interaction stream responsive.

**Encoder-free early fusion**: Audio and video representations are fused with text directly into the central Transformer, without separate encoder towers. This reduces latency by eliminating a modality-encoding bottleneck.

## Relevance to this thesis

### Empathy argument
The micro-turn design operationalises something the [[topics/empathy]] chapter argues for abstractly: *responsiveness*. Pausing, acknowledging interruptions, reacting to visual cues — these are properties of empathic conversation. TML-Interaction-Small shows that at 276B scale these can be implemented; the thesis asks whether the *intent* (constitutional principles around empathy) matters more than the mechanism, even at 0.6B scale on text-only inputs.

### Scale context
The 276B/12B active MoE represents one end of the capability-deployment tradeoff spectrum. [[entities/qwen3-0.6b|Qwen3-0.6B]] is the other end. TML-Interaction-Small cannot run on-device; it necessarily runs on cloud infrastructure, with the same privacy trade-off that motivates the on-device argument in [[experiments/frontier-model-comparison]]. This is a useful framing contrast: real-time empathic AI at scale vs. trustworthy (private, local) empathic AI at 0.6B.

### Not in the frontier comparison experiment
TML-Interaction-Small is a real-time multimodal system, not a text-only chat API. It is not an appropriate comparison model for the dissertation's evaluation (which uses text prompts and a trust/empathy rubric). It belongs in the thesis as a *contextual reference* — what the frontier looks like — rather than as a direct comparator.

## Limitations (as of May 2026)

- High context volume degrades performance in extended sessions.
- Requires stable internet connection — no on-device deployment.
- Larger MoE variants are still too slow for real-time interaction.

## Related

- [[topics/empathy]] — micro-turn design operationalises empathic responsiveness
- [[entities/qwen3-0.6b]] — the opposite end of the scale/privacy tradeoff
- [[experiments/frontier-model-comparison]] — contextual reference, not a direct comparator
- [[topics/security-and-privacy]] — cloud-only deployment implies the same data-exfiltration risk the on-device argument addresses

## Sources

- https://www.trendingtopics.eu/mira-muratis-thinking-machines-lab-challenges-openai-with-real-time-response-model/ (news article, May 2026; no technical report available)
