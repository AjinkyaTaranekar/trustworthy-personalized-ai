---
title: Constitution (SFT v2)
type: entity
tags: [sft, constitution, principles]
sources:
  - pipeline/constitution.md
  - README.md
updated: 2026-04-19
status: current
---

# Constitution (SFT v2)

**The 19-principle document that governs every gold response generated in the
SFT v2 pipeline. Covers capability honesty, tool discipline, and honest
refusal.**

## Summary
The v2 supervised-fine-tuning pipeline replaces a 42-template scenario approach
with a **constitution-driven** generator: the teacher model drafts a response,
critiques it against the 19 principles in `pipeline/constitution.md`, and
revises on violations. Only the revised response enters the training set.
Every `<think>` block in the assembled dataset must contain a
`CAPABILITY_CHECK` — asserted by the dataset assembler quality filter.

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

## Related

- [[topics/reasoning]] — constitution encodes reasoning-honesty rules
- [[topics/tool-use-and-verification]] — constitution encodes tool discipline
- [[sources/code/sft-v2-pipeline]]

## Sources

- `pipeline/constitution.md`
- `README.md`
