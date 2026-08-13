# Constitutional Distillation for Trustworthy, Personalised AI on Small Language Models

MSc dissertation and research pipeline — Ajinkya Taranekar, supervised by Owen Conlan, School of Computer Science and Statistics, Trinity College Dublin.

Full reproduction runbook: **[`pipeline/pipeline.md`](pipeline/pipeline.md)**.

---

## Motivation

An AI assistant that runs on the user's own device keeps private data off the internet. That matters more than it used to: people increasingly bring health, financial and personal questions to an assistant rather than a search engine, and where a search query captures the *what*, *when* and *where* of an event, the conversation captures the *why* and *how* of the thinking behind it. That record is the special-category data GDPR Article 9 protects and the EU AI Act places in its high-risk tier.

Keeping it under the user's control is an architectural problem, not a policy one. Cloud models require the data to leave the device, and the usual safeguards govern what a provider stores rather than what a model absorbs and can later repeat under targeted querying. Federated learning removes the transmission but not the residue, since the trained weights still carry each user's influence. Running the model on the device removes both — but it also forces the model to be small, and a small model may not have the capacity to hold safety, reasoning and tool use at the same time. Apple ships a 3B on-device Foundation Model and Google phone-class Gemma 4 variants, though both need recent hardware. To reach the older and cheaper handsets those leave behind, this work uses a 0.6B model and asks whether a framework can still make it trustworthy.

## Research question

> At 0.6 billion parameters, what does teaching a written constitution into a model's weights buy, what does it cost, and can the cost be recovered by changing the architecture rather than the model size?

Five hypotheses are tested: (H1) constitutional distillation improves compliance even at this scale; (H2) the same distillation degrades reasoning — an alignment tax; (H3) a carefully engineered prompt with tools is less compliant than a constitution encoded in weights; (H4) where the trained model fails, it fails systematically rather than at random; (H5) splitting reasoning from execution across two on-device models recovers capability a single small model cannot reach.

## Experiments

The same base model (`unsloth/Qwen3-0.6B`) is built five ways and every condition answers the same fixed question set, graded by a large independent judge against a 25-principle constitution.

| Condition | What it is |
|---|---|
| `vanilla_base` | Base weights, tools off — the floor |
| `vanilla_tools` | Base weights, careful prompt, live tools — the prompt-engineering rival (H3) |
| `sft_template` | **Experiment 1** — template SFT: format only, no constitutional content |
| `sft_constitution` | **Experiment 2** — constitutional distillation from a frontier teacher, same weights and recipe as Exp 1; only the data differs |
| `thinker_executor` | **Experiment 3** — two 0.6B models: a Thinker that reasons and an Executor that calls tools |

Training is 16-bit LoRA throughout. GRPO and other RL post-training are deliberately out of scope: RL is empirically unstable below 1B parameters.

## What it found

Replacing a format-only corpus with a constitutional one, over identical weights and recipe, raises the head-to-head score by 0.230 — the data decides the outcome. That model does not beat an untrained one given a careful prompt and real tools on average (0.589 vs 0.583), but each is strong where the other is weak. The gain is paid for: empty-reasoning rate goes from 1.4% to 91.3% and mean trace length from 1,309 to 150 characters. Splitting reasoning from execution wins part of that back (36.2% empty traces) but does not match single-model compliance and is weakest over long conversations.

Full tables and figures: `pipeline/reports/` and `docs/Constitutional_AI_in_SLM/`.

## Repository layout

```
pipeline/                Training, inference, benchmarking, analysis
├── pipeline.md          → the end-to-end runbook (start here)
├── constitution.md      The 25 principles (P1–P25)
├── 1_dataset_generator.py … 5_judgement_day.py    Numbered pipeline stages
├── sft_*.py             Corpus generation, assembly, quality gates
├── thinker_executor_orchestrator.py               Experiment 3 two-model loop
├── analyze_experiments.py, *_figures.py           Tables + dissertation assets
├── data/  models/  reports/                       Datasets, checkpoints, results
docs/
├── Constitutional_AI_in_SLM/   Current dissertation (LaTeX)
├── Assets/                     Source PDFs
├── Literature Notes/           Per-paper notes
wiki/                    Obsidian research vault — index.md is the catalog
scripts/                 Repo maintenance utilities
```

## Quickstart

The whole study fits on one consumer GPU (16 GB is enough; roughly $10 of rented compute end to end). Generation needs the GPU; judging and analysis are API/CPU only.

```bash
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai
pip install -r pipeline/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" peft

cp pipeline/.env.example pipeline/.env    # then fill in the keys below
```

`pipeline/.env` needs:

| Variable | Used for |
|---|---|
| `HF_TOKEN` | Publishing checkpoints (optional — benchmarking works from local dirs) |
| `EXA_API_KEY` | Live `web_search` / `read_url` at inference |
| `NVIDIA_NIM_API_KEYS` | LLM judge (comma-separated; rotated automatically) |

Then follow [`pipeline/pipeline.md`](pipeline/pipeline.md), which walks through the seven stages in order: build the corpora → four SFT runs → benchmark five conditions → judge → consolidate tables and figures → comparative ranking.

Note that `3_infererence.py` is a server: start it first, and every other script talks to it over HTTP, so the model is never reloaded between runs.

## Published checkpoints

- Experiment 1: [`trustworthy-ai-sft-template`](https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-sft-template)
- Experiment 2: [`trustworthy-ai-sft-constitution`](https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-sft-constitution)
- Experiment 3: [`trustworthy-ai-thinker`](https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-thinker) + [`trustworthy-ai-executor`](https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-executor)

## Reproducibility

Everything is deterministic under greedy decoding, with one exception: web search is live, so the handful of web-grounded probes are not byte-reproducible. Set `BENCH_MOCK_SEARCH=1` on the server for a fixed offline corpus if exact reproduction is needed. Generation and judging are kept separate on purpose — pay for the GPU once, then judge and re-judge on CPU for free.

The training corpus is synthetic, with target answers produced by a larger model rather than validated by people. This is a stated assumption of the work, not an implementation detail.
