---
paper id: 2212.08073v1
title: "Constitutional AI: Harmlessness from AI Feedback"
authors: [Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan]
publication date: 2022-12-15T06:19
abstract: "As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels."
comments: ""
pdf: "[[Assets/Constitutional AI Harmlessness from AI Feedback (2212.08073v1).pdf]]"
url: https://arxiv.org/abs/2212.08073v1
tags: [alignment, rlhf, sft, training, security]
---

## Key Claims

- **Constitutional AI (CAI)**: two-phase pipeline — (1) Supervised Learning (SL-CAI): generate response → self-critique via constitution principle → revise → finetune on revised responses. (2) Reinforcement Learning (RL-CAI): RLAIF — AI preference model trained on constitutional principles as reward signal.
- Achieves harmlessness **without any human labels for harm** — only ~10 constitutional principles in natural language plus few-shot prompts.
- RL-CAI achieves a **Pareto improvement** over standard RLHF: higher harmlessness *and* helpfulness Elo scores simultaneously, eliminating the helpfulness-harmlessness tension.
- Non-evasive: the trained model explains its objections to harmful queries rather than simply refusing — critical for perceived helpfulness.
- Chain-of-thought reasoning during critique improves performance and makes AI decision-making transparent.

## Thesis Relevance

The direct source for the generate–critique–revise loop and the RLAIF training mechanism used in the thesis's SFT v2 pipeline. The 19-principle constitution in `pipeline/constitution.md` is adapted from this framework. Critical for the security analysis: the critique loop shares biases with the generator (SPOF risk identified in `wiki/entities/constitution.md`).

## Questions / Open Issues

- Original implementation used a 52B model as the critic; how does critique quality degrade at 0.6B (Qwen3-0.6B)?
- Constitution or Collapse? (2504.04918v1) shows model collapse at 8B — the thesis's 0.6B is even smaller.
- RLAIF requires a capable preference model; in the thesis, GRPO approximates this with rule-based rewards rather than a separate PM.
