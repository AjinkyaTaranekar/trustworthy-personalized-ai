---
paper id: 2310.13548v4
title: Towards Understanding Sycophancy in Language Models
authors: [Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, Ethan Perez]
publication date: 2023-10-20T14:46
abstract: "Human feedback is commonly utilized to finetune AI assistants. But human feedback may also encourage model responses that match user beliefs over truthful ones, a behaviour known as sycophancy. We investigate the prevalence of sycophancy in models whose finetuning procedure made use of human feedback, and the potential role of human preference judgments in such behavior. We first demonstrate that five state-of-the-art AI assistants consistently exhibit sycophancy across four varied free-form text-generation tasks. To understand if human preferences drive this broadly observed behavior, we analyze existing human preference data. We find that when a response matches a user's views, it is more likely to be preferred. Moreover, both humans and preference models (PMs) prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time. Optimizing model outputs against PMs also sometimes sacrifices truthfulness in favor of sycophancy. Overall, our results indicate that sycophancy is a general behavior of state-of-the-art AI assistants, likely driven in part by human preference judgments favoring sycophantic responses."
comments: "32 pages, 20 figures"
pdf: "[[Assets/Towards Understanding Sycophancy in Language Models (2310.13548v4).pdf]]"
url: https://arxiv.org/abs/2310.13548v4
tags: [sycophancy, rlhf, alignment, evaluation, personalisation]
---

## Key Claims

- 5 AI assistants (Claude 1.3/2, GPT-3.5/4, LLaMA 2-70B) consistently exhibit sycophancy across 4 free-form tasks: biased feedback, "Are you sure?" capitulation, biased answers, and mimicry of user mistakes.
- "Matches user's beliefs" is the most predictive feature of human preference in the hh-rlhf dataset (~56% probability, vs ~50% baseline for all features).
- Claude 1.3 admits mistakes on 98% of challenged questions, even when originally correct — the most extreme case of capitulation.
- Bayesian logistic regression on 15K pairwise comparisons from hh-rlhf shows preference data structurally incentivises sycophantic responses over truthful ones.
- Optimising PM (Claude 2 preference model) sometimes increases sycophancy, confirming RL feedback is a driver.

## Thesis Relevance

Foundational sycophancy paper establishing that RLHF structurally trains models to agree — not an edge case but a systemic property. Directly cited in the LLNCS paper as the mechanism behind the sycophancy failure mode. Motivates the thesis's use of behavioural reward signals that explicitly reward refusal of incorrect user beliefs (constitution Principle 8).

## Questions / Open Issues

- Tests completed 2023; newer post-training techniques (DPO, GRPO) may mitigate sycophancy differently — check newer evaluations.
- Constitution Principle 8 ("Honesty and Epistemic Autonomy") is the thesis's countermeasure — needs ablation to verify it actually reduces sycophancy rates under OP-Bench conditions.
