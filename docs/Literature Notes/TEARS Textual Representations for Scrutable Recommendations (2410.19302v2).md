---
paper id: 2410.19302v2
title: "TEARS: Textual Representations for Scrutable Recommendations"
authors: [Emiliano Penaloza, Olivier Gouvert, Haolun Wu, Laurent Charlin]
publication date: 2024-10-25T04:26
abstract: "Traditional recommender systems rely on high-dimensional (latent) embeddings for modeling user-item interactions, often resulting in opaque representations that lack interpretability. Moreover, these systems offer limited control to users over their recommendations. Inspired by recent work, we introduce TExtuAl Representations for Scrutable recommendations (TEARS) to address these challenges. Instead of representing a user's interests through a latent embedding, TEARS encodes them in natural text, providing transparency and allowing users to edit them. To do so, TEARS uses a modern LLM to generate user summaries based on user preferences. We find the summaries capture user preferences uniquely. Using these summaries, we take a hybrid approach where we use an optimal transport procedure to align the summaries' representation with the learned representation of a standard VAE for collaborative filtering. We find this approach can surpass the performance of three popular VAE models while providing user-controllable recommendations. We also analyze the controllability of TEARS through three simulated user tasks to evaluate the effectiveness of a user editing its summary."
comments: ""
pdf: "[[Assets/TEARS Textual Representations for Scrutable Recommendations (2410.19302v2).pdf]]"
url: https://arxiv.org/abs/2410.19302v2
tags: [personalisation, scrutability, xai, evaluation]
---

## Key Claims

- **TEARS**: two-encoder hybrid system — a text-summary encoder (scrutable) aligned to a standard VAE collaborative-filtering encoder (high-performance) via optimal transport.
- Interpolation coefficient α: α=1 → pure text-based (fully scrutable), α=0 → pure black-box; users can dial in their own transparency-vs-performance trade-off.
- Surpasses 3 popular VAE models in recommendation accuracy while maintaining user controllability — demonstrating the transparency–performance trade-off can be partially avoided.
- **Controllability validated** through 3 simulated user tasks (large-scope edits, fine-grained edits, guided recommendations via summary tweaks).
- LLM-generated summaries (~200 words) uniquely capture individual user preferences from interaction history; GPT-4-turbo used for generation.

## Thesis Relevance

Companion paper to UPR (Ramos et al. 2024) in the UMAP scrutability tradition cited in the LLNCS paper. TEARS goes further by showing the scrutable approach can *exceed* black-box performance using the hybrid α-interpolation. Relevant to the thesis's 5W+H user model design: NL user profiles with user-editable structure follow the same principle. The α coefficient concept is a useful design pattern for the thesis's autonomy-preserving constraint.

## Questions / Open Issues

- Summary generation uses GPT-4-turbo (proprietary, cloud) — can it be replicated locally for the thesis's on-device privacy argument?
- Tested on movie and book recommendations — domain transfer to conversational AI uncertain.
- Controllability was simulated, not validated with real users — user study needed before claiming users benefit from the scrutable interface.
