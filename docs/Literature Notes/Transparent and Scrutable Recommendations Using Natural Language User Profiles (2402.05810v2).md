---
paper id: 2402.05810v2
title: Transparent and Scrutable Recommendations Using Natural Language User Profiles
authors: [Jerome Ramos, Hossen A. Rahmani, Xi Wang, Xiao Fu, Aldo Lipani]
publication date: 2024-02-08T16:47
abstract: "Recent state-of-the-art recommender systems predominantly rely on either implicit or explicit feedback from users to suggest new items. While effective in recommending novel options, many recommender systems often use uninterpretable embeddings to represent user preferences. This lack of transparency not only limits user understanding of why certain items are suggested but also reduces the user's ability to scrutinize and modify their preferences, thereby affecting their ability to receive a list of preferred recommendations. Given the recent advances in Large Language Models (LLMs), we investigate how a properly crafted prompt can be used to summarize a user's preferences from past reviews and recommend items based only on language-based preferences. In particular, we study how LLMs can be prompted to generate a natural language (NL) user profile that holistically describe a user's preferences. These NL profiles can then be leveraged to fine-tune a LLM using only NL profiles to make transparent and scrutable recommendations. Furthermore, we validate the scrutability of our user profile-based recommender by investigating the impact on recommendation changes after editing NL user profiles. According to our evaluations of the model's rating prediction performance on two benchmarking rating prediction datasets, we observe that this novel approach maintains a performance level on par with established recommender systems in a warm-start setting. With a systematic analysis into the effect of updating user profiles and system prompts, we show the advantage of our approach in easier adjustment of user preferences and a greater autonomy over users' received recommendations."
comments: ACL 2024 (main)
pdf: "[[Assets/Transparent and Scrutable Recommendations Using Natural Language User Profiles (2402.05810v2).pdf]]"
url: https://arxiv.org/abs/2402.05810v2
tags: [personalisation, scrutability, xai, evaluation]
---

## Key Claims

- **User Profile Recommendation (UPR)**: LLM generates a natural-language (NL) summary of user preferences from past reviews; this NL profile is then used as input to a fine-tuned LLM recommender.
- NL profiles achieve **competitive recommendation accuracy** to embedding-based collaborative filtering (UserKNN-BM25, Matrix Factorisation, NeuMF) in a warm-start setting on Amazon Movies/TV and TripAdvisor.
- **Scrutability validated**: editing NL user profiles leads to measurable and expected changes in recommendations — users have real control, not cosmetic transparency.
- All inputs are natural language — no user/item identifiers — so users can inspect and adjust the model's representation of them directly.
- Concurrent with Sanner et al. 2023 (cold-start NL profiles); UPR extends to warm-start and validates editability.

## Thesis Relevance

One of the UMAP scrutability tradition papers cited in the LLNCS paper. Demonstrates that NL user profiles are not just interpretable but also *actionable* — users can edit them and the system responds correctly. Directly supports the thesis's scrutable-user-model design goal and shows it is achievable without sacrificing personalisation performance. Authored by Ramos et al. (UCL/Sheffield, referenced as "Jeromela et al." in the LLNCS paper).

## Questions / Open Issues

- Warm-start requires past interaction history — does the approach extend to cold-start (new users), which is a core thesis scenario?
- Tested on recommender systems (movies, hotels) not conversational AI — transferability to dialogue-based personalisation is an open question.
- Profile generation was done by LLM prompting, not by the user themselves — how does user trust change when the profile is auto-generated vs self-authored?
