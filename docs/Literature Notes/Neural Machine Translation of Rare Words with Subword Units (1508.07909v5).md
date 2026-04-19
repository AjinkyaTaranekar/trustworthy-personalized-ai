---
paper id: 1508.07909v5
title: Neural Machine Translation of Rare Words with Subword Units
authors: Rico Sennrich, Barry Haddow, Alexandra Birch
publication date: 2015-08-31T16:37:31Z
abstract: Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character n-gram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English-German and English-Russian by 1.1 and 1.3 BLEU, respectively.
comments: "accepted at ACL 2016; new in this version: figure 3"
pdf: "[[Assets/Neural Machine Translation of Rare Words with Subword Units (1508.07909v5).pdf]]"
url: https://arxiv.org/abs/1508.07909v5
tags:
  - foundation
  - vectors
---
