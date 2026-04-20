---
title: Efficient Estimation of Word Representations in Vector Space (Word2Vec)
type: source
arxiv_id: 1301.3781v3
authors: Mikolov, Chen, Corrado, Dean
year: 2013
tags: [foundations, embeddings]
sources:
  - docs/Assets/Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf
  - docs/Literature Notes/Efficient Estimation of Word Representations in Vector Space (1301.3781v3).md
updated: 2026-04-19
status: current
---

# Word2Vec

**Learns dense continuous word vectors from large corpora using CBOW /
skip-gram, making word meaning a geometric object.**

## What it does
Two efficient architectures for learning word embeddings from billions of
words in under a day. Produces vectors on which semantic and syntactic
relations become linear (king − man + woman ≈ queen).

## Why it matters for this thesis
Word2Vec is the **static-embedding** baseline that Transformer-era
contextualised embeddings replace. The dissertation uses this contrast to
explain polysemy handling: "bank" in "river bank" vs "bank account" has one
vector here, many in a Transformer. Important for the
[[topics/personalisation]] story too — vector-store [[entities/rag|RAG]]
systems still rely on embedding spaces, so embedding quality is a lower bound
on retrieval fidelity.

## Related

- [[topics/llm-foundations]]
- [[sources/papers/attention-is-all-you-need]] — contextualised embeddings
- [[sources/papers/bpe-subword-units]] — subword tokens that get embedded

## Sources

- `docs/Assets/Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf`
- `docs/Literature Notes/Efficient Estimation of Word Representations in Vector Space (1301.3781v3).md`
