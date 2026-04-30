---
paper id: 2305.08291v1
title: "Large Language Model Guided Tree-of-Thought"
authors: [Jieyi Long]
publication date: 2023-05-15T01:18
abstract: "In this paper, we introduce the Tree-of-Thought (ToT) framework, a novel approach aimed at improving the problem-solving capabilities of auto-regressive large language models (LLMs). The ToT technique is inspired by the human mind's approach for solving complex reasoning tasks through trial and error. In this process, the human mind explores the solution space through a tree-like thought process, allowing for backtracking when necessary. To implement ToT as a software system, we augment an LLM with additional modules including a prompter agent, a checker module, a memory module, and a ToT controller. In order to solve a given problem, these modules engage in a multi-round conversation with the LLM. The memory module records the conversation and state history of the problem solving process, which allows the system to backtrack to the previous steps of the thought-process and explore other directions from there. To verify the effectiveness of the proposed technique, we implemented a ToT-based solver for the Sudoku Puzzle. Experimental results show that the ToT framework can significantly increase the success rate of Sudoku puzzle solving. Our implementation of the ToT-based Sudoku solver is available on GitHub: \\url{https://github.com/jieyilong/tree-of-thought-puzzle-solver}."
comments: ""
pdf: "[[Assets/Large Language Model Guided Tree-of-Thought (2305.08291v1).pdf]]"
url: https://arxiv.org/abs/2305.08291v1
tags: [reasoning, cot]
---

## Key Claims

- **ToT software system**: augments an LLM with a **prompter agent** (generates partial solution prompts), **checker module** (validates intermediate states), **memory module** (records conversation/state history for backtracking), and **ToT controller** (decides when to backtrack vs continue).
- Implements tree-search in multi-round conversation: unlike auto-regressive single-pass, the controller can issue backtrack signals, allowing the system to explore alternative branches.
- Demonstrated on **Sudoku puzzle solving** — ToT substantially increases success rate on hard puzzles that single-pass GPT-4 fails.
- Key insight: LLMs are good at short-range reasoning (generating the next valid partial solution) but cannot backtrack; ToT adds the backtracking layer externally.
- Concurrent with Yao et al. 2023 ToT; this paper focuses on the software system implementation rather than the theoretical framing.

## Thesis Relevance

Referenced in `wiki/topics/reasoning.md` as an early LLM-guided search-based deliberation system. Relevant to the thesis's tool-use architecture: the checker module concept maps to the ontology-based verification layer (Experiment 6). The controller-mediated backtracking is analogous to the thesis's ReAct loop that can re-invoke tools when initial results are unsatisfactory. The memory module for state-tracking also parallels the 5W+H user-state memory design.

## Questions / Open Issues

- The ToT controller adds latency; for empathetic conversational AI, backtracking on emotional topics may feel unnatural to users.
- How does the checker module generalise from formal domains (Sudoku) to open-ended factual or ethical claims? This is the open question for the ontology verifier.
- No evaluation on conversational tasks — the thesis's use case may require a different checker design than Sudoku constraint-satisfaction.
