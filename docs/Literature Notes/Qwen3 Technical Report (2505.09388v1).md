---
paper id: 2505.09388v1
title: Qwen3 Technical Report
authors: [An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu]
publication date: 2025-05-14T13:41
abstract: "In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0."
comments: ""
pdf: "[[Assets/Qwen3 Technical Report (2505.09388v1).pdf]]"
url: https://arxiv.org/abs/2505.09388v1
tags: [foundations, training, reasoning]
---

## Key Claims

- Qwen3 family spans **0.6B to 235B parameters** (dense and MoE), all open-source under Apache 2.0.
- Key innovation: **unified thinking/non-thinking modes** in a single model — no need to switch between a fast chat model and a dedicated reasoning model; dynamic switching via chat templates.
- **Thinking budget mechanism**: users can allocate inference compute adaptively based on task complexity — balancing latency and quality without model switching.
- Pre-trained on 36T tokens covering 119 languages; strong-to-weak distillation transfers flagship knowledge to smaller models.
- Flagship (Qwen3-235B-A22B) achieves 85.7 on AIME'24; SOtA on code, maths, agent tasks; competitive with proprietary models at much smaller activation cost (22B per token for MoE).

## Thesis Relevance

The pipeline's base model is Qwen3-0.6B (the smallest dense model in this family). The unified thinking mode is directly relevant to Experiment 1 (process-reward RL for reasoning): the same model can reason step-by-step or respond rapidly without architectural changes. The thinking budget concept aligns with the thesis's goal of efficient on-device inference. The distillation lineage (large→small) means the 0.6B model benefits from flagship capabilities despite its small size.

## Questions / Open Issues

- The 0.6B model is the smallest; what are its known performance floors on conversational and empathy tasks vs reasoning benchmarks?
- The technical report does not detail per-model sizes for safety evaluation — how does Qwen3-0.6B behave on the security/alignment dimensions of the thesis?
- Thinking budget requires output monitoring to avoid excessive chain-of-thought — integration with the thesis's TTFT (time-to-first-token) empathy requirement.
