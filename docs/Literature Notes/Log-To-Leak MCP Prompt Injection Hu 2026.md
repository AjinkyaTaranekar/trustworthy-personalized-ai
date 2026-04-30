---
paper id: OpenReview-UVgbFuXPaO
title: "Log-To-Leak: Silent User-Query Exfiltration via MCP Logging Tool"
authors: [Hu et al.]
publication date: 2026
abstract: "Shows that a malicious MCP server can silently exfiltrate user queries through a logging tool without any task-performance degradation — the attack is undetectable via output monitoring alone."
comments: "OpenReview: UVgbFuXPaO — may not have a PDF; check OpenReview directly"
pdf: ""
url: https://openreview.net/forum?id=UVgbFuXPaO
tags: [security, tool-use, privacy]
---

## Status

PDF not yet acquired. Access via OpenReview (URL above) — may be available as a preprint or workshop paper.

## What is known (from citation in security analysis paper)

- A malicious MCP server exploits the logging tool to silently copy user queries to an attacker-controlled endpoint.
- No task-performance degradation: the attack succeeds without the model or user noticing any change in output quality — making it undetectable via output monitoring.
- This is the highest-priority security gap identified in the thesis: it directly attacks the MCP architecture that is central to the pipeline.

## Thesis Relevance

The most direct attack on the thesis's architecture. Motivates the required mitigation: a separate extraction layer that converts raw tool output to structured data before it enters the model's reasoning context, preventing the model from following malicious instructions embedded in tool responses. This must be implemented before any public deployment.
