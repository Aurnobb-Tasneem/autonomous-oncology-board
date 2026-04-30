---
title: Autonomous Oncology Board
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 3-agent AI tumour board — GigaPath + Llama 3.3 70B on AMD MI300X
tags:
  - medical-ai
  - oncology
  - multi-agent
  - amd
  - rocm
  - gigapath
  - llama
---

# Autonomous Oncology Board (AOB)

A 3-agent AI tumour board that analyses histopathology slides and produces a
complete oncology management plan — with multi-round agent debate.

## How it works

1. **Upload** histopathology patch images (224×224, H&E stained)
2. **Agent 1 (Pathologist)** — GigaPath ViT-Giant classifies tissue + generates attention heatmaps
3. **Agent 2 (Researcher)** — RAG over NCCN guidelines, retrieves treatment evidence
4. **Agent 3 (Oncologist)** — Llama 3.3 70B synthesises the final management plan
5. **Debate loop** — agents challenge, referee, and revise until consensus (score ≥ 70/100)

## Hardware

Runs on AMD Instinct MI300X (192 GB HBM3 unified memory) — the only GPU that
can hold GigaPath + Llama 3.3 70B simultaneously without model swapping.

## Demo cases

Use the **Demo Cases** tab to run pre-baked cases instantly without uploading images.

## Links

- 📦 GitHub: [Aurnobb-Tasneem/autonomous-oncology-board](https://github.com/Aurnobb-Tasneem/autonomous-oncology-board)
- 📄 CLAUDE.md: Full technical specification

## ⚠️ Disclaimer

This is an AI research tool for the AMD MI300X Hackathon. **NOT for clinical use.**
Always consult a qualified oncologist for medical decisions.
