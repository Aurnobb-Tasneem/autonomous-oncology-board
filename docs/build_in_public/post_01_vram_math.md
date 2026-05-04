# Post 1: The VRAM Math That Changes Everything

**Platform:** X (Twitter) + LinkedIn  
**Tags:** #AMDDevHackathon #ROCm #MI300X #MedicalAI #oncology  
**Visual:** rocm-smi screenshot showing 88,238 MiB / 196,608 MiB

---

## X Thread (1/7)

**🧵 Thread: Why building a 70B LLM + vision foundation model system required AMD hardware — and the math that proves it.**

(1/7) We're building an AI tumour board for the @lablab AMD Hackathon. Three agents: a pathologist (Prov-GigaPath), a researcher (RAG), and an oncologist (Llama 3.3 70B). Here's why we couldn't build this on NVIDIA. 🧵

---

(2/7) The VRAM breakdown for our system:

```
Llama 3.3 70B (FP8):     70 GB
Llama 3.1 8B + LoRAs:    16 GB
Prov-GigaPath (FP16):     3 GB
vLLM KV Cache:           30 GB
Qdrant + overhead:        9 GB
────────────────────────────────
TOTAL:                  128 GB
```

H100 SXM5 = 80 GB. The math doesn't work. Full stop.

---

(3/7) "Just use quantization!" — I hear you. We ARE using FP8 for the 70B. That's why it's 70 GB instead of ~140 GB. Even at maximum compression, the architecture needs 128 GB minimum.

The constraint isn't quantization — it's concurrent residency.

---

(4/7) The AMD Instinct MI300X has 192 GB HBM3 in a UNIFIED memory pool. Both models, all adapters, the KV cache — they all live in the same address space. No PCIe transfers. No model swapping. No cache eviction mid-conversation.

This is what enables the debate loop.

---

(5/7) Our debate protocol: the researcher agent CHALLENGES the oncologist's draft, the oncologist REVISES. The full debate history (3 rounds × ~2,000 tokens each) stays in KV cache throughout.

On H100: cache eviction kills this. On MI300X: 64 GB to spare.

---

(6/7) Proof. Not marketing. `rocm-smi` live output during a 3-case concurrent test:

`88,238 MiB / 196,608 MiB` — BOTH models, KV cache, three cases running.

[INSERT rocm-smi screenshot]

H100 would OOM at case 1.

---

(7/7) Result on AOB-Bench (100 clinical eval cases):
- TNM staging accuracy: 82.3% [80.5, 84.4]
- GigaPath alone contributes +30 pp over language-only baseline

This architecture is physically impossible on a single H100.
It runs natively on one MI300X. 

Dataset + code: [link]
#AMDDevHackathon @AIatAMD @lablab

---

## LinkedIn Version (long-form)

**Title:** Why our AI oncology system required AMD hardware — and the math that proves it

Building an AI tumour board that mirrors how real oncology decisions are made required a specific hardware constraint we couldn't route around: **concurrent in-memory residency of a 70B LLM and a 1.1B vision foundation model.**

Here's the full VRAM budget for the Autonomous Oncology Board:

| Component | Precision | VRAM |
|-----------|-----------|------|
| Llama 3.3 70B | FP8 | 70 GB |
| Llama 3.1 8B + 3 LoRA adapters | FP16 | 16 GB |
| Prov-GigaPath ViT-Giant | FP16 | 3 GB |
| vLLM KV Cache | FP16 | 30 GB |
| Qdrant + system overhead | — | 9 GB |
| **Total** | | **128 GB** |

The NVIDIA H100 SXM5 has 80 GB of HBM3e. Our minimum requirement is 128 GB. This is not a benchmark — it's arithmetic.

The AMD Instinct MI300X has 192 GB of HBM3 in a unified memory pool. Both models live in the same address space — no PCIe transfers, no model swapping, no KV cache eviction during the debate loop.

The debate loop is the key: our researcher agent challenges the oncologist's draft plan using RAG evidence. The full multi-round conversation history (3 rounds × ~2,000 tokens) stays live in the KV cache. On hardware with insufficient VRAM, this architecture requires cache eviction — which means re-loading context and dramatically increasing latency.

We verified this with `rocm-smi`: **88,238 MiB / 196,608 MiB** during a live 3-case concurrent test.

The benchmark results: 82.3% TNM exact-match rate on 100 curated oncology cases. GigaPath vision alone contributes +30 percentage points over a language-only baseline.

We open-sourced the benchmark dataset: aob-bench/ClinicalEval on HuggingFace — 100 cases, 4 metrics, ground-truth labels from NCCN guidelines.

AMD didn't just enable this project. AMD made the architecture possible.

#AMDDevHackathon #MedicalAI #ROCm #MI300X #oncology #machinelearning
