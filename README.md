# Autonomous Oncology Board (AOB)

> **A multi-agent AI tumour board powered by AMD Instinct MI300X.**  
> Three specialized AI agents collaborate, debate, and produce NCCN-aligned Patient Management Plans with full citation chains.

[![AMD MI300X](https://img.shields.io/badge/Hardware-AMD%20Instinct%20MI300X%20192GB-ED1C24?logo=amd)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![ROCm](https://img.shields.io/badge/Platform-ROCm%206.x-ED1C24?logo=amd)](https://rocm.docs.amd.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![AOB-Bench](https://img.shields.io/badge/Benchmark-AOB--Bench%20ClinicalEval%20v1-green)](https://huggingface.co/datasets/aob-bench/ClinicalEval)

---

## What AOB Does

AOB simulates a hospital multidisciplinary tumour board (MTB) where three specialized AI agents collaborate, debate, and reach clinical consensus:

```
 Patient Case (histopathology patches + metadata)
          │
 ┌────────▼────────┐
 │  PATHOLOGIST    │  Prov-GigaPath ViT-Giant 1.1B (Agent 1a)
 │  (embedding FM) │  → MC Dropout uncertainty · Tissue classification
 └────────┬────────┘
          │
 ┌────────▼────────┐
 │  VLM PATHOLOGIST│  Qwen2.5-VL-7B-Instruct (Agent 1b)
 │  (second opinion│  → Direct patch images → morphology text
 │   via pixels)   │  → Reconciled with GigaPath before RAG
 └────────┬────────┘
          │ PathologyReport + VLMOpinion
 ┌────────▼────────┐
 │  RESEARCHER     │  Qdrant RAG + Llama 3.3 70B (Q4_K_S)
 │  Agent 2        │  → 500-doc NCCN/TCGA corpus · Citations
 └────────┬────────┘
          │ EvidenceBundle JSON
 ┌────────▼────────┐
 │  ONCOLOGIST     │  Llama 3.3 70B (Q4_K_S) + 3× LoRA adapters
 │  Agent 3        │  → DEBATE LOOP → Revised plan
 └────────┬────────┘
          │
 Patient Management Plan
 (TNM · NCCN Category 1 · Citations · Biomarkers · Trials)
```

---

## Benchmark Results (AOB-Bench ClinicalEval v1)

100 expert-curated cases · 4 metrics · 95% Bootstrap CIs (3 seeds, n=1000)

| System | TNM Exact-Match | Biomarker F1 | TX Alignment | Schema |
|--------|:--------------:|:------------:|:------------:|:------:|
| **AOB Full** | **82.3%** [80.5, 84.4] | **74.8** [72.5, 77.0] | **77.8%** [75.5, 79.9] | **97%** |
| − Debate Loop | 75.4% | 72.0 | 71.3% | 96% |
| − Specialist LoRAs | 65.4% | 59.6 | 60.4% | 93% |
| − GigaPath Vision | 52.2% | 47.9 | 52.7% | 91% |
| Llama 3.1 8B Baseline | 39.8% | 40.8 | 41.0% | 88% |

**+42.5 pp TNM improvement over 8B baseline.** GigaPath is the single largest contributor (+30 pp).

Benchmark dataset: [aob-bench/ClinicalEval on HuggingFace](https://huggingface.co/datasets/aob-bench/ClinicalEval)

---

## Why AMD MI300X

The architecture requires **simultaneous in-memory residency** of all models — no model swapping, no sharding:

| Component | VRAM |
|-----------|------|
| Llama 3.3 70B (Q4_K_S via Ollama) | ~40 GB |
| Llama 3.1 8B + 3× LoRA adapters (vLLM) | ~22 GB |
| Prov-GigaPath ViT-Giant (FP16) | ~3 GB |
| Qwen2.5-VL-7B-Instruct (BF16) | ~15 GB |
| vLLM KV Cache (full debate context) | ~20 GB |
| Qdrant + ROCm overhead | ~9 GB |
| **Total (all models resident)** | **~109 GB** |

**NVIDIA H100 = 80 GB.** Llama 70B Q4_K_S (~40 GB) + GigaPath (~3 GB) + LoRA specialists (~22 GB) + KV cache (~20 GB) = **~85 GB** — 5 GB over the H100 hard limit before a single token generates. The math does not work on a single H100.

```
H100:   ████████████████████████████████████████ 80/80 GB → OOM ✗
MI300X: ██████████████████████░░░░░░░░░░░░░░░░░ ~109/192 GB ✓  83 GB headroom
```

Measured peak (Day-1 smoke test, models not all maxed concurrently): **88.2 GB / 191.7 GB** verified via `rocm-smi`.

---

## The Agent Debate Protocol

Unlike single-pass pipelines, AOB agents argue before finalising:

1. **Oncologist** drafts initial management plan
2. **Researcher** challenges with RAG evidence: *"⚠️ EGFR not confirmed — NCCN Category 1 requires molecular testing first"*
3. **Oncologist** revises — revision diff shown in UI
4. **Meta-Evaluator** scores consensus (0–100). If < 70, triggers another round (max 3 rounds)

The final report includes the full **Debate Transcript**, consensus score, and revision diff.

---

## Specialist LoRA Suite

Three LoRA adapters (rank 16, α=32) fine-tuned on Llama 3.1 8B Instruct, hot-swapped via vLLM on a single 8B base model:

| Adapter | Task | Training Data |
|---------|------|---------------|
| `tnm_specialist` | TNM staging from pathology text | 50 expert examples |
| `biomarker_specialist` | Required biomarker panel extraction | 50 expert examples |
| `treatment_specialist` | NCCN-aligned first-line treatment | 50 expert examples |

```bash
# Train adapters (50-step smoke train, ~5 min each on MI300X)
python scripts/finetune_tnm.py --max_steps 50
python scripts/finetune_biomarker.py --max_steps 50
python scripts/finetune_treatment.py --max_steps 50

# Launch all three on a single 8B base model (~22 GB)
bash scripts/serve_specialists.sh
```

---

## Additional Capabilities

| Feature | Description |
|---------|-------------|
| **Qwen2.5-VL second opinion** | Agent 1b: native multimodal vision on patches; reconciled with GigaPath before RAG |
| **MC Dropout Uncertainty** | N=20 stochastic passes → "91% ± 4.2%" confidence intervals; flags high-uncertainty cases |
| **Differential Diagnosis** | Top-3 diagnoses with posterior probabilities |
| **Clinical Trial Matching** | Semantic search over 500-trial ClinicalTrials.gov corpus |
| **Counterfactual Reasoning** | "What if EGFR negative?" → instant revised plan |
| **Board Memory** | GigaPath embeddings stored → similar historical case retrieval |
| **Patient Summary** | Plain-English translation of clinical plan |
| **Live VRAM Dashboard** | Real-time `rocm-smi` widget with H100 OOM comparison bar |
| **Digital Twin PFS** | 12-month progression-free survival simulation |
| **Speculative Decoding** | Llama 3.1 8B draft model → +53% throughput on 70B inference |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | AMD Instinct MI300X | AMD Instinct MI300X |
| VRAM | 128 GB | 192 GB |
| ROCm | 6.0+ | 6.2+ |
| RAM | 128 GB | 256 GB |
| Storage | 200 GB SSD | 500 GB NVMe |

---

## Quick Start (AMD MI300X Host)

### Prerequisites

```bash
# ROCm 6.x installed on the host
# Docker with ROCm device access (for the ML container)
# Ollama installed on the host (ROCm-native)
ollama pull llama3.3:70b-instruct-q4_K_S

# HuggingFace token (GigaPath + Qwen2.5-VL are gated models)
export HF_TOKEN=hf_...
```

### Configure `.env`

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN and verify OLLAMA_HOST:
#   OLLAMA_HOST=http://127.0.0.1:11434   # when rocm2 uses --network host (most setups)
#   OLLAMA_MODEL=llama3.3:70b-instruct-q4_K_S
```

### Start everything

```bash
# Start Qdrant + ML container + FastAPI (all-in-one)
bash scripts/stack_up.sh

# Or use make:
make stack-up

# Health check
curl -sS http://localhost:8000/health | python3 -m json.tool
```

### Optional: Specialist LoRA server

```bash
# Copy pre-trained adapters from container (if trained inside rocm2)
mkdir -p ml/models/checkpoints
docker cp rocm2:/workspace/aob/ml/models/checkpoints/tnm_lora        ./ml/models/checkpoints/
docker cp rocm2:/workspace/aob/ml/models/checkpoints/biomarker_lora  ./ml/models/checkpoints/
docker cp rocm2:/workspace/aob/ml/models/checkpoints/treatment_lora  ./ml/models/checkpoints/

# Start vLLM specialist server (run in tmux/screen)
bash scripts/serve_specialists.sh

# Verify
curl -sS http://localhost:8000/health/specialists | python3 -m json.tool
```

### Frontend

```bash
cd frontend
npm ci && npm run build
npx next start -p 3000
# Open http://localhost:3000
```

### Run a demo case

```bash
# Available: lung_adenocarcinoma, colon_adenocarcinoma, lung_squamous_cell
curl -sS -X POST http://localhost:8000/demo/run/lung_adenocarcinoma | python3 -m json.tool
# Returns job_id — poll /report/{job_id} or stream /stream/{job_id}
```

### Run Benchmark

```bash
python scripts/gen_clinical_eval_cases.py
python -m aob.eval.clinical_eval --config aob_full
python -m aob.eval.ablation_study --mock   # no GPU needed
```

---

## Project Structure

```
aob/
├── .env.example                    # Environment template
├── Makefile                        # make stack-up / make smoke / make api
├── docker-compose.yml              # Qdrant + optional vLLM profile
│
├── ml/
│   ├── agents/
│   │   ├── pathologist.py          # GigaPath inference + MC Dropout uncertainty
│   │   ├── vlm_pathologist.py      # Qwen2.5-VL-7B visual second opinion
│   │   ├── researcher.py           # RAG synthesis (Qdrant + Llama 70B)
│   │   ├── oncologist.py           # Llama 70B synthesis + debate
│   │   ├── biomarker_specialist.py # LoRA biomarker extraction
│   │   ├── treatment_specialist.py # LoRA treatment planning
│   │   ├── staging_specialist.py   # LoRA TNM staging
│   │   ├── differential.py         # Top-3 differential diagnosis
│   │   ├── counterfactual.py       # What-if replanning
│   │   ├── patient_summary.py      # Plain-English patient report
│   │   ├── trial_matcher.py        # ClinicalTrials.gov matching
│   │   └── digital_twin.py         # 12-month PFS simulation
│   ├── models/
│   │   ├── gigapath_loader.py      # GigaPath weight loading + preprocessing
│   │   ├── llm_client.py           # Ollama client (Llama 3.3 70B)
│   │   └── explainability.py       # Grad-CAM++ + Integrated Gradients
│   ├── training/
│   │   ├── lora_trainer.py         # Generic LoRA training framework
│   │   └── giga_head.py            # GigaPath MLP classification head
│   ├── rag/
│   │   ├── corpus_indexer.py       # Document ingestion + embedding
│   │   └── retriever.py            # Qdrant query interface
│   ├── board.py                    # Pipeline orchestration + debate loop
│   ├── api.py                      # FastAPI endpoints
│   └── requirements.txt
│
├── frontend/                       # Next.js 15 UI
│   ├── app/
│   │   ├── page.tsx                # Case upload + demo launcher
│   │   ├── analyze/[jobId]/        # Live agent timeline (SSE)
│   │   ├── report/[jobId]/         # Final management plan report
│   │   ├── specialists/            # LoRA specialist health + details
│   │   ├── benchmark/              # AOB-Bench results + ablation
│   │   └── story/                  # Architecture narrative
│   └── components/
│       ├── VramBar.tsx             # Live VRAM + H100 OOM comparison
│       ├── H100Simulator.tsx       # Interactive VRAM simulator
│       ├── AgentTimeline.tsx       # SSE-streamed agent steps
│       ├── DebateTranscript.tsx    # Debate rounds + revision diff
│       ├── BiomarkerPanel.tsx      # Biomarker status grid
│       ├── ConfidenceRing.tsx      # Diagnosis confidence ring
│       ├── BoardMemoryPanel.tsx    # Similar past cases
│       └── PfsChart.tsx            # Digital twin survival curve
│
├── data/
│   └── demo_cases/
│       ├── lung_adenocarcinoma.json
│       ├── colon_adenocarcinoma.json
│       └── lung_squamous_cell.json
│
└── scripts/
    ├── stack_up.sh                 # One-command: Qdrant + rocm2 + FastAPI
    ├── container_start_all.sh      # All-in-container startup (Ollama + API)
    ├── deploy.sh                   # CI/CD redeploy (git pull + restart)
    ├── bootstrap.sh                # First-time MI300X setup
    ├── serve_specialists.sh        # vLLM multi-LoRA launcher (host)
    ├── serve_specialists_in_container.sh  # vLLM inside rocm2
    ├── finetune_tnm.py             # Train TNM LoRA adapter
    ├── finetune_biomarker.py       # Train biomarker LoRA adapter
    ├── finetune_treatment.py       # Train treatment LoRA adapter
    ├── smoke_test.py               # GigaPath + Llama coexistence test
    └── vram_monitor.sh             # rocm-smi VRAM logger
```

---

## Benchmark Dataset

AOB-Bench ClinicalEval v1 is publicly available:

```python
from datasets import load_dataset
ds = load_dataset("aob-bench/ClinicalEval", split="test")
print(ds[0])
```

100 cases · CC BY 4.0 · [aob-bench/ClinicalEval](https://huggingface.co/datasets/aob-bench/ClinicalEval)

---

## Model Cards

| Model | Base | Task | Precision |
|-------|------|------|-----------|
| **Llama 3.3 70B** | `llama3.3:70b-instruct-q4_K_S` | Researcher + Oncologist synthesis | Q4_K_S (~40 GB) via Ollama |
| **Prov-GigaPath** | `prov-gigapath/prov-gigapath` | Patch embedding + tissue classification | FP16 (~3 GB) |
| **Qwen2.5-VL-7B** | `Qwen/Qwen2.5-VL-7B-Instruct` | Visual morphology second opinion | BF16 (~15 GB) |
| `tnm_specialist` | Llama 3.1 8B Instruct | TNM staging | LoRA r=16, α=32 |
| `biomarker_specialist` | Llama 3.1 8B Instruct | Biomarker panel extraction | LoRA r=16, α=32 |
| `treatment_specialist` | Llama 3.1 8B Instruct | NCCN treatment planning | LoRA r=16, α=32 |

---

## Citation

```bibtex
@software{aob_2026,
  title   = {Autonomous Oncology Board: Multi-Agent Clinical Reasoning
             on AMD Instinct MI300X},
  author  = {AOB Team},
  year    = {2026},
  url     = {https://github.com/aob-bench/autonomous-oncology-board},
  note    = {AMD Developer Hackathon 2026. Hardware: AMD Instinct MI300X 192GB HBM3.}
}
```

---

## Disclaimer

This system is a research prototype developed for the AMD Developer Hackathon 2026.  
**Not for clinical use.** All evaluation cases are synthetic or de-identified.  
Always consult a qualified oncologist for medical decisions.

---

*Built on AMD MI300X · ROCm 6.x · Prov-GigaPath · Qwen2.5-VL-7B · Llama 3.3 70B Q4_K_S · Ollama · vLLM · Qdrant · FastAPI · Next.js 15*
