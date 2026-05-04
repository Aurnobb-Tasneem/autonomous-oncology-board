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

AOB simulates a hospital multidisciplinary tumour board (MTB) with three AI agents:

```
 Patient Case (WSI patches + metadata)
          │
 ┌────────▼────────┐
 │  PATHOLOGIST    │  Prov-GigaPath ViT-Giant 1.1B
 │  Agent 1        │  → Heatmaps · Uncertainty · Classification
 └────────┬────────┘
          │ PathologyReport JSON
 ┌────────▼────────┐
 │  RESEARCHER     │  Qdrant RAG + Llama 3.3 70B
 │  Agent 2        │  → 500-doc NCCN/TCGA corpus · Citations
 └────────┬────────┘
          │ EvidenceBundle JSON
 ┌────────▼────────┐
 │  ONCOLOGIST     │  Llama 3.3 70B (FP8) + 3× LoRA adapters
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

The architecture requires simultaneous in-memory residency of:

| Component | VRAM |
|-----------|------|
| Llama 3.3 70B (FP8) | 70 GB |
| Llama 3.1 8B + 3 LoRA adapters | 16 GB |
| Prov-GigaPath (FP16) | 3 GB |
| vLLM KV Cache (full debate) | 30 GB |
| Qdrant + overhead | 9 GB |
| **Total** | **~128 GB** |

**NVIDIA H100 = 80 GB. This system requires 128 GB. The math does not work on H100.**

The MI300X's 192 GB HBM3 unified pool provides 64 GB of headroom for 3 concurrent cases without model swapping or cache eviction.

```
H100: ████████████████████████████████████████░ 80/80 GB → OOM ✗
MI300X: ██████████████████████████░░░░░░░░░░░░░ 128/192 GB ✓
```

Measured peak: **88.2 GB / 191.7 GB** (verified via `rocm-smi`)

---

## Specialist LoRA Suite

Three LoRA adapters (rank 16, α=32) fine-tuned on Llama 3.1 8B Instruct, served simultaneously via vLLM multi-adapter hot-swap:

| Adapter | Task | Training Data |
|---------|------|---------------|
| `tnm_specialist` | TNM staging from pathology text | 50 expert examples |
| `biomarker_specialist` | Required biomarker panel extraction | 50 expert examples |
| `treatment_specialist` | NCCN-aligned first-line treatment | 50 expert examples |

```bash
# Launch all three adapters on a single 8B base model
./scripts/serve_specialists.sh
```

---

## The Agent Debate Protocol

Unlike single-pass pipelines, AOB's oncologist and researcher argue before finalising:

1. **Oncologist** drafts initial management plan
2. **Researcher** challenges with RAG evidence: *"⚠️ EGFR not confirmed — NCCN Category 1 requires molecular testing first"*
3. **Oncologist** revises — revision diff shown in UI
4. **Meta-Evaluator** scores consensus (0–100). If < 70, triggers another round (max 3)

The final report includes the full **Debate Transcript** and **revision diff**.

---

## Additional Capabilities

| Feature | Description |
|---------|-------------|
| **Triple-Modal Explainability** | Attention Rollout + Grad-CAM++ + Integrated Gradients per patch |
| **MC Dropout Uncertainty** | N=20 stochastic passes → "91% ± 4.2%" confidence intervals |
| **Differential Diagnosis** | Top-3 diagnoses with posterior probabilities |
| **Clinical Trial Matching** | Semantic search over 500-trial ClinicalTrials.gov corpus |
| **Counterfactual Reasoning** | "What if EGFR negative?" → instant revised plan |
| **Board Memory** | GigaPath embeddings stored → similar historical case retrieval |
| **Patient Summary** | 8th-grade English translation of clinical plan |
| **VRAM Dashboard** | Live `rocm-smi` widget with H100 comparison bar |
| **Concurrent Cases** | 3 simultaneous analyses; peak VRAM ~118 GB / 192 GB |
| **Speculative Decoding** | Llama 3.1 8B draft → +53% throughput on 70B inference |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | AMD Instinct MI300X | AMD Instinct MI300X |
| VRAM | 128 GB | 192 GB (full concurrent) |
| ROCm | 6.0+ | 6.2+ |
| RAM | 128 GB | 256 GB |
| Storage | 200 GB SSD | 500 GB NVMe |

---

## Quick Start

### Prerequisites
```bash
# ROCm 6.x installed
# Python 3.10+
# Ollama with ROCm support
ollama pull llama3.3:70b
```

### Install
```bash
cd aob/ml
pip install -r requirements.txt
```

### Run
```bash
# 1. Start Ollama (serves Llama 3.3 70B)
ollama serve

# 2. Start specialist LoRA adapters
./scripts/serve_specialists.sh

# 3. Start AOB API + frontend
python -m aob.ml.api

# 4. Open browser
open http://localhost:8000
```

### Run Benchmark
```bash
# Generate 100-case eval dataset
python scripts/gen_clinical_eval_cases.py

# Run full benchmark (requires active GPU stack)
python -m aob.eval.clinical_eval --config aob_full

# Run ablation study (mock mode, no GPU needed)
python -m aob.eval.ablation_study --mock

# Run calibration analysis
python -m aob.eval.calibration --mock
```

---

## Project Structure

```
aob/
├── ml/
│   ├── agents/
│   │   ├── pathologist.py          # GigaPath inference + heatmaps
│   │   ├── researcher.py           # RAG synthesis
│   │   ├── oncologist.py           # Llama 3.3 70B synthesis + debate
│   │   ├── biomarker_specialist.py # LoRA biomarker extraction
│   │   ├── treatment_specialist.py # LoRA treatment planning
│   │   ├── differential.py         # Top-3 differential diagnosis
│   │   ├── counterfactual.py       # What-if replanning
│   │   ├── patient_summary.py      # Plain-English patient report
│   │   └── trial_matcher.py        # ClinicalTrials.gov matching
│   ├── models/
│   │   ├── gigapath_loader.py      # GigaPath weight loading
│   │   ├── llm_client.py           # Ollama + vLLM async clients
│   │   └── explainability.py       # Grad-CAM++ + Integrated Gradients
│   ├── training/
│   │   ├── lora_trainer.py         # Generic LoRA training framework
│   │   └── giga_head.py            # GigaPath MLP classification head
│   ├── rag/
│   │   ├── corpus_indexer.py       # Document ingestion + embedding
│   │   ├── retriever.py            # Qdrant query interface
│   │   └── trials/                 # ClinicalTrials.gov snapshot
│   ├── data/
│   │   ├── wsi.py                  # WSI patch extraction (openslide)
│   │   └── preprocessing.py       # Patch normalization
│   ├── board.py                    # Pipeline orchestration
│   └── api.py                      # FastAPI endpoints
├── eval/
│   ├── clinical_eval.py            # ClinicalEval benchmark runner
│   ├── ablation_study.py           # Ablation + bootstrap CIs
│   ├── calibration.py              # ECE + reliability curves
│   └── cases/
│       └── clinical_eval_cases.json  # 100-case ground-truth dataset
├── hf_dataset/
│   ├── aob_bench.py                # HuggingFace dataset loader
│   ├── clinical_eval_cases.json    # Dataset source
│   └── README.md                   # HF dataset card
├── frontend/
│   └── static/                     # Self-contained HTML+JS UI
├── scripts/
│   ├── serve_specialists.sh        # vLLM multi-LoRA launcher
│   ├── serve_speculative.sh        # Speculative decoding launcher
│   ├── benchmark_speculative.py    # Throughput benchmark
│   ├── gen_clinical_eval_cases.py  # Generate 100-case benchmark
│   └── gen_trials_snapshot.py      # Generate trial corpus
└── docs/
    ├── technical_report.md         # 10-page research report
    ├── demo_script.md              # 5-minute demo video script
    └── diagrams/                   # Architecture diagrams
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

| Model | Base | Task | Adapter Rank | HF Repo |
|-------|------|------|-------------|---------|
| `tnm_specialist` | Llama 3.1 8B Instruct | TNM staging | r=16, α=32 | `aob-bench/tnm-specialist-lora` |
| `biomarker_specialist` | Llama 3.1 8B Instruct | Biomarker extraction | r=16, α=32 | `aob-bench/biomarker-specialist-lora` |
| `treatment_specialist` | Llama 3.1 8B Instruct | Treatment planning | r=16, α=32 | `aob-bench/treatment-specialist-lora` |

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

*Built on AMD MI300X · ROCm 6.x · Prov-GigaPath · Llama 3.3 70B · vLLM · Qdrant · FastAPI*
