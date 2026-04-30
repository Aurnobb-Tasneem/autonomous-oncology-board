# 🔬 Autonomous Oncology Board (AOB)

> **AMD Developer Hackathon 2026** — Multi-agent AI tumour board on AMD Instinct MI300X

[![ROCm](https://img.shields.io/badge/ROCm-6.x-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![GigaPath](https://img.shields.io/badge/GigaPath-ViT--Giant%201.1B-blueviolet)](https://huggingface.co/prov-gigapath/prov-gigapath)
[![Llama](https://img.shields.io/badge/Llama%203.3-70B-orange)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## What Is This?

The AOB simulates a **multi-agent medical consensus meeting** — a digital tumour board. Three specialised AI agents collaborate, challenge each other via a structured debate loop, and produce a complete **Patient Management Plan** from histopathology image patches.

```
  Histopathology Patches (224x224 H&E)
            |
            v
  +-----------------------+    +------------------------+    +----------------------+
  |  AGENT 1              |--->|  AGENT 2               |--->|  AGENT 3             |
  |  Pathologist          |    |  Researcher            |    |  Oncologist          |
  |                       |    |                        |    |                      |
  |  Prov-GigaPath        |    |  RAG + Qdrant          |    |  Llama 3.3 70B       |
  |  ViT-Giant 1.1B       |    |  NCCN Guidelines       |    |  via Ollama (ROCm)   |
  |  + Biomarker Layer    |    |  + Board Memory        |    |  + Similar Cases     |
  |  + MC Dropout         |    |                        |    |                      |
  |  + Attention Maps     |    |  -> ResearchSummary    |    |  -> ManagementPlan   |
  +-----------------------+    +------------------------+    +----------------------+
            |                           |                             |
            +---------------------------+-------- Agent Debate -------+
                                                       |
                                        [up to 3 rounds: challenge -> revise]
                                                       |
                                                 Final Report (JSON)
                                           Consensus score >= 70/100 -> done
```

---

## Why AMD MI300X?

The 192 GB HBM3 unified VRAM pool makes this architecture **physically impossible on a single H100**:

```
Model                            VRAM
-----------------------------------------
Prov-GigaPath ViT-Giant (FP16)   ~3.2 GB
Llama 3.3 70B via Ollama         ~40.0 GB
KV Cache + inference overhead     ~3.0 GB
-----------------------------------------
Total                            ~46.2 GB  OK on MI300X
-----------------------------------------
NVIDIA H100 VRAM limit            80.0 GB
  - After loading Llama 70B:     ~40 GB remaining
  - GigaPath minimum need:       ~3 GB (fits, but no room for KV cache at scale)
  - Concurrent cases:            impossible
```

> A single H100 cannot hold both models **warm simultaneously** at production scale. The MI300X's 192 GB enables zero-compromise dual-model residency with 145+ GB to spare for concurrent cases.

---

## Features

| Feature | Status | Description |
|---|---|---|
| **GigaPath Inference** | Done | 1.1B ViT-Giant, 1536-dim embeddings, FP16 on ROCm |
| **Tissue Classification** | Done | 5 classes (LC25000), prototype cosine similarity |
| **Biomarker Layer** | Done | 8 interpretable oncology biomarkers from centroid |
| **Attention Heatmaps** | Done | Attention rollout across all ViT blocks, red overlay PNGs |
| **MC Dropout Uncertainty** | Done | 20 stochastic passes, confidence intervals + flags |
| **NCCN RAG** | Done | Qdrant vector store, evidence retrieval + synthesis |
| **Agent Debate Loop** | Done | Up to 3 rounds: challenge -> referee -> revise -> MetaEvaluate |
| **Board Memory** | Done | Cosine-similarity retrieval of similar past cases (JSONL) |
| **Demo Cases** | Done | 3 pre-baked JSON cases for instant demo without image upload |
| **VRAM Dashboard** | Done | Live rocm-smi endpoint + H100 comparison |
| **SSE Streaming** | Done | Real-time agent step events |
| **Stress Test** | Done | 10 concurrent cases, P50/P95/P99 latency reporting |
| **HuggingFace Space** | Done | Gradio UI wrapping the live API |

---

## Architecture

| Component | Technology | Notes |
|---|---|---|
| **Vision FM** | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-Giant, 1.1B params, trained on 1.3B pathology patches |
| **LLM** | Llama 3.3 70B via [Ollama](https://ollama.com) | ROCm-native, no CUDA binaries |
| **Vector DB** | [Qdrant](https://qdrant.tech) | In-process, NCCN/TCGA oncology corpus |
| **Board Memory** | JSONL flat file | Cosine similarity retrieval, 1536-dim centroids |
| **Biomarkers** | Fixed seeded projections | 8 interpretable scores (no extra model needed) |
| **API** | FastAPI + SSE | Real-time agent step streaming |
| **HF Space** | Gradio 4.x | Wraps live API, 3-tab UI |
| **GPU Runtime** | ROCm 6.x | AMD MI300X, no NVIDIA dependency |

---

## Quick Start

### Prerequisites
- AMD MI300X (or compatible ROCm GPU)
- ROCm 6.x on host, Docker, Ollama with ROCm support
- HuggingFace account with [GigaPath access](https://huggingface.co/prov-gigapath/prov-gigapath) approved

### 1. Start Ollama + pull model (on HOST)
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
ollama pull llama3.3:70b
ufw allow from 172.17.0.0/16 to any port 11434
```

### 2. Enter the ROCm container
```bash
docker exec -it rocm /bin/bash
cd /workspace/aob
pip install -r ml/requirements.txt
```

### 3. Configure and start
```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, OLLAMA_HOST
export $(cat .env | xargs)
export PYTHONPATH=/workspace/aob
python scripts/smoke_test.py          # validate both models in VRAM
uvicorn ml.api:app --host 0.0.0.0 --port 8000
```

### 4. Run a demo case instantly
```bash
# No image upload needed
curl -X POST http://localhost:8000/demo/run/lung_adenocarcinoma
# Returns: {"job_id": "job_...", "status": "queued"}

# Then stream live agent steps:
curl -N http://localhost:8000/stream/<job_id>

# Or fetch the completed report:
curl http://localhost:8000/report/<job_id>
```

### 5. Stress test (10 concurrent cases)
```bash
pip install aiohttp
python scripts/stress_test.py --host http://localhost:8000 --concurrency 10
```

---

## Repository Structure

```
aob/
├── ml/
│   ├── agents/
│   │   ├── pathologist.py       <- Agent 1: GigaPath + biomarkers + MC dropout + heatmaps
│   │   ├── researcher.py        <- Agent 2: RAG evidence + challenge()
│   │   ├── oncologist.py        <- Agent 3: Llama 70B plan synthesis + revise()
│   │   ├── meta_evaluator.py    <- Debate consensus scorer (0-100)
│   │   ├── board_memory.py      <- Similar case retrieval (cosine, JSONL)
│   │   ├── biomarker.py         <- 8 interpretable biomarker scores
│   │   └── uncertainty.py       <- MC Dropout (20 passes)
│   ├── models/
│   │   ├── gigapath_loader.py   <- GigaPath load + attention rollout
│   │   └── llm_client.py        <- Ollama REST wrapper
│   ├── rag/retriever.py         <- Qdrant + NCCN corpus
│   ├── board.py                 <- Main orchestrator + debate loop
│   └── api.py                   <- FastAPI endpoints
├── data/
│   ├── demo_cases/              <- 3 pre-baked JSON cases
│   └── board_memory.jsonl       <- Auto-generated case store
├── hf_space/
│   ├── app.py                   <- Gradio HuggingFace Space
│   └── README.md                <- HF Space config
├── docs/build_in_public/        <- Twitter/X thread drafts
└── scripts/
    ├── smoke_test.py            <- VRAM validation
    └── stress_test.py           <- Concurrent load test
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API + Ollama + board status |
| `POST` | `/analyze` | Submit image patches -> `job_id` |
| `GET` | `/stream/{job_id}` | SSE stream of live agent steps |
| `GET` | `/report/{job_id}` | Complete ManagementPlan JSON |
| `GET` | `/heatmaps/{job_id}` | Attention heatmap PNGs (base64) |
| `GET` | `/cases` | List all jobs |
| `GET` | `/memory/cases` | Board memory — all stored past cases |
| `GET` | `/api/vram` | Live VRAM from rocm-smi + H100 comparison |
| `GET` | `/demo/cases` | List available pre-baked demo cases |
| `POST` | `/demo/run/{case_name}` | Run demo case without image upload |

---

## Output Schema (abbreviated)

```json
{
  "case_id": "case_001",
  "total_time_s": 187.4,
  "debate_rounds_completed": 1,
  "pathology_report": {
    "tissue_type": "lung_adenocarcinoma",
    "confidence": 0.87,
    "biomarkers": {
      "nuclear_pleomorphism": {"score": 0.73, "level": "High"},
      "mitotic_index": {"score": 0.61, "level": "Moderate"}
    },
    "uncertainty_interval": "87.0% +/- 3.8%"
  },
  "management_plan": {
    "diagnosis": {"primary": "Lung Adenocarcinoma", "tnm_stage": "Stage IV NSCLC"},
    "treatment_plan": {
      "first_line": "EGFR/ALK/ROS1 panel FIRST -> osimertinib if EGFR+ / pembrolizumab if PD-L1 >=50%",
      "rationale": "NCCN Category 1 evidence for targeted therapy in driver-mutation positive NSCLC"
    },
    "consensus_score": 82
  }
}
```

---

## Disclaimer

Research demonstration for AMD Developer Hackathon 2026.

**NOT a medical device. NOT for clinical use.** All outputs are AI-generated and must not be used for actual patient diagnosis or treatment decisions. Always consult a qualified oncologist.

---

## License

Copyright 2026 **Aurnobb Tasneem** · [Apache License 2.0](LICENSE)

> Commercial use requires written permission. Contact: taurnobb@gmail.com
