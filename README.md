# 🔬 Autonomous Oncology Board (AOB)

> **AMD Developer Hackathon 2026** — Multi-agent AI tumour board running on AMD Instinct MI300X

[![ROCm](https://img.shields.io/badge/ROCm-6.x-ED1C24?logo=amd)](https://rocm.docs.amd.com)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Hackathon%20Only-lightgrey)](#license)

---

## What Is This?

The AOB simulates a **multi-agent medical consensus meeting** — a digital tumour board. Three specialized AI agents collaborate, challenge each other, and produce a structured **Patient Management Plan** from histopathology image patches.

```
  Pathology Patches (224×224)
          │
          ▼
  ┌──────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
  │  AGENT 1         │────▶│  AGENT 2             │────▶│  AGENT 3           │
  │  Pathologist     │     │  Researcher          │     │  Oncologist        │
  │                  │     │                      │     │                    │
  │  Prov-GigaPath   │     │  RAG + Qdrant        │     │  Llama 3.3 70B     │
  │  ViT-Giant 1.1B  │     │  NCCN / TCGA / NEJM  │     │  via Ollama ROCm   │
  │                  │     │                      │     │                    │
  │  → PathologyReport│    │  → ResearchSummary   │     │  → ManagementPlan  │
  └──────────────────┘     └──────────────────────┘     └────────────────────┘
          │                        │                              │
          └────────────────────────┴──── Agent Debate ───────────┘
                                           │
                                    Final Report (JSON)
```

---

## Why AMD MI300X?

The 192 GB HBM3 unified VRAM pool makes this specific architecture **physically possible**:

```
Model                      VRAM
───────────────────────────────────────────
GigaPath ViT-Giant (FP16)   ~3 GB
Llama 3.3 70B via Ollama   ~40 GB
KV Cache + inference        ~45 GB
───────────────────────────────────────────
Total (measured Day 1)      88.2 GB  ✅
Headroom remaining         103.5 GB  ✅
───────────────────────────────────────────
H100 VRAM limit              80 GB   ❌ OOM
```

> **A single NVIDIA H100 (80 GB) cannot hold the full pipeline.** The MI300X enables zero-compromise dual-model residency with 103 GB to spare — room for concurrent cases.

---

## Architecture

| Component | Technology | Notes |
|---|---|---|
| **Vision FM** | [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) | ViT-Giant, 1.1B params, trained on 1.3B pathology tokens |
| **LLM** | Llama 3.3 70B via [Ollama](https://ollama.com) | ROCm-native, no CUDA binaries |
| **Vector DB** | [Qdrant](https://qdrant.tech) | In-process, corpus of NCCN/TCGA guidelines |
| **Embedder** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim for RAG |
| **API** | FastAPI + SSE streaming | Real-time agent step updates |
| **Frontend** | Self-contained HTML | Served from FastAPI, no build step |
| **GPU Runtime** | ROCm 6.x | AMD MI300X, no NVIDIA dependency |

---

## Quick Start

### Prerequisites
- AMD MI300X (or compatible ROCm GPU)
- ROCm 6.x installed on host
- Docker (for the ML container)
- Ollama with ROCm support
- HuggingFace account with [GigaPath access approved](https://huggingface.co/prov-gigapath/prov-gigapath)

### 1. Start Ollama + pull Llama 3.3 70B (on HOST)
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
ollama pull llama3.3:70b
ufw allow from 172.17.0.0/16 to any port 11434   # allow Docker → host
```

### 2. Enter the ROCm container
```bash
docker exec -it rocm /bin/bash
cd /workspace/aob
```

### 3. Install dependencies
```bash
pip install -r ml/requirements.txt
```

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env — set HF_TOKEN and OLLAMA_HOST
export $(cat .env | xargs)
```

### 5. Run the smoke test (validates both models in VRAM)
```bash
export PYTHONPATH=/workspace/aob
python scripts/smoke_test.py
```
Expected: `88.2 GB / 191.7 GB used · All checks PASS`

### 6. Index the oncology corpus
```bash
python ml/rag/corpus_indexer.py
```

### 7. Start the API server
```bash
uvicorn ml.api:app --host 0.0.0.0 --port 8000
```

### 8. Open the demo UI
```
http://<server-ip>:8000
```

---

## Repository Structure

```
aob/
├── CLAUDE.md                    ← Project bible (AI assistant: read this first)
├── README.md                    ← This file
├── .env.example                 ← Environment variable template
├── docker-compose.yml           ← Local dev orchestration
│
├── ml/                          ← Python ML/AI layer
│   ├── agents/
│   │   ├── pathologist.py       ← Agent 1: GigaPath patch inference
│   │   ├── researcher.py        ← Agent 2: RAG evidence synthesis
│   │   └── oncologist.py        ← Agent 3: Llama 70B plan synthesis
│   ├── models/
│   │   ├── gigapath_loader.py   ← GigaPath loading + preprocessing
│   │   └── llm_client.py        ← Ollama REST client wrapper
│   ├── rag/
│   │   ├── corpus_indexer.py    ← One-time corpus ingestion → Qdrant
│   │   ├── retriever.py         ← Qdrant query interface + mock corpus
│   │   └── corpus/              ← Raw oncology documents (gitignored if large)
│   ├── static/
│   │   └── index.html           ← Self-contained demo frontend
│   ├── board.py                 ← Sequential state machine orchestrating 3 agents
│   ├── api.py                   ← FastAPI endpoints + SSE streaming
│   └── requirements.txt
│
└── scripts/
    ├── smoke_test.py            ← Day 1: validates GigaPath + Llama in VRAM together
    └── vram_monitor.sh          ← Continuous rocm-smi VRAM logger
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Submit image patches for analysis. Returns `job_id`. |
| `GET` | `/status/{job_id}` | Poll job status and completed agent steps. |
| `GET` | `/stream/{job_id}` | SSE stream of live agent step events. |
| `GET` | `/report/{job_id}` | Retrieve completed `ManagementPlan` JSON. |
| `GET` | `/health` | Check API + Ollama connectivity. |
| `GET` | `/` | Serve demo UI. |

### Example: Submit a case
```python
import base64, requests
from PIL import Image
import io, numpy as np

# Prepare patches
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()

# Submit
r = requests.post("http://localhost:8000/analyze", json={
    "case_id": "case_001",
    "patches_b64": [b64] * 6,
    "metadata": {"patient_age": 67, "sex": "M", "clinical_notes": "Persistent cough, weight loss"}
})
job_id = r.json()["job_id"]

# Stream live updates
import sseclient
for event in sseclient.SSEClient(f"http://localhost:8000/stream/{job_id}"):
    print(event.data)
```

---

## Output Schema

```json
{
  "management_plan": {
    "case_id": "case_001",
    "diagnosis": {
      "primary": "Lung Adenocarcinoma",
      "tnm_stage": "Stage IV NSCLC — pending molecular workup",
      "confidence": 0.85
    },
    "immediate_actions": ["Order EGFR mutation testing", "..."],
    "treatment_plan": {
      "first_line": "Osimertinib 80mg/day (EGFR-mutant) or Pembrolizumab 200mg Q3W (PD-L1 ≥50%)",
      "rationale": "Based on NCCN Category 1 evidence...",
      "alternatives": ["Alectinib 600mg BID for ALK-positive", "..."]
    },
    "citations": [
      "NCCN Clinical Practice Guidelines in Oncology: NSCLC v4.2024",
      "Reck M, et al. N Engl J Med 2016;375:1823-1833."
    ],
    "disclaimer": "AI research tool. NOT for clinical use."
  }
}
```

---

## Disclaimer

This project is a research demonstration built for the AMD Developer Hackathon 2026.

**It is NOT a medical device and NOT approved for clinical use.** All outputs are generated by AI models and must not be used for actual patient diagnosis or treatment decisions. Always consult a qualified oncologist.

---

## License

Submitted for hackathon evaluation. Not licensed for production or clinical use.
