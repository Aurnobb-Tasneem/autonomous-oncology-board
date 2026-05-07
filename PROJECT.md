# Autonomous Oncology Board (AOB) - Comprehensive Project Document

## 1. Project Overview
**Autonomous Oncology Board (AOB)** is an advanced multi-agent clinical reasoning system simulating a hospital multidisciplinary tumour board. Developed for the **AMD Developer Hackathon 2026**, the system leverages the massive 192GB unified VRAM of the **AMD Instinct MI300X** to concurrently run multiple heavy-weight foundation and large language models. The result is a robust, explainable, and consensus-driven Patient Management Plan that mirrors human clinical workflows.

## 2. The Core Concept
Instead of a single monolithic model outputting a classification, AOB uses specialized AI agents that collaborate and debate. 
- **Input**: Whole Slide Image (WSI) patches and clinical metadata.
- **Process**: Vision models extract morphological features and flag abnormalities. A research agent retrieves relevant clinical guidelines (RAG). An oncologist agent synthesizes the findings and debates with the researcher before finalizing the plan.
- **Output**: A structured Patient Management Plan including TNM staging, NCCN-aligned treatment recommendations, citations, and confidence scores.

## 3. The Multi-Agent Architecture
The system consists of three primary roles played by four discrete models:

### Agent 1a: The Digital Pathologist (Vision Foundation Model)
- **Model**: `Prov-GigaPath` (ViT-Giant 1.1B, pre-trained on 1.3B pathology tokens)
- **Role**: Analyzes WSI patches, extracts morphological embeddings, classifies tissue regions, and generates visual attention heatmaps (explainability).

### Agent 1b: The VLM Second Opinion
- **Model**: `Qwen2.5-VL-7B-Instruct`
- **Role**: Provides direct pixel-level second opinions on patches, reconciling findings with GigaPath using a Meta-Evaluator before passing data downstream.

### Agent 2: The Clinical Researcher (RAG Pipeline)
- **Model**: `Qdrant` (Vector DB) + `Llama 3.3 70B`
- **Role**: Queries a pre-indexed local corpus of ~500 documents including NCCN guidelines and TCGA studies to retrieve cited protocols and formulate an evidence bundle.

### Agent 3: The Lead Oncologist (Orchestrator)
- **Model**: `Llama 3.3 70B` (FP8) + 3 specialized `Llama 3.1 8B` LoRA adapters (TNM staging, Biomarkers, Treatment).
- **Role**: Synthesizes the pathology reports and evidence bundle into a Patient Management Plan. Participates in a **Debate Loop** with the Researcher to refine the plan, ensuring clinical accuracy (e.g., verifying required biomarker testing before recommending targeted therapies).

## 4. Hardware & Infrastructure: The AMD MI300X Advantage
AOB's architecture is physically impossible on a standard 80GB NVIDIA H100. It requires the simultaneous in-memory residency of:
- Llama 3.3 70B (FP8): ~70 GB
- Llama 3.1 8B + 3 LoRAs: ~16 GB
- Prov-GigaPath (FP16): ~3 GB
- Qwen2.5-VL-7B-Instruct (BF16): ~15 GB
- vLLM KV Cache: ~30 GB
- System Overhead (Qdrant, etc.): ~9 GB
- **Total Concurrent VRAM**: ~143 GB

The **AMD Instinct MI300X** provides 192GB of HBM3 unified memory, easily accommodating this full stack while leaving headroom for processing up to 3 concurrent cases. A live VRAM dashboard visually proves this utilization during the demo.

## 5. Technology Stack
- **Backend/ML Layer**: Python 3.10+, ROCm 6.x, PyTorch (ROCm), vLLM / Ollama (ROCm-native), Transformers, FastAPI.
- **RAG & Data Layer**: Qdrant (in-process vector DB), sentence-transformers, openslide-python.
- **Frontend Layer**: Next.js 15 (App Router), TypeScript, Tailwind CSS, shadcn/ui, Server-Sent Events (SSE) for streaming agent reasoning.

## 6. Winning Differentiators
- **Multi-Round Agent Debate**: Agents challenge each other's reasoning (e.g., catching missing EGFR tests) and revise the plan. The UI displays the full debate transcript and a revision diff.
- **Visual Explainability**: GigaPath Attention Heatmaps project model attention back onto tissue patches, highlighting "suspicious" regions.
- **Monte Carlo Dropout Uncertainty**: Runs stochastic passes to quantify confidence intervals (e.g., "91% ± 4.2%"), triggering secondary reviews on high variance.
- **Biomarker-Guided Precision Oncology**: Gates treatments behind mandatory biomarker testing, mirroring real-world NCCN guidelines.
- **Board Memory**: Retrieves similar historical cases using vector similarity on pathology embeddings to inform current decisions.

## 7. Performance & Benchmarks
Evaluated on **AOB-Bench ClinicalEval v1** (100 expert-curated cases):
- **TNM Exact-Match**: 82.3%
- **Biomarker F1**: 74.8
- **Treatment Alignment**: 77.8%
The multi-agent approach provides a massive **+42.5 percentage point improvement** in TNM staging accuracy compared to a baseline 8B model.

## 8. Project Structure
The repository (`aob/`) is structured as a monorepo containing:
- `ml/`: The core Python ML pipeline, containing the agents, model loaders, RAG implementation, and the `board.py` orchestration logic.
- `backend/`: The API gateway (FastAPI/NestJS) serving the models and handling UI requests.
- `eval/`: The clinical evaluation suites and benchmark runners.
- `frontend/`: The Next.js UI providing the live dashboard and timeline visualization.
- `scripts/`: Operational tools for standing up the stack (e.g., serving LoRAs, VRAM monitoring).

The system relies purely on local inference (no external API calls for models), fully demonstrating the standalone power and capability of the AMD platform.
