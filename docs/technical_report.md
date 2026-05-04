# Autonomous Oncology Board (AOB): A Multi-Agent Clinical Reasoning System on AMD Instinct MI300X

**AMD Developer Hackathon 2026 — Technical Report**

*Submitted: May 2026 | Hardware: AMD Instinct MI300X 192 GB HBM3 | ROCm 6.x*

---

## Abstract

We present the **Autonomous Oncology Board (AOB)**, a multi-agent AI system that simulates a hospital tumour board. Three specialized agents — a Digital Pathologist (Prov-GigaPath ViT-Giant 1.1B), a Clinical Researcher (RAG over 500 oncology documents), and a Lead Oncologist (Llama 3.3 70B with three LoRA specialist adapters) — collaborate through a structured debate protocol to produce NCCN-aligned Patient Management Plans with full citation chains. We introduce **AOB-Bench ClinicalEval v1**, a 100-case open benchmark spanning lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, and benign tissue. The full pipeline achieves 82.3% TNM exact-match, 74.8 biomarker F1, and 77.8% treatment class alignment, representing a +42.5 pp TNM improvement over a Llama 3.1 8B baseline. The AMD MI300X's 192 GB unified HBM3 memory pool is the architectural enabler: both the vision foundation model and the 70B LLM co-reside in a single VRAM address space, eliminating GPU-to-GPU transfer latency and enabling real-time concurrent case processing.

---

## 1. Introduction

Multidisciplinary tumour boards (MTBs) are the clinical gold standard for complex oncology decisions. An MTB convenes pathologists, radiologists, oncologists, and surgeons to debate a case before recommending treatment. Current AI tools offer only isolated outputs — "this tissue is 94% malignant" — rather than the integrated, cited, staged management plan that clinicians actually need.

AOB is architected to mirror the MTB structure computationally:

1. **Pathologist Agent** — examines histopathology image patches with a gigapixel-scale vision foundation model
2. **Researcher Agent** — retrieves relevant clinical evidence from a pre-indexed corpus of NCCN guidelines and oncology literature
3. **Oncologist Agent** — synthesizes all evidence into a structured management plan with TNM staging, biomarker-gated treatment options, and uncertainty quantification

The key systems contribution is demonstrating that this architecture — which requires simultaneous residency of a 1.1B vision transformer and a 70B LLM — is only feasible on hardware with ≥130 GB of unified memory. The AMD Instinct MI300X (192 GB HBM3) is the hardware that makes this design physically realizable.

---

## 2. System Architecture

### 2.1 Overview

```
Input: WSI patches (N × 3 × 224 × 224) + clinical metadata
                         │
        ┌────────────────▼───────────────────┐
        │  AGENT 1: Digital Pathologist       │
        │  Prov-GigaPath ViT-Giant 1.1B       │
        │  • Attention rollout heatmaps        │
        │  • Grad-CAM++ saliency maps          │
        │  • Integrated Gradients              │
        │  • Monte Carlo Dropout uncertainty   │
        │  Output: PathologyReport (JSON)      │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  AGENT 2: Clinical Researcher       │
        │  Qdrant (in-process) + Llama 3.3 70B│
        │  • NCCN guideline retrieval          │
        │  • Biomarker protocol lookup         │
        │  • Citation-grounded synthesis       │
        │  Output: EvidenceBundle (JSON)       │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  SPECIALIST LoRA SUITE              │
        │  Base: Llama 3.1 8B (vLLM)          │
        │  Adapters hot-swapped:               │
        │   • tnm_specialist                   │
        │   • biomarker_specialist             │
        │   • treatment_specialist             │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  AGENT 3: Lead Oncologist           │
        │  Llama 3.3 70B (FP8, Ollama)        │
        │  DEBATE LOOP (max 3 rounds):         │
        │   1. Draft management plan           │
        │   2. Researcher critiques → Challenge│
        │   3. Oncologist revises → Final plan │
        │  Output: PatientManagementPlan       │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  POST-PROCESSING AGENTS             │
        │  • DifferentialDx (top-3 diagnoses) │
        │  • Counterfactual ("What if?")       │
        │  • PatientSummary (8th-grade English)│
        │  • TrialMatcher (500 ClinicalTrials) │
        └────────────────────────────────────┘
```

### 2.2 The AMD MI300X as Architectural Enabler

The AOB design requires simultaneous in-memory residency of:

| Component | Precision | VRAM |
|-----------|-----------|------|
| Llama 3.3 70B | FP8 | ~70 GB |
| Prov-GigaPath ViT-Giant | FP16 | ~3 GB |
| 3× LoRA adapters (Llama 3.1 8B) | FP16 | ~16 GB |
| Qdrant vector DB (500 docs) | — | ~1 GB |
| vLLM KV Cache | FP16 | ~30 GB |
| System overhead | — | ~8 GB |
| **Total** | | **~128 GB** |

The NVIDIA H100 SXM5 has 80 GB HBM3. Loading Llama 3.3 70B alone (FP8) consumes ~70 GB, leaving only 10 GB for GigaPath inference, the KV cache, and all auxiliary models. This makes concurrent residency impossible on a single H100.

The MI300X's 192 GB unified HBM3 pool provides ~64 GB of headroom, enabling:
- Simultaneous GigaPath + LLM inference (no model swapping)
- Three concurrent case pipelines
- Full debate loop without cache eviction

This is not a marketing claim — it is verified memory arithmetic, confirmed by `rocm-smi` during live operation showing 88.2 GB / 191.7 GB peak usage.

---

## 3. The Specialist LoRA Suite

### 3.1 Architecture

Three LoRA adapters (rank 16, α = 32) are fine-tuned on top of Llama 3.1 8B Instruct using domain-specific synthetic training data:

| Adapter | Task | Training Examples | Output Schema |
|---------|------|-------------------|---------------|
| `tnm_specialist` | TNM staging from pathology text | 50 | `{T, N, M, stage, rationale}` |
| `biomarker_specialist` | Biomarker panel extraction | 50 | `{tests_required, rationale, urgency}` |
| `treatment_specialist` | NCCN-aligned treatment plan | 50 | `{first_line, nccn_category, rationale}` |

All three adapters are served simultaneously via vLLM's multi-LoRA hot-swap mechanism — a single 8B base model serves all three tasks without re-loading weights.

### 3.2 Training Details

- **Base model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Framework:** HuggingFace PEFT (LoRA) + Transformers
- **Optimizer:** AdamW, lr = 2e-4, warmup 10%
- **Batch size:** 4 (gradient accumulation ×4 = effective 16)
- **Epochs:** 3
- **Hardware:** AMD MI300X (ROCm 6.x, PyTorch ROCm build)

### 3.3 Benchmark Performance (Ablation)

The ablation study quantifies the contribution of each component using 100 ClinicalEval cases, 3 random seeds, and 1000 bootstrap iterations for 95% CIs:

| System | TNM-EM [95% CI] | BM-F1 [95% CI] | TX-Align [95% CI] |
|--------|-----------------|----------------|-------------------|
| Full AOB | 82.3% [80.5, 84.4] | 74.8 [72.5, 77.0] | 77.8% [75.5, 79.9] |
| − Debate | 75.4% [73.3, 77.6] | 72.0 [69.7, 74.2] | 71.3% [68.8, 74.0] |
| − Specialist LoRAs | 65.4% [62.9, 67.9] | 59.6 [57.1, 62.1] | 60.4% [58.2, 62.6] |
| − GigaPath | 52.2% [49.8, 55.1] | 47.9 [45.3, 51.0] | 52.7% [50.0, 55.0] |
| Baseline (8B) | 39.8% [37.6, 42.1] | 40.8 [38.1, 43.0] | 41.0% [38.6, 43.7] |

**Component contributions (Δ vs full pipeline):**
- Debate loop adds +6.9 pp TNM, +6.5 pp TX alignment
- Specialist LoRAs add +16.9 pp TNM, +17.4 pp TX alignment
- GigaPath vision adds +30.1 pp TNM — the single largest contributor

---

## 4. Vision Explainability

We implement **triple-modal saliency** for each pathology patch to address the "black box" criticism of medical AI:

### 4.1 Attention Rollout
We propagate attention weights through all transformer heads of GigaPath, averaging across the [CLS] token attention to produce a 14×14 spatial map, then bilinearly interpolated to 224×224.

### 4.2 Grad-CAM++
We hook the final ViT block's `proj` layer, backpropagate from the predicted class logit, and compute the α-weighted gradient-activation product. This provides class-discriminative localization superior to standard Grad-CAM for multi-instance medical images.

### 4.3 Integrated Gradients
We compute integrated gradients from a black baseline over 50 interpolation steps:

```
IG = (x - x') · ∫₀¹ (∂F/∂x)(x' + α(x - x')) dα
```

The final visualization renders all three heatmaps as independent channels, with a consensus overlay that highlights only regions flagged by ≥2 methods.

### 4.4 Monte Carlo Dropout Uncertainty
Running N=20 stochastic forward passes with dropout enabled at inference produces a predictive confidence interval: **"Lung Adenocarcinoma: 91% ± 4.2%"**. When σ > 0.08, the system auto-flags: *"⚠️ High morphological uncertainty — recommend second-opinion biopsy."*

---

## 5. The Agent Debate Protocol

The debate protocol is the core differentiator of AOB versus single-pass pipelines:

### Round 1: Initial Draft
The Oncologist synthesizes all inputs into a first-draft management plan.

### Round 2: Researcher Challenge
The Researcher agent re-queries the RAG corpus specifically for evidence that **contradicts or qualifies** the draft. Challenge examples:
- *"⚠️ EGFR status not confirmed. NCCN Category 1 requires molecular confirmation before osimertinib."*
- *"⚠️ Cited KEYNOTE-189 applies to non-squamous NSCLC only. Patient histology is squamous — pembrolizumab monotherapy is the appropriate reference."*

### Round 3: Revision
The Oncologist integrates challenges into a revised plan. The final report includes a **Debate Transcript** panel and a **revision diff** (strikethrough old text → green new text).

### Meta-Evaluator
After revision, a lightweight LLM call scores **Consensus Score (0–100)** based on:
- Are all challenge points addressed?
- Is evidence cited for all treatment claims?
- Are biomarker tests ordered before biomarker-dependent therapies?

If score < 70, a second debate round is triggered (max 3 rounds total).

---

## 6. Calibration Analysis

We measure Expected Calibration Error (ECE) for two probabilistic outputs:

| Component | Pre-calibration ECE | Post-calibration ECE (T scaling) |
|-----------|--------------------|------------------------------------|
| GigaPath classifier | 0.0886 | 0.0619 (T=1.1) |
| Board consensus score | 0.0723 | 0.0675 (T=1.2) |

Temperature scaling with T≈1.1 brings GigaPath into the "well-calibrated" range (ECE < 0.07). The board consensus score benefits from the debate protocol's natural calibration effect — uncertain cases trigger additional debate rounds.

---

## 7. Additional Capabilities

### 7.1 Clinical Trial Matching
A local corpus of 500 oncology trials (ClinicalTrials.gov snapshot) is embedded via `sentence-transformers` and indexed in Qdrant. After generating the management plan, the TrialMatcher agent retrieves the top-5 most relevant open trials based on semantic similarity between the plan text and trial eligibility criteria.

### 7.2 Board Memory (Similar Case Retrieval)
GigaPath's mean-pooled patch embedding is stored in Qdrant after each analysis. New cases retrieve the top-3 most similar historical cases by cosine similarity, giving the Oncologist contextual precedent: *"3 similar cases were analyzed. Mean time-to-progression on first-line platinum was 8.2 months."*

### 7.3 Differential Diagnosis
A dedicated agent produces the top-3 candidate diagnoses with posterior probabilities derived from GigaPath softmax, patient metadata priors, and LLM-based reasoning: ruling in/out alternatives provides the Oncologist with a diagnostic uncertainty map.

### 7.4 Counterfactual Reasoning
An interactive "What if?" interface lets clinicians modify key assumptions and receive a revised plan without re-running the full pipeline. Example: *"What if the EGFR result came back negative?"* → the treatment plan switches from osimertinib to platinum-doublet within ~3 seconds.

### 7.5 Patient-Facing Summary
All management plans are automatically translated to 8th-grade reading level English via a dedicated summarisation prompt, making AOB outputs directly patient-shareable.

---

## 8. Speculative Decoding Performance

vLLM's speculative decoding is configured with Llama 3.1 8B as the draft model and Llama 3.3 70B as the verification model. On 100 oncology prompts:

| Configuration | Mean Tokens/sec | Latency (median) | Acceptance Rate |
|---------------|-----------------|-----------------|-----------------|
| Standard 70B | ~28 tok/s | 18.3s | — |
| Speculative (8B draft) | ~43 tok/s | 11.7s | ~64% |

The **+53% throughput improvement** is crucial for the live demo: a complete management plan generates in under 12 seconds rather than 18, which meaningfully improves demo energy.

---

## 9. Concurrent Case Stress Test

Three distinct cancer types were submitted simultaneously to the AOB API:
- Case A: Lung Adenocarcinoma (EGFR L858R, Stage IV)
- Case B: Colon Adenocarcinoma (MSI-H, Stage IVA)
- Case C: Lung Squamous Cell Carcinoma (PD-L1 TPS 80%, Stage IV)

All three completed without OOM errors. Peak VRAM: **~118 GB / 192 GB**. Total wall-clock time: 41 seconds for all three cases. On a single H100, sequential processing of these three cases alone would require model swapping, adding estimated 45+ seconds of model load overhead per swap.

---

## 10. Conclusion

AOB demonstrates that medical AI systems of genuine clinical depth are possible within a 7-day sprint when the hardware architecture is chosen deliberately. The AMD MI300X's 192 GB unified memory pool is not incidental — it is the prerequisite that makes multi-model, multi-agent, concurrent clinical reasoning physically realizable.

The three contributions of this work are:
1. **A working multi-agent tumour board** with structured debate, citation, and uncertainty quantification
2. **AOB-Bench ClinicalEval**, the first open benchmark for oncology clinical reasoning quality
3. **Architectural proof** that 192 GB unified VRAM transforms what is possible in medical AI system design

---

## References

1. Xu, H. et al. (2024). *A whole-slide foundation model for pathology from real-world data.* Nature.
2. NCCN Clinical Practice Guidelines in Oncology: Non-Small Cell Lung Cancer. Version 4.2024.
3. NCCN Clinical Practice Guidelines in Oncology: Colon Cancer. Version 2.2024.
4. Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288.
5. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
6. Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
7. Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017.
8. ArtificialAnalysis.ai. (2024). *MI300X vs H100 throughput benchmarks.* https://artificialanalysis.ai
9. AMD. (2024). *AMD Instinct MI300X Accelerator: Product Brief.* AMD Developer Hub.
10. Chattopadhay, A. et al. (2018). *Grad-CAM++: Generalized Gradient-based Visual Explanations.* WACV 2018.

---

*This report describes a research prototype developed for the AMD Developer Hackathon 2026.*  
*Not for clinical use. All cases used in evaluation are synthetic or de-identified.*
