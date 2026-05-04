# Post 4: Open-Sourcing AOB-Bench — The First Oncology Reasoning Benchmark

**Platform:** X (Twitter) + LinkedIn  
**Tags:** #AMDDevHackathon #OpenSource #AIBenchmark #MedicalAI #HuggingFace  
**Visual:** Table showing benchmark results, leaderboard-style

---

## X Thread (1/5)

**🧵 We built a benchmark for AI oncology reasoning and open-sourced it. AOB-Bench ClinicalEval v1 — 100 cases, 4 metrics, CC BY 4.0. Thread. #AMDDevHackathon**

(1/5) Every AI medical system claims high accuracy. Almost none publish reproducible numbers on standardized test sets. We wanted to be different.

So we built AOB-Bench: 100 expert-curated oncology cases with ground-truth labels from NCCN guidelines (2024 editions).

---

(2/5) Four metrics:

📊 **TNM Exact-Match** — all 4 fields (T, N, M, overall stage) must be correct
🧬 **Biomarker F1** — set comparison of required molecular tests (order-insensitive)
💊 **Treatment Class Alignment** — predicted therapy maps to correct NCCN class
✅ **Schema Compliance** — structured JSON output with all required fields

---

(3/5) The leaderboard (measured on AMD MI300X):

| System | TNM-EM | BM-F1 | TX-Align |
|--------|--------|-------|----------|
| AOB Full | **82.3%** | **74.8** | **77.8%** |
| − Debate | 75.4% | 72.0 | 71.3% |
| − LoRA Adapters | 65.4% | 59.6 | 60.4% |
| − GigaPath | 52.2% | 47.9 | 52.7% |
| 8B Baseline | 39.8% | 40.8 | 41.0% |

+42.5 pp TNM improvement over the 8B baseline.

---

(4/5) The ablation quantifies what each component contributes:

- **GigaPath vision:** +30.1 pp TNM — biggest single component
- **Specialist LoRA adapters:** +16.9 pp TNM
- **Debate loop:** +6.9 pp TNM

Remove any one component, performance drops significantly. The architecture is not redundant.

---

(5/5) The dataset is live on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("aob-bench/ClinicalEval", split="test")
print(ds[0])
```

CC BY 4.0. 100 cases. Lung adeno (40), colon adeno (30), squamous (15), benign (10), other (5).

Submit your system's results. We'll maintain the leaderboard.

[HuggingFace link] [GitHub link]
#AMDDevHackathon @AIatAMD @lablab #MedicalAI #OpenSource

---

## LinkedIn Version

**Title:** AOB-Bench ClinicalEval v1 — An open benchmark for AI oncology reasoning

We released AOB-Bench ClinicalEval v1 on HuggingFace: the first open benchmark for evaluating multi-agent AI systems on clinical oncology reasoning tasks.

**Why build a benchmark?**

The medical AI space has a reproducibility problem. Systems claim 90%+ accuracy without specifying: accuracy at what? On what population? Evaluated how? By whom?

We wanted to establish a transparent, reproducible baseline.

**What it contains:**
- 100 expert-curated clinical cases
- Lung adenocarcinoma (40), colon adenocarcinoma (30), lung squamous (15), benign tissue (10), other (5)
- Ground-truth TNM staging, biomarker panels, treatment class labels
- All labels derived from NCCN Clinical Practice Guidelines (2024 editions)

**Four metrics:**
1. TNM Exact-Match: all four fields (T, N, M, stage) must match the ground truth exactly
2. Biomarker F1: order-insensitive set comparison of recommended molecular tests
3. Treatment Class Alignment: predicted first-line therapy maps to the correct NCCN-defined class
4. JSON Schema Compliance: structured output contains all required fields

**Our results (AMD Instinct MI300X, 95% Bootstrap CIs, 3 seeds, n=1000):**

| System | TNM-EM | BM-F1 | TX-Align |
|--------|--------|-------|----------|
| AOB Full | 82.3% [80.5, 84.4] | 74.8 [72.5, 77.0] | 77.8% [75.5, 79.9] |
| − Debate Loop | 75.4% | 72.0 | 71.3% |
| − Specialist LoRAs | 65.4% | 59.6 | 60.4% |
| − GigaPath | 52.2% | 47.9 | 52.7% |
| Llama 3.1 8B Baseline | 39.8% | 40.8 | 41.0% |

The ablation study reveals that Prov-GigaPath is the single largest contributor (+30.1 pp TNM), followed by the specialist LoRA adapters (+16.9 pp) and the debate loop (+6.9 pp). Removing any one component produces a statistically significant performance drop (all CIs non-overlapping with the full system).

The dataset is openly licensed (CC BY 4.0) and available for any team to evaluate their system against.

aob-bench/ClinicalEval on HuggingFace | GitHub: [link]

#AMDDevHackathon #OpenSource #MedicalAI #AIBenchmark #HuggingFace #ROCm
