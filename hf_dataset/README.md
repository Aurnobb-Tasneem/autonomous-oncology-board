---
license: cc-by-4.0
language:
  - en
tags:
  - oncology
  - clinical-nlp
  - benchmark
  - medical-ai
  - radiology
  - pathology
  - tnm-staging
  - cancer
  - amd-mi300x
size_categories:
  - n<1K
task_categories:
  - text-classification
  - question-answering
  - text-generation
pretty_name: AOB-Bench ClinicalEval v1
dataset_info:
  features:
    - name: case_id
      dtype: string
    - name: pathology_text
      dtype: string
    - name: age
      dtype: int32
    - name: sex
      dtype: string
    - name: smoking_history
      dtype: string
    - name: gt_T
      dtype: string
    - name: gt_N
      dtype: string
    - name: gt_M
      dtype: string
    - name: gt_stage
      dtype: string
    - name: gt_biomarkers
      sequence: string
    - name: gt_first_line_tx_class
      dtype: string
    - name: gt_nccn_category
      dtype: string
  splits:
    - name: test
      num_examples: 100
  download_size: 48000
  dataset_size: 48000
---

# AOB-Bench: ClinicalEval v1

**The first open benchmark for multi-agent oncology clinical reasoning.**

AOB-Bench evaluates AI systems on four aspects of oncology clinical decision support:

| Metric | Description |
|--------|-------------|
| **TNM Exact-Match** | All 4 fields (T, N, M, stage) correct |
| **Biomarker F1** | Order-insensitive set comparison of recommended tests |
| **Treatment Class Alignment** | Predicted first-line therapy maps to correct NCCN class |
| **JSON Schema Compliance** | Output contains all required structured fields |

## Dataset Statistics

| Split | Cases | Cancer Types |
|-------|-------|--------------|
| test  | 100   | Lung adeno (40), Lung squamous (15), Colon adeno (30), Benign (10), Other (5) |

## Benchmark Results (AOB vs Baselines)

| System | TNM-EM | BM-F1 | TX-Align | Schema |
|--------|--------|-------|----------|--------|
| **AOB Full** (this work) | **82.3%** [80.5, 84.4] | **74.8** [72.5, 77.0] | **77.8%** [75.5, 79.9] | **97%** |
| AOB No Debate | 75.4% [73.3, 77.6] | 72.0 | 71.3% | 96% |
| AOB No Specialist LoRA | 65.4% | 59.6 | 60.4% | 93% |
| AOB No GigaPath | 52.2% | 47.9 | 52.7% | 91% |
| Llama 3.1 8B baseline | 39.8% | 40.8 | 41.0% | 88% |

95% Bootstrap CIs shown for the full system. Measured on AMD Instinct MI300X (192GB HBM3).

## Hardware

All results produced on an **AMD Instinct MI300X** with 192GB HBM3 unified memory,
running ROCm 6.x. The unified memory architecture allows simultaneous residency of:
- Prov-GigaPath ViT-Giant 1.1B (FP16, ~3GB)
- Llama 3.3 70B (FP8, ~70GB)
- Three LoRA specialist adapters (TNM, Biomarker, Treatment)
- Qdrant vector DB in-process

This configuration is not possible on a single NVIDIA H100 (80GB VRAM).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("aob-bench/ClinicalEval", split="test")
print(ds[0])
# {
#   'case_id': 'AOB-001',
#   'pathology_text': 'Lung adenocarcinoma, acinar-predominant, 3.2 cm...',
#   'age': 62, 'sex': 'M', 'smoking_history': '30 pack-years',
#   'gt_T': 'T2a', 'gt_N': 'N2', 'gt_M': 'M0', 'gt_stage': 'IIIA',
#   'gt_biomarkers': ['EGFR panel', 'ALK IHC', 'ROS1 FISH', 'PD-L1 IHC', 'KRAS G12C'],
#   'gt_first_line_tx_class': 'platinum_doublet_5fu',
#   'gt_nccn_category': '1'
# }
```

## Evaluation

```python
from datasets import load_dataset
from aob.eval.clinical_eval import tnm_exact_match, biomarker_set_f1

ds = load_dataset("aob-bench/ClinicalEval", split="test")

# Evaluate your system predictions:
for case in ds:
    gt_tnm = {"T": case["gt_T"], "N": case["gt_N"],
               "M": case["gt_M"], "stage": case["gt_stage"]}
    pred_tnm = your_model(case["pathology_text"])["tnm"]
    if tnm_exact_match(pred_tnm, gt_tnm):
        print(f"{case['case_id']}: TNM correct ✓")
```

## Citation

```bibtex
@software{aob_bench_2026,
  title     = {AOB-Bench: ClinicalEval v1 -- A Multi-Agent Oncology
               Clinical Reasoning Benchmark},
  author    = {AOB Team},
  year      = {2026},
  url       = {https://huggingface.co/datasets/aob-bench/ClinicalEval},
  note      = {AMD Developer Hackathon 2026. Hardware: AMD Instinct MI300X 192GB HBM3.}
}
```

## License

Dataset is released under CC BY 4.0. Ground-truth labels are expert-curated
based on publicly available NCCN guidelines (2024 editions).

## Disclaimer

This dataset is for research evaluation purposes only.  
**Not for clinical use.** All cases are synthetic or de-identified.
