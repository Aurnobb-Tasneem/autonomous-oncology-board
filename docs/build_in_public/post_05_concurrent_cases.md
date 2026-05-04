# Post 5: 3 Cancer Types, Simultaneously — The Concurrent Case Demo

**Platform:** X (Twitter) + LinkedIn  
**Tags:** #AMDDevHackathon #AMD #MI300X #ROCm #MedicalAI #VRAM  
**Visual:** VRAM timeseries chart showing three concurrent cases climbing to 118 GB

---

## X Thread (1/5)

**🧵 We ran 3 different cancer types through our AI tumour board simultaneously. 118 GB peak VRAM. 31 seconds. All done. Here's what that looks like. #AMDDevHackathon**

(1/5) Production medical AI doesn't analyze one patient at a time. A real hospital might have 30 cases queued for a Monday morning tumour board.

We wanted to demonstrate that our architecture scales. So we ran three simultaneously.

---

(2/5) The three cases:

🫁 **Case A:** Lung Adenocarcinoma — EGFR L858R, Stage IV, treatment-naive
🫀 **Case B:** Colon Adenocarcinoma — MSI-H, Stage IVA, Lynch syndrome suspected  
🫁 **Case C:** Lung Squamous — PD-L1 TPS 80%, Stage IV, treatment-naive

All three submitted simultaneously via `/api/concurrent/run`.

---

(3/5) Memory usage during concurrent execution:

```
0 GB   →   GigaPath loads:       3 GB
3 GB   →   Llama 3.3 70B loads: 73 GB  
73 GB  →   Case A starts:       88 GB
88 GB  →   Cases B+C add batch: 112 GB
Peak: 118 GB / 192 GB (61% utilization)
```

H100 would OOM during Case A. We still had 74 GB to spare at peak.

[INSERT: VRAM timeseries screenshot]

---

(4/5) All three completed without errors:
- Case A: 23 seconds
- Case B: 28 seconds  
- Case C: 31 seconds
- Total wall-clock: 31 seconds (parallel)

Sequential on H100 (if it could fit): estimated 90+ seconds with model swap overhead per case.

---

(5/5) The live VRAM dashboard widget shows this in real time during the demo. Judges can watch the memory bar climb as each agent loads.

And next to it: the H100 bar. Capped at 80 GB. Labeled "OOM."

Not a claim. A visualization. Of math.

#AMDDevHackathon @AIatAMD @lablab #MI300X #ROCm #MedicalAI

---

## LinkedIn Version

**Title:** Three simultaneous cancer cases on one GPU — the concurrent case stress test

For the AMD Developer Hackathon, we submitted three different cancer analyses to our AI tumour board simultaneously:

- **Case A:** Lung Adenocarcinoma, EGFR L858R, Stage IV (treatment-naive)
- **Case B:** Colon Adenocarcinoma, MSI-H dMMR, Stage IVA (Lynch syndrome)
- **Case C:** Lung Squamous Cell Carcinoma, PD-L1 TPS 80%, Stage IV

All three cases submitted at the same time via the `/api/concurrent/run` endpoint.

**VRAM progression:**

| Event | VRAM Used |
|-------|-----------|
| GigaPath loads | 3 GB |
| Llama 3.3 70B loads | 73 GB |
| Case A pipeline active | 88 GB |
| Cases B + C active | 112 GB |
| **Peak (3 concurrent)** | **118 GB** |
| Headroom remaining | 74 GB |

All three completed successfully. Total wall-clock time: 31 seconds. No OOM errors. No model swapping. No cache eviction.

**Why this matters:**

In a real hospital deployment, a tumour board coordinator might queue 10–30 cases to be analyzed overnight. A system that processes them concurrently can deliver results before morning rounds. A system limited to sequential processing with model swaps cannot.

The AMD MI300X's unified 192 GB memory pool is the prerequisite for this capability. The same architecture on a single H100 (80 GB) would fail during Case A — before Cases B and C even begin.

We built a real-time VRAM dashboard into the AOB UI. It polls `rocm-smi` every 2 seconds and displays:
1. A live animated bar showing current VRAM usage
2. A static comparison bar showing the H100's 80 GB limit (labeled "OOM")
3. Model breakdown labels (GigaPath, 70B, KV Cache, adapters)

During the demo, judges watch this bar fill from 0 to 118 GB as three patients are analyzed simultaneously. That visualization is the "No-NVIDIA" proof — live, in real time, during the presentation.

Open source: [GitHub link] | Benchmark: aob-bench/ClinicalEval on HuggingFace

#AMDDevHackathon #MI300X #ROCm #MedicalAI #ProductionAI #ConcurrentInference
