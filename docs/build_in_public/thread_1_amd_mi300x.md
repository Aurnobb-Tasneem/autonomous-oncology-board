# Thread 1: Why AMD MI300X Changes AI in Medicine
**Build in Public — Autonomous Oncology Board | @AurnabbTasneem**

---

**1/12**
We just built something that's physically impossible on a single NVIDIA H100.

An AI tumour board that runs GigaPath (1.1B ViT) + Llama 3.3 70B *simultaneously* — no model swapping, no API calls.

Here's how the AMD MI300X changes what's possible in clinical AI 🧵

---

**2/12**
The problem with medical AI today:

→ Pathology models need GPU inference on gigapixel images
→ LLMs need 40+ GB VRAM for 70B parameter models
→ Both at once = impossible on consumer or even enterprise NVIDIA hardware

The H100 has 80 GB VRAM. Llama 70B alone takes ~40 GB.
That leaves zero room for a pathology model.

---

**3/12**
The AMD MI300X has 192 GB unified HBM3 memory.

Not 80 GB. Not 96 GB. **192 GB.**

That's not an incremental improvement. That's a different category of hardware.
It changes the architecture of what you can build.

---

**4/12**
Our model breakdown:

| Model | VRAM |
|-------|------|
| Prov-GigaPath ViT-Giant | ~3.2 GB |
| Llama 3.3 70B (via Ollama) | ~40 GB |
| KV cache + overhead | ~3 GB |
| **Total** | **~46 GB** |

On H100: you'd have to choose one. On MI300X: both run simultaneously, always warm.

---

**5/12**
Why does "always warm" matter for medicine?

When a clinician submits a case, both models are already loaded in HBM3.
No cold start. No model loading delay.

In an emergency tumour board scenario, latency isn't inconvenience — it's clinical outcome.

---

**6/12**
The MI300X uses ROCm — AMD's open-source GPU stack.

We ran everything in Docker with PyTorch ROCm builds.
Zero code changes from CUDA — just swap the base image.

```bash
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.11_pytorch_release_2.3.0
```

That's the entire hardware migration.

---

**7/12**
GigaPath is a 1.1 billion parameter Vision Transformer trained on 1.3 billion histopathology patches.

It understands tissue at a level no radiologist can process manually at scale.

We load it once at startup. It stays in HBM3. Every case gets full inference.

---

**8/12**
The pathologist pipeline:

1. Preprocess patches → FP16 tensors (224×224)
2. GigaPath embeds → 1536-dim vectors
3. Cosine similarity to tissue class prototypes
4. MC Dropout (20 passes) → uncertainty intervals
5. Attention rollout → heatmaps highlighting suspicious regions

All in one forward pass, on one GPU.

---

**9/12**
Then Llama 3.3 70B takes the embedding report and acts as a senior oncologist.

It reads:
→ Tissue classification + confidence
→ Abnormality scores per patch
→ NCCN guideline evidence (RAG)
→ Similar past cases from board memory

And produces a complete patient management plan.

---

**10/12**
The key insight: unified memory architecture changes the software design too.

On NVIDIA, you'd architect for GPU scarcity — cache models, queue requests, use smaller models.

On MI300X, you architect for capability — run the best model, keep it warm, add features.

Different hardware → different system design philosophy.

---

**11/12**
For healthcare AI specifically, this matters enormously:

→ Rural hospitals can't afford 4x H100 clusters
→ A single MI300X node can run a complete diagnostic AI suite
→ Open-source stack (ROCm) means no vendor lock-in
→ 192 GB means you can add models as medicine advances

The democratisation of clinical AI infrastructure.

---

**12/12**
We built this for the AMD MI300X hackathon.

The code is open-source: github.com/Aurnobb-Tasneem/autonomous-oncology-board

→ GigaPath on ROCm ✅
→ Llama 3.3 70B ✅
→ Multi-agent debate ✅
→ 192 GB HBM3 utilisation ✅

What would you build if VRAM wasn't a constraint?

#AMD #MI300X #MedicalAI #ROCm #BuildInPublic
