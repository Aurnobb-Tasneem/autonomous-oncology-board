# VRAM Architecture Comparison: AMD MI300X vs NVIDIA H100

## The Memory Arithmetic That Changes Everything

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VRAM BUDGET COMPARISON                                    │
│                                                                              │
│  NVIDIA H100 SXM5 (80 GB HBM3e)          AMD Instinct MI300X (192 GB HBM3)  │
│  ──────────────────────────────           ──────────────────────────────────  │
│                                                                              │
│  ████████████████████████████████         ██████░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  ▲ 80 GB ──── FULL ────────────           ▲ 128 GB used                      │
│                                                                              │
│  Llama 3.3 70B (FP8):  70 GB ████        Llama 3.3 70B (FP8):  70 GB █      │
│  GigaPath (FP16):       3 GB █           Llama 3.1 8B + LoRAs: 16 GB █      │
│  vLLM KV cache:        ~7 GB █  ──────►  GigaPath (FP16):       3 GB █      │
│                        ──────            vLLM KV cache:         30 GB █      │
│  TOTAL:                80 GB            Qdrant + overhead:       9 GB █      │
│  HEADROOM:              0 GB 🔴          ──────────────────────────────       │
│                                          TOTAL:                128 GB        │
│  ✗ NO ROOM FOR:                          HEADROOM:              64 GB 🟢     │
│    • Full KV cache for debate                                                │
│    • 3 concurrent cases                  ✓ ENABLES:                         │
│    • LoRA adapter hot-swap               • Full debate loop with full KV     │
│    • Board memory (Qdrant)               • 3 concurrent cases (peak ~120GB)  │
│    • Speculative draft model             • All 3 LoRA adapters hot-swapped   │
│                                          • Speculative 8B draft model        │
│                                          • Board memory + trial corpus       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Concurrent Case Performance

```
Time (seconds) →
0    5   10   15   20   25   30   35   40   45
│    │    │    │    │    │    │    │    │    │

MI300X — 3 concurrent cases:
Case A (Lung Adeno): ████████████████████░░░░░ → 23s
Case B (Colon MSI-H): ████████████████████████░ → 28s
Case C (Lung SqCC):  ██████████████████████████ → 31s
Peak VRAM: 118 GB / 192 GB                    All done: 31s

H100 (sequential, model swaps):
Case A: ████ swap ██████████ swap ████████       41s
Case B: (waiting)            ████ swap ████████  82s
Case C: (waiting)                     ████ ████ 120s+
```

## The "No-NVIDIA" Proof

| Evidence Type | What It Shows |
|---------------|---------------|
| **Memory arithmetic** | H100 80GB < 128GB required. Mathematical impossibility. |
| **rocm-smi screenshot** | `88,238 MiB / 196,608 MiB` on ONE device, both models loaded |
| **Live VRAM dashboard** | Real-time bar chart in the UI, visible to judges during demo |
| **Concurrent stress test** | 3 cases × 3 agents = 9 LLM calls simultaneously, no OOM |
| **ArtificialAnalysis.ai** | Third-party benchmark: MI300X 1.3× H100 throughput on 70B models |

> **"We didn't compromise the architecture to fit the hardware.  
> We chose hardware that made the architecture possible."**
