#!/usr/bin/env bash
# scripts/serve_specialists.sh
# ==============================
# Launch vLLM with all three LoRA specialist adapters hot-swapped on one
# Llama-3.1-8B-Instruct base model on AMD MI300X.
#
# VRAM budget (MI300X 192 GB HBM3):
#   Llama-3.1-8B base (bf16)    ~16 GB
#   Three LoRA adapters         ~0.3 GB total (adapter weights are tiny)
#   KV cache + inference         ~6 GB
#   Total:                      ~22 GB  (leaves >170 GB for GigaPath + 70B)
#   vLLM gpu-memory-utilization: 0.12 = ~23 GB on 192 GB device
#
# Adapters served:
#   tnm_specialist      → POST /v1/chat/completions with model "tnm_specialist"
#   biomarker_specialist → POST /v1/chat/completions with model "biomarker_specialist"
#   treatment_specialist → POST /v1/chat/completions with model "treatment_specialist"
#
# Usage:
#   # On the MI300X host (ROCm-native):
#   export HF_TOKEN=hf_...
#   bash scripts/serve_specialists.sh
#
#   # Test any adapter:
#   curl http://localhost:8006/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "tnm_specialist",
#       "messages": [{"role":"user","content":"3.2 cm lung adenocarcinoma, 2/15 nodes positive."}],
#       "max_tokens": 80, "temperature": 0
#     }'
#
# Fallback (if vLLM multi-LoRA hot-swap fails on ROCm):
#   Run three separate servers:  serve_specialists_fallback.sh
#   Ports: 8006 (tnm), 8007 (biomarker), 8008 (treatment)
#
# Environment variables:
#   SPECIALISTS_PORT         Port for the multi-LoRA vLLM server   [default: 8006]
#   GPU_MEM_UTIL             vLLM GPU memory utilization fraction   [default: 0.12]
#   TNM_LORA_DIR             Path to TNM adapter                    [default: aob/ml/models/checkpoints/tnm_lora]
#   BIOMARKER_LORA_DIR       Path to biomarker adapter              [default: aob/ml/models/checkpoints/biomarker_lora]
#   TREATMENT_LORA_DIR       Path to treatment adapter              [default: aob/ml/models/checkpoints/treatment_lora]

set -euo pipefail

SPECIALISTS_PORT="${SPECIALISTS_PORT:-8006}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"
TNM_LORA_DIR="${TNM_LORA_DIR:-aob/ml/models/checkpoints/tnm_lora}"
BIOMARKER_LORA_DIR="${BIOMARKER_LORA_DIR:-aob/ml/models/checkpoints/biomarker_lora}"
TREATMENT_LORA_DIR="${TREATMENT_LORA_DIR:-aob/ml/models/checkpoints/treatment_lora}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# ── Validate adapter directories exist ───────────────────────────────────────
for dir in "$TNM_LORA_DIR" "$BIOMARKER_LORA_DIR" "$TREATMENT_LORA_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️  Adapter directory not found: $dir"
        echo "   Train the adapters first:"
        echo "     python scripts/finetune_tnm.py"
        echo "     python scripts/finetune_biomarker.py"
        echo "     python scripts/finetune_treatment.py"
        echo ""
        echo "   Or run in smoke-test mode (50 steps each):"
        echo "     python scripts/finetune_tnm.py --max_steps 50"
        echo "     python scripts/finetune_biomarker.py --max_steps 50"
        echo "     python scripts/finetune_treatment.py --max_steps 50"
        exit 1
    fi
done

echo "============================================================"
echo "  AOB Specialist Suite — vLLM Multi-LoRA Server"
echo "  AMD MI300X · ROCm · Three Adapters · Hot-Swap"
echo "============================================================"
echo "  Base model        : $BASE_MODEL"
echo "  Port              : $SPECIALISTS_PORT"
echo "  GPU mem util      : $GPU_MEM_UTIL  (~$(echo "$GPU_MEM_UTIL * 192" | bc)GB on MI300X)"
echo "  TNM adapter       : $TNM_LORA_DIR"
echo "  Biomarker adapter : $BIOMARKER_LORA_DIR"
echo "  Treatment adapter : $TREATMENT_LORA_DIR"
echo "============================================================"
echo ""
echo "Model aliases:"
echo "  'tnm_specialist'       → TNM staging JSON"
echo "  'biomarker_specialist' → Biomarker panel JSON"
echo "  'treatment_specialist' → Treatment plan JSON"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
    tnm_specialist="$TNM_LORA_DIR" \
    biomarker_specialist="$BIOMARKER_LORA_DIR" \
    treatment_specialist="$TREATMENT_LORA_DIR" \
  --port "$SPECIALISTS_PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --trust-remote-code
