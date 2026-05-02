#!/usr/bin/env bash
# =============================================================================
# scripts/serve_tnm_adapter.sh
# =============================================================================
# Serve the TNM LoRA adapter via vLLM with an OpenAI-compatible REST API
# on AMD Instinct MI300X (ROCm).
#
# Prerequisites
# -------------
#   1. ROCm-compatible vLLM installed:
#        pip install vllm  # uses the ROCm wheel automatically on MI300X
#   2. HF_TOKEN set (for gated model download on first run):
#        export HF_TOKEN=hf_...
#   3. LoRA adapter trained:
#        python scripts/finetune_tnm.py --output_dir aob/ml/models/checkpoints/tnm_lora
#
# Usage
# -----
#   bash scripts/serve_tnm_adapter.sh
#
# Configuration via environment variables
# ----------------------------------------
#   VLLM_PORT          vLLM listen port          (default: 8006)
#   VLLM_GPU_MEM_UTIL  GPU memory fraction       (default: 0.15  ~29 GB on MI300X)
#   BASE_MODEL         HF model ID               (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   ADAPTER_DIR        Path to LoRA adapter dir  (default: aob/ml/models/checkpoints/tnm_lora)
#
# VRAM budget
# -----------
#   Llama-3-8B-Instruct (bf16)  ~16 GB
#   LoRA adapter overhead       < 1 GB
#   vLLM KV cache (2048 ctx)    ~12 GB
#   ─────────────────────────────────────
#   Total                       ~29 GB   (gpu-memory-utilization 0.15 on 192 GB)
#   Remaining for GigaPath+70B  ~163 GB  ✓
#
# Once running, verify:
#   curl http://localhost:8006/v1/models
#
# Test staging:
#   curl http://localhost:8006/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "tnm_specialist",
#       "messages": [{"role": "user", "content":
#         "3.2 cm lung adenocarcinoma, 2/15 nodes positive, no metastasis."}],
#       "max_tokens": 80, "temperature": 0
#     }'
# =============================================================================

set -euo pipefail

PORT="${VLLM_PORT:-8006}"
GPU_MEM="${VLLM_GPU_MEM_UTIL:-0.15}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-aob/ml/models/checkpoints/tnm_lora}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is not set. Export it before running:"
  echo "  export HF_TOKEN=hf_..."
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "[ERROR] LoRA adapter not found at '${ADAPTER_DIR}'."
  echo "  Run fine-tuning first:"
  echo "    python scripts/finetune_tnm.py --output_dir ${ADAPTER_DIR}"
  exit 1
fi

echo "============================================================"
echo "  AOB TNM Specialist — vLLM OpenAI-compatible server"
echo "  AMD Instinct MI300X · ROCm"
echo "============================================================"
echo "  Base model    : ${BASE_MODEL}"
echo "  Adapter       : ${ADAPTER_DIR}"
echo "  Port          : ${PORT}"
echo "  GPU mem util  : ${GPU_MEM} ($(echo "${GPU_MEM} * 192" | bc -l | xargs printf '%.0f') GB on 192 GB MI300X)"
echo "============================================================"
echo ""
echo "Starting vLLM server (OpenAI-compatible endpoint at port ${PORT})..."
echo "Endpoint: http://localhost:${PORT}/v1/chat/completions"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules "tnm_specialist=${ADAPTER_DIR}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len 2048 \
  --dtype bfloat16
