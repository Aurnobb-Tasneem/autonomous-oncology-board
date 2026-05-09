#!/usr/bin/env bash
# scripts/serve_specialists_fallback.sh
# =======================================
# Fallback: runs each LoRA adapter on a separate vLLM port if multi-LoRA
# hot-swap is not available on the current ROCm/vLLM build.
#
# This costs ~30 GB extra VRAM (3 × ~10 GB) but still fits in MI300X 192 GB.
#
# Ports:
#   8006 — tnm_specialist
#   8007 — biomarker_specialist  (also used for speculative decoding target)
#   8008 — treatment_specialist
#
# Usage:  bash scripts/serve_specialists_fallback.sh
# Then update env vars:
#   TNM_VLLM_BASE_URL=http://localhost:8006/v1
#   BIOMARKER_VLLM_BASE_URL=http://localhost:8007/v1
#   TREATMENT_VLLM_BASE_URL=http://localhost:8008/v1

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.05}"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Neither python3 nor python is on PATH. Set PYTHON=/path/to/python"
  exit 1
fi

echo "Starting TNM specialist on :8006 ..."
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora \
  --lora-modules tnm_specialist=ml/models/checkpoints/tnm_lora \
  --port 8006 --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 --max-model-len 2048 &

sleep 10

echo "Starting biomarker specialist on :8007 ..."
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora \
  --lora-modules biomarker_specialist=ml/models/checkpoints/biomarker_lora \
  --port 8007 --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 --max-model-len 2048 &

sleep 10

echo "Starting treatment specialist on :8008 ..."
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --enable-lora \
  --lora-modules treatment_specialist=ml/models/checkpoints/treatment_lora \
  --port 8008 --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 --max-model-len 2048 &

echo ""
echo "All three specialist adapters starting..."
echo "TNM:        http://localhost:8006/v1 (model: tnm_specialist)"
echo "Biomarker:  http://localhost:8007/v1 (model: biomarker_specialist)"
echo "Treatment:  http://localhost:8008/v1 (model: treatment_specialist)"
wait
