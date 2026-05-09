#!/usr/bin/env bash
# scripts/container_start_all.sh
# =================================
# Run inside the rocm2 container — starts Ollama + FastAPI + vLLM specialists
# all on localhost, no host networking required.
#
# Usage (from the HOST):
#   CONTAINER=rocm2 bash scripts/container_start_all.sh
#
# Or directly inside the container (cd /workspace/aob first):
#   bash scripts/container_start_all.sh --inside

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER="${CONTAINER:-rocm2}"
OLLAMA_PORT=11434
API_PORT=8000
SPECIALISTS_PORT=8006
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.3:70b-instruct-q4_K_S}"

# ── If called from the HOST, re-exec ourselves inside the container ──────────
if [[ "${1:-}" != "--inside" ]]; then
  echo "==> [host] Copying latest ML code into $CONTAINER..."
  docker cp "$REPO_DIR/ml"      "$CONTAINER":/workspace/aob/
  docker cp "$REPO_DIR/scripts" "$CONTAINER":/workspace/aob/
  [ -d "$REPO_DIR/data" ] && docker cp "$REPO_DIR/data"   "$CONTAINER":/workspace/aob/ || true
  [ -d "$REPO_DIR/eval" ] && docker cp "$REPO_DIR/eval"   "$CONTAINER":/workspace/aob/ || true

  # Read HF_TOKEN from .env if not set
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
  fi

  echo "==> [host] Launching container_start_all.sh inside $CONTAINER..."
  docker exec -it \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e OLLAMA_MODEL="$OLLAMA_MODEL" \
    -e SPECIALISTS_PORT="$SPECIALISTS_PORT" \
    "$CONTAINER" bash -lc \
    "cd /workspace/aob && export PYTHONPATH=/workspace/aob && bash scripts/container_start_all.sh --inside"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Everything below runs INSIDE the container
# ─────────────────────────────────────────────────────────────────────────────
cd /workspace/aob
export PYTHONPATH=/workspace/aob

echo ""
echo "============================================================"
echo "  AOB All-in-One Container Startup"
echo "  Ollama + FastAPI + Specialists — all on localhost"
echo "============================================================"
echo ""

# ── 1. Install Ollama if missing ─────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "--> Installing Ollama (ROCm-native)..."
  # zstd is required by the Ollama installer since 0.6+
  if ! command -v zstd &>/dev/null; then
    echo "    Installing zstd (required by Ollama installer)..."
    apt-get install -y -qq zstd 2>/dev/null || true
  fi
  curl -fsSL https://ollama.com/install.sh | sh
  echo "    OK"
else
  echo "--> Ollama: $(ollama --version 2>&1 | head -1)"
fi

# ── 2. Start Ollama server (if not already running) ──────────────────────────
if ! pgrep -x ollama &>/dev/null; then
  echo "--> Starting Ollama server on 127.0.0.1:${OLLAMA_PORT}..."
  OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" nohup ollama serve \
    > /var/log/ollama_container.log 2>&1 &
  sleep 5
else
  echo "--> Ollama already running (PID $(pgrep -x ollama | head -1))"
fi

# ── 3. Pull model if missing ─────────────────────────────────────────────────
echo "--> Checking model: $OLLAMA_MODEL"
if ! OLLAMA_HOST=http://127.0.0.1:${OLLAMA_PORT} ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
  echo "    Pulling ${OLLAMA_MODEL} (may take a few minutes)..."
  OLLAMA_HOST=http://127.0.0.1:${OLLAMA_PORT} ollama pull "$OLLAMA_MODEL"
else
  echo "    Model already present."
fi

# Quick ping
if curl -fsS --max-time 5 http://127.0.0.1:${OLLAMA_PORT}/api/tags &>/dev/null; then
  echo "--> Ollama: reachable at http://127.0.0.1:${OLLAMA_PORT} ✓"
else
  echo "    WARN: Ollama not responding — check /var/log/ollama_container.log"
fi

# ── 4. Start vLLM Specialists (if adapters present) ──────────────────────────
TNM_DIR="ml/models/checkpoints/tnm_lora"
BIO_DIR="ml/models/checkpoints/biomarker_lora"
TRX_DIR="ml/models/checkpoints/treatment_lora"

if [ -d "$TNM_DIR" ] && [ -d "$BIO_DIR" ] && [ -d "$TRX_DIR" ]; then
  if command -v python3 &>/dev/null && python3 -c "import vllm" &>/dev/null; then
    echo "--> Starting vLLM specialist server on :${SPECIALISTS_PORT}..."
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 1
    SPECIALISTS_PORT="$SPECIALISTS_PORT" nohup python3 -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --enable-lora \
      --max-lora-rank 16 \
      --lora-modules \
        tnm_specialist="${TNM_DIR}" \
        biomarker_specialist="${BIO_DIR}" \
        treatment_specialist="${TRX_DIR}" \
      --port "${SPECIALISTS_PORT}" \
      --gpu-memory-utilization 0.12 \
      --dtype bfloat16 \
      --max-model-len 2048 \
      --trust-remote-code \
      > /var/log/aob_specialists.log 2>&1 &
    sleep 3
    echo "    Specialists server PID=$! (log: /var/log/aob_specialists.log)"
  else
    echo "--> vLLM not installed inside container — skipping specialists."
    echo "    (Specialists run in degraded mode; core board still works via Ollama)"
  fi
else
  echo "--> Adapter dirs missing — skipping specialists."
fi

# ── 5. Stop any existing uvicorn ─────────────────────────────────────────────
echo "--> Stopping old uvicorn (if any)..."
pkill -f uvicorn 2>/dev/null || true
sleep 2

# ── 6. Start FastAPI ─────────────────────────────────────────────────────────
echo "--> Starting FastAPI on :${API_PORT}..."
export OLLAMA_HOST="http://127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_MODEL="$OLLAMA_MODEL"
export OLLAMA_REQUEST_TIMEOUT="${OLLAMA_REQUEST_TIMEOUT:-600}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export HF_TOKEN="${HF_TOKEN:-}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export TNM_VLLM_BASE_URL="http://127.0.0.1:${SPECIALISTS_PORT}/v1"
export BIOMARKER_VLLM_BASE_URL="http://127.0.0.1:${SPECIALISTS_PORT}/v1"
export TREATMENT_VLLM_BASE_URL="http://127.0.0.1:${SPECIALISTS_PORT}/v1"

nohup uvicorn ml.api:app --host 0.0.0.0 --port ${API_PORT} \
  > /var/log/aob_api.log 2>&1 &
API_PID=$!

# ── 7. Health check ───────────────────────────────────────────────────────────
echo "--> Waiting for API to become healthy..."
for i in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:${API_PORT}/health &>/dev/null; then
    echo ""
    echo "==> All services running:"
    echo "    Ollama    : http://127.0.0.1:${OLLAMA_PORT}"
    echo "    FastAPI   : http://127.0.0.1:${API_PORT}/health"
    echo "    Specialists: http://127.0.0.1:${SPECIALISTS_PORT}/v1/models"
    echo ""
    curl -sS http://127.0.0.1:${API_PORT}/health 2>/dev/null | python3 -m json.tool 2>/dev/null || true
    exit 0
  fi
  echo "    ($i/30) not ready — sleeping 2s..."
  sleep 2
done

echo ""
echo "==> API did not become healthy in 60s. Showing logs:"
tail -40 /var/log/aob_api.log
exit 1
