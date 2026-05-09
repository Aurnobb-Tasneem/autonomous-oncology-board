#!/usr/bin/env bash
# scripts/bootstrap.sh
# ====================
# Run ONCE on a fresh AMD MI300X instance to set up the AOB project.
#
# Usage:
#   ssh root@<amd-ip>
#   git clone https://github.com/Aurnobb-Tasneem/autonomous-oncology-board.git /root/aob
#   bash /root/aob/scripts/bootstrap.sh

set -euo pipefail

REPO_DIR="/root/aob"
CONTAINER="rocm"
OLLAMA_PORT=11434

echo ""
echo "============================================"
echo "  AOB Bootstrap — AMD MI300X"
echo "============================================"
echo ""

# ── 1. Verify Docker container is running ─────────────────────────────────
echo "--> Checking Docker container '$CONTAINER'..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "ERROR: Docker container '$CONTAINER' is not running."
  echo "Start it first, then re-run this script."
  exit 1
fi
echo "    OK: container is running"

# ── 2. Set up .env ─────────────────────────────────────────────────────────
echo "--> Setting up .env..."
if [ ! -f "$REPO_DIR/.env" ]; then
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
  echo "    Created .env from .env.example"
  echo "    IMPORTANT: Edit /root/aob/.env and set your HF_TOKEN before continuing!"
  read -p "    Press Enter once you've set HF_TOKEN in .env..."
fi
export $(grep -v '^#' "$REPO_DIR/.env" | xargs)

# ── 2b. Detect Docker bridge gateway and persist OLLAMA_HOST to .env ──────
# The Docker bridge gateway varies by machine configuration. Detect it from
# the running container rather than assuming 172.17.0.1 (which breaks on
# custom bip= networks, rootless Docker, or overlapping subnets).
detect_docker_gateway() {
  local gw nm
  nm=$(docker inspect "$CONTAINER" --format '{{.HostConfig.NetworkMode}}' 2>/dev/null || true)
  if [ "$nm" = "host" ]; then
    echo "127.0.0.1"
    return
  fi
  # 1. Ask the running container what its default gateway is (most reliable).
  gw=$(docker exec "$CONTAINER" sh -c "ip route 2>/dev/null | awk '/default/ {print \$3; exit}'" 2>/dev/null || true)
  if [ -n "$gw" ]; then echo "$gw"; return; fi
  # 2. Fall back to the host's docker0 inet address.
  gw=$(ip -4 addr show docker0 2>/dev/null | awk '/inet / {sub(/\/.*/,"",$2); print $2; exit}' || true)
  if [ -n "$gw" ]; then echo "$gw"; return; fi
  # 3. Last resort: the historical default.
  echo "172.17.0.1"
}
DOCKER_GW=$(detect_docker_gateway)
DETECTED_OLLAMA_HOST="http://${DOCKER_GW}:${OLLAMA_PORT}"
echo "--> Detected Docker gateway: ${DOCKER_GW}  →  OLLAMA_HOST=${DETECTED_OLLAMA_HOST}"

# Write detected value back to .env so deploy.sh also picks it up.
if ! grep -q '^OLLAMA_HOST=' "$REPO_DIR/.env"; then
  echo "OLLAMA_HOST=${DETECTED_OLLAMA_HOST}" >> "$REPO_DIR/.env"
elif grep -qE '^OLLAMA_HOST=http://172\.(17|18)\.0\.1' "$REPO_DIR/.env"; then
  # Refresh stale bridge defaults (common mismatch when the ML container uses host networking).
  sed -i "s|^OLLAMA_HOST=.*|OLLAMA_HOST=${DETECTED_OLLAMA_HOST}|" "$REPO_DIR/.env"
fi
export OLLAMA_HOST="${DETECTED_OLLAMA_HOST}"

# Also open firewall for the detected subnet (not just the hardcoded /16).
DOCKER_SUBNET=$(docker exec "$CONTAINER" sh -c \
  "ip route 2>/dev/null | awk '/default/ {print \$3}' | sed 's/\.[0-9]*$/.0\/24/'" 2>/dev/null || true)

# ── 3. Start Ollama on HOST ────────────────────────────────────────────────
echo "--> Checking Ollama..."
if ! pgrep -x ollama > /dev/null; then
  echo "    Starting Ollama server..."
  OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" nohup ollama serve > /var/log/ollama.log 2>&1 &
  sleep 3
fi

echo "--> Pulling Llama 3.3 70B FP16 (large download; first run may take a long time)..."
ollama pull llama3.3:70b-instruct-q4_K_S

# Allow Docker → host Ollama traffic (use detected subnet, fallback to /16).
echo "--> Opening firewall port ${OLLAMA_PORT} for Docker subnet..."
if [ -n "${DOCKER_SUBNET:-}" ]; then
  ufw allow from "${DOCKER_SUBNET}" to any port "${OLLAMA_PORT}" 2>/dev/null || true
fi
# Also allow the classic bridge subnet as a belt-and-suspenders backup.
ufw allow from 172.17.0.0/16 to any port "${OLLAMA_PORT}" 2>/dev/null || true

# ── 4. Copy code into container ────────────────────────────────────────────
echo "--> Copying project into container..."
docker exec "$CONTAINER" mkdir -p /workspace/aob
docker cp "$REPO_DIR/." "$CONTAINER":/workspace/aob/
echo "    OK"

# ── 5. Install Python dependencies ────────────────────────────────────────
echo "--> Installing Python dependencies inside container..."
docker exec "$CONTAINER" pip install -r /workspace/aob/ml/requirements.txt -q
echo "    OK"

# ── 5b. Build & start Next.js frontend on the HOST ────────────────────────
# The Next.js frontend runs on the HOST (not inside the ROCm container) so it
# can serve traffic on port 3000 without ROCm/CUDA constraints. The frontend
# talks to the FastAPI backend on port 8000 via Next.js rewrites.
echo "--> Checking Node.js (need >=20 for Next.js 16)..."
NODE_OK=false
if command -v node > /dev/null; then
  NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo "0")
  if [ "${NODE_MAJOR}" -ge 20 ]; then
    NODE_OK=true
  fi
fi
if [ "$NODE_OK" = "false" ]; then
  echo "    Installing Node.js 20.x via NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null
  apt-get install -y nodejs > /dev/null
fi
echo "    Node $(node -v) / npm $(npm -v)"

echo "--> Configuring frontend env..."
cd "$REPO_DIR/frontend"
if [ ! -f ".env.local" ]; then
  {
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000"
    echo "BACKEND_INTERNAL_URL=http://localhost:8000"
  } > .env.local
fi

echo "--> Installing frontend dependencies (npm ci)..."
npm ci --prefer-offline --no-audit --no-fund

echo "--> Building Next.js frontend (this takes ~1–2 min)..."
npm run build

echo "--> Starting Next.js server on :3000..."
pkill -f "next-server" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
nohup npx next start -p 3000 > /var/log/aob_frontend.log 2>&1 &
sleep 4
ufw allow 3000 2>/dev/null || true

if curl -sf http://localhost:3000 > /dev/null; then
  echo "    OK — frontend live on :3000"
else
  echo "    WARNING: frontend health check failed — see /var/log/aob_frontend.log"
fi
cd "$REPO_DIR"

# ── 6. Index the corpus ────────────────────────────────────────────────────
echo "--> Indexing oncology corpus into Qdrant..."
docker exec "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export HF_TOKEN=${HF_TOKEN:-''} &&
  python scripts/index_corpus.py
"
echo "    OK"

# ── 7. Run smoke test ──────────────────────────────────────────────────────
echo "--> Running smoke test..."
docker exec "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export OLLAMA_HOST=${OLLAMA_HOST} &&
  export OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.3:70b-instruct-q4_K_S} &&
  export OLLAMA_REQUEST_TIMEOUT=${OLLAMA_REQUEST_TIMEOUT:-600} &&
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m} &&
  export HF_TOKEN=${HF_TOKEN:-''} &&
  python scripts/smoke_test.py
" && echo "    Smoke test PASSED" || echo "    WARNING: Smoke test had issues — check output above"

# ── 8. Open port 8000 in firewall ─────────────────────────────────────────
echo "--> Opening port 8000 in firewall..."
ufw allow 8000 2>/dev/null || true

# ── 9. Start the API ───────────────────────────────────────────────────────
echo "--> Starting FastAPI server..."
docker exec "$CONTAINER" pkill -f uvicorn 2>/dev/null || true
sleep 1
docker exec -d "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export OLLAMA_HOST=${OLLAMA_HOST} &&
  export OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.3:70b-instruct-q4_K_S} &&
  export OLLAMA_REQUEST_TIMEOUT=${OLLAMA_REQUEST_TIMEOUT:-600} &&
  export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m} &&
  export HF_TOKEN=${HF_TOKEN:-''} &&
  export QWEN_VL_MODEL=${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct} &&
  export TNM_VLLM_BASE_URL=${TNM_VLLM_BASE_URL:-http://localhost:8006/v1} &&
  export BIOMARKER_VLLM_BASE_URL=${BIOMARKER_VLLM_BASE_URL:-http://localhost:8006/v1} &&
  export TREATMENT_VLLM_BASE_URL=${TREATMENT_VLLM_BASE_URL:-http://localhost:8006/v1} &&
  uvicorn ml.api:app --host 0.0.0.0 --port 8000 > /var/log/aob_api.log 2>&1
"
sleep 4

# ── 9b. Start vLLM specialist adapters (LoRA), if trained ─────────────────
# The three LoRA adapters (TNM / Biomarker / Treatment) are served via vLLM's
# multi-LoRA hot-swap on a single Llama-3.1-8B base. Skip silently if the
# adapter checkpoints are not present (they require training first).
if [ -d "$REPO_DIR/ml/models/checkpoints/tnm_lora" ]; then
  echo "--> Starting vLLM specialist server on :8006 (multi-LoRA hot-swap)..."
  docker exec "$CONTAINER" pkill -f "vllm.entrypoints" 2>/dev/null || true
  sleep 1
  docker exec -d "$CONTAINER" bash -c "
    cd /workspace/aob &&
    export PYTHONPATH=/workspace/aob &&
    export HF_TOKEN=${HF_TOKEN:-''} &&
    bash scripts/serve_specialists.sh > /var/log/aob_specialists.log 2>&1
  "
  ufw allow 8006 2>/dev/null || true
  echo "    OK (warmup ~60s; check /var/log/aob_specialists.log)"
else
  echo "    Skipping specialists server — LoRA adapters not trained yet."
  echo "    (Run scripts/finetune_tnm.py / finetune_biomarker.py / finetune_treatment.py first.)"
fi

# ── 10. Health check ───────────────────────────────────────────────────────
echo "--> Health check..."
API_OK=false
FE_OK=false
if curl -sf http://localhost:8000/health > /dev/null; then
  API_OK=true
fi
if curl -sf http://localhost:3000 > /dev/null; then
  FE_OK=true
fi

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<host>")

echo ""
echo "============================================"
echo "  AOB Bootstrap Complete"
echo "============================================"
if [ "$API_OK" = "true" ]; then
  echo "  FastAPI :8000  → http://${PUBLIC_IP}:8000  [OK]"
else
  echo "  FastAPI :8000  → FAILED — docker exec $CONTAINER tail -50 /var/log/aob_api.log"
fi
if [ "$FE_OK" = "true" ]; then
  echo "  Next.js :3000  → http://${PUBLIC_IP}:3000  [OK]"
else
  echo "  Next.js :3000  → FAILED — tail -50 /var/log/aob_frontend.log"
fi
echo "============================================"
echo ""
