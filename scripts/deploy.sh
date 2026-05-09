#!/usr/bin/env bash
# scripts/deploy.sh
# =================
# Pull latest code from GitHub and restart the AOB API.
# Called automatically by GitHub Actions on every push to main.
# Can also be run manually on the AMD host (from any clone path):
#
#   bash scripts/deploy.sh
#
# Optional: CONTAINER=rocm2 bash scripts/deploy.sh
# Flags: --skip-pull  --skip-frontend

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER="${CONTAINER:-rocm}"
API_URL="http://localhost:8000/health"
HEALTH_RETRIES=30      # 30 × 2s = 60s max wait
HEALTH_INTERVAL=2
SKIP_PULL=false
SKIP_FRONTEND=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --skip-pull) SKIP_PULL=true ;;
    --skip-frontend) SKIP_FRONTEND=true ;;
  esac
done

echo ""
echo "==> [AOB Deploy] $(date '+%Y-%m-%d %H:%M:%S')"

# ── 1. Pull latest code ────────────────────────────────────────────────────
if [ "$SKIP_PULL" = "true" ]; then
  echo "--> Skipping pull (already done by caller)"
  cd "$REPO_DIR"
  echo "    At $(git rev-parse --short HEAD)"
else
  echo "--> Pulling latest code from GitHub..."
  cd "$REPO_DIR"
  git fetch origin main
  git reset --hard origin/main
  echo "    OK — now at $(git rev-parse --short HEAD)"
fi

# ── 2. Sync ML code into Docker container ─────────────────────────────────
echo "--> Syncing ML code into container..."
docker cp "$REPO_DIR/ml"      "$CONTAINER":/workspace/aob/
docker cp "$REPO_DIR/scripts" "$CONTAINER":/workspace/aob/
docker cp "$REPO_DIR/data"    "$CONTAINER":/workspace/aob/
# eval/ holds reproducible benchmark results consumed by /api/benchmark/* endpoints
[ -d "$REPO_DIR/eval" ] && docker cp "$REPO_DIR/eval" "$CONTAINER":/workspace/aob/ || true
echo "    OK"

# ── 2b. Rebuild & restart Next.js frontend on the HOST ────────────────────
# The frontend runs on the host (not in the ROCm container). On every deploy
# we install deps, rebuild, and restart `next start` on :3000.
if [ "$SKIP_FRONTEND" = true ]; then
  echo "--> Skipping Next.js frontend (--skip-frontend)"
elif [ -d "$REPO_DIR/frontend" ] && command -v npm >/dev/null 2>&1; then
  echo "--> Rebuilding Next.js frontend..."
  cd "$REPO_DIR/frontend"
  if [ ! -f ".env.local" ]; then
    {
      echo "NEXT_PUBLIC_API_URL=http://localhost:8000"
      echo "BACKEND_INTERNAL_URL=http://localhost:8000"
    } > .env.local
  fi
  npm ci --prefer-offline --no-audit --no-fund --silent
  npm run build
  echo "    OK"

  echo "--> Restarting Next.js server on :3000..."
  pkill -f "next-server" 2>/dev/null || true
  pkill -f "next start" 2>/dev/null || true
  sleep 1
  nohup npx next start -p 3000 > /var/log/aob_frontend.log 2>&1 &
  sleep 3
  cd "$REPO_DIR"
  echo "    OK"
elif [ -d "$REPO_DIR/frontend" ]; then
  echo "--> Skipping Next.js frontend (npm not on PATH — install Node.js or use --skip-frontend)"
fi

# ── 3. Restart FastAPI ─────────────────────────────────────────────────────
echo "--> Restarting FastAPI server..."
docker exec "$CONTAINER" pkill -f uvicorn 2>/dev/null || true
sleep 2

# Read all env vars from .env if present (host side).
# OLLAMA_HOST is written here by bootstrap.sh after auto-detecting the Docker
# bridge gateway — so we never need to hardcode 172.17.0.1 again.
HF_TOKEN=""
OLLAMA_HOST=""
OLLAMA_MODEL=""
QWEN_VL_MODEL=""
TNM_VLLM_BASE_URL=""
BIOMARKER_VLLM_BASE_URL=""
TREATMENT_VLLM_BASE_URL=""
if [ -f "$REPO_DIR/.env" ]; then
  HF_TOKEN=$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  OLLAMA_HOST=$(grep -E '^OLLAMA_HOST=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  OLLAMA_MODEL=$(grep -E '^OLLAMA_MODEL=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  QWEN_VL_MODEL=$(grep -E '^QWEN_VL_MODEL=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  TNM_VLLM_BASE_URL=$(grep -E '^TNM_VLLM_BASE_URL=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  BIOMARKER_VLLM_BASE_URL=$(grep -E '^BIOMARKER_VLLM_BASE_URL=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
  TREATMENT_VLLM_BASE_URL=$(grep -E '^TREATMENT_VLLM_BASE_URL=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
fi

# Default Q4_K_S 70B (~40 GB) when .env omits OLLAMA_MODEL (matches api.py / llm_client.py)
if [ -z "$OLLAMA_MODEL" ]; then
  OLLAMA_MODEL="llama3.3:70b-instruct-q4_K_S"
fi

# If OLLAMA_HOST wasn't persisted by bootstrap yet, re-detect the gateway now.
if [ -z "$OLLAMA_HOST" ]; then
  DOCKER_GW=$(docker exec "$CONTAINER" sh -c \
    "ip route 2>/dev/null | awk '/default/ {print \$3; exit}'" 2>/dev/null || echo "172.17.0.1")
  OLLAMA_HOST="http://${DOCKER_GW}:11434"
  echo "    OLLAMA_HOST not in .env — detected as ${OLLAMA_HOST}"
fi
echo "--> OLLAMA_HOST: ${OLLAMA_HOST}"
echo "--> OLLAMA_MODEL: ${OLLAMA_MODEL}"

docker exec -d "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export OLLAMA_HOST='${OLLAMA_HOST}' &&
  export OLLAMA_MODEL='${OLLAMA_MODEL}' &&
  export HF_TOKEN='${HF_TOKEN}' &&
  export QWEN_VL_MODEL='${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}' &&
  export TNM_VLLM_BASE_URL='${TNM_VLLM_BASE_URL:-http://localhost:8006/v1}' &&
  export BIOMARKER_VLLM_BASE_URL='${BIOMARKER_VLLM_BASE_URL:-http://localhost:8006/v1}' &&
  export TREATMENT_VLLM_BASE_URL='${TREATMENT_VLLM_BASE_URL:-http://localhost:8006/v1}' &&
  uvicorn ml.api:app --host 0.0.0.0 --port 8000 > /var/log/aob_api.log 2>&1
"

# ── 4. Health check with retries ───────────────────────────────────────────
echo "--> Waiting for API to become healthy (up to ${HEALTH_RETRIES}×${HEALTH_INTERVAL}s)..."
HEALTHY=0
for i in $(seq 1 "$HEALTH_RETRIES"); do
  if curl -fsS --max-time 2 "$API_URL" > /dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  echo "    ($i/$HEALTH_RETRIES) not ready yet — sleeping ${HEALTH_INTERVAL}s..."
  sleep "$HEALTH_INTERVAL"
done

# ── 5. Report result + dump logs on failure ────────────────────────────────
if [ "$HEALTHY" -eq 1 ]; then
  echo "    OK — API is live at http://localhost:8000"
  if curl -fsS --max-time 2 http://localhost:3000 > /dev/null 2>&1; then
    echo "    OK — Frontend is live at http://localhost:3000"
  else
    echo "    WARNING: Frontend health check failed — tail -50 /var/log/aob_frontend.log"
  fi
else
  echo ""
  echo "==> HEALTH CHECK FAILED after $((HEALTH_RETRIES * HEALTH_INTERVAL))s. Dumping diagnostics..."
  echo ""

  echo "--- docker ps -a ---"
  docker ps -a || true

  echo ""
  echo "--- docker logs rocm (last 100 lines) ---"
  docker logs --tail=100 "$CONTAINER" 2>&1 || true

  echo ""
  echo "--- /var/log/aob_api.log (last 100 lines) ---"
  docker exec "$CONTAINER" sh -c 'tail -100 /var/log/aob_api.log 2>/dev/null || echo "(log file not found)"' || true

  echo ""
  echo "--- /var/log/aob_frontend.log (last 50 lines, host) ---"
  tail -50 /var/log/aob_frontend.log 2>/dev/null || echo "(log file not found)"

  echo ""
  echo "--- Python / pip version inside container ---"
  docker exec "$CONTAINER" sh -c 'python3 -V && pip3 -V' || true

  echo ""
  echo "--- Test import of ml.api inside container ---"
  docker exec "$CONTAINER" sh -c 'cd /workspace/aob && PYTHONPATH=/workspace/aob python3 -c "import ml.api; print(\"import OK\")"' || true

  exit 1
fi

echo ""
echo "==> Deploy complete."
