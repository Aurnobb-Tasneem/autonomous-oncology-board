#!/usr/bin/env bash
# scripts/deploy.sh
# =================
# Pull latest code from GitHub and restart the AOB API.
# Called automatically by GitHub Actions on every push to main.
# Can also be run manually on the AMD host:
#
#   bash /root/aob/scripts/deploy.sh

set -euo pipefail

REPO_DIR="/root/aob"
CONTAINER="rocm"
API_URL="http://localhost:8000/health"
HEALTH_RETRIES=30      # 30 × 2s = 60s max wait
HEALTH_INTERVAL=2
SKIP_PULL=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --skip-pull) SKIP_PULL=true ;;
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

# ── 2. Sync into Docker container ─────────────────────────────────────────
echo "--> Syncing code into container..."
docker cp "$REPO_DIR/ml"      "$CONTAINER":/workspace/aob/
docker cp "$REPO_DIR/scripts" "$CONTAINER":/workspace/aob/
docker cp "$REPO_DIR/data"    "$CONTAINER":/workspace/aob/
echo "    OK"

# ── 3. Restart FastAPI ─────────────────────────────────────────────────────
echo "--> Restarting FastAPI server..."
docker exec "$CONTAINER" pkill -f uvicorn 2>/dev/null || true
sleep 2

# Read HF_TOKEN from .env if present (host side — passed into container below)
HF_TOKEN=""
if [ -f "$REPO_DIR/.env" ]; then
  HF_TOKEN=$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'") || true
fi

docker exec -d "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export OLLAMA_HOST=http://172.17.0.1:11434 &&
  export HF_TOKEN='${HF_TOKEN}' &&
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
  echo "--- Python / pip version inside container ---"
  docker exec "$CONTAINER" sh -c 'python3 -V && pip3 -V' || true

  echo ""
  echo "--- Test import of ml.api inside container ---"
  docker exec "$CONTAINER" sh -c 'cd /workspace/aob && PYTHONPATH=/workspace/aob python3 -c "import ml.api; print(\"import OK\")"' || true

  exit 1
fi

echo ""
echo "==> Deploy complete."
