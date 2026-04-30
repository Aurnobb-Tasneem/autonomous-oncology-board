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

echo ""
echo "==> [AOB Deploy] $(date '+%Y-%m-%d %H:%M:%S')"

# ── 1. Pull latest code ────────────────────────────────────────────────────
echo "--> Pulling latest code from GitHub..."
cd "$REPO_DIR"
git fetch origin main
git reset --hard origin/main
echo "    OK — now at $(git rev-parse --short HEAD)"

# ── 2. Sync into Docker container ─────────────────────────────────────────
echo "--> Syncing code into container..."
docker cp "$REPO_DIR/ml" "$CONTAINER":/workspace/aob/
docker cp "$REPO_DIR/scripts" "$CONTAINER":/workspace/aob/
echo "    OK"

# ── 3. Restart FastAPI ─────────────────────────────────────────────────────
echo "--> Restarting FastAPI server..."
docker exec "$CONTAINER" pkill -f uvicorn 2>/dev/null || true
sleep 2

source "$REPO_DIR/.env" 2>/dev/null || true

docker exec -d "$CONTAINER" bash -c "
  cd /workspace/aob &&
  export PYTHONPATH=/workspace/aob &&
  export OLLAMA_HOST=http://172.17.0.1:11434 &&
  export HF_TOKEN=${HF_TOKEN:-''} &&
  uvicorn ml.api:app --host 0.0.0.0 --port 8000 > /var/log/aob_api.log 2>&1
"
sleep 4

# ── 4. Health check ────────────────────────────────────────────────────────
echo "--> Health check..."
if curl -sf http://localhost:8000/health > /dev/null; then
  echo "    OK — API is live at http://localhost:8000"
else
  echo "    FAILED — check logs: docker exec $CONTAINER tail -50 /var/log/aob_api.log"
  exit 1
fi

echo ""
echo "==> Deploy complete."
