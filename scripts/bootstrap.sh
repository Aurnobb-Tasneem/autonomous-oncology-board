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

# ── 3. Start Ollama on HOST ────────────────────────────────────────────────
echo "--> Checking Ollama..."
if ! pgrep -x ollama > /dev/null; then
  echo "    Starting Ollama server..."
  OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}" nohup ollama serve > /var/log/ollama.log 2>&1 &
  sleep 3
fi

echo "--> Pulling Llama 3.3 70B (this may take 30-60 min on first run)..."
ollama pull llama3.3:70b

# Allow Docker → host Ollama traffic
echo "--> Opening firewall port ${OLLAMA_PORT} for Docker subnet..."
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
  export OLLAMA_HOST=http://172.17.0.1:11434 &&
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
  export OLLAMA_HOST=http://172.17.0.1:11434 &&
  export HF_TOKEN=${HF_TOKEN:-''} &&
  uvicorn ml.api:app --host 0.0.0.0 --port 8000 > /var/log/aob_api.log 2>&1
"
sleep 4

# ── 10. Health check ───────────────────────────────────────────────────────
echo "--> Health check..."
if curl -sf http://localhost:8000/health > /dev/null; then
  echo ""
  echo "============================================"
  echo "  AOB is LIVE!"
  echo "  http://$(curl -s ifconfig.me):8000"
  echo "============================================"
  echo ""
else
  echo "WARNING: Health check failed. Check logs:"
  echo "  docker exec $CONTAINER tail -50 /var/log/aob_api.log"
fi
