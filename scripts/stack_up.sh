#!/usr/bin/env bash
# scripts/stack_up.sh
# ====================
# Bring up local dependencies + ROCm ML container + FastAPI (same path as deploy).
#
# On the AMD host (e.g. syntroph):
#   cd ~/autonomous-oncology-board
#   bash scripts/stack_up.sh
#
# Optional:
#   CONTAINER=my_rocm bash scripts/stack_up.sh
#   SKIP_QDRANT=1 bash scripts/stack_up.sh
#
# Prereqs (host): Ollama systemd service, Docker, repo .env with HF_TOKEN + OLLAMA_HOST.
# Optional: vLLM specialists are NOT started here — run `bash scripts/serve_specialists.sh`
# in another terminal if you need :8006 multi-LoRA (requires adapter checkpoints).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

SKIP_QDRANT="${SKIP_QDRANT:-0}"

# ── Resolve ROCm container name ─────────────────────────────────────────────
CONTAINER="${CONTAINER:-}"
if [ -z "$CONTAINER" ]; then
  if docker ps -a --format '{{.Names}}' | grep -qx 'rocm2'; then
    CONTAINER=rocm2
  elif docker ps -a --format '{{.Names}}' | grep -qx 'rocm'; then
    CONTAINER=rocm
  else
    echo "ERROR: No Docker container named 'rocm2' or 'rocm'." >&2
    echo "  Create/start your ROCm dev container first, or run:  CONTAINER=<name> bash scripts/stack_up.sh" >&2
    exit 1
  fi
fi
export CONTAINER

echo ""
echo "==> [AOB stack_up] CONTAINER=$CONTAINER  REPO=$REPO_DIR"

# ── Qdrant (optional; API can use in-process Qdrant per env) ────────────────
if [ "$SKIP_QDRANT" != "1" ]; then
  if docker compose version >/dev/null 2>&1; then
    echo "--> docker compose up -d qdrant"
    docker compose -f "$REPO_DIR/docker-compose.yml" up -d qdrant || {
      echo "    WARN: qdrant compose step failed (non-fatal if you do not use standalone Qdrant)."
    }
  else
    echo "    WARN: docker compose not found — skipping qdrant."
  fi
else
  echo "--> Skipping Qdrant (SKIP_QDRANT=1)"
fi

# ── ROCm ML container ──────────────────────────────────────────────────────
echo "--> docker start $CONTAINER"
docker start "$CONTAINER"

echo "--> Waiting for container exec..."
for _ in $(seq 1 30); do
  if docker exec "$CONTAINER" sh -c "true" 2>/dev/null; then
    break
  fi
  sleep 1
done

# ── Ollama (host) — soft check ──────────────────────────────────────────────
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "--> Ollama: reachable at http://127.0.0.1:11434"
  else
    echo "    WARN: Ollama not responding on 127.0.0.1:11434 — start with:  sudo systemctl start ollama" >&2
  fi
fi

# ── Sync + uvicorn (reuse deploy) ───────────────────────────────────────────
echo "--> deploy.sh --skip-pull --skip-frontend"
bash "$REPO_DIR/scripts/deploy.sh" --skip-pull --skip-frontend

echo ""
echo "==> Stack summary"
echo "    ML container : $CONTAINER (running)"
echo "    FastAPI      : http://localhost:8000/health"
echo "    Specialists  : curl -sS http://localhost:8006/v1/models  (optional; start serve_specialists.sh)"
echo ""
