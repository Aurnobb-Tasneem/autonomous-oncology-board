#!/usr/bin/env bash
# =============================================================================
# vram_monitor.sh — Continuous VRAM logger using rocm-smi
# =============================================================================
# USAGE:
#   chmod +x scripts/vram_monitor.sh
#   ./scripts/vram_monitor.sh                  # default: log every 2s indefinitely
#   ./scripts/vram_monitor.sh 5 300            # poll every 5s for 300s (5 min)
#
# OUTPUT:
#   logs/vram_monitor_<timestamp>.csv          # timestamped CSV
#   stdout                                     # human-readable live display
#
# Run this in a separate terminal alongside smoke_test.py:
#   Terminal 1: python scripts/smoke_test.py
#   Terminal 2: ./scripts/vram_monitor.sh
# =============================================================================

POLL_INTERVAL=${1:-2}       # seconds between polls
MAX_DURATION=${2:-0}        # 0 = run indefinitely

LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="${LOG_DIR}/vram_monitor_${TIMESTAMP}.csv"

echo "# AOB VRAM Monitor — started at $(date)" | tee "$CSV_FILE"
echo "# Poll interval: ${POLL_INTERVAL}s" | tee -a "$CSV_FILE"
echo "timestamp,device,used_gb,total_gb,pct_used" | tee -a "$CSV_FILE"
echo "─────────────────────────────────────────────────────"

START_TIME=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$MAX_DURATION" -gt 0 ] && [ "$ELAPSED" -ge "$MAX_DURATION" ]; then
        echo "Monitor duration (${MAX_DURATION}s) reached. Exiting."
        break
    fi

    NOW=$(date +"%Y-%m-%dT%H:%M:%S")

    # Check if rocm-smi is available
    if ! command -v rocm-smi &> /dev/null; then
        echo "${NOW} — rocm-smi not found. Is ROCm installed?"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Parse rocm-smi JSON output
    JSON=$(rocm-smi --showmeminfo vram --json 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$JSON" ]; then
        echo "${NOW} — rocm-smi failed or returned empty output"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Use python to parse JSON (available anywhere PyTorch is installed)
    python3 - <<PYEOF
import json, sys

data = json.loads('''${JSON}''')
rows = []
for device_id, info in data.items():
    used_b  = int(info.get("VRAM Total Used Memory (B)", 0))
    total_b = int(info.get("VRAM Total Memory (B)", 0))
    used_gb  = used_b  / (1024**3)
    total_gb = total_b / (1024**3)
    pct = (100 * used_gb / total_gb) if total_gb > 0 else 0
    rows.append((device_id, used_gb, total_gb, pct))

for dev, used, total, pct in rows:
    csv_line = f"${NOW},{dev},{used:.2f},{total:.2f},{pct:.1f}"
    print(csv_line)
    # Also print human-readable to stdout
    bar_len = int(pct / 2)
    bar = "█" * bar_len + "░" * (50 - bar_len)
    print(f"  [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%)  {dev}", file=sys.stderr)

PYEOF

    sleep "$POLL_INTERVAL"
done

echo ""
echo "VRAM log saved to: ${CSV_FILE}"
