#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_ipcam.sh  —  Quick-start script for IP camera production mode
#  Edit RTSP_URL and MODEL below, then:
#    chmod +x scripts/run_ipcam.sh
#    ./scripts/run_ipcam.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
RTSP_URL="rtsp://user:pass@<camera-ip>:554/stream1"
MODEL="./models/best225v3.engine"   # use .engine for best performance
IMGSZ=960
CONF=0.20
DETECT_EVERY=2                       # infer every 2 frames to save compute

# ── Auto-detect virtualenv ────────────────────────────────────────────────────
if [ -d "./venv/bin" ]; then
    source ./venv/bin/activate
elif [ -d "$HOME/.virtualenvs/yarnbreak/bin" ]; then
    source "$HOME/.virtualenvs/yarnbreak/bin/activate"
fi

echo "========================================"
echo "  YarnBreak IP Camera  –  $(date)"
echo "  RTSP:  $RTSP_URL"
echo "  Model: $MODEL"
echo "========================================"

python3 main.py \
    --source ipcam \
    --rtsp "$RTSP_URL" \
    --model "$MODEL" \
    --imgsz "$IMGSZ" \
    --conf "$CONF" \
    --detect-every "$DETECT_EVERY" \
    --headless \
    --save-snapshot \
    --log-events \
    --device 0 \
    --half \
    --reconnect-delay 3
