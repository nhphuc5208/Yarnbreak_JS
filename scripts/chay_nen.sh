#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# YarnBreak 24/7 — chạy NỀN, KHÔNG hiện cửa sổ camera (headless).
# - Tự khởi động lại khi chương trình thoát/crash (ổn định 24/24).
# - Khóa chống chạy trùng (chỉ 1 tiến trình giữ camera tại một thời điểm).
# - Ghi log ra file trong outputs/logs/.
# Dừng: đóng cửa sổ terminal này, hoặc nhấn Ctrl+C.
# ──────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."   # scripts/ -> project root

# Chống chạy trùng: nếu đã có tiến trình khác đang giữ khóa thì thoát.
exec 9>/tmp/yarnbreak_24_7.lock
if ! flock -n 9; then
    echo "⚠  YarnBreak đang chạy ở tiến trình khác (camera đang bận). Thoát."
    echo "Nhấn Enter để đóng."; read
    exit 1
fi

# MVS GigE SDK runtime (cần cho camera MV-CS050-10GC)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs
LOG="outputs/logs/run24_7_$(date +%Y%m%d_%H%M%S).log"
echo "YarnBreak 24/7 (headless) — log: $LOG"

source venv/bin/activate

# Dừng sạch khi nhận Ctrl+C / đóng terminal.
trap 'echo "[$(date +%F\ %T)] Nhận tín hiệu dừng, thoát."; exit 0' INT TERM

# Vòng tự khởi động lại: nếu chương trình thoát vì bất kỳ lý do gì thì chạy lại.
# Output hiện TRỰC TIẾP trên terminal (cảnh báo + trạng thái theo thời gian)
# đồng thời lưu vào file log. KHÔNG hiện cửa sổ camera.
while true; do
    echo "[$(date +%F\ %T)] Khởi động YarnBreak (headless)…" | tee -a "$LOG"
    python3 main.py --config config.yaml --headless 2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    echo "[$(date +%F\ %T)] Chương trình dừng (exit=$code). Khởi động lại sau 5s…" | tee -a "$LOG"
    sleep 5
done
