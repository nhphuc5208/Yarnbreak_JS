#!/usr/bin/env bash
# Theo dõi TRỰC TIẾP luồng alarm + nút đỏ trên phần cứng thật.
#
# Bám vào log mới nhất của pipeline và lọc đúng các sự kiện cần quan sát:
#   - YARN BREAK DETECTED   → alarm nổ
#   - Relay CH1 → CLOSE/OPEN → tín hiệu dừng máy/còi đóng/nhả
#   - RED button             → nhấn nút đỏ
#   - Alert cleared / paused / stop
#
# KHÔNG mở cổng serial nên chạy song song an toàn với pipeline (pipeline đang
# giữ /dev/ttyACM0). Chỉ đọc file log.
#
# Dùng:
#   ./scripts/watch_alarm_test.sh            # tail log file mới nhất
#   ./scripts/watch_alarm_test.sh --journal  # nếu chạy bằng systemd (journalctl)

set -euo pipefail
cd "$(dirname "$0")/.."

# Regex các dòng đáng quan tâm (màu để dễ nhìn)
PAT='YARN BREAK|Alert cleared|Relay CH1|Relay CH2|RED button|GREEN button|Pausing|Stopping|paused|force_off|red button'

hl() {
    # tô màu: đỏ cho CLOSE/YARN BREAK/RED, xanh cho OPEN/cleared/GREEN
    sed -E \
        -e 's/(CLOSE|YARN BREAK DETECTED|RED button|Stopping)/\x1b[1;31m\1\x1b[0m/g' \
        -e 's/(OPEN|Alert cleared|GREEN button)/\x1b[1;32m\1\x1b[0m/g'
}

if [[ "${1:-}" == "--journal" ]]; then
    echo ">> Theo dõi journalctl -u yarnbreak (Ctrl-C để thoát)"
    exec stdbuf -oL journalctl -u yarnbreak -f -n 0 | grep --line-buffered -E "$PAT" | hl
fi

LOG=$(ls -t outputs/logs/run_*.log 2>/dev/null | head -1 || true)
if [[ -z "$LOG" ]]; then
    echo "Không tìm thấy outputs/logs/run_*.log — hãy khởi động pipeline trước." >&2
    exit 1
fi
echo ">> Đang theo dõi: $LOG"
echo ">> Lọc sự kiện alarm + nút đỏ. Ctrl-C để thoát."
echo "--------------------------------------------------------------"
exec tail -n 0 -F "$LOG" | grep --line-buffered -E "$PAT" | hl
