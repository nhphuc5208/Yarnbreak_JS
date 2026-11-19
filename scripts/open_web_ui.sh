#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  open_web_ui.sh — Tự động mở web UI trên màn hình desktop sau khi bật nguồn.
#  Được gọi bởi GNOME autostart (~/.config/autostart/yarnbreak-web-ui.desktop).
#  Chờ web server (cổng 8000) sẵn sàng rồi mới mở Chromium ở cửa sổ thường.
# ──────────────────────────────────────────────────────────────────────────────
set -u

URL="http://localhost:8000"
HOST="localhost"
PORT="8000"
MAX_WAIT=120          # giây — tối đa chờ server lên (bình thường ~45s)

LOG="/tmp/yarnbreak-web-ui.log"
echo "[$(date '+%F %T')] open_web_ui.sh start; chờ server $HOST:$PORT ..." >> "$LOG"

# Chờ cổng web server mở (poll mỗi 1s)
waited=0
until (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "[$(date '+%F %T')] hết $MAX_WAIT s vẫn chưa thấy server — mở browser anyway" >> "$LOG"
        break
    fi
done
exec 3>&- 2>/dev/null || true
echo "[$(date '+%F %T')] server sẵn sàng sau ${waited}s — mở Chromium (cửa sổ thường)" >> "$LOG"

# Mở Chromium ở CỬA SỔ THƯỜNG (không toàn màn hình).
# --app  : vẫn giữ dáng "ứng dụng" — không thanh địa chỉ, không tab.
# KHÔNG dùng --kiosk/--start-fullscreen: người vận hành cần thu nhỏ / dùng máy việc khác.
# Muốn quay lại toàn màn hình thì thêm --kiosk vào đây (và bỏ --window-size/-position).
exec chromium \
    --app="$URL" \
    --window-size=1600,900 \
    --window-position=100,50 \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    >> "$LOG" 2>&1
