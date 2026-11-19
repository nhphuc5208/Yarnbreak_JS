#!/bin/bash
# Click-to-run: Công cụ CHỤP ẢNH MẪU từ camera lược (độc lập chương trình gốc).
#
# ⚠ Camera mở ở chế độ độc quyền → phải DỪNG chương trình gốc trước:
#       sudo systemctl stop yarnbreak
#   Xong việc lấy mẫu thì bật lại:
#       sudo systemctl start yarnbreak
cd "$(dirname "$0")/.."   # scripts/ -> project root

# MVS GigE SDK runtime
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH

source venv/bin/activate

OUT="${1:-./dataset_comb}"
echo "=== Chụp mẫu camera lược — ảnh lưu vào: $OUT ==="
echo "=== Mở trình duyệt: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8010 ==="
python3 scripts/comb_capture.py --out "$OUT" --port 8010

echo ""
echo "=== Đã dừng. Nhấn Enter để đóng ==="
read
