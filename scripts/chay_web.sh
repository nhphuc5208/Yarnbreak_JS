#!/bin/bash
# Click-to-run: Khởi chạy YarnBreak Web Server.
cd "$(dirname "$0")/.."   # scripts/ -> project root

# MVS GigE SDK runtime (cần cho camera MV-CS050-10GC)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH

source venv/bin/activate
python3 web_server.py

echo ""
echo "=== Máy chủ đã dừng. Nhấn Enter để đóng ==="
read
