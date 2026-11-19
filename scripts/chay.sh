#!/bin/bash
# Click-to-run: YarnBreak Detection từ camera Hikrobot MVS (có cửa sổ dashboard).
cd "$(dirname "$0")/.."   # scripts/ -> project root

# MVS GigE SDK runtime (cần cho camera MV-CS050-10GC)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH

source venv/bin/activate
python3 main.py --config config.yaml

echo ""
echo "=== Chương trình đã dừng. Nhấn Enter để đóng ==="
read
