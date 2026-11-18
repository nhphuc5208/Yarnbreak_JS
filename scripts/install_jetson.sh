#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  install_jetson.sh  —  Full setup guide for Jetson Orin Nano 8 GB
#  Ubuntu 22.04 / JetPack 6.x
#
#  Run each section manually — DO NOT run this as a blind bash script.
#  Some steps require reboot or interactive confirmation.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
echo "=== YarnBreak Jetson Setup ==="

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – System update
# ══════════════════════════════════════════════════════════════════════════════
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-rtsp \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    v4l-utils


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – Verify JetPack / L4T version
# ══════════════════════════════════════════════════════════════════════════════
cat /etc/nv_tegra_release 2>/dev/null || echo "nv_tegra_release not found"
dpkg -l | grep -i jetpack 2>/dev/null || true
# Expected output example:
#   R36 (release), REVISION: 3.0  → JetPack 6.0, L4T 36.3


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – Check CUDA
# ══════════════════════════════════════════════════════════════════════════════
nvcc --version
nvidia-smi  # may not work on Jetson; use tegrastats instead
# Or:
# python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – Create virtualenv
# ══════════════════════════════════════════════════════════════════════════════
# Navigate to project directory first:
# cd ~/yarnbreak_jetson

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – Install PyTorch for Jetson (NVIDIA wheel)
#
#  ⚠  DO NOT run "pip install torch" — that installs the x86 CPU build.
#  Use the official NVIDIA wheel for JetPack 6.x (L4T R36):
# ══════════════════════════════════════════════════════════════════════════════

# Option A – From NVIDIA PyPI index (JetPack 6.x / L4T 36.x):
pip install torch torchvision torchaudio \
    --index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# Option B – Manual .whl download (if Option A fails):
# 1. Go to: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# 2. Download the correct wheel for your JetPack version
# 3. pip install torch-*.whl

# Verify GPU access:
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – Install remaining requirements
# ══════════════════════════════════════════════════════════════════════════════
pip install -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – Verify OpenCV GStreamer support
#  If "GStreamer: NO" → must rebuild OpenCV from source or install system package
# ══════════════════════════════════════════════════════════════════════════════
python3 - <<'EOF'
import cv2
info = cv2.getBuildInformation()
gst_line = [l for l in info.splitlines() if "GStreamer" in l]
print("OpenCV version:", cv2.__version__)
print("GStreamer support:")
for l in gst_line:
    print(" ", l.strip())
EOF


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 – Test with video file
# ══════════════════════════════════════════════════════════════════════════════
# python3 main.py \
#     --source video \
#     --video ./videos/YB3.mp4 \
#     --model ./models/best225v3.pt \
#     --imgsz 960 \
#     --conf 0.20 \
#     --display


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 – Test with IP camera
# ══════════════════════════════════════════════════════════════════════════════
# python3 main.py \
#     --source ipcam \
#     --rtsp "rtsp://user:pass@<camera-ip>:554/stream1" \
#     --model ./models/best225v3.pt \
#     --display \
#     --conf 0.20


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 – Run headless (no GUI)
# ══════════════════════════════════════════════════════════════════════════════
# python3 main.py \
#     --source video \
#     --video ./videos/YB3.mp4 \
#     --model ./models/best225v3.pt \
#     --headless \
#     --save-snapshot \
#     --log-events


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 – Save output video
# ══════════════════════════════════════════════════════════════════════════════
# python3 main.py \
#     --source video \
#     --video ./videos/YB3.mp4 \
#     --model ./models/best225v3.pt \
#     --headless \
#     --save-video \
#     --save-snapshot


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 – Export TensorRT engine (run once on Jetson)
# ══════════════════════════════════════════════════════════════════════════════
# python3 scripts/export_tensorrt.py \
#     --model ./models/best225v3.pt \
#     --imgsz 960 \
#     --half


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 13 – Benchmark .pt vs .engine
# ══════════════════════════════════════════════════════════════════════════════
# python3 scripts/benchmark.py \
#     --video ./videos/YB3.mp4 \
#     --models ./models/best225v3.pt ./models/best225v3.engine \
#     --imgsz 640 960 \
#     --half \
#     --frames 300


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 14 – Monitor Jetson performance during inference
# ══════════════════════════════════════════════════════════════════════════════
# sudo tegrastats --interval 1000
# # Shows: CPU, GPU, RAM, temperature in real time


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 15 – Install as systemd service (24/7 factory mode)
# ══════════════════════════════════════════════════════════════════════════════
# Edit yarnbreak.service: set correct User, WorkingDirectory, RTSP URL, model path
# sudo cp yarnbreak.service /etc/systemd/system/
# sudo systemctl daemon-reload
# sudo systemctl enable yarnbreak.service
# sudo systemctl start  yarnbreak.service
# sudo systemctl status yarnbreak.service
# journalctl -u yarnbreak -f

echo "=== Setup complete ==="
