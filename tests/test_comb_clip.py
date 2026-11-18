"""Test CombEventClip bằng ĐỒNG HỒ GIẢ: kiểm tra đúng luật tối thiểu 30s.

Chạy: python test_comb_clip.py <duong-dan-du-an>
Không cần camera, không cần Jetson — chỉ dùng frame numpy sinh tại chỗ.
"""
import os
import sys
import shutil
import tempfile

proj = sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")
sys.path.insert(0, proj)

import numpy as np
import comb_pipeline as cp

FAKE = {"t": 1000.0}
cp.time.monotonic = lambda: FAKE["t"]          # đồng hồ giả cho logic thời lượng

out = tempfile.mkdtemp(prefix="combclip_")
cfg = {
    "event_clip": True,
    "event_clip_dir": out,
    "event_buffer_scale": 0.5,
    "event_pre_seconds": 3.0,
    "event_min_seconds": 30.0,
    "event_post_seconds": 3.0,
    "event_max_seconds": 120.0,
    "event_clip_fps": 0,
}

FPS = 4.0
DT = 1.0 / FPS
frame = np.full((200, 640, 3), 40, dtype=np.uint8)

fails = []


def tick(clip, alarm_on, just_on=False):
    FAKE["t"] += DT
    clip.feed(frame, alarm_on, just_on)


def check(label, cond):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        fails.append(label)


def mp4s():
    d = os.path.join(out, "comb_1")
    return sorted(f for f in os.listdir(d)) if os.path.isdir(d) else []


print("1) Alarm tat som (2s) -> clip van phai keo du 30s")
clip = cp.CombEventClip("comb_1", cfg)
check("bat dau chua ghi gi", not mp4s())

for _ in range(int(10 * FPS)):          # 10s binh thuong -> chi pre-roll
    tick(clip, False)
check("chay 10s binh thuong: chua co file", not mp4s())
check("pre-roll khong vuot 3s", len(clip._buf) <= int(3 * FPS) + 1)

tick(clip, True, just_on=True)          # BAO DUT
check("vua bao dut: da mo 1 file", len(mp4s()) == 1)
opened = clip.current_path is not None
check("writer dang mo", opened)

for _ in range(int(2 * FPS)):           # bao 2s roi tat
    tick(clip, True)
for _ in range(int(20 * FPS)):          # 20s sau khi tat
    tick(clip, False)
check("sau 22s: VAN con ghi (chua du 30s)", clip.current_path is not None)

for _ in range(int(9 * FPS)):           # tong ~31s
    tick(clip, False)
check("sau ~31s: da dong file", clip.current_path is None)

n_frames = None
import cv2
path = os.path.join(out, "comb_1", mp4s()[0])
cap = cv2.VideoCapture(path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_file = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
dur = n_frames / fps_file if fps_file else 0
print(f"     file: {os.path.basename(path)}  {w}x{h}  {n_frames} frame  "
      f"{fps_file:.2f} fps  ~{dur:.1f}s")
check("thoi luong file >= 30s", dur >= 30.0)
check("kich thuoc = 0.5 goc (320x100)", (w, h) == (320, 100))

print("2) Alarm keo dai -> chan tren max_seconds (120s)")
shutil.rmtree(out, ignore_errors=True)
clip2 = cp.CombEventClip("comb_1", cfg)
tick(clip2, True, just_on=True)
for _ in range(int(130 * FPS)):
    tick(clip2, True)
check("da dong khi cham 120s", clip2.current_path is None)
check("chi tao 1 file", len(mp4s()) == 1)

print("3) Bao lai trong luc cho dong -> giu CUNG mot file")
shutil.rmtree(out, ignore_errors=True)
clip3 = cp.CombEventClip("comb_1", cfg)
tick(clip3, True, just_on=True)
for _ in range(int(5 * FPS)):
    tick(clip3, False)
for _ in range(int(5 * FPS)):           # bao lai
    tick(clip3, True)
for _ in range(int(25 * FPS)):
    tick(clip3, False)
check("van chi 1 file cho ca su kien", len(mp4s()) == 1)

print("4) event_clip = false -> khong ghi gi")
shutil.rmtree(out, ignore_errors=True)
off = dict(cfg); off["event_clip"] = False
clip4 = cp.CombEventClip("comb_1", off)
tick(clip4, True, just_on=True)
for _ in range(int(40 * FPS)):
    tick(clip4, True)
check("khong tao file nao", not os.path.isdir(os.path.join(out, "comb_1")))

shutil.rmtree(out, ignore_errors=True)
print()
print("KET QUA:", "TAT CA PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
