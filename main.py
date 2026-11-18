#!/usr/bin/env python3
"""
YarnBreak Detection – Production build for Jetson Orin Nano 8 GB
Ubuntu 22.04 / JetPack 6.x

Architecture:
  FrameGrabber thread
      ↓  frame_queue (maxsize=1)
  YoloDetector thread
      ↓  result_queue (maxsize=2)
  Main loop  →  DashboardRenderer  →  cv2.imshow / VideoWriter
                    ↓
              AlertNotifier / EventLogger

Usage examples:
  python3 main.py --source video  --video ./videos/YB3.mp4 --model ./models/best225v3.pt
  python3 main.py --source ipcam  --rtsp "rtsp://user:pass@<camera-ip>:554/stream1" --headless
  python3 main.py --source mvs    --model ./models/best225v3.pt   # Hikrobot MVS camera
  python3 main.py --config config.yaml
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from live_view import live_stream

# ─── OpenCV built-in optimisations (SIMD, etc.) ───────────────────────────────
cv2.setUseOptimized(True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging(log_dir: Optional[str] = None) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"run_{ts}.log")
        fh.setFormatter(logging.Formatter(fmt))
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("YarnBreak")


logger = logging.getLogger("YarnBreak")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    # ── Source ────────────────────────────────────────────────────────────────
    source: str          = "video"           # "video" | "ipcam" | "mvs"
    video: str           = "./videos/YB3.mp4"
    rtsp: str            = ""
    codec: str           = "h264"            # "h264" | "h265" (RTSP codec hint)
    latency_ms: int      = 200               # GStreamer rtspsrc latency (ms)
    reconnect_delay: float = 3.0

    # ── Hikrobot MVS camera (GigE/USB, e.g. MV-CS050-10GC) ─────────────────────
    mvs_sdk_path: str    = "/opt/MVS/Samples/aarch64/Python/MvImport"
    mvs_serial: str      = ""                # "" = first camera found; else match serial
    mvs_exposure_us: float = 0.0             # >0 = fixed exposure (µs); 0 = leave camera default
    mvs_gain: float      = -1.0              # >=0 = fixed gain (dB); <0 = leave camera default
    mvs_fps: float       = 0.0               # >0 = cap AcquisitionFrameRate; 0 = leave default
    mvs_packet_size: int = 0                 # GigE packet size; 0 = auto-optimal (jumbo)

    # ── Model ─────────────────────────────────────────────────────────────────
    model: str           = "./models/best225v3.pt"
    imgsz: int           = 960               # 640 / 800 / 960
    conf: float          = 0.20
    device: str          = "auto"            # "auto" | "0" | "cpu"
    half: bool           = True              # FP16, CUDA only
    detect_every: int    = 1                 # 1=every frame, 2=every other, …

    # ── Alert ─────────────────────────────────────────────────────────────────
    # Đường 1 (conf cao): bật alarm khi TRUNG BÌNH conf của alert_on_consec lần
    # detect gần nhất ≥ alert_on_avg_conf. (Trước đây là "conf ≥ cfg.conf liên
    # tục N lần"; đổi sang trung bình để một frame trồi/sụt lẻ không tự quyết định.)
    alert_on_consec: int    = 6      # số lần detect gần nhất lấy trung bình
    alert_on_avg_conf: float = 0.82  # ngưỡng trung bình để bật alarm (đường 1)
    alert_off_miss: int   = 5

    # Auto-off khi alarm ĐÃ ON quá lâu: nếu alarm bật liên tục ≥ alert_long_on_seconds
    # mà trong alert_long_off_window lần detect gần nhất KHÔNG có conf nào > alert_long_off_conf
    # thì tắt alarm (dọn báo dai dẳng ở conf trung bình mà luật alert_off_miss không bắt được,
    # vì detection vẫn liên tục nên _miss không đủ). 0 = tắt luật này.
    alert_long_on_seconds: float = 40.0
    alert_long_off_window: int   = 10
    alert_long_off_conf: float   = 0.90

    # Đường báo thứ hai: conf THẤP nhưng TĂNG DẦN. Sợi đứt thật thì vết tách rộng
    # dần nên conf lớn dần đều; nhiễu thì conf nhảy lung tung. Nhờ vậy bắt được sớm
    # ở mức conf thấp mà không phải hạ ngưỡng chung (vốn sẽ kéo theo báo giả).
    alert_rise_enabled: bool   = True
    alert_rise_conf_min: float = 0.60   # mọi mẫu trong cửa sổ phải > mức này
    alert_rise_window: int     = 5      # số lần detect LIÊN TIẾP đem ra xét
    alert_rise_min_gain: float = 0.10   # mẫu cuối phải cao hơn mẫu đầu ít nhất ngần này
    alert_rise_max_dip: float  = 0.05   # không mẫu nào được tụt quá ngần này so với mẫu trước
    # Cửa XÁC NHẬN cho đường 2: khi tăng dần đủ điều kiện, KHÔNG bật báo ngay mà vào
    # trạng thái "chờ". Chỉ bật khi đã qua alert_rise_confirm_seconds VÀ trong lúc chờ
    # có ≥1 lần detect conf > alert_rise_confirm_conf. Mất target (miss) giữa chừng →
    # huỷ chờ, phải tăng dần lại từ đầu. Đường 1 (TB conf cao) không qua cửa này.
    alert_rise_confirm_seconds: float = 3.0   # thời gian chờ trước khi được phép báo
    alert_rise_confirm_conf: float    = 0.80  # trong lúc chờ phải có conf vượt mức này

    # ── Smoothing ─────────────────────────────────────────────────────────────
    ema_alpha: float      = 0.35             # 0.1=very smooth / 0.6=responsive
    hold_frames: int      = 8

    # ── Display ───────────────────────────────────────────────────────────────
    display: bool         = True
    headless: bool        = False
    fast_draw: bool       = False
    enable_mask: bool     = True
    mask_alpha: float     = 0.35
    overlay_scale: float  = 1.0               # nhân thêm cỡ chữ/box overlay (1.0=auto theo độ phân giải)
    window_width: int     = 1600
    window_height: int    = 820
    window_name: str      = "Industrial AI – Yarn Break Detection"
    enable_dashboard_cache: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    save_video: bool      = False
    save_snapshot: bool   = False
    snapshot_dir: str     = "./outputs/snapshots"
    video_out_dir: str    = "./outputs/videos"
    output_path: str      = ""               # explicit file path overrides video_out_dir
    log_dir: str          = "./outputs/logs"
    log_events: bool      = True

    # ── Event clip recording (only while alarm is ON) ─────────────────────────
    # Ghi 1 clip video riêng cho MỖI lần alarm ON (khác save_video: ghi liên tục).
    # Clip gồm pre-roll (vài giây TRƯỚC khi alarm bật, nhờ buffer trong RAM) +
    # khoảng alarm ON + post-roll. Buffer được hạ độ phân giải để tiết kiệm RAM.
    event_clip: bool          = True
    event_clip_dir: str       = "./outputs/events"
    event_pre_seconds: float  = 3.0          # số giây pre-roll giữ trong buffer
    event_post_seconds: float = 3.0          # số giây tiếp tục ghi sau khi alarm OFF
    event_buffer_scale: float = 0.5          # hệ số hạ độ phân giải clip (1.0=gốc)
    # Giới hạn thời lượng mỗi file clip. Nếu alarm ON kéo dài vượt ngưỡng này, đóng
    # file hiện tại và mở tiếp event_<ts>_partN.mp4 (cắt khúc) để không có 1 clip
    # phình vô hạn. 0 = không giới hạn (ghi liền suốt thời gian alarm).
    event_max_seconds: float  = 15.0
    # Đảm bảo thời lượng tối thiểu: nếu trong 1 sự kiện có > event_min_hold_frames
    # frame với conf > event_min_hold_conf, thì clip phải ghi ít nhất
    # event_min_hold_seconds giây TÍNH TỪ frame conf-cao ĐẦU TIÊN — kể cả khi
    # alarm đã tắt sớm (hoãn đóng file cho tới khi đủ). 0 giây = tắt tính năng.
    event_min_hold_conf: float    = 0.91
    event_min_hold_frames: int    = 3        # kích hoạt khi số frame vượt ngưỡng > số này
    event_min_hold_seconds: float = 60.0

    # ── Video file playback ───────────────────────────────────────────────────
    pace_video: bool      = True             # pace grabber to source FPS (avoid fast-forward)
    loop_video: bool      = False            # rewind to frame 0 on EOF (display demo mode)

    # ── Modbus RTU relay output (Waveshare USB-RS485 → 8DI/8DO relay) ─────────
    # Đóng relay CH (relay_channel) khi phát hiện sợi đứt, mở khi hết alarm.
    relay_enabled: bool   = False            # toggle Modbus relay output on alert
    relay_port: str       = "/dev/ttyACM0"   # serial device of the USB-RS485 adapter
    relay_baud: int       = 9600             # board default 9600 8N1
    relay_slave: int      = 1                # Modbus slave address of the relay board
    relay_channel: int    = 1                # 1-based channel to drive (CH1)
    relay_alarm_light_channel: int = 2       # CH2: đèn báo, bật cùng CH1 khi sợi đứt
    relay_active_close: bool = True          # true: alert→CLOSE relay; false: alert→OPEN
    # Đèn VÀNG khi TẠM DỪNG. Trước đây chỉ dựa vào set_tower_lights("PAUSED")
    # (reason "status:tower" trên CH3) — reason đó CÒN bị watchdog đèn tháp ghi
    # đè ("RUNNING" → tắt CH3) nên có lúc bấm PAUSE mà đèn vàng không sáng.
    # Nay PAUSE có reason RIÊNG ("status:paused") trên kênh này: kênh = OR của
    # mọi reason nên không nguồn nào tắt được đèn vàng khi đang tạm dừng.
    # Đổi số kênh nếu đèn vàng đấu vào DO khác.
    relay_pause_yellow_channel: int = 3      # CH3 (vàng) — 0 = tắt tính năng

    # ── Modbus RTU digital inputs (RED = STOP/PAUSE, GREEN = START/RESUME) ────
    relay_di_enabled: bool = False
    relay_di_red: int = 1
    relay_di_green: int = 2
    relay_di_poll_interval: float = 0.2
    relay_di_control_mode: str = "pause_resume"
    relay_status_enabled: bool = False
    relay_status_channel: int = 8
    relay_status2_enabled: bool = False
    relay_status2_channel: int = 7
    # CH7 "đi theo có trễ" CH1: CH1 (sợi đứt) đóng liên tục đủ
    # relay_follow_delay_s giây thì CH7 bắt đầu NHÁY (blink_on/blink_off),
    # tới khi CH1 mở thì CH7 mở ngay.
    relay_follow_enabled: bool = False
    relay_follow_channel: int = 7            # kênh DO đi theo (CH7)
    relay_follow_source_channel: int = 0     # 0 = dùng relay_channel (CH1)
    relay_follow_delay_s: float = 7.0        # CH1 phải ON liên tục bao lâu
    relay_follow_blink_on_s: float = 0.5     # thời gian ĐÓNG mỗi nhịp nháy
    relay_follow_blink_off_s: float = 0.5    # thời gian MỞ mỗi nhịp nháy (0/0 = giữ đóng)
    camera_timeout_shutdown_enabled: bool = False
    camera_timeout_minutes: float = 15.0
    system_shutdown_delay_minutes: float = 5.0
    enable_tower_watchdog: bool = True
    tower_watchdog_timeout_s: float = 5.0

    # ── [multicam] Hệ nhiều camera (camera trục + các camera lược) ────────────
    # _load_config chỉ nạp key YAML nào CÓ field tương ứng ở đây, nên các key mới
    # của block `comb:` phải được khai báo sẵn mới đọc được từ config.yaml.
    camera_id: str = "cam_main"              # định danh camera trục (state/DB/relay reason)
    # Kênh DO camera trục bật khi báo đứt: CH1 (đỏ) + CH2 (đèn báo / tắt đèn xanh).
    alarm_outputs: list = field(default_factory=lambda: [1, 2])
    # Cấu hình camera lược; rỗng = không bật. Xem block `comb:` trong config.yaml.
    comb: dict = field(default_factory=dict)
    # Vượt ngưỡng này: nhân đôi detect_every của camera lược (nhường GPU cho cam trục).
    thermal_throttle_c: float = 75.0


    # ── Region of interest (chỉ nhận diện trong vùng trục) ────────────────────
    # Khi khung có nhiều vật ngoài vùng trục gây nhiễu, bật roi_enabled và khai
    # báo hình chữ nhật vùng trục. Detection nào có TÂM box nằm ngoài vùng này bị
    # loại trước khi chọn box conf cao nhất → alert chỉ theo vật trong trục.
    # Toạ độ là TỈ LỆ 0.0–1.0 theo khung [x1, y1, x2, y2] (0,0=góc trên-trái,
    # 1,1=góc dưới-phải) nên không phụ thuộc độ phân giải camera.
    roi_enabled: bool     = False
    roi: List[float]      = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])

    # ── Vùng NHẠY phía trên ROI ───────────────────────────────────────────────
    # Chia ROI làm 2 phần theo chiều dọc:
    #   • roi_top_fraction (mặc định 0.10 = 10% CHIỀU CAO ROI tính từ cạnh trên):
    #     VÙNG NHẠY. Detection có TÂM box rơi vào đây được báo sợi đứt khi CHỈ CẦN
    #     roi_top_frames lần detect LIÊN TIẾP đều có conf > roi_top_conf → báo NGAY
    #     (không cần TB cao, không cần tăng dần, không qua cửa xác nhận). Đây là chỗ
    #     sợi thường đứt nên bắt rất sớm ở conf thấp.
    #   • 90% còn lại (phía dưới): giữ NGUYÊN logic cũ (TB ≥ alert_on_avg_conf, hoặc
    #     rise qua cửa xác nhận alert_rise_confirm_*).
    # Ranh giới chia = y_min + roi_top_fraction*(y_max − y_min) của đa giác ROI.
    # 0 = tắt (cả ROI dùng logic cũ).
    # LƯU Ý: conf_gate của model hạ xuống ≤ roi_top_conf để nhìn thấy detection yếu
    # tới mức này (áp dụng cho toàn khung, nhưng vùng dưới có ngưỡng riêng nên vẫn lọc).
    roi_top_fraction: float = 0.10
    roi_top_conf: float     = 0.65   # mỗi frame trong vùng trên phải có conf > mức này
    roi_top_frames: int     = 5      # số frame LIÊN TIẾP (conf > roi_top_conf) để báo
    # TRƯỢT riêng góc dưới-PHẢI của vùng trên DỌC theo cạnh phải ROI (điểm luôn bám
    # cạnh gốc, không lệch ra ngoài). Đơn vị PX theo phương y: DƯƠNG = trượt XUỐNG,
    # ÂM = lên. Px quy chiếu khung gốc cao 2048 (_ROI_REF_HEIGHT_PX). 0 = chia ngang.
    roi_top_right_drop_px: float = 15.0

    # ── Target class ──────────────────────────────────────────────────────────
    target_class: str     = "YarnBreak"
    display_label: str    = "Yarn Break"

    # Chỉ coi là YarnBreak khi cạnh TRÊN của bounding box (chiều rộng box = x2−x1)
    # rộng hơn ngưỡng này. Pixel đo trên KHUNG GỐC camera (trục ngang 2448px của
    # frame 2448×2048) — cùng hệ toạ độ box mà model trả về (r.orig_shape), không
    # quy đổi. Box hẹp hơn bị bỏ, không tính vào chuỗi báo động. 0 = tắt lọc.
    # Đây là ngưỡng của VÙNG DƯỚI (90%). Vùng trên dùng roi_top_min_box_width.
    min_box_top_width: float = 272.0
    # Ngưỡng bề rộng box RIÊNG cho VÙNG TRÊN (nhạy): box có tâm ở vùng trên chỉ cần
    # rộng hơn mức này (thấp hơn nhiều vì bắt sợi đứt sớm ở đầu trục). 0 = tắt lọc.
    roi_top_min_box_width: float = 100.0

    # Loại box "đứng" — cạnh DỌC (cao = y2−y1) lớn hơn cạnh NGANG (rộng = x2−x1).
    # Sợi đứt hiện thành vệt NGANG (rộng > cao); box cao hơn rộng thường là nhiễu
    # dọc (trục, khe sáng…) nên KHÔNG báo. True = bật lọc, False = tắt.
    reject_tall_box: bool = True

    # ── Đếm số lần báo sợi đứt theo CA (hiển thị trên web UI) ──────────────────
    # Mỗi khi alarm BẬT, detector tính TRUNG BÌNH conf của shift_count_frames lần
    # detect gần nhất. Giá trị này đi kèm callback on_alarm_trigger để lớp web
    # quyết định có +1 vào bộ đếm ca hay không (điều kiện conf TB + debounce nằm ở
    # inference_worker/alarm_logger). Ở đây chỉ cần biết lấy trung bình mấy frame.
    shift_count_frames: int = 10

    # ── Performance / power ───────────────────────────────────────────────────
    # Cap detector loop rate (fps). 0 = no cap.
    # Set to source_fps (e.g. 25) to limit GPU duty cycle and avoid overcurrent.
    max_infer_fps: float  = 0.0

    # CPU thread cap for OpenCV + PyTorch.  Inference runs on the GPU; the CPU
    # only does frame copies, letterbox preprocessing and NMS.  Letting those
    # libraries spread across all 6 Orin cores spikes CPU temperature for little
    # speed gain.  Capping the pool keeps cores cooler.
    #   0 → leave library defaults (all cores)
    #   4 → recommended on Orin Nano (leaves headroom for grabber + main loop)
    cpu_threads: int      = 4

    # ── Thermal guard ─────────────────────────────────────────────────────────
    # Monitor CPU/GPU thermal zones. If either crosses temp_limit_c, pause the
    # grabber + detector (GPU/CPU load drops so the SoC cools) for cooldown_s,
    # then resume automatically. Monitoring continues, so it re-pauses if still hot.
    thermal_guard: bool   = True
    temp_limit_c: float   = 80.0      # pause when CPU or GPU reaches this (°C)
    cooldown_s: float     = 1200.0    # pause duration before resuming (s) = 20 min
    thermal_check_s: float = 5.0      # how often to sample temps (s)

    # ── Profile ───────────────────────────────────────────────────────────────
    show_profile: bool    = True
    profile_every: int    = 30


def _parse_roi(roi) -> Optional[List[Tuple[float, float]]]:
    """Chuẩn hoá cấu hình ROI về danh sách đỉnh đa giác [(x,y), …] tỉ lệ 0–1.

    Chấp nhận 2 dạng:
      • Đa giác: [[x1,y1], [x2,y2], …]  (≥3 đỉnh, theo thứ tự vòng)
      • Chữ nhật: [x1, y1, x2, y2]       (4 số phẳng → tự dựng 4 góc)
    Trả về None nếu không hợp lệ hoặc bằng toàn khung (không cần lọc)."""
    if not roi:
        return None
    try:
        # Đa giác: mọi phần tử là cặp [x, y].
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in roi):
            pts = [(float(x), float(y)) for x, y in roi]
            if len(pts) < 3:
                return None
        # Chữ nhật phẳng [x1,y1,x2,y2].
        elif len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
            x1, y1, x2, y2 = (float(v) for v in roi)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
                return None
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            return None
    except (ValueError, TypeError):
        return None
    return pts


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting: True nếu điểm (x,y) nằm trong đa giác poly [(px,py), …]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and \
           x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _roi_top_split_y(poly: List[Tuple[float, float]], frac: float) -> Optional[float]:
    """Trả về toạ độ y (tỉ lệ 0–1) chia ROI: mọi điểm có y ≤ giá trị này thuộc
    'vùng trên' (frac% chiều cao ROI tính từ cạnh trên). None nếu không áp dụng."""
    if not poly or frac <= 0.0:
        return None
    ys = [y for _, y in poly]
    ymin, ymax = min(ys), max(ys)
    if ymax <= ymin:
        return None
    return ymin + frac * (ymax - ymin)


def _clip_poly_half(poly: List[Tuple[float, float]], y_line: float,
                    keep_top: bool) -> List[Tuple[float, float]]:
    """Cắt đa giác bằng đường ngang y=y_line (Sutherland–Hodgman). keep_top=True
    giữ phần y ≤ y_line (phía trên), False giữ phần y ≥ y_line (phía dưới)."""
    inside = (lambda p: p[1] <= y_line) if keep_top else (lambda p: p[1] >= y_line)
    out: List[Tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[i - 1]
        cur_in, prv_in = inside(cur), inside(prv)
        if cur_in:
            if not prv_in:
                out.append(_intersect_y(prv, cur, y_line))
            out.append(cur)
        elif prv_in:
            out.append(_intersect_y(prv, cur, y_line))
    return out


def _intersect_y(a: Tuple[float, float], b: Tuple[float, float],
                 y_line: float) -> Tuple[float, float]:
    """Giao điểm của đoạn a–b với đường ngang y=y_line."""
    ax, ay = a
    bx, by = b
    if by == ay:
        return (ax, y_line)
    t = (y_line - ay) / (by - ay)
    return (ax + t * (bx - ax), y_line)


# Chiều cao khung GỐC camera (px) mà toạ độ tỉ lệ ROI quy chiếu tới — dùng để đổi
# "nâng góc phải bao nhiêu px" sang tỉ lệ 0–1 (khớp hệ đo của min_box_top_width).
_ROI_REF_HEIGHT_PX = 2048.0


def _right_edge_point(poly: List[Tuple[float, float]], y0: float,
                      drop: float) -> Optional[Tuple[float, float]]:
    """Điểm nằm TRÊN cạnh phải ROI (cạnh cắt đường y0 tại x lớn nhất), trượt DỌC theo
    cạnh xuống 'drop' theo phương y (dương = xuống, âm = lên). Nhờ bám cạnh gốc nên
    điểm không bao giờ lệch ra ngoài ROI."""
    best = None  # (x_tại_y0, đỉnh_a, đỉnh_b)
    n = len(poly)
    for i in range(n):
        a, b = poly[i - 1], poly[i]
        ay, by = a[1], b[1]
        if (ay > y0) != (by > y0):          # cạnh a–b cắt ngang y0
            t = (y0 - ay) / (by - ay)
            x = a[0] + t * (b[0] - a[0])
            if best is None or x > best[0]:
                best = (x, a, b)
    if best is None:
        return None
    _, a, b = best
    yt = y0 + drop
    t = (yt - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), yt)


def _replace_right_y0_vertex(band: List[Tuple[float, float]], y0: float,
                             newpt: Tuple[float, float], eps: float = 1e-6) -> None:
    """Thay đỉnh có x LỚN NHẤT nằm trên đường chia (y≈y0) của 'band' bằng newpt. Dùng
    cho cả band trên lẫn dưới để 2 band vẫn dùng chung đúng một điểm → không hở/chồng."""
    idxs = [i for i, (_, y) in enumerate(band) if abs(y - y0) < eps]
    if not idxs:
        return
    ri = max(idxs, key=lambda i: band[i][0])
    band[ri] = newpt


def _roi_split_polys(poly: Optional[List[Tuple[float, float]]], frac: float,
                     right_drop: float = 0.0):
    """Chia ROI thành (đa giác VÙNG TRÊN, đa giác VÙNG DƯỚI) theo tỉ lệ frac.
    right_drop ≠ 0 → TRƯỢT riêng góc dưới-PHẢI của vùng trên DỌC theo cạnh phải ROI:
    dương = xuống, âm = lên. Trả về (None, None) nếu không áp dụng."""
    y0 = _roi_top_split_y(poly, frac)
    if y0 is None:
        return None, None
    top = _clip_poly_half(poly, y0, keep_top=True)
    bot = _clip_poly_half(poly, y0, keep_top=False)
    if right_drop != 0.0:
        r2 = _right_edge_point(poly, y0, right_drop)
        if r2 is not None:
            _replace_right_y0_vertex(top, y0, r2)
            _replace_right_y0_vertex(bot, y0, r2)
    return top, bot


def _load_config(yaml_path: Optional[str], cli: argparse.Namespace) -> AppConfig:
    """Merge YAML → CLI overrides → resolved device."""
    cfg = AppConfig()

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        logger.info(f"Config loaded from: {yaml_path}")

    # CLI args that were explicitly provided override YAML / defaults
    cli_dict = vars(cli)
    for k, v in cli_dict.items():
        attr = k.replace("-", "_")
        if v is not None and hasattr(cfg, attr):
            setattr(cfg, attr, v)

    # Resolve device
    if cfg.device == "auto":
        cfg.device = "0" if torch.cuda.is_available() else "cpu"

    # headless implies no display
    if cfg.headless:
        cfg.display = False

    # FP16 only meaningful with CUDA
    if cfg.device == "cpu":
        cfg.half = False

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME GRABBER
# ══════════════════════════════════════════════════════════════════════════════

class FrameGrabber:
    """Base class – subclasses implement _run()."""

    def __init__(self, cfg: AppConfig, frame_q: queue.Queue, stop: threading.Event,
                 display_q: Optional[queue.Queue] = None,
                 pause: Optional[threading.Event] = None,
                 on_camera_connected: Optional[callable] = None):
        self.cfg = cfg
        self.frame_q = frame_q
        self.display_q = display_q
        self.stop = stop
        self.pause = pause
        self.on_camera_connected = on_camera_connected
        self.source_fps: float = 30.0
        self._thread: Optional[threading.Thread] = None

    def _wait_if_paused(self) -> bool:
        """Block while the thermal guard holds the pause. Returns True if a pause
        actually happened (caller should resync any frame-pacing clock)."""
        paused = False
        while self.pause is not None and self.pause.is_set() and not self.stop.is_set():
            paused = True
            time.sleep(0.5)
        return paused

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="FrameGrabber")
        self._thread.start()

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _push(self, frame: np.ndarray):
        """Drop oldest frame, push newest (keep queue at maxsize=1)."""
        try:
            self.frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            pass
        if self.display_q is not None:
            try:
                self.display_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.display_q.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self):
        raise NotImplementedError


# ─── Video file ───────────────────────────────────────────────────────────────

class VideoFileGrabber(FrameGrabber):
    def _run(self):
        path = self.cfg.video
        if not os.path.exists(path):
            logger.error(f"Video file not found: {path}")
            self.stop.set()
            return

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {path}")
            self.stop.set()
            return

        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Video: {path}  FPS={self.source_fps:.2f}  frames={total}  "
            f"pace={self.cfg.pace_video}  loop={self.cfg.loop_video}"
        )
        if self.on_camera_connected:
            self.on_camera_connected(True)

        frame_period = 1.0 / max(self.source_fps, 1.0)
        next_t = time.perf_counter()

        try:
            while not self.stop.is_set():
                # Thermal pause: stop decoding so the CPU cools; resync clock after.
                if self._wait_if_paused():
                    next_t = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.cfg.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_t = time.perf_counter()
                        continue
                    logger.info("End of video.")
                    self.stop.set()
                    break

                self._push(frame)

                if self.cfg.pace_video:
                    # Sleep until next frame's real-time slot
                    next_t += frame_period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        # We're behind schedule; resync to avoid drift
                        next_t = time.perf_counter()
        finally:
            cap.release()
            if self.on_camera_connected:
                self.on_camera_connected(False)


# ─── IP Camera (RTSP) ─────────────────────────────────────────────────────────

def build_rtsp_gstreamer_pipeline(rtsp_url: str, codec: str = "h264", latency: int = 200) -> str:
    """
    Hardware-accelerated GStreamer pipeline for Jetson RTSP capture.
    codec   : "h264" | "h265" – choose based on your camera stream.
    latency : rtspsrc latency in milliseconds (lower = less delay, more jitter).
    Requires: OpenCV built with GStreamer support (check cv2.getBuildInformation()).
    """
    codec = (codec or "h264").lower()
    if codec == "h265":
        depay_parse = "rtph265depay ! h265parse"
    else:
        depay_parse = "rtph264depay ! h264parse"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} "
        f"drop-on-latency=true buffer-mode=0 "
        f"! {depay_parse} "
        "! nvv4l2decoder "
        "! nvvidconv "
        "! video/x-raw,format=BGRx "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=1 max-buffers=1 sync=false"
    )


# Backward-compatible alias for the previous private name.
_gst_rtsp_pipeline = build_rtsp_gstreamer_pipeline


class IPCamGrabber(FrameGrabber):
    def _run(self):
        url = self.cfg.rtsp
        delay = self.cfg.reconnect_delay
        fail_count = 0

        while not self.stop.is_set():
            cap = self._open(url)

            if cap is None or not cap.isOpened():
                fail_count += 1
                if fail_count == 1 or fail_count % 10 == 0:
                    logger.warning(f"[Attempt {fail_count}] Cannot connect to {url}. Retry in {delay}s")
                time.sleep(delay)
                continue

            self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            logger.info(f"Camera connected: {url}  FPS={self.source_fps:.1f}")
            if self.on_camera_connected:
                self.on_camera_connected(True)
            fail_count = 0

            try:
                consec_fail = 0
                while not self.stop.is_set():
                    # Thermal pause: drop the live stream (buffers flush) while cooling.
                    self._wait_if_paused()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consec_fail += 1
                        if consec_fail >= 5:
                            logger.warning("Camera stream interrupted. Reconnecting…")
                            break
                        continue
                    consec_fail = 0
                    self._push(frame)
            finally:
                cap.release()
                if self.on_camera_connected:
                    self.on_camera_connected(False)

            if not self.stop.is_set():
                logger.info(f"Reconnecting in {delay}s…")
                time.sleep(delay)

    def _open(self, url: str) -> Optional[cv2.VideoCapture]:
        # Codec order: preferred from config first, the other as fallback.
        preferred = (self.cfg.codec or "h264").lower()
        codec_order = [preferred] + [c for c in ("h264", "h265") if c != preferred]
        latency = int(self.cfg.latency_ms or 200)

        logger.info(f"Connecting RTSP… codec_pref={preferred} latency={latency}ms")

        # 1) Try GStreamer hardware pipeline (Jetson)
        for codec in codec_order:
            pipe = build_rtsp_gstreamer_pipeline(url, codec=codec, latency=latency)
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info(f"Connected: GStreamer pipeline ({codec}).")
                return cap

        # 2) OpenCV software RTSP fallback
        logger.warning("GStreamer pipeline unavailable – falling back to cv2 RTSP…")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set timeouts to avoid blocking forever (OpenCV 4.5.4+)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass
        if cap.isOpened():
            logger.info("Connected: cv2 RTSP fallback.")
            return cap

        return None


# ─── Hikrobot MVS industrial camera (GigE / USB3 Vision) ──────────────────────

class MVSCameraGrabber(FrameGrabber):
    """Grab frames directly from a Hikrobot camera via the MVS SDK.

    Tested with MV-CS050-10GC (GigE, 2448×2048 BayerRG8). Each raw frame is
    converted to BGR8 on the host with MV_CC_ConvertPixelType so the rest of the
    pipeline sees the same np.ndarray (H, W, 3) BGR frames as the video/RTSP paths.
    """

    def _import_sdk(self):
        """Import the MVS Python bindings lazily so the module loads on machines
        without the SDK installed. Returns the MvImport module namespace."""
        sdk_path = self.cfg.mvs_sdk_path
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
        import MvCameraControl_class as mvs  # noqa: E402
        return mvs

    def _run(self):
        try:
            mvs = self._import_sdk()
        except Exception as e:
            logger.error(f"Cannot import MVS SDK (mvs_sdk_path={self.cfg.mvs_sdk_path}): {e}")
            self.stop.set()
            return

        delay = self.cfg.reconnect_delay
        fail_count = 0

        while not self.stop.is_set():
            cam = self._open_camera(mvs)
            if cam is None:
                fail_count += 1
                if fail_count == 1 or fail_count % 10 == 0:
                    logger.warning(f"[Attempt {fail_count}] No Hikrobot camera. Retry in {delay}s")
                time.sleep(delay)
                continue
            fail_count = 0
            if self.on_camera_connected:
                self.on_camera_connected(True)

            try:
                self._grab_loop(mvs, cam)
            except Exception as e:
                logger.warning(f"MVS grab loop error: {e}. Reconnecting…")
            finally:
                self._close_camera(cam)
                if self.on_camera_connected:
                    self.on_camera_connected(False)

            if not self.stop.is_set():
                logger.info(f"Reconnecting MVS camera in {delay}s…")
                time.sleep(delay)

    def _open_camera(self, mvs):
        """Enumerate, pick the camera (by serial if configured), open and configure it."""
        from ctypes import cast, POINTER, byref, c_bool

        dev_list = mvs.MV_CC_DEVICE_INFO_LIST()
        tlayer = mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE
        ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
        if ret != 0 or dev_list.nDeviceNum == 0:
            return None

        # Select device: matching serial if requested, else the first one.
        want = (self.cfg.mvs_serial or "").strip()
        chosen = None
        for i in range(dev_list.nDeviceNum):
            info = cast(dev_list.pDeviceInfo[i], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            serial = self._dev_serial(mvs, info)
            if not want or serial == want:
                chosen = (info, serial)
                if not want:
                    break
                if serial == want:
                    break
        if chosen is None:
            logger.warning(f"MVS: serial '{want}' not found among {dev_list.nDeviceNum} device(s)")
            return None

        info, serial = chosen
        cam = mvs.MvCamera()
        if cam.MV_CC_CreateHandle(info) != 0:
            return None
        if cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0) != 0:
            logger.warning("MVS: OpenDevice failed (camera busy or wrong subnet?)")
            cam.MV_CC_DestroyHandle()
            return None

        # GigE: use an optimal jumbo packet size for max throughput.
        if info.nTLayerType == mvs.MV_GIGE_DEVICE:
            ps = self.cfg.mvs_packet_size or cam.MV_CC_GetOptimalPacketSize()
            if ps > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", ps)

        # Continuous, free-running acquisition (no hardware trigger).
        cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")

        # Optional manual exposure / gain (otherwise leave the camera's own auto).
        if self.cfg.mvs_exposure_us and self.cfg.mvs_exposure_us > 0:
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            cam.MV_CC_SetFloatValue("ExposureTime", float(self.cfg.mvs_exposure_us))
        if self.cfg.mvs_gain is not None and self.cfg.mvs_gain >= 0:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            cam.MV_CC_SetFloatValue("Gain", float(self.cfg.mvs_gain))

        # Frame-rate cap. >0 caps AcquisitionFrameRate (lighter GPU/thermal load);
        # <=0 explicitly DISABLES the cap so the camera free-runs at its max. The
        # disable is important because the enable flag is stored in the camera and
        # would otherwise stick at a previous run's value.
        if self.cfg.mvs_fps and self.cfg.mvs_fps > 0:
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.cfg.mvs_fps))
        else:
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

        # Report resulting frame rate / resolution.
        fr = mvs.MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("ResultingFrameRate", fr) == 0 and fr.fCurValue > 0:
            self.source_fps = float(fr.fCurValue)
        w = mvs.MVCC_INTVALUE(); h = mvs.MVCC_INTVALUE()
        cam.MV_CC_GetIntValue("Width", w); cam.MV_CC_GetIntValue("Height", h)

        if cam.MV_CC_StartGrabbing() != 0:
            logger.warning("MVS: StartGrabbing failed")
            cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
            return None

        logger.info(f"MVS camera connected: serial={serial}  "
                    f"{w.nCurValue}x{h.nCurValue} @ {self.source_fps:.1f}fps")
        return cam

    @staticmethod
    def _dev_serial(mvs, info) -> str:
        if info.nTLayerType == mvs.MV_GIGE_DEVICE:
            chars = info.SpecialInfo.stGigEInfo.chSerialNumber
        else:
            chars = info.SpecialInfo.stUsb3VInfo.chSerialNumber
        return "".join(chr(c) for c in chars if c)

    def _grab_loop(self, mvs, cam):
        from ctypes import byref, memset, sizeof, cast, POINTER, c_ubyte

        frame_out = mvs.MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        dst_buf = None
        dst_size = 0
        consec_fail = 0

        while not self.stop.is_set():
            # Thermal pause: stop the stream so the sensor/SoC load drops, then resume.
            if self.pause is not None and self.pause.is_set():
                cam.MV_CC_StopGrabbing()
                self._wait_if_paused()
                if self.stop.is_set():
                    break
                cam.MV_CC_StartGrabbing()
                consec_fail = 0
                continue

            ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
            if ret != 0:
                consec_fail += 1
                if consec_fail >= 5:
                    raise RuntimeError(f"GetImageBuffer failed (0x{ret & 0xffffffff:X})")
                continue
            consec_fail = 0

            try:
                fi = frame_out.stFrameInfo
                need = fi.nWidth * fi.nHeight * 3
                if dst_buf is None or dst_size < need:
                    dst_buf = (c_ubyte * need)()
                    dst_size = need

                cvt = mvs.MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(cvt), 0, sizeof(cvt))
                cvt.nWidth = fi.nWidth
                cvt.nHeight = fi.nHeight
                cvt.pSrcData = frame_out.pBufAddr
                cvt.nSrcDataLen = fi.nFrameLen
                cvt.enSrcPixelType = fi.enPixelType
                cvt.enDstPixelType = mvs.PixelType_Gvsp_BGR8_Packed
                cvt.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
                cvt.nDstBufferSize = dst_size

                if cam.MV_CC_ConvertPixelType(cvt) != 0:
                    continue

                # BGR8 packed → contiguous (H, W, 3) BGR frame for the pipeline.
                frame = np.frombuffer(dst_buf, dtype=np.uint8,
                                      count=need).reshape(fi.nHeight, fi.nWidth, 3).copy()
            finally:
                cam.MV_CC_FreeImageBuffer(frame_out)

            self._push(frame)

    def _close_camera(self, cam):
        try:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass


def make_grabber(cfg: AppConfig, fq: queue.Queue, stop: threading.Event,
                 display_q: Optional[queue.Queue] = None,
                 pause: Optional[threading.Event] = None,
                 on_camera_connected: Optional[callable] = None) -> FrameGrabber:
    if cfg.source == "ipcam":
        if not cfg.rtsp:
            raise ValueError("--rtsp URL is required when --source ipcam")
        return IPCamGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)
    if cfg.source == "mvs":
        return MVSCameraGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)
    # source == "video"
    if not cfg.video:
        raise ValueError("--video PATH is required when --source video")
    if not Path(cfg.video).is_file():
        raise FileNotFoundError(f"Video file not found: {cfg.video}")
    return VideoFileGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO DETECTOR  (inference worker thread)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_class_id(names, target: str) -> int:
    key = _normalize(target)
    mapping = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    for cid, cname in mapping.items():
        if _normalize(cname) == key:
            return int(cid)
    raise ValueError(f"Class '{target}' not found in model names: {names}")


class YoloDetector:
    def __init__(self, cfg: AppConfig, in_q: queue.Queue, out_q: queue.Queue,
                 stop: threading.Event, notifier: "Optional[AlertNotifier]" = None,
                 pause: Optional[threading.Event] = None,
                 reset_alarm_evt: Optional[threading.Event] = None,
                 on_model_loaded: Optional[callable] = None):
        self.cfg = cfg
        self.in_q = in_q
        self.out_q = out_q
        self.stop = stop
        self.pause = pause
        self._notifier = notifier
        self.reset_alarm_evt = reset_alarm_evt
        self.on_model_loaded = on_model_loaded
        self.model: Optional[YOLO] = None
        self.target_id: int = -1
        self._thread: Optional[threading.Thread] = None
        # The segmentation polygon is only ever drawn (display / saved video).
        # Computing masks.xy runs CPU contour extraction on every inference, so
        # skip it when nothing consumes the polygon — saves CPU heat in headless.
        # Chỉ tính masks.xy (contour extraction trên CPU, rất nặng ở 5MP, chạy mỗi
        # frame có detection) khi polygon thực sự được vẽ. Tắt enable_mask hoặc bật
        # fast_draw → bỏ bước này, lấy lại FPS khi đang detect (chỉ vẽ bounding box).
        self._need_poly = ((cfg.display or cfg.save_video or cfg.event_clip)
                           and cfg.enable_mask and not cfg.fast_draw)
        # ROI (vùng trục) dạng đa giác, toạ độ tỉ lệ 0–1. Bỏ qua nếu tắt/toàn khung.
        self._roi = _parse_roi(cfg.roi) if cfg.roi_enabled else None
        if self._roi is not None:
            pts = ", ".join(f"({x:.3f},{y:.3f})" for x, y in self._roi)
            logger.info(f"ROI enabled ({len(self._roi)} pts, fractions): {pts}")
        elif cfg.roi_enabled:
            logger.warning(f"Invalid/empty roi={cfg.roi!r}; ROI filter disabled.")
        # Đa giác VÙNG TRÊN (nhạy) — box có TÂM nằm trong đây thuộc vùng trên. Đường
        # chia có thể nghiêng (trượt góc phải dọc cạnh ROI, roi_top_right_drop_px). Phân
        # loại bằng điểm-trong-đa-giác để luôn khớp đúng hình được vẽ.
        drop = cfg.roi_top_right_drop_px / _ROI_REF_HEIGHT_PX
        self._roi_top_poly, _ = (_roi_split_polys(self._roi, cfg.roi_top_fraction, drop)
                                 if self._roi is not None else (None, None))
        if self._roi_top_poly is not None:
            y0 = _roi_top_split_y(self._roi, cfg.roi_top_fraction)
            logger.info(f"ROI vùng trên (nhạy) ~y0={y0:.3f} "
                        f"({cfg.roi_top_fraction*100:.0f}% trên, trượt phải "
                        f"{cfg.roi_top_right_drop_px:.0f}px dọc cạnh, "
                        f"{cfg.roi_top_frames} frame conf>{cfg.roi_top_conf} → báo)")
        # Vùng trên bật? → cần hạ gate model để thấy detection tới roi_top_conf.
        self._top_enabled = self._roi_top_poly is not None
        # Alert state owned by detector (worker thread only – no mutex needed)
        self._miss = 0
        self._alert = False
        self._prev_alert = False
        self._alert_just_on = False
        # Ngưỡng ĐƯA VÀO model. Các luật cần nhìn thấy detection yếu mà cfg.conf đang
        # chặn, nên hạ ngưỡng model xuống mức THẤP NHẤT mà một luật nào đó còn quan tâm:
        # rise cần >alert_rise_conf_min (0.5); vùng trên cần >roi_top_conf (0.4). cfg.conf
        # VẪN là mức "conf cao" của luật gốc — việc lọc theo mức nào là do _update_alert
        # quyết định, không phải model.
        gates = [cfg.conf]
        if cfg.alert_rise_enabled:
            gates.append(cfg.alert_rise_conf_min)
        if self._top_enabled:
            gates.append(cfg.roi_top_conf)
        self._conf_gate = min(gates)
        # conf của các lần detect LIÊN TIẾP (mọi mức ≥ gate) — nguyên liệu cho luật rise.
        self._conf_hist: deque = deque(maxlen=max(2, cfg.alert_rise_window))
        # conf của alert_on_consec lần detect gần nhất — nguyên liệu cho luật trung bình (đường 1).
        self._avg_hist: deque = deque(maxlen=max(1, cfg.alert_on_consec))
        # conf của alert_long_off_window lần detect gần nhất — nguyên liệu cho luật auto-off
        # khi alarm kéo dài. KHÔNG xoá khi miss (rolling last-N) để luôn có dữ liệu đánh giá.
        self._hi_hist: deque = deque(maxlen=max(1, cfg.alert_long_off_window))
        # conf của shift_count_frames lần detect gần nhất — để tính TRUNG BÌNH tại
        # thời điểm alarm bật (điều kiện đếm số lần báo theo ca). Rolling last-N,
        # KHÔNG xoá khi miss để avg phản ánh chất lượng detect quanh lúc báo.
        self._count_hist: deque = deque(maxlen=max(1, cfg.shift_count_frames))
        # TB conf 10 lần detect gần nhất, chốt tại cạnh lên alarm (0 nếu chưa có).
        self._alert_avg_conf: float = 0.0
        # thời điểm alarm bật (cạnh lên) — để đo thời lượng alarm ON cho luật auto-off.
        self._alert_on_time: Optional[float] = None
        # Cửa xác nhận đường 2: mốc thời gian bắt đầu chờ (None = không chờ) và cờ đã
        # từng thấy conf > alert_rise_confirm_conf trong lúc chờ hay chưa.
        self._rise_pending_time: Optional[float] = None
        self._rise_pending_hi: bool = False
        # Đếm số frame LIÊN TIẾP có target ở VÙNG TRÊN với conf > roi_top_conf.
        # Reset khi miss / frame ở vùng dưới / conf ≤ roi_top_conf. ≥ roi_top_frames → báo.
        self._top_hits: int = 0
        # Target gần nhất từ một lần inference THẬT. Frame skip (detect_every) đẩy lại
        # target này thay vì None, để bounding box bám theo đúng nhịp detection (giống
        # alert) — nếu đẩy None, frame skip sẽ ghi đè/xóa box trong display purge.
        self._last_target: Optional[Dict] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self):
        path = self.cfg.model
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        logger.info(f"Loading model: {path}")
        # TensorRT .engine không tự đoán được task → ép 'segment' (model này có mask).
        # File .pt vẫn để ultralytics tự nhận diện như cũ.
        if str(path).endswith(".engine"):
            self.model = YOLO(path, task="segment")
        else:
            self.model = YOLO(path)
        self.target_id = _find_class_id(self.model.names, self.cfg.target_class)
        logger.info(f"Target class '{self.cfg.target_class}' → id={self.target_id}")

        c = self.cfg
        logger.info(f"Alert luật 1 (conf cao): trung bình {c.alert_on_consec} lần detect ≥ {c.alert_on_avg_conf}")
        if c.alert_rise_enabled:
            logger.info(f"Alert luật 2 (tăng dần): {c.alert_rise_window} lần liên tiếp đều "
                        f"> {c.alert_rise_conf_min}, cuối−đầu ≥ {c.alert_rise_min_gain}, "
                        f"không tụt quá {c.alert_rise_max_dip}/lần → chờ "
                        f"{c.alert_rise_confirm_seconds:.0f}s VÀ có conf > {c.alert_rise_confirm_conf} mới báo")
        else:
            logger.info("Alert luật 2 (tăng dần): TẮT")
        logger.info(f"Ngưỡng đưa vào model: {self._conf_gate}  |  nhả alarm sau "
                    f"{c.alert_off_miss} lần không thấy gì")
        if c.alert_long_on_seconds > 0:
            logger.info(f"Auto-off: alarm ON ≥ {c.alert_long_on_seconds:.0f}s mà "
                        f"{c.alert_long_off_window} lần detect gần nhất đều ≤ {c.alert_long_off_conf} → tắt")

        self._check_device()
        # Warmup + on_model_loaded được chuyển sang _run() (luồng detector). Lần
        # predict đầu (warmup) mới thực sự tạo execution context của AutoBackend/
        # TensorRT; context này KHÔNG thread-safe. Nếu tạo ở luồng main rồi inference
        # ở luồng detector, .engine chạy được vài frame rồi treo cứng (heartbeat đứng
        # → tower watchdog báo lỗi). Warmup trong chính luồng inference để tránh việc đó.

    def _check_device(self):
        d = str(self.cfg.device)
        if d == "0":
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {name}  VRAM: {mem_gb:.1f} GB")
                logger.info(f"FP16: {'enabled' if self.cfg.half else 'disabled'}")
            else:
                logger.warning("CUDA device=0 requested but CUDA unavailable. Switching to CPU.")
                self.cfg.device = "cpu"
                self.cfg.half = False
        else:
            logger.info("Running on CPU.")
            self.cfg.half = False

    def _warmup(self):
        logger.info(f"Warming up model (imgsz={self.cfg.imgsz})…")
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        with torch.inference_mode():
            self.model.predict(
                source=dummy, conf=self._conf_gate, imgsz=self.cfg.imgsz,
                classes=[self.target_id], device=self.cfg.device,
                half=self.cfg.half, verbose=False,
            )
        logger.info("Warmup complete.")

    # ── Thread ────────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="YoloWorker")
        self._thread.start()

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        # Warmup NGAY trong luồng này để execution context TensorRT gắn đúng luồng
        # inference (xem chú thích ở load()). Nếu lỗi → dừng chương trình.
        try:
            self._warmup()
        except Exception as exc:
            logger.error(f"Model warmup failed in detector thread: {exc}")
            self.stop.set()
            return
        if self.on_model_loaded:
            self.on_model_loaded()

        # Luồng này chết = hệ thống MÙ: không còn inference, nhưng grabber/display vẫn
        # chạy nên log trông vẫn khoẻ (fps bình thường), chỉ có infer=0.0ms và heartbeat
        # đứng im. Traceback mặc định ra stderr — chạy qua web_server thì stderr là
        # socket nên KHÔNG vào file log → lỗi vô hình. Ghi lại rồi dừng hẳn pipeline.
        try:
            self._loop()
        except Exception:
            logger.exception("Detector thread crashed — stopping pipeline")
            self.stop.set()

    def _loop(self):
        n = self.cfg.detect_every
        frame_counter = 0
        min_period = (1.0 / self.cfg.max_infer_fps) if self.cfg.max_infer_fps > 0 else 0.0
        _next_slot = time.perf_counter()

        was_paused = False
        while True:
            if self.reset_alarm_evt is not None and self.reset_alarm_evt.is_set():
                self._miss = 0
                self._avg_hist.clear()
                self._conf_hist.clear()
                self._hi_hist.clear()
                self._top_hits = 0
                self._alert_on_time = None
                self._rise_pending_time = None
                self._rise_pending_hi = False
                self._alert = self._prev_alert = self._alert_just_on = False
                if self._notifier:
                    self._notifier.force_off("manual reset")
                self.reset_alarm_evt.clear()

            # Thermal pause: hold off all GPU inference so the SoC can cool.
            if self.pause is not None and self.pause.is_set():
                if not was_paused:
                    # Entering pause: clear the alert latch so that when we resume,
                    # a still-present break re-fires a fresh ON edge. Also force the
                    # alarm output OFF so ANY pause (red button / web) silences an
                    # active alarm — the thermal guard already does this on its path.
                    # Safe: only the detector thread owns this state.
                    self._miss = 0
                    self._avg_hist.clear()
                    self._conf_hist.clear()
                    self._hi_hist.clear()
                    self._top_hits = 0
                    self._alert_on_time = None
                    self._rise_pending_time = None
                    self._rise_pending_hi = False
                    self._alert = self._prev_alert = self._alert_just_on = False
                    self._last_target = None
                    was_paused = True
                    if self._notifier is not None:
                        # Chỉ nhả CH1; CH2/CH3 để _sync_pause bật theo trạng thái paused
                        # (DO1 tắt, DO2 bật, DO3 bật).
                        self._notifier.release_stop_output("paused")
                if self.stop.is_set():
                    break
                time.sleep(0.5)
                _next_slot = time.perf_counter()  # resync rate-cap clock on resume
                continue
            was_paused = False

            try:
                item = self.in_q.get(timeout=0.5)
                try:
                    from state_manager import state_manager
                    state_manager.last_heartbeat = time.monotonic()
                except Exception:
                    pass
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue

            if item is None:
                break

            frame_counter += 1

            # detect_every_n: skip inference on intermediate frames.
            # Đẩy lại target của lần inference gần nhất (không phải None) để box hiển
            # thị liên tục giữa các lần inference, đồng bộ với badge alarm.
            if n > 1 and (frame_counter % n) != 0:
                self._push_result(None, self._last_target, 0.0)
                continue

            # Optional rate cap — prevents GPU from running at 100% duty cycle,
            # reducing peak power draw and avoiding overcurrent on marginal supplies.
            if min_period > 0:
                wait = _next_slot - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                _next_slot = max(time.perf_counter(), _next_slot) + min_period

            t0 = time.perf_counter()
            with torch.inference_mode():
                results = self.model.predict(
                    source=item, conf=self._conf_gate, imgsz=self.cfg.imgsz,
                    classes=[self.target_id], device=self.cfg.device,
                    half=self.cfg.half, verbose=False,
                )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            target = self._parse(results[0])
            self._last_target = target
            self._update_alert(target)

            # Fire alert immediately from detector thread — avoids result_q drain delay.
            # Không (tái) kích alarm nếu một lệnh pause đang tới (nút đỏ/web): tránh
            # sợi vẫn đứt làm CH1 đóng lại ngay sau khi nút đỏ vừa nhả relay.
            _pausing = self.pause is not None and self.pause.is_set()
            if self._alert_just_on and self._notifier is not None and not _pausing:
                conf = target["conf"] if target else 0.0
                self._notifier.on_alert(frame_counter, conf)
            elif (self._prev_alert and not self._alert) and self._notifier is not None:
                self._notifier.on_clear(frame_counter)

            self._push_result(item, target, infer_ms)

    def _box_zone(self, bx1: float, by1: float, bx2: float, by2: float,
                  W: int, H: int) -> str:
        """'top' nếu TÂM box nằm trong đa giác vùng trên (nhạy), ngược lại 'bottom'."""
        if self._roi_top_poly is None:
            return "bottom"
        cx = (bx1 + bx2) * 0.5 / W
        cy = (by1 + by2) * 0.5 / H
        return "top" if _point_in_poly(cx, cy, self._roi_top_poly) else "bottom"

    def _parse(self, r) -> Optional[Dict]:
        boxes, masks = r.boxes, r.masks
        if boxes is None or len(boxes) == 0:
            return None
        confs = boxes.conf
        xyxy = boxes.xyxy

        H, W = r.orig_shape  # (H, W) của frame gốc

        # Lọc vùng trục: chỉ giữ box có TÂM nằm trong ROI, rồi mới chọn conf cao
        # nhất trong số còn lại. Đặt trước argmax để một vật ngoài trục (dù conf
        # cao) không lấn át được detection thật trong vùng trục.
        cand = list(range(len(boxes)))
        if self._roi is not None:
            roi_cand = []
            for k in cand:
                bx1, by1, bx2, by2 = xyxy[k].tolist()
                # Tâm box quy về tỉ lệ 0–1 rồi kiểm tra nằm trong đa giác ROI.
                cx = (bx1 + bx2) * 0.5 / W
                cy = (by1 + by2) * 0.5 / H
                if _point_in_poly(cx, cy, self._roi):
                    roi_cand.append(k)
            cand = roi_cand
            if not cand:
                return None

        # Luật bề rộng THEO VÙNG: box ở vùng trên chỉ cần rộng > roi_top_min_box_width
        # (mặc định 100px); box ở vùng dưới phải rộng > min_box_top_width (272px). Loại
        # box hẹp TRƯỚC khi chọn conf cao nhất để box hẹp (dù conf cao) không lấn box
        # rộng thật. Ngưỡng ≤ 0 = không lọc ở vùng đó.
        kept = []
        for k in cand:
            bx1, by1, bx2, by2 = xyxy[k].tolist()
            # Loại box "đứng": cạnh dọc (cao) > cạnh ngang (rộng) → không báo.
            if self.cfg.reject_tall_box and (by2 - by1) > (bx2 - bx1):
                continue
            z = self._box_zone(bx1, by1, bx2, by2, W, H)
            thr = (self.cfg.roi_top_min_box_width if z == "top"
                   else self.cfg.min_box_top_width)
            if thr <= 0 or (bx2 - bx1) > thr:
                kept.append(k)
        cand = kept
        if not cand:
            return None

        i = max(cand, key=lambda k: float(confs[k].item()))
        conf = float(confs[i].item())
        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        zone = self._box_zone(x1, y1, x2, y2, W, H)
        poly = None
        if self._need_poly and masks is not None and i < len(masks.xy):
            poly = masks.xy[i]
        return {"conf": conf, "box": (x1, y1, x2, y2), "poly": poly, "zone": zone}

    def _is_high_avg(self) -> bool:
        """Luật 1 (vùng dưới): TRUNG BÌNH conf của alert_on_consec lần detect liên tiếp
        gần nhất ≥ alert_on_avg_conf. Cần đủ alert_on_consec mẫu (một lần miss xoá cửa
        sổ nên chuỗi phải liên tiếp) → một frame trồi/sụt lẻ không đủ tự bật alarm.
        """
        c = self.cfg
        if len(self._avg_hist) < c.alert_on_consec:
            return False
        return (sum(self._avg_hist) / len(self._avg_hist)) >= c.alert_on_avg_conf

    def _is_rising(self) -> bool:
        """Luật 2: conf THẤP nhưng TĂNG DẦN qua alert_rise_window lần detect liên tiếp.

        Có dung sai vì conf thực tế luôn nhiễu — đòi hỏi tăng nghiêm ngặt từng bước thì
        luật gần như không bao giờ kích hoạt. Thay vào đó xét 'dáng' tổng thể:
          • mọi mẫu phải > alert_rise_conf_min  (đủ đáng ngờ để xét)
          • mẫu cuối cao hơn mẫu đầu ≥ alert_rise_min_gain  (xu hướng tăng rõ rệt)
          • không mẫu nào tụt quá alert_rise_max_dip so với mẫu ngay trước (không nhấp
            nhô kiểu nhiễu — nhiễu thường tụt sâu rồi vọt lên lại)
        """
        c = self.cfg
        if not c.alert_rise_enabled or len(self._conf_hist) < c.alert_rise_window:
            return False
        h = list(self._conf_hist)
        if any(v <= c.alert_rise_conf_min for v in h):
            return False
        if h[-1] - h[0] < c.alert_rise_min_gain:
            return False
        return all(h[i] - h[i - 1] >= -c.alert_rise_max_dip for i in range(1, len(h)))

    def _fire_alert(self):
        """Bật alarm (cạnh lên): ghi mốc thời gian, mở cửa sổ auto-off mới, xoá chờ."""
        self._alert = True
        self._alert_on_time = time.time()
        self._hi_hist.clear()          # bắt đầu cửa sổ mới cho luật auto-off
        self._rise_pending_time = None
        self._rise_pending_hi = False

    def _update_alert(self, target: Optional[Dict]):
        prev = self._alert
        conf = target["conf"] if target is not None else 0.0

        zone = target["zone"] if target is not None else None
        if target is not None:
            self._miss = 0
            # Ghi conf của mỗi lần detect (mọi mức ≥ gate) vào cả hai cửa sổ: cửa sổ
            # trung bình (đường 1) và cửa sổ tăng dần (đường 2). Một lần MISS xoá sạch
            # cả hai để trung bình luôn tính trên chuỗi detect LIÊN TIẾP gần nhất.
            self._avg_hist.append(conf)
            self._conf_hist.append(conf)
            self._hi_hist.append(conf)     # rolling last-N cho luật auto-off (không xoá khi miss)
            self._count_hist.append(conf)  # rolling last-N cho TB conf đếm ca (không xoá khi miss)
            # Đếm chuỗi frame LIÊN TIẾP ở VÙNG TRÊN với conf > roi_top_conf.
            if zone == "top" and conf > self.cfg.roi_top_conf:
                self._top_hits += 1
            else:
                self._top_hits = 0
        else:
            self._miss += 1
            self._avg_hist.clear()
            self._conf_hist.clear()
            self._top_hits = 0             # MISS ngắt chuỗi vùng trên
            # MISS huỷ trạng thái chờ xác nhận của đường 2 (phải tăng dần lại từ đầu).
            self._rise_pending_time = None
            self._rise_pending_hi = False

        if not self._alert:
            c = self.cfg
            if zone == "top":
                # VÙNG TRÊN (10% nhạy): chỉ cần roi_top_frames frame LIÊN TIẾP có
                # conf > roi_top_conf → báo NGAY (không cần TB cao/tăng dần/xác nhận).
                if self._top_hits >= c.roi_top_frames:
                    self._fire_alert()
            elif self._is_high_avg():
                # Đường 1 (TB conf cao): bật ngay, không qua cửa xác nhận.
                self._fire_alert()
            elif target is not None:
                # Đường 2 (tăng dần): KHÔNG bật ngay — vào/duy trì cửa xác nhận.
                if self._rise_pending_time is None:
                    # Chưa chờ: bắt đầu chờ khi vừa đủ điều kiện tăng dần.
                    if self._is_rising():
                        self._rise_pending_time = time.time()
                        self._rise_pending_hi = (conf > c.alert_rise_confirm_conf)
                else:
                    # Đang chờ: theo dõi có lần nào conf vượt ngưỡng xác nhận chưa,
                    # rồi chỉ bật khi đã qua đủ thời gian chờ VÀ đã thấy conf cao.
                    if conf > c.alert_rise_confirm_conf:
                        self._rise_pending_hi = True
                    if (time.time() - self._rise_pending_time >= c.alert_rise_confirm_seconds
                            and self._rise_pending_hi):
                        self._fire_alert()

        if self._alert and (self._miss >= self.cfg.alert_off_miss or self._long_on_timeout()):
            self._alert = False
            self._alert_on_time = None
            self._avg_hist.clear()
            self._conf_hist.clear()
            self._hi_hist.clear()
            self._top_hits = 0
            self._rise_pending_time = None
            self._rise_pending_hi = False

        self._alert_just_on = (not prev) and self._alert
        # Chốt TB conf 10 lần detect gần nhất tại cạnh lên → dùng làm điều kiện đếm ca.
        if self._alert_just_on:
            self._alert_avg_conf = (sum(self._count_hist) / len(self._count_hist)
                                    if self._count_hist else 0.0)
        self._prev_alert = prev

    def _long_on_timeout(self) -> bool:
        """Auto-off: alarm đã ON ≥ alert_long_on_seconds mà trong alert_long_off_window lần
        detect gần nhất KHÔNG có conf nào > alert_long_off_conf → tắt (báo dai dẳng conf vừa).
        """
        c = self.cfg
        if c.alert_long_on_seconds <= 0 or self._alert_on_time is None:
            return False
        if (time.time() - self._alert_on_time) < c.alert_long_on_seconds:
            return False
        if len(self._hi_hist) < c.alert_long_off_window:
            return False
        return max(self._hi_hist) <= c.alert_long_off_conf

    def _push_result(self, frame: Optional[np.ndarray], target: Optional[Dict], infer_ms: float):
        # In headless mode, embed a snapshot copy in the result so the main loop
        # doesn't need to read from frame_q (eliminating queue competition).
        snap_frame = None
        if (self._alert_just_on
                and self.cfg.save_snapshot
                and not self.cfg.display
                and frame is not None):
            snap_frame = frame.copy()
        result = {
            "target": target,
            "alert": self._alert,
            "alert_just_on": self._alert_just_on,
            "alert_just_off": self._prev_alert and not self._alert,
            "alert_avg_conf": self._alert_avg_conf,   # TB conf 10 detect gần nhất tại cạnh lên
            "infer_ms": infer_ms,
            "snap_frame": snap_frame,
        }
        self._alert_just_on = False  # clear so skip frames don't inherit this edge
        # Display mode: purge stale result so the UI always shows the latest frame.
        # Headless mode: keep as FIFO — dropping would silently discard inference
        # results that the main loop hasn't read yet (skip frames push immediately
        # after inference, overwriting the result before the main loop wakes).
        if self.cfg.display:
            try:
                stale = self.out_q.get_nowait()
                # CRITICAL: preserve one-shot edge events from the purged result.
                # alert_just_on/off are True for a single frame only; if the main
                # loop hasn't consumed that result yet, dropping it silently loses
                # the edge → no ALERT_ON log, no snapshot, no event clip START.
                if stale.get("alert_just_on"):
                    result["alert_just_on"] = True
                    # Giữ luôn TB conf của cạnh lên bị purge để không mất điều kiện đếm ca.
                    result["alert_avg_conf"] = stale.get("alert_avg_conf", result.get("alert_avg_conf", 0.0))
                    if result.get("snap_frame") is None and stale.get("snap_frame") is not None:
                        result["snap_frame"] = stale["snap_frame"]
                if stale.get("alert_just_off"):
                    result["alert_just_off"] = True
            except queue.Empty:
                pass
        try:
            self.out_q.put_nowait(result)
        except queue.Full:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION SMOOTHER  (EMA + hold)
# ══════════════════════════════════════════════════════════════════════════════

class DetectionSmoother:
    """EMA-smoothed bounding box + hold box N frames after detection loss."""

    def __init__(self, alpha: float = 0.35, hold_max: int = 8):
        self.alpha = alpha
        self.hold_max = hold_max
        self._sx1 = self._sy1 = self._sx2 = self._sy2 = self._sconf = None
        self._hold = 0
        self._last_poly = None

    def update(self, raw: Optional[Dict]) -> Optional[Dict]:
        if raw is not None:
            x1, y1, x2, y2 = raw["box"]
            c = raw["conf"]
            if self._sx1 is None:
                self._sx1, self._sy1, self._sx2, self._sy2 = float(x1), float(y1), float(x2), float(y2)
                self._sconf = c
            else:
                a = self.alpha
                self._sx1 = a * x1 + (1 - a) * self._sx1
                self._sy1 = a * y1 + (1 - a) * self._sy1
                self._sx2 = a * x2 + (1 - a) * self._sx2
                self._sy2 = a * y2 + (1 - a) * self._sy2
                self._sconf = a * c + (1 - a) * self._sconf
            if raw["poly"] is not None:
                self._last_poly = raw["poly"]
            self._hold = self.hold_max
            return self._smooth()

        # No detection
        self._hold -= 1
        if self._hold > 0 and self._sx1 is not None:
            return self._smooth()

        # Reset
        self._sx1 = self._sy1 = self._sx2 = self._sy2 = self._sconf = None
        self._last_poly = None
        return None

    def _smooth(self) -> Dict:
        return {
            "conf": self._sconf,
            "box": (int(self._sx1), int(self._sy1), int(self._sx2), int(self._sy2)),
            "poly": self._last_poly,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT NOTIFIER  (Linux / Jetson – no winsound)
# ══════════════════════════════════════════════════════════════════════════════

class AlertNotifier:
    """
    Platform-agnostic alert for Ubuntu / Jetson.
    - Logs WARNING to console & file.
    - Optional terminal bell (ASCII BEL).
    - Optional Modbus RTU relay output (Waveshare USB-RS485 → 8DI/8DO relay):
      close the alert channel while a break is active, release it on clear.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._relay = None     # ModbusRelay handle (None when disabled/unavailable)
        self._hub = None       # RelayHub bọc quanh _relay — MỌI lệnh ghi/đọc đi qua đây
        # [multicam] Reason của camera trục trên CH1/CH2 (camera lược dùng
        # "alarm:<cam_id>" riêng trên CH1/CH3 nên không đụng nhau).
        self._alarm_reason = f"alarm:{getattr(cfg, 'camera_id', 'cam_main')}"
        if (cfg.relay_enabled or getattr(cfg, "relay_di_enabled", False)
                or getattr(cfg, "relay_status2_enabled", False)
                or getattr(cfg, "relay_follow_enabled", False)):
            self._setup_relay()

    def _setup_relay(self):
        """Open the Modbus RTU relay board over the USB-RS485 adapter and force
        the alert channel to its idle (released) state."""
        try:
            from modbus_relay import ModbusRelay
        except ImportError as exc:
            logger.error(f"Relay requested but modbus_relay/pyserial missing: {exc}")
            return
        relay = ModbusRelay(
            port=self.cfg.relay_port, baudrate=self.cfg.relay_baud,
            slave=self.cfg.relay_slave,
        )
        if not relay.open():
            return
        # [multicam] Bọc transport bằng RelayHub: mỗi kênh DO = OR các reason, chỉ
        # ghi Modbus khi giá trị đổi, dùng chung một lock RS485 cho cả ghi DO và
        # đọc DI (nút bấm). Hub là chủ DUY NHẤT của cổng trong lúc pipeline chạy.
        from relay_hub import RelayHub
        hub = RelayHub(relay, active_close=self.cfg.relay_active_close)
        self._relay = relay
        self._hub = hub
        # Idle state: alert OFF. Lần set_reason đầu luôn ghi xuống board (trạng thái
        # nội bộ khởi tạo là None) → tương đương 2 lệnh ghi idle của bản cũ.
        alarm_ch = getattr(self.cfg, "relay_alarm_light_channel", 2)
        hub.set_reason(self.cfg.relay_channel, self._alarm_reason, False)
        hub.set_reason(alarm_ch, self._alarm_reason, False)
        # [multicam] QUYẾT ĐỊNH ĐÃ CHỐT: camera LƯỢC báo đứt thì đèn xanh cũng
        # phải TẮT. Camera lược chỉ ghi CH1/CH3 (alarm_outputs=[1,3]), nên CH2
        # được khai báo là kênh "gương" của mọi reason "alarm:*" — tắt đèn xanh
        # cho MỌI camera mà không phải sửa alarm_outputs của comb.
        hub.set_mirror(alarm_ch, "alarm:")

        # CH7 đi theo CH1 có trễ: CH1 đóng (sợi đứt) liên tục đủ
        # relay_follow_delay_s giây → CH7 bắt đầu NHÁY theo chu kỳ
        # blink_on_s + blink_off_s (mặc định 0,5 + 0,5 = 1 s); CH1 mở → CH7 mở
        # ngay. Follower nhìn giá trị LOGIC của CH1 nên mọi nguồn kích CH1
        # (camera trục, camera lược, …) đều tính — không cần sửa chỗ gọi nào.
        if getattr(self.cfg, "relay_follow_enabled", False):
            follow_ch = getattr(self.cfg, "relay_follow_channel", 7)
            src_ch = getattr(self.cfg, "relay_follow_source_channel", 0) or self.cfg.relay_channel
            hub.set_delayed_follow(
                src_ch, follow_ch,
                getattr(self.cfg, "relay_follow_delay_s", 7.0),
                reason=f"follow:CH{src_ch}",
                blink_on_s=getattr(self.cfg, "relay_follow_blink_on_s", 0.5),
                blink_off_s=getattr(self.cfg, "relay_follow_blink_off_s", 0.5),
            )

        # Synchronize initial tower light state
        try:
            from state_manager import state_manager
            initial_state = "STOPPED" if state_manager.state.value in ("STOPPED", "PAUSED") else "RUNNING"
            self.set_tower_lights(initial_state)
            # Khởi động: chắc chắn chưa có reason PAUSE treo trên kênh đèn vàng.
            yellow_ch = getattr(self.cfg, "relay_pause_yellow_channel", 3)
            if yellow_ch:
                hub.set_reason(yellow_ch, "status:paused", False)
                logger.info(f"Đèn vàng khi PAUSE: CH{yellow_ch} "
                            f"(reason 'status:paused'); đèn tháp: CH3 "
                            f"(reason 'status:tower'); đèn xanh: CH{alarm_ch}")
        except Exception as exc:
            logger.warning(f"Failed to set initial tower lights state: {exc}")

    def _ensure_hub(self):
        """Trả về RelayHub đang dùng. Nếu đã có `_relay` nhưng chưa có hub (ví dụ
        test gán thẳng `notifier._relay = FakeRelay()` mà không qua _setup_relay)
        thì bọc lazily — nhờ vậy mọi lối gọi cũ vẫn chạy đúng."""
        if self._hub is None and self._relay is not None:
            from relay_hub import RelayHub
            self._hub = RelayHub(self._relay, active_close=self.cfg.relay_active_close)
        return self._hub

    def set_tower_lights(self, state: str) -> bool:
        """Set tower light on DO3 based on machine state.

        CH2 (đèn báo đứt) do alert điều khiển riêng, không liên quan hàm này.

        State Mapping:
        - RUNNING / ALARM: DO3=OFF
        - STOPPED / PAUSED: DO3=ON (Yellow Steady)
        - OFF: DO3=OFF
        """
        hub = self._ensure_hub()
        if hub is None:
            return False

        if state in ("RUNNING", "ALARM"):
            value = False
        elif state in ("STOPPED", "PAUSED"):
            value = True
        elif state == "OFF":
            value = False
        else:
            logger.warning(f"set_tower_lights: unknown state '{state}'")
            return False

        logger.info(f"Tower light transition to {state} (DO3 → {value})")
        # [multicam] CH3 = OR(status:tower, alarm:comb_*). "status:tower" giữ đúng
        # ngữ nghĩa cũ (nguồn duy nhất của bản cũ, ai ghi sau thắng) cho PAUSED/
        # STOPPED/watchdog; alarm của camera lược là reason RIÊNG nên vẫn giữ CH3
        # sáng khi lược đang báo đứt dù tower về RUNNING.
        return hub.set_reason(3, "status:tower", value)

    def _relay_set(self, alert_on: bool):
        hub = self._ensure_hub()
        if hub is None:
            return
        # [multicam] Camera trục bật/tắt reason của NÓ trên CH1 (đỏ) + CH2 (đèn báo).
        # Hub tự lo đảo cực theo relay_active_close và chỉ ghi khi kênh đổi giá trị.
        # Camera lược dùng reason khác trên CH1/CH3 → không ghi đè lẫn nhau.
        alarm_ch = getattr(self.cfg, "relay_alarm_light_channel", 2)
        hub.set_reason(self.cfg.relay_channel, self._alarm_reason, alert_on)
        hub.set_reason(alarm_ch, self._alarm_reason, alert_on)

    def on_alert(self, frame_index: int, conf: float):
        logger.warning(f"⚠  YARN BREAK DETECTED  frame={frame_index}  conf={conf:.3f}")
        self._beep()
        self._relay_set(True)

    def on_clear(self, frame_index: int):
        logger.info(f"✓  Alert cleared  frame={frame_index}")
        self._relay_set(False)

    def force_off(self, reason: str = "forced off"):
        """Immediately release the relay. Used by the thermal guard so the alert
        output can't stay asserted through a cooldown pause."""
        hub = self._ensure_hub()
        if hub is None:
            return
        # [multicam] Xoá MỌI reason alarm:* (cả camera trục lẫn camera lược) →
        # CH1 nhả chắc chắn, không còn nguồn nào giữ ngõ dừng máy.
        hub.clear_reason_prefix("alarm:")

    def release_stop_output(self, reason: str = ""):
        """Nhả CHỈ CH1 (ngõ còi / tín hiệu dừng máy). KHÔNG chạm CH2 (đèn báo).
        Dùng khi PAUSE: trạng thái tạm dừng cần DO1 tắt, DO2 BẬT, DO3 BẬT — trong
        đó CH2/CH3 do _sync_pause làm chủ (bật khi paused). Nếu nhả cả CH2 ở đây
        sẽ đua với _sync_pause và làm DO2 tắt oan."""
        hub = self._ensure_hub()
        if hub is None:
            return
        # [multicam] Xoá alarm:* CHỈ TRÊN CH1 (mọi camera) — giữ nguyên nguyên tắc
        # cũ là không chạm CH2 để đèn xanh không nháy trong lúc chờ _sync_pause.
        hub.clear_reason_prefix("alarm:", channel=self.cfg.relay_channel)

    def close(self):
        """Release the relay. Call from finally block in main()."""
        if self._relay is not None:
            try:
                hub = self._ensure_hub()
                # Set tower lights to OFF state (CH3 tắt — reason "status:tower")
                self.set_tower_lights("OFF")

                # When stopping: CH8 -> ON, CH7 -> OFF
                if getattr(self.cfg, "relay_status_enabled", False):
                    status_ch = getattr(self.cfg, "relay_status_channel", 8)
                    hub.set_status(status_ch, True)
                if getattr(self.cfg, "relay_status2_enabled", False):
                    status2_ch = getattr(self.cfg, "relay_status2_channel", 7)
                    hub.set_status(status2_ch, False)
                # Dừng luồng follower TRƯỚC khi nhả CH7, kẻo nó ghi lại ngay sau.
                if getattr(self.cfg, "relay_follow_enabled", False):
                    follow_ch = getattr(self.cfg, "relay_follow_channel", 7)
                    hub.stop_followers()
                    hub.clear_reason_prefix("follow:", channel=follow_ch)

                # Dừng hẳn: nhả CH1 (không kích máy) nhưng BẬT CH2 để TẮT đèn xanh
                # (đèn xanh đấu vào NC của DO2). CH3 đã tắt ở set_tower_lights("OFF").
                # Relay giữ (latch) trạng thái này sau khi đóng cổng serial.
                # [multicam] Xoá mọi alarm:* (CH1 nhả, kể cả camera lược), rồi bật
                # CH2 bằng reason "status:stopped" — khớp dòng STOPPED bảng sự thật.
                alarm_ch = getattr(self.cfg, "relay_alarm_light_channel", 2)
                hub.clear_reason_prefix("alarm:")
                hub.set_reason(alarm_ch, "status:paused", False)
                hub.set_reason(alarm_ch, "status:stopped", True)
                # Nhả reason PAUSE của đèn vàng, kẻo relay latch giữ CH3 sáng
                # mãi sau khi chương trình đã dừng.
                yellow_ch = getattr(self.cfg, "relay_pause_yellow_channel", 3)
                if yellow_ch:
                    hub.set_reason(yellow_ch, "status:paused", False)
                hub.close()
                logger.info("Relay serial closed.")
            except Exception as exc:
                logger.warning(f"Relay close error: {exc}")
            finally:
                self._relay = None
                self._hub = None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _beep():
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class EventLogger:
    def __init__(self, cfg: AppConfig,
                 on_alarm_trigger: Optional[callable] = None,
                 on_alarm_clear: Optional[callable] = None,
                 on_snapshot_saved: Optional[callable] = None):
        self.cfg = cfg
        self.on_alarm_trigger = on_alarm_trigger
        self.on_alarm_clear = on_alarm_clear
        self.on_snapshot_saved = on_snapshot_saved
        self._ev_log = logging.getLogger("YarnBreak.Events")
        self._ev_log.propagate = False

        if cfg.log_events:
            d = Path(cfg.log_dir)
            d.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(d / f"events_{ts}.csv")
            fh.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
            self._ev_log.addHandler(fh)
            self._ev_log.setLevel(logging.INFO)
            logger.info(f"Event log: {d / f'events_{ts}.csv'}")

        if cfg.save_snapshot:
            Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)

    def alert_on(self, frame_index: int, conf: float, avg_conf: float = 0.0):
        self._ev_log.info(f"ALERT_ON,{frame_index},{conf:.4f}")
        if self.on_alarm_trigger:
            self.on_alarm_trigger(frame_index, conf, avg_conf)

    def alert_off(self, frame_index: int):
        self._ev_log.info(f"ALERT_OFF,{frame_index},")
        if self.on_alarm_clear:
            self.on_alarm_clear(frame_index)

    def save_snapshot(self, frame: np.ndarray, frame_index: int):
        if not self.cfg.save_snapshot:
            return
        # Khung hình rỗng/None thì bỏ qua, tránh ghi file 0 byte.
        if frame is None or getattr(frame, "size", 0) == 0:
            logger.warning(f"Snapshot skipped (empty frame) at idx={frame_index}")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = Path(self.cfg.snapshot_dir) / f"yarnbreak_{ts}_{frame_index:08d}.jpg"
        # Bảo đảm thư mục tồn tại (phòng khi bị xoá lúc đang chạy).
        p.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]))
        # Chỉ báo "đã lưu" khi file thực sự tồn tại trên đĩa => DB không còn
        # trỏ tới ảnh không có thật (tránh 404 khi xem lại sau này).
        if not ok or not p.exists():
            logger.error(f"Snapshot write FAILED, skip DB update: {p}")
            return
        logger.info(f"Snapshot: {p}")
        if self.on_snapshot_saved:
            self.on_snapshot_saved(frame_index, str(p))

    def close(self):
        for h in self._ev_log.handlers[:]:
            h.close()
            self._ev_log.removeHandler(h)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO WRITER
# ══════════════════════════════════════════════════════════════════════════════

class VideoWriter:
    def __init__(self, cfg: AppConfig, fps: float, size: Tuple[int, int]):
        self._writer: Optional[cv2.VideoWriter] = None
        if cfg.save_video:
            if cfg.output_path:
                path = Path(cfg.output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out = Path(cfg.video_out_dir)
                out.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = out / f"output_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
            logger.info(f"Output video: {path}  size={size}  fps={fps:.1f}")

    def write(self, frame: np.ndarray):
        if self._writer:
            self._writer.write(frame)

    def release(self):
        if self._writer:
            self._writer.release()
            self._writer = None


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT CLIP RECORDER  (ghi video riêng cho mỗi lần alarm ON)
# ══════════════════════════════════════════════════════════════════════════════

class EventClipRecorder:
    """Ghi một clip .mp4 cho mỗi sự kiện alarm.

    - Luôn giữ một ring-buffer vài giây frame gần nhất (pre-roll) trong RAM ở
      độ phân giải đã hạ (event_buffer_scale) để tiết kiệm bộ nhớ trên Jetson.
    - Khi alarm bật (just_on): mở file mới, đổ toàn bộ pre-roll vào, rồi ghi
      tiếp từng frame trong suốt thời gian alarm.
    - Khi alarm tắt (just_off): ghi thêm post_seconds rồi đóng file.
    - Nếu alarm bật lại trong lúc post-roll: tiếp tục cùng một clip.

    Không đụng tới cấu trúc pipeline gốc — chỉ tiêu thụ frame ở main loop giống
    như VideoWriter.
    """

    def __init__(self, cfg: AppConfig, fps: float):
        self.cfg = cfg
        self.enabled = cfg.event_clip
        self.fps = max(fps, 1.0)
        self.scale = cfg.event_buffer_scale if cfg.event_buffer_scale > 0 else 1.0
        self._buf: deque = deque(maxlen=max(1, int(self.cfg.event_pre_seconds * self.fps)))
        self._post_frames = max(0, int(self.cfg.event_post_seconds * self.fps))
        self._max_frames = max(0, int(self.cfg.event_max_seconds * self.fps))  # 0 = không giới hạn
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._post_left: Optional[int] = None
        self._size: Optional[Tuple[int, int]] = None
        self._written = 0                       # số frame đã ghi vào file hiện tại (cho cắt khúc)
        self._seg = 0                           # chỉ số khúc trong cùng một sự kiện
        self._event_ts: Optional[str] = None    # timestamp dùng chung cho mọi khúc của sự kiện
        # Đảm bảo thời lượng tối thiểu khi conf cao lặp lại.
        self._min_hold_conf = float(getattr(cfg, "event_min_hold_conf", 0.91))
        self._min_hold_frames = int(getattr(cfg, "event_min_hold_frames", 3))
        self._min_hold_target = max(0, int(
            getattr(cfg, "event_min_hold_seconds", 0.0) * self.fps))  # số frame cho N giây
        self._hi_count = 0                      # số frame conf>ngưỡng trong sự kiện hiện tại
        self._min_hold_active = False           # đã đạt >min_hold_frames chưa
        self._since_hi: Optional[int] = None    # frame đã ghi kể từ frame conf-cao ĐẦU (không reset khi rollover)
        if self.enabled:
            Path(cfg.event_clip_dir).mkdir(parents=True, exist_ok=True)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * self.scale), int(h * self.scale)),
                          interpolation=cv2.INTER_AREA)

    def _next_path(self) -> Path:
        """Đường dẫn khúc kế tiếp. Khúc đầu: event_<ts>.mp4; các khúc sau cùng
        sự kiện: event_<ts>_partN.mp4 (N≥2)."""
        out = Path(self.cfg.event_clip_dir)
        self._seg += 1
        if self._seg == 1:
            self._event_ts = time.strftime("%Y%m%d_%H%M%S")
            return out / f"event_{self._event_ts}.mp4"
        return out / f"event_{self._event_ts}_part{self._seg}.mp4"

    def _start(self):
        # Kích thước clip = kích thước frame đã hạ (lấy từ buffer hoặc đặt sau).
        sample = self._buf[-1] if self._buf else None
        if sample is not None:
            h, w = sample.shape[:2]
            self._size = (w, h)
        if self._size is None:
            return  # chưa có frame nào để biết kích thước; sẽ thử lại frame sau
        self._seg = 0
        path = self._next_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, self.fps, self._size)
        self._recording = True
        self._post_left = None
        self._written = 0
        self._hi_count = 0
        self._min_hold_active = False
        self._since_hi = None
        # Đổ pre-roll vào clip
        for f in self._buf:
            self._writer.write(f)
            self._written += 1
        self._buf.clear()
        logger.info(f"Event clip START: {path}  size={self._size}  fps={self.fps:.1f}")

    def _roll(self):
        """Đóng khúc hiện tại và mở ngay khúc kế tiếp (alarm vẫn ON). Không đổ
        pre-roll — ghi liền mạch sang file mới."""
        if self._writer:
            self._writer.release()
        path = self._next_path()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, self.fps, self._size)
        self._written = 0
        logger.info(f"Event clip ROLLOVER → {path}")

    def _min_hold_satisfied(self) -> bool:
        """True nếu KHÔNG cần giữ thêm: hoặc chưa kích hoạt đảm bảo tối thiểu, hoặc
        đã ghi đủ event_min_hold_seconds tính từ frame conf-cao đầu tiên."""
        if not self._min_hold_active or self._min_hold_target <= 0:
            return True
        if self._since_hi is None:
            return True
        return self._since_hi >= self._min_hold_target

    def _stop(self):
        if self._writer:
            self._writer.release()
            logger.info("Event clip STOP (saved).")
        self._writer = None
        self._recording = False
        self._post_left = None

    def feed(self, frame: np.ndarray, just_on: bool, just_off: bool,
             alarm_on: Optional[bool] = None, conf: Optional[float] = None):
        if not self.enabled:
            return
        small = self._resize(frame)
        if self._size is None:
            h, w = small.shape[:2]
            self._size = (w, h)

        if just_on:
            if self._recording:
                self._post_left = None     # huỷ đóng đang chờ, giữ cùng clip
            else:
                self._buf.append(small)    # đảm bảo frame hiện tại nằm trong pre-roll
                self._start()
                return                     # _start đã ghi frame hiện tại qua buffer

        if self._recording:
            self._writer.write(small)
            self._written += 1

            # Theo dõi conf cao để đảm bảo thời lượng tối thiểu. _since_hi đếm mọi
            # frame đã ghi kể từ (và gồm) frame conf-cao ĐẦU TIÊN; không reset khi
            # rollover nên đo đúng 60s dù clip bị cắt thành nhiều part.
            if conf is not None and conf > self._min_hold_conf:
                self._hi_count += 1
                if self._hi_count > self._min_hold_frames:
                    self._min_hold_active = True
                self._since_hi = 1 if self._since_hi is None else self._since_hi + 1
            elif self._since_hi is not None:
                self._since_hi += 1

            # Bắt đầu post-roll khi: có cạnh OFF, HOẶC báo động đã tắt ở trạng thái
            # ổn định (alarm_on=False). Điều kiện thứ hai là chốt an toàn: nếu cạnh
            # just_off bị mất (result_q đầy → detector drop result), recorder vẫn
            # tự đóng file thay vì ghi vô hạn.
            close_now = just_off or (alarm_on is False and not just_on)
            if close_now and self._post_left is None:
                self._post_left = self._post_frames
            elif self._post_left is not None and self._post_left > 0:
                self._post_left -= 1

            # Post-roll đã hết → được phép dừng, TRỪ KHI đảm bảo tối thiểu chưa đủ
            # (giữ _post_left=0 để mỗi frame kế tiếp lại kiểm tra tới khi đủ 60s).
            if self._post_left is not None and self._post_left <= 0:
                if self._min_hold_satisfied():
                    self._stop()
                    return
                self._post_left = 0

            # Cắt khúc: chỉ khi đang ghi chủ động (chưa vào post-roll) và vượt ngưỡng.
            if (self._max_frames > 0 and self._written >= self._max_frames
                    and self._post_left is None):
                self._roll()
        else:
            self._buf.append(small)

    def release(self):
        self._stop()


# ══════════════════════════════════════════════════════════════════════════════
#  FPS METER
# ══════════════════════════════════════════════════════════════════════════════

class FPSMeter:
    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self._ema = 0.0
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        raw = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now
        self._ema = self.alpha * raw + (1 - self.alpha) * self._ema if self._ema > 0 else raw
        return self._ema


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD RENDERER
#  Carries all drawing logic from the original Windows code, fully refactored
#  into a class to eliminate global state.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  THERMAL GUARD  (CPU/GPU over-temperature pause)
# ══════════════════════════════════════════════════════════════════════════════

class ThermalGuard:
    """Background monitor that pauses the pipeline when the SoC runs too hot.

    Reads the CPU and GPU thermal zones from sysfs. When either reaches
    cfg.temp_limit_c, it sets `pause_evt` — the grabber and detector then idle,
    dropping GPU/CPU load so the chip cools — waits cfg.cooldown_s, clears the
    event so work resumes, and keeps monitoring (re-pausing if still hot).

    Structure/principle of the detection pipeline is untouched: this only gates
    when the existing threads are allowed to run.
    """

    THERMAL_DIR = "/sys/devices/virtual/thermal"

    def __init__(self, cfg: AppConfig, pause_evt: threading.Event, stop: threading.Event,
                 notifier: "Optional[AlertNotifier]" = None):
        self.cfg = cfg
        self.pause = pause_evt
        self.stop = stop
        self._notifier = notifier
        self._thread: Optional[threading.Thread] = None
        self._cpu_path = self._find_zone("cpu-thermal")
        self._gpu_path = self._find_zone("gpu-thermal")

    def _find_zone(self, ztype: str) -> Optional[str]:
        for zd in sorted(glob.glob(os.path.join(self.THERMAL_DIR, "thermal_zone*"))):
            try:
                with open(os.path.join(zd, "type")) as f:
                    if f.read().strip() == ztype:
                        return os.path.join(zd, "temp")
            except OSError:
                continue
        return None

    @staticmethod
    def _read_c(path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        try:
            with open(path) as f:
                s = f.read().strip()
            return int(s) / 1000.0 if s else None
        except (OSError, ValueError):
            return None

    def available(self) -> bool:
        return self._cpu_path is not None or self._gpu_path is not None

    def start(self):
        if not self.cfg.thermal_guard:
            return
        if not self.available():
            logger.warning("Thermal guard requested but no CPU/GPU thermal zone found — disabled.")
            return
        logger.info(
            f"Thermal guard ON: limit={self.cfg.temp_limit_c:.0f}°C  "
            f"cooldown={self.cfg.cooldown_s/60:.0f}min  check every {self.cfg.thermal_check_s:.0f}s"
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="ThermalGuard")
        self._thread.start()

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _interruptible_sleep(self, seconds: float):
        """Sleep in 1 s steps so a stop request ends the wait promptly."""
        end = time.perf_counter() + seconds
        while not self.stop.is_set() and time.perf_counter() < end:
            time.sleep(min(1.0, end - time.perf_counter()))

    def _run(self):
        limit = self.cfg.temp_limit_c
        while not self.stop.is_set():
            cpu = self._read_c(self._cpu_path)
            gpu = self._read_c(self._gpu_path)
            hot = max([t for t in (cpu, gpu) if t is not None], default=0.0)

            if hot >= limit:
                logger.warning(
                    f"🔥 OVER-TEMP: cpu={cpu}°C gpu={gpu}°C ≥ {limit:.0f}°C — "
                    f"pausing pipeline for {self.cfg.cooldown_s/60:.0f} min to cool down."
                )
                self.pause.set()
                # Force the alert output OFF so a latched siren/lamp can't stay
                # asserted through the cooldown. The detector clears its alert
                # latch on pause, so a still-present break re-triggers on resume.
                if self._notifier is not None:
                    # Nhả CH1; CH2/CH3 do _sync_pause bật theo trạng thái paused.
                    self._notifier.release_stop_output("thermal pause")
                self._interruptible_sleep(self.cfg.cooldown_s)
                self.pause.clear()
                if not self.stop.is_set():
                    c2, g2 = self._read_c(self._cpu_path), self._read_c(self._gpu_path)
                    logger.info(f"✓ Cooldown done — resuming. cpu={c2}°C gpu={g2}°C")
                continue

            self._interruptible_sleep(self.cfg.thermal_check_s)


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


class DashboardRenderer:
    # ── Colour palette (BGR) ──────────────────────────────────────────────────
    _STYLE: Dict = {
        "dashboard_bg":       (14,  16,  20),
        "panel_bg":           (24,  27,  32),
        "video_bg":           ( 8,  10,  13),
        "header_bg":          (28,  32,  38),
        "footer_bg":          (22,  25,  30),
        "divider_color":      (45,  50,  58),
        "padding": 2,  "gap": 4,
        "header_height": 30,  "footer_height": 30,
        "border_thickness": 1,
        "font":               cv2.FONT_HERSHEY_SIMPLEX,
        "title_scale":        0.52,  "title_thickness": 2,
        "footer_label_scale": 0.43,  "footer_value_scale": 0.48,
        "footer_thickness":   1,
        "status_pill_scale":  0.42,  "status_pill_thickness": 1,
        "det_label_min":      0.48,  "det_label_max": 2.8,
        "det_label_mul":      1.08,  "det_label_t_plus": 0,
        "label_pad_x":        11,    "label_pad_y": 7,
        "warn_title_mul":     1.12,  "warn_title_min": 0.62, "warn_title_max": 2.8,
        "warn_sub_mul":       1.00,  "warn_sub_min": 0.42,   "warn_sub_max": 1.8,
        "warn_icon_mul":      1.20,  "warn_t_plus": 1,
        "original_accent":    (220, 220, 220),
        "detected_accent":    (120, 255, 180),
        "normal_color":       (120, 255, 180),
        "warning_color":      ( 40, 120, 255),
        "warning_red":        ( 40,  40, 255),
        "box_color":          ( 90, 210, 255),
        "mask_color":         ( 90, 190, 255),
        "roi_color":          ( 80, 230,  80),   # xanh lá – vùng dưới ROI (logic cũ)
        "roi_top_color":      ( 60, 180, 255),   # cam – vùng trên ROI (15% nhạy)
        "label_bg":           ( 30,  34,  40),
        "label_text":         (245, 245, 245),
        "badge_bg":           ( 30,  75, 210),
        "badge_alpha":        0.76,
        "badge_text":         (255, 255, 255),
        "status_text":        (230, 235, 240),
        "muted_text":         (165, 172, 182),
        "fps_color":          (120, 255, 180),
    }

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.S = self._STYLE
        self._vscale_cache:   Dict = {}
        self._tscale_cache:   Dict = {}
        self._layout_cache:   Dict = {}
        self._resize_cache:   Dict = {}
        self._dash_buf:       Optional[np.ndarray] = None
        self._src_label = "IPCAM" if cfg.source == "ipcam" else "VIDEO"
        # Đa giác ROI (vùng trục) dạng tỉ lệ 0–1, để vẽ viền lên video khi bật.
        self._roi = _parse_roi(cfg.roi) if cfg.roi_enabled else None
        # Hai đa giác vùng trên / dưới (đường chia có thể nghiêng) để vẽ 2 màu — phải
        # dựng GIỐNG HỆT detector để hình vẽ khớp đúng vùng phân loại.
        _drop = cfg.roi_top_right_drop_px / _ROI_REF_HEIGHT_PX
        self._roi_top_poly, self._roi_bot_poly = (
            _roi_split_polys(self._roi, cfg.roi_top_fraction, _drop)
            if self._roi is not None else (None, None))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vscale(self, h: int, w: int) -> Tuple:
        """Returns (font_scale, small_scale, thickness, box_thickness). Cached."""
        key = (h, w)
        if key not in self._vscale_cache:
            base = min(h, w)
            # Overlay được vẽ trên frame gốc (5MP → base~2048) rồi mới thu nhỏ vào
            # panel, nên trần phải đủ lớn để chữ/box không bị teo. Tỉ lệ ~tuyến tính
            # theo độ phân giải: 720p→~0.82, 1080p→~1.2, 5MP→~2.3.
            scl = self.cfg.overlay_scale
            fs = _clamp(base / 880.0,  0.48, 2.6) * scl
            ss = _clamp(base / 1050.0, 0.40, 2.0) * scl
            t  = int(_clamp(round(base / 390.0 * scl), 1, 7))
            bt = int(_clamp(round(base / 330.0 * scl), 2, 9))
            self._vscale_cache[key] = (fs, ss, t, bt)
        return self._vscale_cache[key]

    def _fit(self, text: str, desired: float, thick: int, max_w: int, min_s: float = 0.30) -> float:
        """Shrink font scale until text fits max_w. Cached."""
        key = (text, round(desired, 4), thick, max_w, round(min_s, 4))
        if key in self._tscale_cache:
            return self._tscale_cache[key]
        s = desired
        font = self.S["font"]
        while s > min_s:
            (tw, _), _ = cv2.getTextSize(text, font, s, thick)
            if tw <= max_w:
                break
            s -= 0.03
        r = max(s, min_s)
        self._tscale_cache[key] = r
        return r

    def _alpha_rect(self, img, x1, y1, x2, y2, color, alpha=0.6, border=None, bt=1):
        H, W = img.shape[:2]
        x1 = int(_clamp(x1, 0, W - 1)); y1 = int(_clamp(y1, 0, H - 1))
        x2 = int(_clamp(x2, 0, W - 1)); y2 = int(_clamp(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        layer = np.full(roi.shape, color, dtype=np.uint8)
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)
        if border and bt > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), border, bt, cv2.LINE_AA)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _layout(self, dw: int, dh: int) -> Dict:
        key = (dw, dh)
        if key in self._layout_cache:
            return self._layout_cache[key]
        S = self.S
        p, g, hh, fh = S["padding"], S["gap"], S["header_height"], S["footer_height"]
        uw = dw - 2 * p - g
        uh = dh - 2 * p
        lw = uw // 2
        rw = uw - lw
        ph = uh
        left = {
            "panel":  (p, p, lw, ph),
            "header": (p, p, lw, hh),
            "video":  (p, p + hh, lw, ph - hh - fh),
            "footer": (p, p + ph - fh, lw, fh),
        }
        rx = p + lw + g
        right = {
            "panel":  (rx, p, rw, ph),
            "header": (rx, p, rw, hh),
            "video":  (rx, p + hh, rw, ph - hh - fh),
            "footer": (rx, p + ph - fh, rw, fh),
        }
        result = {"left": left, "right": right}
        self._layout_cache[key] = result
        return result

    def _fit_video(self, canvas, img, rect, bg):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        ih, iw = img.shape[:2]
        if ih == 0 or iw == 0 or w <= 0 or h <= 0:
            return
        key = (iw, ih, w, h)
        if key not in self._resize_cache:
            sc = min(w / iw, h / ih)
            nw = max(1, int(iw * sc))
            nh = max(1, int(ih * sc))
            self._resize_cache[key] = (nw, nh, (w - nw) // 2, (h - nh) // 2)
        nw, nh, dx, dy = self._resize_cache[key]
        rsz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[y + dy:y + dy + nh, x + dx:x + dx + nw] = rsz

    # ── Panel sub-components ──────────────────────────────────────────────────

    def _header(self, canvas, rect, title, accent, status=None, sc=None):
        S = self.S
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), S["header_bg"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 2), accent, -1)
        ty = y + int(h * 0.70)
        ts = self._fit(title, S["title_scale"], S["title_thickness"], w - 160, 0.38)
        cv2.putText(canvas, title, (x + 10, ty), S["font"], ts, accent, S["title_thickness"], cv2.LINE_AA)
        if status:
            txt = status.upper()
            ps = self._fit(txt, S["status_pill_scale"], S["status_pill_thickness"], 120, 0.32)
            (tw, _), _ = cv2.getTextSize(txt, S["font"], ps, S["status_pill_thickness"])
            pw, ph2 = tw + 22, max(20, h - 8)
            px1 = x + w - pw - 8
            py1 = y + (h - ph2) // 2
            px2, py2 = px1 + pw, py1 + ph2
            c = sc or S["normal_color"]
            self._alpha_rect(canvas, px1, py1, px2, py2, c, 0.18, c, 1)
            cv2.putText(canvas, txt, (px1 + 10, py1 + ph2 - 6), S["font"], ps, c, S["status_pill_thickness"], cv2.LINE_AA)

    def _footer(self, canvas, rect, items):
        S = self.S
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), S["footer_bg"], -1)
        cx = x + 10
        ty = y + int(h * 0.72)
        ls, vs, t = S["footer_label_scale"], S["footer_value_scale"], S["footer_thickness"]
        for lbl, val, col in items:
            lt, vt = f"{lbl}:", str(val)
            cv2.putText(canvas, lt, (cx, ty), S["font"], ls, S["muted_text"], t, cv2.LINE_AA)
            (lw2, _), _ = cv2.getTextSize(lt, S["font"], ls, t)
            cv2.putText(canvas, vt, (cx + lw2 + 5, ty), S["font"], vs, col, t, cv2.LINE_AA)
            (vw2, _), _ = cv2.getTextSize(vt, S["font"], vs, t)
            cx += lw2 + vw2 + 24
            if cx > x + w - 80:
                break

    def _panel(self, canvas, img, lo, title, accent, fitems, status=None, sc=None):
        S = self.S
        px, py, pw, ph = lo["panel"]
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), S["panel_bg"], -1)
        self._header(canvas, lo["header"], title, accent, status, sc)
        self._fit_video(canvas, img, lo["video"], S["video_bg"])
        self._footer(canvas, lo["footer"], fitems)
        cv2.rectangle(canvas, (px, py), (px + pw - 1, py + ph - 1), S["divider_color"], S["border_thickness"], cv2.LINE_AA)

    # ── Detection drawing ─────────────────────────────────────────────────────

    def draw_roi(self, img: np.ndarray):
        """Vẽ viền vùng trục (ROI) lên frame để căn chỉnh. Toạ độ ROI là tỉ lệ
        0–1 nên nhân với kích thước frame hiện tại. Không làm gì nếu ROI tắt.
        Nếu bật vùng trên nhạy → tô 2 màu: cam = vùng trên (nhạy), xanh = vùng dưới."""
        if not self._roi:
            return
        h, w = img.shape[:2]
        _, _, _, bt = self._vscale(h, w)
        c_bot = self.S["roi_color"]
        c_top = self.S["roi_top_color"]

        def _to_px(poly_frac):
            if poly_frac is None or len(poly_frac) < 3:
                return None
            arr = np.array([(x * w, y * h) for x, y in poly_frac], dtype=np.int32)
            return arr.reshape((-1, 1, 2))

        # Không chia vùng → vẽ như cũ, một màu.
        if self._roi_top_poly is None:
            bands = [(self._roi, c_bot)]
        else:
            bands = [(self._roi_bot_poly, c_bot), (self._roi_top_poly, c_top)]

        overlay = img.copy()
        for poly_frac, c in bands:
            pts = _to_px(poly_frac)
            if pts is not None:
                cv2.fillPoly(overlay, [pts], c)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
        for poly_frac, c in bands:
            pts = _to_px(poly_frac)
            if pts is None:
                continue
            cv2.polylines(img, [pts], True, c, max(2, bt + 1), cv2.LINE_AA)
        # Chỉ chấm đỉnh của đa giác ROI GỐC (không chấm điểm giao do cắt band).
        outer = _to_px(self._roi)
        if outer is not None:
            for p in outer.reshape(-1, 2):
                cv2.circle(img, (int(p[0]), int(p[1])), max(4, bt + 3), c_bot, -1, cv2.LINE_AA)

    def draw_detections(self, img: np.ndarray, target: Optional[Dict]) -> int:
        if target is None:
            return 0
        S = self.S
        h, w = img.shape[:2]
        fs, ss, t, bt = self._vscale(h, w)
        conf = target["conf"]
        x1, y1, x2, y2 = target["box"]
        poly = target["poly"]
        box_c = S["box_color"]
        msk_c = S["mask_color"]

        # ── Segmentation mask ─────────────────────────────────────────────────
        if not self.cfg.fast_draw and self.cfg.enable_mask:
            if poly is not None and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                arc = cv2.arcLength(pts, True)
                if arc > 0:
                    pts = cv2.approxPolyDP(pts, 0.004 * arc, True)
                bx, by, bw2, bh2 = cv2.boundingRect(pts)
                bx, by = max(0, bx), max(0, by)
                bx2, by2 = min(w, bx + bw2), min(h, by + bh2)
                if bx2 > bx and by2 > by:
                    roi = img[by:by2, bx:bx2]
                    roi_cp = roi.copy()
                    shifted = pts - np.array([[[bx, by]]], dtype=np.int32)
                    cv2.fillPoly(roi_cp, [shifted], msk_c)
                    cv2.addWeighted(roi_cp, self.cfg.mask_alpha, roi, 1 - self.cfg.mask_alpha, 0, roi)
                cv2.polylines(img, [pts], True, box_c, max(1, bt - 1), cv2.LINE_AA)
            else:
                bx, by = max(0, x1), max(0, y1)
                bx2, by2 = min(w, x2), min(h, y2)
                if bx2 > bx and by2 > by:
                    roi = img[by:by2, bx:bx2]
                    layer = np.full(roi.shape, msk_c, dtype=np.uint8)
                    cv2.addWeighted(layer, self.cfg.mask_alpha, roi, 1 - self.cfg.mask_alpha, 0, roi)

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), box_c, bt, cv2.LINE_AA)

        # ── Corner highlights ─────────────────────────────────────────────────
        cl = max(16, int(min(x2 - x1, y2 - y1) * 0.16))
        ct = bt + 1
        for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx1 = cl if px == x1 else -cl
            dy1 = cl if py == y1 else -cl
            cv2.line(img, (px, py), (px + dx1, py), box_c, ct, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy1), box_c, ct, cv2.LINE_AA)

        # ── Label ─────────────────────────────────────────────────────────────
        self._draw_label(img, f"{self.cfg.display_label} {conf:.2f}", x1, y1, box_c, h, w, t)
        return 1

    def _draw_label(self, img, text, x1, y1, color, H, W, t):
        S = self.S
        fs, _, _, _ = self._vscale(H, W)
        ds = _clamp(fs * S["det_label_mul"], S["det_label_min"], S["det_label_max"])
        t2 = max(1, t + S["det_label_t_plus"])
        px, py = S["label_pad_x"], S["label_pad_y"]
        fs2 = self._fit(text, ds, t2, int(W * 0.45), S["det_label_min"])
        (tw, th), bl = cv2.getTextSize(text, S["font"], fs2, t2)
        lw2, lh2 = tw + px * 2, th + py * 2 + bl
        lx1 = int(_clamp(x1, 8, W - lw2 - 8))
        ly1 = int(y1 - lh2 - 7)
        if ly1 < 8:
            ly1 = int(_clamp(y1 + 7, 8, H - lh2 - 8))
        lx2, ly2 = lx1 + lw2, ly1 + lh2
        self._alpha_rect(img, lx1, ly1, lx2, ly2, S["label_bg"], 0.88, color, 1)
        cv2.rectangle(img, (lx1, ly1), (lx1 + 5, ly2), color, -1)
        cv2.putText(img, text, (lx1 + px + 4, ly2 - py - bl), S["font"], fs2, S["label_text"], t2, cv2.LINE_AA)

    def draw_warning_badge(self, img: np.ndarray):
        S = self.S
        h, w = img.shape[:2]
        fs, ss, t, _ = self._vscale(h, w)
        badge_text, sub_text, icon = "YARN BREAK DETECTED", "CHECK YARN LINE", "!"
        bh = int(_clamp(h * 0.095, 54, 230))
        bw = int(_clamp(w * 0.42, 360, 1150))
        x1, y1, x2, y2 = 16, 16, 16 + bw, 16 + bh
        self._alpha_rect(img, x1, y1, x2, y2, S["badge_bg"], S["badge_alpha"], S["warning_color"], 1)
        isz = bh - 18
        ix1, iy1 = x1 + 10, y1 + 9
        ix2, iy2 = ix1 + isz, iy1 + isz
        self._alpha_rect(img, ix1, iy1, ix2, iy2, S["warning_red"], 0.82, (255, 255, 255), 1)
        iscale = _clamp(fs * S["warn_icon_mul"], 0.68, 3.2)
        wt = max(2, t + S["warn_t_plus"])
        cv2.putText(img, icon, (ix1 + int(isz * 0.36), iy2 - int(isz * 0.20)), S["font"], iscale, S["badge_text"], wt, cv2.LINE_AA)
        tx = ix2 + 14
        avw = x2 - tx - 14
        ts2 = self._fit(badge_text, _clamp(fs * S["warn_title_mul"], S["warn_title_min"], S["warn_title_max"]), wt, avw, S["warn_title_min"])
        ss2 = self._fit(sub_text,   _clamp(ss * S["warn_sub_mul"],   S["warn_sub_min"],   S["warn_sub_max"]),   1,  avw, S["warn_sub_min"])
        cv2.putText(img, badge_text, (tx, y1 + int(bh * 0.52)), S["font"], ts2, S["badge_text"], wt, cv2.LINE_AA)
        cv2.putText(img, sub_text,   (tx, y1 + int(bh * 0.82)), S["font"], ss2, (235, 238, 242), 1,  cv2.LINE_AA)

    def draw_warning_corners(self, img: np.ndarray):
        h, w = img.shape[:2]
        c = self.S["warning_color"]
        t  = int(_clamp(min(h, w) / 300, 2, 8))
        ln = int(_clamp(min(h, w) * 0.07, 28, 190))
        m  = int(_clamp(min(h, w) * 0.022, 10, 60))
        for px, py in [(m, m), (w - m, m), (m, h - m), (w - m, h - m)]:
            dx = ln if px == m else -ln
            dy = ln if py == m else -ln
            cv2.line(img, (px, py), (px + dx, py), c, t, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy), c, t, cv2.LINE_AA)

    # ── Full 2-panel dashboard ─────────────────────────────────────────────────

    def make_dashboard(
        self,
        original: np.ndarray,
        detected: np.ndarray,
        fps: float,
        detected_count: int,
        alert: bool,
        infer_ms: float,
    ) -> np.ndarray:
        S = self.S
        dw, dh = self.cfg.window_width, self.cfg.window_height

        if self.cfg.enable_dashboard_cache:
            if self._dash_buf is None or self._dash_buf.shape[:2] != (dh, dw):
                self._dash_buf = np.empty((dh, dw, 3), dtype=np.uint8)
            self._dash_buf[:] = S["dashboard_bg"]
            canvas = self._dash_buf
        else:
            canvas = np.full((dh, dw, 3), S["dashboard_bg"], dtype=np.uint8)

        lo = self._layout(dw, dh)
        status = "WARNING" if alert else "NORMAL"
        sc = S["warning_color"] if alert else S["normal_color"]

        det_rows = [
            ("STATUS", status, sc),
            ("FPS", f"{fps:.1f}", S["fps_color"]),
            ("CLS", detected_count, S["status_text"]),
            ("INF", f"{infer_ms:.0f}ms", S["muted_text"]),
        ]

        self._panel(
            canvas, original, lo["left"],
            "ORIGINAL FEED", S["original_accent"],
            [("MODE", "RAW", S["status_text"]), ("SRC", self._src_label, S["status_text"])],
            "LIVE", S["original_accent"],
        )
        self._panel(
            canvas, detected, lo["right"],
            "AI DETECTION", S["detected_accent"],
            det_rows,
            status, sc,
        )
        return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YarnBreak Detection — Jetson Orin Nano production",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",          default=None,           help="Path to config.yaml")
    p.add_argument("--source",          choices=["video","ipcam","mvs"], help="Input source")
    p.add_argument("--video",           help="Path to video file (.mp4/.avi/…)")
    p.add_argument("--rtsp",            help="RTSP URL  rtsp://user:pass@host/stream")
    p.add_argument("--codec",           choices=["h264", "h265"], help="RTSP codec hint for GStreamer pipeline")
    p.add_argument("--latency-ms",      type=int, dest="latency_ms", help="GStreamer rtspsrc latency (ms)")
    p.add_argument("--mvs-serial",      dest="mvs_serial", help="Hikrobot camera serial ('' = first found)")
    p.add_argument("--mvs-exposure-us", type=float, dest="mvs_exposure_us", help="Fixed exposure µs (0=auto)")
    p.add_argument("--mvs-gain",        type=float, dest="mvs_gain", help="Fixed gain dB (<0=camera default)")
    p.add_argument("--mvs-fps",         type=float, dest="mvs_fps", help="Cap camera frame rate (0=default)")
    p.add_argument("--model",           help="Path to .pt or .engine model")
    p.add_argument("--imgsz",           type=int, choices=[640, 800, 960], help="Inference size")
    p.add_argument("--conf",            type=float, help="Confidence threshold (0–1)")
    p.add_argument("--device",          help="Device: auto / 0 / cpu")
    p.add_argument("--half",            action="store_true", default=None, help="FP16 (CUDA)")
    p.add_argument("--detect-every",    type=int, dest="detect_every", help="Infer every N frames")
    p.add_argument("--display",         action="store_true", default=None, help="Show OpenCV window")
    p.add_argument("--headless",        action="store_true", default=None, help="No GUI")
    p.add_argument("--fast-draw",       action="store_true", default=None, dest="fast_draw")
    p.add_argument("--enable-mask",     action="store_true", default=None, dest="enable_mask")
    p.add_argument("--overlay-scale",   type=float, dest="overlay_scale", help="Phóng to chữ/box overlay (1.0=auto)")
    p.add_argument("--window-width",    type=int, dest="window_width")
    p.add_argument("--window-height",   type=int, dest="window_height")
    p.add_argument("--save-video",      action="store_true", default=None, dest="save_video")
    p.add_argument("--save-snapshot",   action="store_true", default=None, dest="save_snapshot")
    p.add_argument("--snapshot-dir",    dest="snapshot_dir")
    p.add_argument("--output-path",     dest="output_path",
                   help="Explicit output video file path (overrides --snapshot-dir/video_out_dir)")
    p.add_argument("--log-dir",         dest="log_dir")
    p.add_argument("--reconnect-delay", type=float, dest="reconnect_delay")
    p.add_argument("--no-pace",         action="store_false", default=None, dest="pace_video",
                   help="Disable source-FPS pacing for video files (default: paced)")
    p.add_argument("--loop",            action="store_true", default=None, dest="loop_video",
                   help="Loop video file (rewind on EOF)")
    p.add_argument("--max-infer-fps",   type=float, dest="max_infer_fps",
                   help="Cap detector rate (fps) to limit GPU duty cycle; 0=no cap")
    p.add_argument("--cpu-threads",     type=int, dest="cpu_threads",
                   help="Cap OpenCV + PyTorch CPU threads (thermal); 0=library default")
    p.add_argument("--no-thermal-guard", action="store_false", default=None, dest="thermal_guard",
                   help="Disable the CPU/GPU over-temperature pause")
    p.add_argument("--temp-limit",      type=float, dest="temp_limit_c",
                   help="Pause pipeline when CPU or GPU reaches this temp (°C)")
    p.add_argument("--cooldown",        type=float, dest="cooldown_s",
                   help="Pause duration in seconds when over-temp (e.g. 1200 = 20 min)")
    p.add_argument("--relay-di-enabled", action="store_true", default=None, dest="relay_di_enabled")
    p.add_argument("--relay-di-red", type=int, dest="relay_di_red")
    p.add_argument("--relay-di-green", type=int, dest="relay_di_green")
    p.add_argument("--relay-di-poll-interval", type=float, dest="relay_di_poll_interval")
    p.add_argument("--relay-di-control-mode", choices=["pause_resume", "start_stop"], dest="relay_di_control_mode")
    p.add_argument("--relay-status-enabled", action="store_true", default=None, dest="relay_status_enabled")
    p.add_argument("--relay-status-channel", type=int, dest="relay_status_channel")
    p.add_argument("--relay-status2-enabled", action="store_true", default=None, dest="relay_status2_enabled")
    p.add_argument("--relay-status2-channel", type=int, dest="relay_status2_channel")
    p.add_argument("--relay-follow-enabled", action="store_true", default=None, dest="relay_follow_enabled")
    p.add_argument("--relay-follow-channel", type=int, dest="relay_follow_channel")
    p.add_argument("--relay-follow-source-channel", type=int, dest="relay_follow_source_channel")
    p.add_argument("--relay-follow-delay-s", type=float, dest="relay_follow_delay_s")
    p.add_argument("--relay-follow-blink-on-s", type=float, dest="relay_follow_blink_on_s")
    p.add_argument("--relay-follow-blink-off-s", type=float, dest="relay_follow_blink_off_s")
    p.add_argument("--camera-timeout-shutdown-enabled", action="store_true", default=None, dest="camera_timeout_shutdown_enabled")
    p.add_argument("--camera-timeout-minutes", type=float, dest="camera_timeout_minutes")
    p.add_argument("--system-shutdown-delay-minutes", type=float, dest="system_shutdown_delay_minutes")
    p.add_argument("--enable-tower-watchdog", action="store_true", default=None, dest="enable_tower_watchdog")
    p.add_argument("--tower-watchdog-timeout-s", type=float, dest="tower_watchdog_timeout_s")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main(
    stop_evt: Optional[threading.Event] = None,
    user_pause_evt: Optional[threading.Event] = None,
    reset_alarm_evt: Optional[threading.Event] = None,
    on_camera_connected: Optional[callable] = None,
    on_model_loaded: Optional[callable] = None,
    on_fps_update: Optional[callable] = None,
    on_alarm_trigger: Optional[callable] = None,
    on_alarm_clear: Optional[callable] = None,
    on_snapshot_saved: Optional[callable] = None,
):
    parser = _build_parser()
    args = parser.parse_args()

    # Bootstrap logging (log dir may come from CLI)
    log_dir_early = args.log_dir or "./outputs/logs"
    _setup_logging(log_dir_early)

    cfg = _load_config(args.config, args)

    # ── CPU thread cap (thermal) ──────────────────────────────────────────────
    # Limit OpenCV + PyTorch CPU pools so preprocessing / NMS / frame copies
    # don't saturate every core. Heavy lifting stays on the GPU; this only
    # trims CPU temperature. Must run before the first inference / cv2 op.
    if cfg.cpu_threads and cfg.cpu_threads > 0:
        try:
            cv2.setNumThreads(cfg.cpu_threads)
            torch.set_num_threads(cfg.cpu_threads)
            logger.info(f"CPU thread cap: cv2={cfg.cpu_threads} torch={cfg.cpu_threads}")
        except Exception as exc:
            logger.warning(f"Could not set CPU thread cap: {exc}")

    logger.info("=" * 62)
    logger.info("  YarnBreak Detection  —  Jetson Production Build")
    logger.info(f"  source={cfg.source}  model={cfg.model}  imgsz={cfg.imgsz}")
    logger.info(f"  CUDA={torch.cuda.is_available()}  device={cfg.device}  half={cfg.half}")
    logger.info(f"  headless={cfg.headless}  save_video={cfg.save_video}  save_snap={cfg.save_snapshot}")
    logger.info("=" * 62)

    # ── Queues & events ───────────────────────────────────────────────────────
    # Khi chạy STANDALONE (chay.sh/chay_nen.sh) main() không được truyền event.
    # Mặc định về CHÍNH event của state_manager để nút đỏ/web (gọi
    # state_manager.pause_pipeline()/stop_pipeline()) điều khiển ĐÚNG detector —
    # nếu tạo Event local rời rạc thì pause_pipeline() sẽ không pause được detector
    # (sợi vẫn đứt → CH1 đóng lại). Xem memory di_pause_event_wiring.
    from state_manager import state_manager as _sm
    stop_evt = stop_evt or _sm.stop_event
    thermal_pause_evt = threading.Event()      # ← thermal guard holds this
    user_pause_evt = user_pause_evt or _sm.user_pause_event
    reset_alarm_evt = reset_alarm_evt or _sm.reset_alarm_event
    combined_pause_evt = threading.Event()

    # Pause synchronization thread
    def _sync_pause():
        last_paused = None
        while not stop_evt.is_set():
            paused = thermal_pause_evt.is_set() or user_pause_evt.is_set()
            if paused:
                combined_pause_evt.set()
            else:
                combined_pause_evt.clear()
            
            # Update status light CH8 and CH7 when pause state changes
            if paused != last_paused:
                try:
                    if notifier is not None and notifier._hub is not None:
                        if getattr(cfg, "relay_status_enabled", False):
                            status_ch = getattr(cfg, "relay_status_channel", 8)
                            notifier._hub.set_status(status_ch, paused)
                        # CH7 đã chuyển sang "đi theo CH1" (relay_follow_*) nên
                        # KHÔNG ghi trạng thái paused lên kênh đó nữa, tránh hai
                        # nguồn tranh nhau giữ CH7 khi CH1 đã mở.
                        status2_ch = getattr(cfg, "relay_status2_channel", 7)
                        follow_ch = getattr(cfg, "relay_follow_channel", 7)
                        if (getattr(cfg, "relay_status2_enabled", False)
                                and not (getattr(cfg, "relay_follow_enabled", False)
                                         and status2_ch == follow_ch)):
                            notifier._hub.set_status(status2_ch, paused)
                        # Đèn xanh đấu vào tiếp điểm NC của DO2 (relay_alarm_light_channel):
                        #   DO2 ON  → NC hở → đèn xanh TẮT (khi tạm dừng hoặc đang báo lỗi)
                        #   DO2 OFF → NC đóng → đèn xanh SÁNG (khi đang chạy bình thường)
                        # Giữ DO2 ON nếu vẫn còn báo lỗi để không bật lại đèn xanh khi resume.
                        from state_manager import state_manager
                        alarm_ch = getattr(cfg, "relay_alarm_light_channel", 2)
                        # [multicam] Giữ ĐÚNG công thức cũ CH2 = paused OR alarm_active,
                        # nhưng viết bằng 2 reason độc lập để camera lược không bị ảnh
                        # hưởng. Ghi lại cả reason alarm theo alarm_active (trị tuyệt đối)
                        # để khi RESUME mà hết lỗi thì reason alarm cũ không còn treo
                        # làm CH2 sáng oan (bản cũ ghi trực tiếp nên tự hết).
                        notifier._hub.set_reason(alarm_ch, notifier._alarm_reason,
                                                 state_manager.alarm_active)
                        notifier._hub.set_reason(alarm_ch, "status:paused", paused)
                        # Update tower lights (PAUSED/STOPPED maps to all ON, RUNNING to all OFF)
                        notifier.set_tower_lights("PAUSED" if paused else "RUNNING")
                        # Đèn VÀNG khi tạm dừng — reason RIÊNG của PAUSE nên
                        # watchdog đèn tháp (ghi "status:tower") không tắt được.
                        # Đặt SAU set_tower_lights để lượt ghi cuối là của PAUSE.
                        yellow_ch = getattr(cfg, "relay_pause_yellow_channel", 3)
                        if yellow_ch:
                            notifier._hub.set_reason(yellow_ch, "status:paused", paused)
                    last_paused = paused
                except (NameError, AttributeError):
                    pass # notifier is not yet initialized in the main thread
                
            time.sleep(0.05)

    pause_sync_thread = threading.Thread(target=_sync_pause, daemon=True, name="PauseSync")
    pause_sync_thread.start()

    frame_q:    queue.Queue     = queue.Queue(maxsize=1)  # ← detector only
    display_q:  queue.Queue     = queue.Queue(maxsize=1)  # ← display/save only
    result_q:   queue.Queue     = queue.Queue(maxsize=4)

    # ── Build components ──────────────────────────────────────────────────────
    try:
        # display_q luôn được cấp: ngoài display/save/event_clip, luồng xem camera
        # trực tiếp (live view) cũng cần frame. _push() không chặn (drop-old) nên
        # chi phí khi không ai dùng là không đáng kể.
        grabber  = make_grabber(cfg, frame_q, stop_evt,
                                display_q,
                                combined_pause_evt, on_camera_connected)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    notifier    = AlertNotifier(cfg)
    detector    = YoloDetector(cfg, frame_q, result_q, stop_evt, notifier, combined_pause_evt, reset_alarm_evt, on_model_loaded)
    thermal     = ThermalGuard(cfg, thermal_pause_evt, stop_evt, notifier)
    smoother    = DetectionSmoother(cfg.ema_alpha, cfg.hold_frames)
    ev_logger   = EventLogger(cfg, on_alarm_trigger, on_alarm_clear, on_snapshot_saved)
    fps_meter   = FPSMeter()
    # Renderer cần cho: dashboard/imshow (display) VÀ vẽ overlay lên frame ghi ra
    # (save_video / event_clip). Nhờ vậy clip chạy headless-web vẫn có bounding box
    # + badge cảnh báo. Cửa sổ/dashboard vẫn chỉ bật khi cfg.display (xem bên dưới).
    # Renderer luôn được tạo: cần cho display/save/event_clip VÀ cho live view
    # (vẽ ROI + bounding box lên frame phát cho web). Bản thân đối tượng nhẹ; chỉ
    # tốn CPU khi vòng lặp thực sự vẽ (đường frame-driven).
    renderer    = DashboardRenderer(cfg)
    vid_writer: Optional[VideoWriter] = None
    clip_recorder: Optional[EventClipRecorder] = None
    comb_pipeline = None       # [multicam] gán thật sau khi các thread đã chạy

    # ── Load model (main thread – before spawning worker) ─────────────────────
    try:
        detector.load()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Start threads ─────────────────────────────────────────────────────────
    grabber.start()
    detector.start()
    thermal.start()

    # ── [multicam] Pipeline camera lược (YarnBreakComb) ───────────────────────
    # Chạy chung tiến trình nhưng ĐỘC LẬP: lỗi của nó không được làm dừng camera
    # trục, nên bọc toàn bộ trong try/except và không bao giờ set stop_evt.
    _comb_cfg = getattr(cfg, "comb", None) or {}
    if _comb_cfg.get("enabled"):
        try:
            from comb_pipeline import CombPipeline
            comb_pipeline = CombPipeline(
                _comb_cfg,
                sdk_path=cfg.mvs_sdk_path,
                hub=notifier._ensure_hub(),
                pause_event=combined_pause_evt,
            )
            if not comb_pipeline.start():
                comb_pipeline = None
        except Exception as exc:
            logger.error(f"Không khởi động được pipeline camera lược: {exc}",
                         exc_info=True)
            comb_pipeline = None

    # ── Modbus DI Buttons Watcher Thread ──────────────────────────────────────
    if getattr(cfg, "relay_di_enabled", False) and notifier._hub is not None:
        def _di_poll_loop():
            last_red = False
            last_green = False
            red_idx = cfg.relay_di_red - 1
            green_idx = cfg.relay_di_green - 1
            poll_interval = cfg.relay_di_poll_interval
            red_press_start = 0.0
            
            logger.info(f"Modbus DI watcher started in main (mode: {cfg.relay_di_control_mode})")
            logger.info(f"DI pause wiring: user_pause_evt is state_manager.user_pause_event = "
                        f"{user_pause_evt is _sm.user_pause_event}")
            while not stop_evt.is_set():
                try:
                    # [multicam] Đọc DI qua hub để dùng CHUNG lock RS485 với các
                    # lệnh ghi DO của mọi camera (tránh đụng khung Modbus).
                    states = notifier._hub.read_inputs()
                    if states is not None:
                        # 1. Check RED button (STOP/PAUSE)
                        if 0 <= red_idx < len(states):
                            red_state = states[red_idx]
                            if red_state:
                                if not last_red:
                                    logger.warning("🔴 Modbus RED button pressed! Pausing pipeline...")
                                    from state_manager import state_manager
                                    state_manager.pause_pipeline()
                                    # Chạm THẲNG event mà detector/grabber dùng để pause NGAY
                                    # (zero-lag, không đợi _sync_pause 50ms).
                                    user_pause_evt.set()
                                    combined_pause_evt.set()
                                    # Nhả CH1 (còi/dừng máy) NGAY. KHÔNG chạm CH2 — _sync_pause
                                    # sẽ bật CH2/CH3 theo trạng thái paused → DO1 tắt, DO2 bật,
                                    # DO3 bật.
                                    notifier.release_stop_output("red button pause")
                                    from alarm_logger import alarm_logger
                                    alarm_logger.log_audit("Physical Button", "PAUSE", "SUCCESS", "Tạm dừng nhận diện (Nút Đỏ)")
                                    red_press_start = time.perf_counter()
                                elif cfg.relay_di_control_mode == "pause_resume":
                                    # Check hold duration for shutdown in pause_resume mode
                                    held_time = time.perf_counter() - red_press_start
                                    if held_time >= 5.0:
                                        logger.warning("🔴 Modbus RED button held for 5s! Stopping program completely...")
                                        from alarm_logger import alarm_logger
                                        alarm_logger.log_audit("Physical Button", "STOP", "SUCCESS", "Tắt hoàn toàn chương trình (Nút Đỏ giữ 5s)")
                                        notifier.force_off("red button stop")  # tắt alarm ngay, không đợi finally
                                        stop_evt.set()
                                else:
                                    # In start_stop mode, stop immediately
                                    logger.warning("🔴 Modbus RED button pressed! Stopping program...")
                                    from alarm_logger import alarm_logger
                                    alarm_logger.log_audit("Physical Button", "STOP", "SUCCESS", "Tắt chương trình (Nút Đỏ)")
                                    notifier.force_off("red button stop")  # tắt alarm ngay, không đợi finally
                                    stop_evt.set()
                            last_red = red_state
                            
                        # 2. Check GREEN button (START/RESUME)
                        if 0 <= green_idx < len(states):
                            green_state = states[green_idx]
                            if green_state and not last_green:
                                logger.info("🟢 Modbus GREEN button pressed!")
                                if cfg.relay_di_control_mode == "pause_resume":
                                    from state_manager import state_manager
                                    state_manager.resume_pipeline()
                                    # Clear THẲNG event local để detector chạy lại (standalone
                                    # không nối state_manager.user_pause_event với local).
                                    user_pause_evt.clear()
                                    combined_pause_evt.clear()
                                    from alarm_logger import alarm_logger
                                    alarm_logger.log_audit("Physical Button", "RESUME", "SUCCESS", "Tiếp tục nhận diện (Nút Xanh)")
                            last_green = green_state
                except Exception as e:
                    logger.error(f"Error in Modbus DI poll: {e}")
                time.sleep(poll_interval)
                
        di_thread = threading.Thread(target=_di_poll_loop, daemon=True, name="ModbusDIWatcher")
        di_thread.start()

    # ── Tower Light Heartbeat Watchdog Thread ────────────────────────────────
    if getattr(cfg, "enable_tower_watchdog", True) and notifier._hub is not None:
        def _watchdog_loop():
            logger.info("Tower light heartbeat watchdog thread started.")
            timeout = getattr(cfg, "tower_watchdog_timeout_s", 5.0)
            override_active = False
            
            # Allow time for model loading and initial grabber connection
            time.sleep(3.0)
            
            from state_manager import state_manager
            state_manager.last_heartbeat = time.monotonic()
            
            while not stop_evt.is_set():
                from state_manager import SystemState
                current_state = state_manager.state
                
                # We only check heartbeat timeout if program is in RUNNING or ALARM state
                if current_state in (SystemState.RUNNING, SystemState.ALARM):
                    elapsed = time.monotonic() - state_manager.last_heartbeat
                    if elapsed > timeout:
                        if not override_active:
                            logger.critical(
                                f"❌ Tower Light Watchdog: Detection heartbeat lost for {elapsed:.1f}s "
                                f"(>{timeout}s)! Overriding tower lights to STOPPED/Safety state (Yellow Steady)."
                            )
                            notifier.set_tower_lights("STOPPED")
                            override_active = True
                    else:
                        if override_active:
                            logger.info("✓ Tower Light Watchdog: Heartbeat recovered. Restoring RUNNING state (Green Flashing).")
                            notifier.set_tower_lights("RUNNING")
                            override_active = False
                else:
                    override_active = False
                    
                time.sleep(0.5)
                
        watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True, name="TowerWatchdog")
        watchdog_thread.start()

    # ── Camera Connection Monitor Thread ──────────────────────────────────────
    if getattr(cfg, "camera_timeout_shutdown_enabled", False):
        def _camera_monitor_loop():
            import subprocess
            from state_manager import state_manager
            
            timeout_seconds = getattr(cfg, "camera_timeout_minutes", 15.0) * 60.0
            shutdown_delay = getattr(cfg, "system_shutdown_delay_minutes", 5.0)
            
            logger.info(f"Camera connection monitor started. Timeout: {timeout_seconds/60:.1f} min, Shutdown delay: {shutdown_delay:.1f} min")
            last_connected_time = time.perf_counter()
            
            while not stop_evt.is_set():
                from state_manager import SystemState
                if state_manager.state not in (SystemState.RUNNING, SystemState.ALARM):
                    last_connected_time = time.perf_counter()
                elif state_manager.camera_connected:
                    last_connected_time = time.perf_counter()
                else:
                    elapsed = time.perf_counter() - last_connected_time
                    if elapsed >= timeout_seconds:
                        logger.critical(f"❌ Camera connection lost or failed for {timeout_seconds/60:.1f} minutes! Initiating shutdown procedure...")
                        
                        # 1. Write audit log
                        try:
                            from alarm_logger import alarm_logger
                            alarm_logger.log_audit(
                                "System",
                                "SHUTDOWN_TIMEOUT",
                                "SUCCESS",
                                f"Mất kết nối camera {timeout_seconds/60:.1f}p -> Shutdown Jetson sau {shutdown_delay:.1f}p"
                            )
                        except Exception as ex:
                            logger.error(f"Failed to log audit for camera timeout shutdown: {ex}")
                        
                        # 2. Schedule system shutdown in delay minutes
                        try:
                            delay_str = f"+{int(shutdown_delay)}"
                            subprocess.Popen(["sudo", "shutdown", "-h", delay_str, "Camera connection failed for 15 minutes. System shutting down."])
                            logger.info(f"Scheduled system shutdown in {shutdown_delay:.1f} minutes.")
                        except Exception as ex:
                            logger.error(f"Failed to run shutdown command: {ex}")
                            
                        # 3. Stop the yarnbreak service cleanly so systemd doesn't restart it
                        try:
                            subprocess.Popen(["sudo", "systemctl", "stop", "yarnbreak"])
                            logger.info("Triggered stop command for yarnbreak service.")
                        except Exception as ex:
                            logger.error(f"Failed to stop yarnbreak service: {ex}")
                        
                        stop_evt.set()
                        break
                        
                time.sleep(5.0)
                
        monitor_thread = threading.Thread(target=_camera_monitor_loop, daemon=True, name="CameraMonitor")
        monitor_thread.start()

    logger.info("Threads running. Press  Q / ESC  or  Ctrl-C  to quit.")

    # ── Window ────────────────────────────────────────────────────────────────
    if cfg.display and renderer:
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, cfg.window_width, cfg.window_height)

    # ── State ─────────────────────────────────────────────────────────────────
    frame_idx        = 0
    cached_target:   Optional[Dict] = None
    last_target:     Optional[Dict] = None   # box gần nhất, giữ để vẽ suốt khi alarm ON
    alert_latched    = False
    prev_alert       = False
    infer_ms         = 0.0
    ui_ms            = 0.0

    # When headless and not saving video / event clips, the main loop must NOT
    # consume from frame_q — that queue belongs exclusively to the detector
    # thread.  Results (including an embedded snapshot when alert fires) come via
    # result_q.  event_clip needs frames (via display_q, not frame_q) to record,
    # so it forces the frame-driven path even in headless mode.
    # Tĩnh: các tính năng luôn cần frame. Live view thêm nhu cầu ĐỘNG (chỉ khi có
    # người đang xem) nên _needs_frame được tính lại mỗi vòng lặp bên dưới.
    _static_needs_frame = cfg.save_video or cfg.display or cfg.event_clip

    # Wall-clock FPS counter for headless mode.
    # fps_meter uses EMA which gives misleading values when skip frames and
    # inference frames arrive in tight bursts; a windowed rate is more accurate.
    _hd_fps_t0 = time.perf_counter()
    _hd_fps_n  = 0

    try:
        while not stop_evt.is_set():
            fps = fps_meter.tick()

            # Live view bật đường frame-driven ngay cả khi headless thuần: cần frame
            # để phát cho web. Khi không ai xem, quay lại đường result_q nhẹ nhàng.
            _needs_frame = _static_needs_frame or live_stream.active()

            if not _needs_frame:
                # ── Headless path: detector owns frame_q entirely ─────────────
                # Block on result_q; no frame competition with the detector.
                try:
                    res = result_q.get(timeout=1.0)
                except queue.Empty:
                    if stop_evt.is_set():
                        break
                    continue

                frame_idx   += 1
                _hd_fps_n   += 1
                prev_alert    = alert_latched
                alert_latched = res["alert"]
                # Keep the last actual inference time; skip frames carry 0.
                if res["infer_ms"] > 0:
                    infer_ms = res["infer_ms"]
                cached_target = smoother.update(res["target"])

                if res["alert_just_on"]:
                    conf = res["target"]["conf"] if res["target"] else 0.0
                    ev_logger.alert_on(frame_idx, conf, res.get("alert_avg_conf", 0.0))
                    snap = res.get("snap_frame")
                    if snap is not None:
                        ev_logger.save_snapshot(snap, frame_idx)

                if res.get("alert_just_off"):
                    ev_logger.alert_off(frame_idx)

                if frame_idx % cfg.profile_every == 0:
                    _elapsed = time.perf_counter() - _hd_fps_t0
                    fps_log  = _hd_fps_n / max(_elapsed, 1e-6)
                    _hd_fps_t0 = time.perf_counter()
                    _hd_fps_n  = 0
                    if cfg.show_profile:
                        logger.info(
                            f"frame={frame_idx:>6d}  fps={fps_log:5.1f}  "
                            f"infer={infer_ms:5.1f}ms  "
                            f"alert={'ON ' if alert_latched else 'off'}"
                        )
                    if on_fps_update:
                        on_fps_update(fps_log, infer_ms)
                continue

            # ── Display / video-save path ─────────────────────────────────────
            # Block until a frame arrives (1 s timeout to check stop_evt)
            try:
                frame = display_q.get(timeout=1.0)
            except queue.Empty:
                if stop_evt.is_set():
                    break
                continue

            if frame is None:
                break

            frame_idx += 1

            # Init video writer + event clip recorder on first frame
            if vid_writer is None and cfg.save_video:
                h, w = frame.shape[:2]
                vid_writer = VideoWriter(cfg, grabber.source_fps or fps, (w, h))
            if clip_recorder is None and cfg.event_clip:
                clip_recorder = EventClipRecorder(cfg, grabber.source_fps or fps)

            # ── Consume latest inference result ───────────────────────────────
            alert_just_on_evt  = False
            alert_just_off_evt = False
            cur_conf = None            # conf của detect MỚI frame này (None nếu không có result mới)
            try:
                res = result_q.get_nowait()
                prev_alert    = alert_latched
                alert_latched = res["alert"]
                if res["infer_ms"] > 0:
                    infer_ms = res["infer_ms"]
                cur_conf = res["target"]["conf"] if res["target"] else None
                cached_target = smoother.update(res["target"])

                if res["alert_just_on"]:
                    conf = res["target"]["conf"] if res["target"] else 0.0
                    # notifier.on_alert() already fired from detector thread (zero-delay)
                    ev_logger.alert_on(frame_idx, conf, res.get("alert_avg_conf", 0.0))
                    alert_just_on_evt = True

                if res.get("alert_just_off"):
                    # notifier.on_clear() already fired from detector thread
                    ev_logger.alert_off(frame_idx)
                    alert_just_off_evt = True

            except queue.Empty:
                pass  # keep using cached_target until new result arrives

            # ── Draw on output copy ───────────────────────────────────────────
            t_ui  = time.perf_counter()
            output = frame.copy()

            if renderer:
                # Bám box theo trạng thái alarm. Vì detect_every đẩy target=None ở các
                # frame skip, smoother (hold_frames) hết hạn sớm hơn nhiều so với alert
                # hysteresis (alert_off_miss) → badge "YARN BREAK DETECTED" hiện nhưng
                # mất bounding box. Giữ box gần nhất và vẽ lại suốt khi alarm còn ON để
                # box và badge luôn xuất hiện cùng nhau.
                if cached_target is not None:
                    last_target = cached_target
                draw_target = cached_target
                if draw_target is None and alert_latched and last_target is not None:
                    draw_target = last_target
                if not alert_latched:
                    last_target = None
                if cfg.display:
                    renderer.draw_roi(output)   # viền vùng trục để căn chỉnh (chỉ khi xem desktop)
                det_count = renderer.draw_detections(output, draw_target)
                if alert_latched:
                    renderer.draw_warning_badge(output)
                    renderer.draw_warning_corners(output)
            else:
                det_count = 1 if cached_target is not None else 0

            # ── Save output frame ─────────────────────────────────────────────
            if vid_writer:
                vid_writer.write(output)

            # ── Snapshot (ảnh ĐÃ vẽ bounding box) + event clip (video GỐC sạch) ──
            # Snapshot giữ bounding box để biết AI phát hiện ở đâu; nhưng video sự
            # kiện ghi từ frame GỐC (không box/badge/góc đỏ) theo yêu cầu.
            if alert_just_on_evt:
                ev_logger.save_snapshot(output, frame_idx)
            if clip_recorder:
                clip_recorder.feed(frame, alert_just_on_evt, alert_just_off_evt,
                                   alert_latched, cur_conf)

            # ── Live view (web) — phát frame đã vẽ box + viền ROI để căn camera ──
            # want_frame() đã gồm rate-limit nên chỉ copy/vẽ khi thực sự cần publish.
            if renderer and live_stream.want_frame():
                live_img = output.copy()      # giữ nguyên frame ghi/clip (không burn ROI)
                renderer.draw_roi(live_img)   # viền vùng trục để căn chỉnh vị trí camera
                live_stream.publish(live_img)

            # ── Display dashboard ─────────────────────────────────────────────
            if cfg.display and renderer:
                dash = renderer.make_dashboard(
                    original=frame, detected=output,
                    fps=fps, detected_count=det_count,
                    alert=alert_latched, infer_ms=infer_ms,
                )
                cv2.imshow(cfg.window_name, dash)

            ui_ms = (time.perf_counter() - t_ui) * 1000.0

            # ── Profile log ───────────────────────────────────────────────────
            if frame_idx % cfg.profile_every == 0:
                if cfg.show_profile:
                    logger.info(
                        f"frame={frame_idx:>6d}  fps={fps:5.1f}  "
                        f"infer={infer_ms:5.1f}ms  ui={ui_ms:4.1f}ms  "
                        f"alert={'ON ' if alert_latched else 'off'}"
                    )
                if on_fps_update:
                    on_fps_update(fps, infer_ms)

            # ── Key handler ───────────────────────────────────────────────────
            if cfg.display:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):          # ESC or Q
                    logger.info("Quit key pressed.")
                    stop_evt.set()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt – shutting down.")
        stop_evt.set()

    finally:
        logger.info("Cleaning up…")
        stop_evt.set()

        # Signal detector worker to exit
        try:
            frame_q.put_nowait(None)
        except queue.Full:
            pass

        grabber.join(timeout=3.0)
        detector.join(timeout=3.0)
        thermal.join(timeout=3.0)
        try:
            pause_sync_thread.join(timeout=1.0)
        except Exception:
            pass

        # [multicam] Dừng camera lược TRƯỚC khi đóng relay: stop() nhả mọi reason
        # alarm:<comb_*> nên CH1/CH3 không bị giữ lại sau khi tắt.
        if comb_pipeline is not None:
            try:
                comb_pipeline.stop()
            except Exception as exc:
                logger.warning(f"Lỗi khi dừng pipeline camera lược: {exc}")

        if vid_writer:
            vid_writer.release()
        if clip_recorder:
            clip_recorder.release()
        ev_logger.close()
        notifier.close()

        if cfg.display:
            cv2.destroyAllWindows()

        logger.info("Done.")


if __name__ == "__main__":
    main()
