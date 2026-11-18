"""Application configuration (AppConfig dataclass + YAML/CLI loader).

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml

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
