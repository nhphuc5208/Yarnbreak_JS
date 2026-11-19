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
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch

from yarnbreak.live_view import live_stream
from yarnbreak.alerts import AlertNotifier
from yarnbreak.config import _load_config
from yarnbreak.dashboard import DashboardRenderer
from yarnbreak.detector import DetectionSmoother, YoloDetector
from yarnbreak.grabbers import make_grabber
from yarnbreak.monitoring import FPSMeter, ThermalGuard
from yarnbreak.recording import EventClipRecorder, EventLogger, VideoWriter

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
    from yarnbreak.state_manager import state_manager as _sm
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
                        from yarnbreak.state_manager import state_manager
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
            from yarnbreak.comb_pipeline import CombPipeline
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
                                    from yarnbreak.state_manager import state_manager
                                    state_manager.pause_pipeline()
                                    # Chạm THẲNG event mà detector/grabber dùng để pause NGAY
                                    # (zero-lag, không đợi _sync_pause 50ms).
                                    user_pause_evt.set()
                                    combined_pause_evt.set()
                                    # Nhả CH1 (còi/dừng máy) NGAY. KHÔNG chạm CH2 — _sync_pause
                                    # sẽ bật CH2/CH3 theo trạng thái paused → DO1 tắt, DO2 bật,
                                    # DO3 bật.
                                    notifier.release_stop_output("red button pause")
                                    from yarnbreak.alarm_logger import alarm_logger
                                    alarm_logger.log_audit("Physical Button", "PAUSE", "SUCCESS", "Tạm dừng nhận diện (Nút Đỏ)")
                                    red_press_start = time.perf_counter()
                                elif cfg.relay_di_control_mode == "pause_resume":
                                    # Check hold duration for shutdown in pause_resume mode
                                    held_time = time.perf_counter() - red_press_start
                                    if held_time >= 5.0:
                                        logger.warning("🔴 Modbus RED button held for 5s! Stopping program completely...")
                                        from yarnbreak.alarm_logger import alarm_logger
                                        alarm_logger.log_audit("Physical Button", "STOP", "SUCCESS", "Tắt hoàn toàn chương trình (Nút Đỏ giữ 5s)")
                                        notifier.force_off("red button stop")  # tắt alarm ngay, không đợi finally
                                        stop_evt.set()
                                else:
                                    # In start_stop mode, stop immediately
                                    logger.warning("🔴 Modbus RED button pressed! Stopping program...")
                                    from yarnbreak.alarm_logger import alarm_logger
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
                                    from yarnbreak.state_manager import state_manager
                                    state_manager.resume_pipeline()
                                    # Clear THẲNG event local để detector chạy lại (standalone
                                    # không nối state_manager.user_pause_event với local).
                                    user_pause_evt.clear()
                                    combined_pause_evt.clear()
                                    from yarnbreak.alarm_logger import alarm_logger
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
            
            from yarnbreak.state_manager import state_manager
            state_manager.last_heartbeat = time.monotonic()
            
            while not stop_evt.is_set():
                from yarnbreak.state_manager import SystemState
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
            from yarnbreak.state_manager import state_manager
            
            timeout_seconds = getattr(cfg, "camera_timeout_minutes", 15.0) * 60.0
            shutdown_delay = getattr(cfg, "system_shutdown_delay_minutes", 5.0)
            
            logger.info(f"Camera connection monitor started. Timeout: {timeout_seconds/60:.1f} min, Shutdown delay: {shutdown_delay:.1f} min")
            last_connected_time = time.perf_counter()
            
            while not stop_evt.is_set():
                from yarnbreak.state_manager import SystemState
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
                            from yarnbreak.alarm_logger import alarm_logger
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
