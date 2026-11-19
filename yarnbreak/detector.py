"""YOLO / TensorRT detector thread and detection smoothing.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

from yarnbreak.alerts import AlertNotifier
from yarnbreak.config import AppConfig
from yarnbreak.roi import _ROI_REF_HEIGHT_PX, _parse_roi, _point_in_poly, _roi_split_polys, _roi_top_split_y

logger = logging.getLogger("YarnBreak")


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
                    from yarnbreak.state_manager import state_manager
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
