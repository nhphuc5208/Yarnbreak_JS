#!/usr/bin/env python3
"""
comb_pipeline.py — Pipeline camera "lược" (YarnBreakComb) chạy CHUNG tiến trình
với camera trục, nhưng ĐỘC LẬP hoàn toàn về lỗi.

Thiết kế:
  • Mỗi camera lược = 1 luồng đọc frame (CombCameraStream) giữ frame mới nhất.
  • MỘT luồng detector duy nhất nạp 1 engine và xử lý LUÂN PHIÊN (round-robin)
    các camera. Lý do: TensorRT KHÔNG thread-safe — engine phải warmup VÀ
    inference trong CÙNG một luồng (xem memory tensorrt_thread_context_hang).
  • Camera lược báo đứt → bật các kênh trong comb.alarm_outputs ([1, 3] = ĐỎ +
    VÀNG) qua RelayHub bằng reason riêng "alarm:<cam_id>" nên không ghi đè
    camera trục (CH1 là OR của mọi camera).
  • Lỗi/crash của pipeline lược KHÔNG được làm dừng camera trục: mọi vòng lặp
    đều bọc try/except, luồng chết thì tự khởi động lại sau vài giây. Tuyệt đối
    không gọi stop_event.set().
  • Camera lược mất kết nối → chỉ log + hiện trên web, KHÔNG tắt Jetson,
    KHÔNG bật CH3.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("YarnBreak.Comb")


# ══════════════════════════════════════════════════════════════════════════════
#  LUẬT CẢNH BÁO (port nguyên ý nghĩa từ comb_forcamera.py:_inference_worker)
# ══════════════════════════════════════════════════════════════════════════════
class CombAlertEngine:
    """Bật/tắt cảnh báo theo conf của box cao nhất mỗi lần detect.

    BẬT khi một trong hai điều kiện:
      A) Tăng dần: cửa sổ đủ `trend_window` lần liên tiếp, lần ĐẦU > `trend_min_conf`
         và dãy TĂNG NGHIÊM NGẶT.
      B) Conf cao: `high_window` lần gần nhất đều > `high_conf`.
    TẮT khi `off_miss` lần liên tiếp không thấy gì.
    Miss 1 lần là XOÁ cửa sổ (giống bản tham chiếu).
    """

    def __init__(self, trend_window: int = 5, trend_min_conf: float = 0.6,
                 high_window: int = 3, high_conf: float = 0.8,
                 off_miss: int = 5):
        self.trend_window = int(trend_window)
        self.trend_min_conf = float(trend_min_conf)
        self.high_window = int(high_window)
        self.high_conf = float(high_conf)
        self.off_miss = int(off_miss)
        self._cap = max(self.trend_window, self.high_window)
        self._win: List[float] = []
        self._miss = 0
        self.alert = False

    @staticmethod
    def _strictly_increasing(seq: List[float]) -> bool:
        return all(b > a for a, b in zip(seq, seq[1:]))

    def update(self, top_conf: Optional[float]) -> tuple:
        """Nạp conf cao nhất của lần detect (None = không thấy gì).
        Trả về (alert, just_on, just_off)."""
        prev = self.alert
        if top_conf is not None:
            self._miss = 0
            self._win.append(float(top_conf))
            if len(self._win) > self._cap:
                self._win.pop(0)
        else:
            self._miss += 1
            self._win.clear()

        if not self.alert:
            cond_trend = (
                len(self._win) == self.trend_window
                and self._win[0] > self.trend_min_conf
                and self._strictly_increasing(self._win)
            )
            cond_high = (
                len(self._win) >= self.high_window
                and all(c > self.high_conf for c in self._win[-self.high_window:])
            )
            if cond_trend or cond_high:
                self.alert = True

        if self.alert and self._miss >= self.off_miss:
            self.alert = False

        return self.alert, (not prev and self.alert), (prev and not self.alert)

    def reset(self):
        """Xoá sạch trạng thái — dùng khi PAUSE để khi RESUME phát hiện lại từ đầu."""
        self._win.clear()
        self._miss = 0
        self.alert = False


# ══════════════════════════════════════════════════════════════════════════════
#  GHI VIDEO SỰ KIỆN CHO CAMERA LƯỢC
# ══════════════════════════════════════════════════════════════════════════════
class CombEventClip:
    """Ghi MỘT file .mp4 cho mỗi lần camera lược báo đứt sợi.

    Lưu vào folder RIÊNG của camera lược (comb.event_clip_dir, mặc định
    ./outputs/events_comb/<cam_id>/) — không trộn với clip của camera trục
    (./outputs/events/).

    Thời lượng:
      • pre-roll: vài giây TRƯỚC khi báo (giữ sẵn trong RAM theo MỐC THỜI GIAN,
        không theo số frame, vì nhịp frame camera lược thay đổi theo tải GPU);
      • tối thiểu `event_min_seconds` giây tính TỪ LÚC BÁO — kể cả khi cảnh báo
        tắt sớm hoặc người vận hành bấm TẮT CẢNH BÁO (yêu cầu: tối thiểu 30s);
      • còn báo thì ghi tiếp, hết báo thì ghi thêm `event_post_seconds` giây;
      • `event_max_seconds` chặn trên để một file không phình vô hạn.

    fps ghi vào file là nhịp frame THỰC ĐO được (camera lược chỉ tới ~4 fps vì
    max_infer_fps + detect_every), nhờ vậy clip phát lại đúng thời lượng thật.
    """

    FPS_MIN = 1.0
    FPS_MAX = 30.0
    FPS_DEFAULT = 4.0

    def __init__(self, cam_id: str, cfg: dict):
        self.cam_id = cam_id
        self.enabled = bool(cfg.get("event_clip", True))
        base_dir = cfg.get("event_clip_dir") or "./outputs/events_comb"
        self.dir = os.path.join(base_dir, cam_id)
        scale = float(cfg.get("event_buffer_scale") or 0)
        self.scale = scale if scale > 0 else 1.0
        self.pre_s = max(0.0, float(cfg.get("event_pre_seconds", 3.0)))
        self.min_s = max(0.0, float(cfg.get("event_min_seconds", 30.0)))
        self.post_s = max(0.0, float(cfg.get("event_post_seconds", 3.0)))
        self.max_s = max(self.min_s, float(cfg.get("event_max_seconds", 120.0)))
        self.fallback_fps = float(cfg.get("event_clip_fps", 0) or 0)

        self._times: deque = deque(maxlen=40)   # mốc thời gian các frame gần nhất
        self._buf: deque = deque()              # pre-roll: (t, frame đã hạ cỡ)
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[str] = None
        self._size: Optional[tuple] = None
        self._fps = 0.0
        self._frames = 0
        self._t_start = 0.0
        self._t_alarm_off: Optional[float] = None

        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
            except Exception as exc:
                logger.warning(f"[{cam_id}] không tạo được {self.dir}: {exc} "
                               "→ tắt ghi clip.")
                self.enabled = False

    # ── Thông tin ────────────────────────────────────────────────────────────
    @property
    def current_path(self) -> Optional[str]:
        """Đường dẫn file đang ghi (None nếu không ghi)."""
        return self._path if self._writer is not None else None

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return frame.copy()
        h, w = frame.shape[:2]
        return cv2.resize(frame, (max(2, int(w * self.scale)), max(2, int(h * self.scale))),
                          interpolation=cv2.INTER_AREA)

    def _measure_fps(self) -> float:
        """Nhịp frame thực tế đo từ các mốc thời gian gần nhất."""
        if len(self._times) >= 5:
            span = self._times[-1] - self._times[0]
            if span > 0:
                fps = (len(self._times) - 1) / span
                return min(max(fps, self.FPS_MIN), self.FPS_MAX)
        if self.fallback_fps > 0:
            return min(max(self.fallback_fps, self.FPS_MIN), self.FPS_MAX)
        return self.FPS_DEFAULT

    def _stamp(self, img: np.ndarray):
        """Đóng dấu thời gian lên góc dưới-trái để clip dùng được làm bằng chứng."""
        txt = time.strftime("%Y-%m-%d %H:%M:%S")
        h = img.shape[0]
        fs = max(0.5, 0.9 * self.scale)
        cv2.putText(img, txt, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), max(3, int(round(4 * self.scale))), cv2.LINE_AA)
        cv2.putText(img, txt, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), max(1, int(round(2 * self.scale))), cv2.LINE_AA)

    def _start(self, now: float) -> bool:
        sample = self._buf[-1][1] if self._buf else None
        if sample is None:
            return False
        h, w = sample.shape[:2]
        self._size = (w, h)
        fps = self._measure_fps()
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.dir, f"{self.cam_id}_{ts}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self._size)
        if not writer.isOpened():
            logger.warning(f"[{self.cam_id}] không mở được VideoWriter {path} "
                           "→ bỏ qua clip lần này.")
            return False

        self._writer = writer
        self._path = path
        self._fps = fps
        self._frames = 0
        self._t_start = now
        self._t_alarm_off = None
        for _, f in self._buf:                  # đổ pre-roll vào clip
            writer.write(f)
            self._frames += 1
        self._buf.clear()
        logger.info(f"[{self.cam_id}] Clip START: {path} size={self._size} "
                    f"fps={fps:.1f} (pre-roll {self._frames} frame)")
        return True

    def close(self, why: str = ""):
        """Đóng file đang ghi (dùng khi PAUSE / STOP / lỗi pipeline)."""
        if self._writer is None:
            self._buf.clear()
            return
        try:
            self._writer.release()
        except Exception:
            pass
        dur = self._frames / self._fps if self._fps > 0 else 0.0
        logger.info(f"[{self.cam_id}] Clip STOP ({why}): {self._path} — "
                    f"{self._frames} frame ≈ {dur:.1f}s")
        self._writer = None
        self._frames = 0
        self._t_alarm_off = None
        self._buf.clear()

    # ── Nạp frame ────────────────────────────────────────────────────────────
    def feed(self, frame: np.ndarray, alarm_on: bool, just_on: bool,
             draw: Optional[Callable[[np.ndarray, float], np.ndarray]] = None):
        """Nạp một frame vào clip.

        frame    : frame GỐC (chưa vẽ gì) — hàm tự hạ cỡ theo event_buffer_scale
        alarm_on : camera này ĐANG báo đứt hay không
        just_on  : frame này là lúc CHUYỂN sang báo đứt (mở clip mới)
        draw     : hàm vẽ chú thích lên frame ĐÃ HẠ CỠ, nhận (img, scale)
        """
        if not self.enabled or frame is None:
            return
        try:
            now = time.monotonic()
            self._times.append(now)
            small = self._resize(frame)
            if draw is not None:
                small = draw(small, self.scale)
            self._stamp(small)

            # Chưa ghi: giữ pre-roll theo số giây, mở clip khi vừa có báo đứt
            if self._writer is None:
                self._buf.append((now, small))
                while self._buf and (now - self._buf[0][0]) > self.pre_s:
                    self._buf.popleft()
                if just_on:
                    self._start(now)
                return

            self._writer.write(small)
            self._frames += 1

            # Mốc "hết báo" để tính post-roll; báo lại thì xoá mốc, giữ cùng clip
            if alarm_on or just_on:
                self._t_alarm_off = None
            elif self._t_alarm_off is None:
                self._t_alarm_off = now

            elapsed = now - self._t_start
            if elapsed >= self.max_s:
                self.close(f"chạm giới hạn {self.max_s:.0f}s")
                return
            if (self._t_alarm_off is not None
                    and elapsed >= self.min_s
                    and (now - self._t_alarm_off) >= self.post_s):
                self.close(f"đủ tối thiểu {self.min_s:.0f}s")
        except Exception as exc:
            # Ghi clip KHÔNG được phép làm chết pipeline lược.
            logger.warning(f"[{self.cam_id}] ghi clip lỗi: {exc}")
            self.close("lỗi")


# ══════════════════════════════════════════════════════════════════════════════
#  GRABBER CAMERA LƯỢC (Hikrobot MVS)
# ══════════════════════════════════════════════════════════════════════════════
class CombCameraStream:
    """Đọc frame từ 1 camera Hikrobot, giữ frame MỚI NHẤT (LatestImagesOnly).

    Dùng mvs_sdk (SDK chung, init 1 lần/tiến trình, chọn theo serial bắt buộc).
    Tự kết nối lại khi mất tín hiệu; KHÔNG bao giờ gọi Finalize (phá camera khác).
    """

    GET_TIMEOUT_MS = 1000
    RECONNECT_MAX_FAIL = 15
    RECONNECT_DELAY_S = 3.0
    FPS_WINDOW_S = 2.0          # cửa sổ đo nhịp frame rồi đẩy lên web

    def __init__(self, cam_cfg: dict, sdk_path: str):
        self.cfg = cam_cfg
        self.cam_id = cam_cfg.get("id", "comb")
        self.name = cam_cfg.get("name", self.cam_id)
        self.serial = str(cam_cfg.get("mvs_serial") or "").strip()
        self._sdk_path = sdk_path

        self.mvs = None
        self.cam = None
        self._dev_info = None
        self.connected = False
        self.fail_streak = 0
        self.reconnect_count = 0

        # Nhịp frame THỰC TẾ của camera (web hiển thị). Đo ngay ở luồng đọc nên
        # vẫn cập nhật kể cả khi detector đang bận hoặc đang PAUSE.
        self.fps = 0.0
        self._fps_t0 = 0.0
        self._fps_n = 0

        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Buffer tái sử dụng cho convert Bayer→BGR (tránh cấp phát 35MB/frame)
        self._cvt_buf = None
        self._cvt_buf_size = 0
        self._cvt_view = None

    # ── Vòng đời ──────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_guard, daemon=True,
                                        name=f"CombGrab-{self.cam_id}")
        self._thread.start()

    def stop(self):
        self._running = False
        self._close_device()
        self._report_status(connected=False, fps=0.0)

    def read(self):
        """Trả về (frame, seq) — frame là array riêng, an toàn dùng ở luồng khác."""
        with self._lock:
            return self._frame, self._seq

    # ── Mở / cấu hình camera ────────────────────────────────────────────────
    def _open_device(self):
        import mvs_sdk
        self.mvs = mvs_sdk.load_sdk(self._sdk_path)
        mvs_sdk.sdk_init()
        mvs = self.mvs

        self._dev_info = mvs_sdk.find_by_serial(self.serial)

        cam = mvs.MvCamera()
        ret = cam.MV_CC_CreateHandle(self._dev_info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle lỗi 0x{ret:x}")
        ret = cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"MV_CC_OpenDevice lỗi 0x{ret:x} — thường do app khác (MVS Client) "
                "đang giữ camera."
            )
        self.cam = cam
        self._apply_settings()

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            self._close_device()
            raise RuntimeError(f"MV_CC_StartGrabbing lỗi 0x{ret:x}")

        self.connected = True
        self.fail_streak = 0
        self._fps_t0 = time.monotonic()
        self._fps_n = 0
        self._log_info()
        self._report_status(connected=True, fps=0.0)

    def _nic_mtu_for_camera(self) -> Optional[int]:
        """MTU của card mạng dùng để tới camera này.

        GigE Vision gửi payload bằng UDP: packet size LỚN HƠN MTU sẽ bị rớt gói
        hàng loạt → frame rách (ảnh đen kèm vệt nhiễu ngang) chứ KHÔNG báo lỗi.
        Vì vậy phải tự giới hạn packet size theo MTU thật của cổng mạng."""
        try:
            import subprocess
            mvs = self.mvs
            if self._dev_info.nTLayerType != mvs.MV_GIGE_DEVICE:
                return None
            ip_int = self._dev_info.SpecialInfo.stGigEInfo.nCurrentIp
            ip = "%d.%d.%d.%d" % ((ip_int >> 24) & 0xFF, (ip_int >> 16) & 0xFF,
                                  (ip_int >> 8) & 0xFF, ip_int & 0xFF)
            out = subprocess.run(["ip", "route", "get", ip], capture_output=True,
                                 text=True, timeout=3).stdout.split()
            dev = out[out.index("dev") + 1]
            with open(f"/sys/class/net/{dev}/mtu") as f:
                mtu = int(f.read().strip())
            logger.info(f"[{self.cam_id}] camera {ip} đi qua {dev} (MTU {mtu})")
            return mtu
        except Exception as exc:
            logger.warning(f"[{self.cam_id}] không đọc được MTU: {exc}")
            return None

    def _apply_settings(self):
        mvs, cam, c = self.mvs, self.cam, self.cfg

        def warn(label, ret):
            if ret != 0:
                logger.warning(f"[{self.cam_id}] set {label} lỗi 0x{ret:x}")

        # Packet size: lấy nMax của node rồi clamp theo nMin/nInc (KHÔNG tin
        # GetOptimalPacketSize — trên link-local nó hay trả 1500 dù card bật jumbo).
        if self._dev_info.nTLayerType in (mvs.MV_GIGE_DEVICE,
                                          getattr(mvs, "MV_GENTL_GIGE_DEVICE", -1)):
            want = int(c.get("packet_size") or 0)
            if want <= 0:
                try:
                    want = int(cam.MV_CC_GetOptimalPacketSize())
                except Exception:
                    want = 1500
            # Giới hạn theo MTU thật của cổng mạng TRƯỚC khi clamp theo node.
            mtu = self._nic_mtu_for_camera()
            if mtu and want > mtu:
                logger.warning(
                    f"[{self.cam_id}] packet_size {want} > MTU {mtu} của cổng mạng "
                    f"→ hạ về {mtu}. (Muốn dùng jumbo: đặt MTU 9000 cho cổng đó.)"
                )
                want = mtu

            iv = mvs.MVCC_INTVALUE_EX()
            mvs.memset(mvs.byref(iv), 0, mvs.sizeof(iv))
            if cam.MV_CC_GetIntValueEx("GevSCPSPacketSize", iv) == 0:
                inc = int(iv.nInc) or 1
                want = min(int(want), int(iv.nMax))
                want = max(want, int(iv.nMin))
                want = (want // inc) * inc
            if want > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", want)
                if ret != 0:
                    logger.warning(f"[{self.cam_id}] packet size {want} lỗi 0x{ret:x} "
                                   "→ tụt về 1500")
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
                elif want <= 1500:
                    logger.warning(f"[{self.cam_id}] packet size chỉ {want} — kiểm tra "
                                   "MTU/Jumbo Frame của cổng mạng.")

        warn("TriggerMode=Off", cam.MV_CC_SetEnumValue("TriggerMode", mvs.MV_TRIGGER_MODE_OFF))

        binning = int(c.get("binning") or 1)
        if binning > 1:
            warn("BinningHorizontal", cam.MV_CC_SetIntValue("BinningHorizontal", binning))
            warn("BinningVertical", cam.MV_CC_SetIntValue("BinningVertical", binning))

        # ROI cắt NGAY TRÊN CẢM BIẾN: phải thu nhỏ Width/Height TRƯỚC rồi mới set
        # Offset, ngược lại camera từ chối vì offset+width vượt kích thước hiện tại.
        hw_roi = c.get("hw_roi")
        if hw_roi:
            ox, oy, rw, rh = hw_roi
            warn("OffsetX=0", cam.MV_CC_SetIntValue("OffsetX", 0))
            warn("OffsetY=0", cam.MV_CC_SetIntValue("OffsetY", 0))
            warn("Width", cam.MV_CC_SetIntValue("Width", int(rw)))
            warn("Height", cam.MV_CC_SetIntValue("Height", int(rh)))
            warn("OffsetX", cam.MV_CC_SetIntValue("OffsetX", int(ox)))
            warn("OffsetY", cam.MV_CC_SetIntValue("OffsetY", int(oy)))

        if c.get("exposure_us"):
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            warn("ExposureTime", cam.MV_CC_SetFloatValue("ExposureTime",
                                                         float(c["exposure_us"])))
        if c.get("gain") is not None:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            warn("Gain", cam.MV_CC_SetFloatValue("Gain", float(c["gain"])))

        if c.get("fps"):
            warn("AcquisitionFrameRateEnable",
                 cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True))
            warn("AcquisitionFrameRate",
                 cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(c["fps"])))

        # Chỉ giữ frame mới nhất (tương đương CAP_PROP_BUFFERSIZE=1)
        cam.MV_CC_SetImageNodeNum(2)
        cam.MV_CC_SetGrabStrategy(mvs.MV_GrabStrategy_LatestImagesOnly)

    def _log_info(self):
        mvs, cam = self.mvs, self.cam

        def rd_int(key):
            v = mvs.MVCC_INTVALUE()
            mvs.memset(mvs.byref(v), 0, mvs.sizeof(v))
            return int(v.nCurValue) if cam.MV_CC_GetIntValue(key, v) == 0 else 0

        def rd_float(key):
            v = mvs.MVCC_FLOATVALUE()
            mvs.memset(mvs.byref(v), 0, mvs.sizeof(v))
            return float(v.fCurValue) if cam.MV_CC_GetFloatValue(key, v) == 0 else 0.0

        logger.info(
            f"[{self.cam_id}] đã kết nối: serial={self.serial} "
            f"{rd_int('Width')}x{rd_int('Height')} "
            f"ResultingFrameRate={rd_float('ResultingFrameRate'):.1f} "
            f"packet={rd_int('GevSCPSPacketSize')}"
        )

    def _close_device(self):
        if self.cam is not None:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception as exc:
                logger.warning(f"[{self.cam_id}] lỗi khi đóng camera: {exc}")
            self.cam = None
        self.connected = False
        self.fps = 0.0
        # CỐ Ý không gọi mvs_sdk.sdk_finalize() ở đây: Finalize sẽ phá các camera
        # khác đang mở trong cùng tiến trình.

    # ── Chuyển đổi pixel ────────────────────────────────────────────────────
    def _to_bgr(self, st_frame):
        """Trả về (ảnh BGR, borrowed). borrowed=True → trỏ vào buffer dùng chung."""
        mvs = self.mvs
        fi = st_frame.stFrameInfo
        w, h = fi.nWidth, fi.nHeight

        if fi.enPixelType == mvs.PixelType_Gvsp_Mono8:
            ptr = mvs.cast(st_frame.pBufAddr, mvs.POINTER(mvs.c_ubyte * (w * h)))
            gray = np.frombuffer(ptr.contents, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), False

        if fi.enPixelType == mvs.PixelType_Gvsp_BGR8_Packed:
            ptr = mvs.cast(st_frame.pBufAddr, mvs.POINTER(mvs.c_ubyte * (w * h * 3)))
            return np.frombuffer(ptr.contents, dtype=np.uint8).reshape(h, w, 3), True

        # BayerRG8 (camera CU120) → để SDK convert: tên pattern GenICam lệch 1
        # pixel so với OpenCV nên cv2.cvtColor(COLOR_BAYER_*) rất dễ ra sai màu.
        need = w * h * 3
        if self._cvt_buf is None or self._cvt_buf_size < need:
            self._cvt_buf = (mvs.c_ubyte * need)()
            self._cvt_buf_size = need
            self._cvt_view = None

        param = mvs.MV_CC_PIXEL_CONVERT_PARAM_EX()
        mvs.memset(mvs.byref(param), 0, mvs.sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.pSrcData = st_frame.pBufAddr
        param.nSrcDataLen = fi.nFrameLen
        param.enSrcPixelType = fi.enPixelType
        param.enDstPixelType = mvs.PixelType_Gvsp_BGR8_Packed
        param.pDstBuffer = self._cvt_buf
        param.nDstBufferSize = self._cvt_buf_size

        ret = self.cam.MV_CC_ConvertPixelTypeEx(param)
        if ret != 0:
            raise RuntimeError(f"MV_CC_ConvertPixelTypeEx lỗi 0x{ret:x}")

        if self._cvt_view is None or self._cvt_view.shape != (h, w, 3):
            self._cvt_view = np.frombuffer(self._cvt_buf, dtype=np.uint8,
                                           count=need).reshape(h, w, 3)
        return self._cvt_view, True

    # ── Vòng đọc ────────────────────────────────────────────────────────────
    def _reader_guard(self):
        """Bọc vòng đọc: lỗi gì cũng không được làm chết camera trục."""
        while self._running:
            try:
                self._reader()
            except Exception as exc:
                logger.error(f"[{self.cam_id}] luồng đọc lỗi: {exc} → thử lại sau "
                             f"{self.RECONNECT_DELAY_S}s")
                self._close_device()
                time.sleep(self.RECONNECT_DELAY_S)

    def _reader(self):
        mvs = None
        st_frame = None
        while self._running:
            if self.cam is None:
                try:
                    self._open_device()
                    mvs = self.mvs
                    st_frame = mvs.MV_FRAME_OUT()
                    mvs.memset(mvs.byref(st_frame), 0, mvs.sizeof(st_frame))
                except Exception as exc:
                    self.reconnect_count += 1
                    logger.warning(f"[{self.cam_id}] mở camera thất bại: {exc} "
                                   f"(lần {self.reconnect_count})")
                    self._report_status(connected=False, fps=0.0)
                    time.sleep(self.RECONNECT_DELAY_S)
                    continue

            ret = self.cam.MV_CC_GetImageBuffer(st_frame, self.GET_TIMEOUT_MS)
            if ret != 0:
                self.fail_streak += 1
                if self.fail_streak >= self.RECONNECT_MAX_FAIL:
                    logger.warning(f"[{self.cam_id}] GetImageBuffer lỗi liên tục "
                                   f"(0x{ret:x}) → kết nối lại")
                    self._close_device()
                    self._report_status(connected=False, fps=0.0)
                self._tick_fps(got_frame=False)
                continue

            try:
                bgr, borrowed = self._to_bgr(st_frame)
                frame = bgr.copy() if borrowed else bgr
            except Exception as exc:
                logger.warning(f"[{self.cam_id}] lỗi chuyển frame: {exc}")
                frame = None
            finally:
                self.cam.MV_CC_FreeImageBuffer(st_frame)

            if frame is None:
                continue

            self.fail_streak = 0
            with self._lock:
                self._frame = frame
                self._seq += 1
            self._tick_fps()

    def _tick_fps(self, got_frame: bool = True):
        """Đo nhịp frame theo cửa sổ FPS_WINDOW_S giây rồi đẩy lên web.

        Gọi cả khi GetImageBuffer timeout (got_frame=False) để camera đứng hình
        tụt về 0 fps chứ không treo ở giá trị cũ."""
        if got_frame:
            self._fps_n += 1
        now = time.monotonic()
        dt = now - self._fps_t0
        if dt < self.FPS_WINDOW_S:
            return
        self.fps = self._fps_n / dt
        self._fps_t0 = now
        self._fps_n = 0
        self._report_status(connected=self.connected, fps=round(self.fps, 1))

    def _report_status(self, **fields):
        try:
            from yarnbreak.state_manager import state_manager
            state_manager.set_camera_status(self.cam_id, name=self.name, **fields)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE (1 luồng detector dùng chung engine cho mọi camera lược)
# ══════════════════════════════════════════════════════════════════════════════
class CombPipeline:
    def __init__(self, comb_cfg: dict, sdk_path: str, hub=None,
                 pause_event: Optional[threading.Event] = None,
                 stop_event: Optional[threading.Event] = None):
        self.cfg = comb_cfg or {}
        self.sdk_path = sdk_path
        self.hub = hub
        self.pause_event = pause_event
        self.stop_event = stop_event or threading.Event()

        self.alarm_outputs = list(self.cfg.get("alarm_outputs") or [1, 3])
        self.target_class = self.cfg.get("target_class", "Yarn-Break-Comb")
        self.imgsz = int(self.cfg.get("imgsz", 1024))
        self.conf = float(self.cfg.get("conf", 0.3))
        self.max_det = int(self.cfg.get("max_det", 20))
        self.draw_conf_min = float(self.cfg.get("draw_conf_min", 0.6))
        self.max_infer_fps = float(self.cfg.get("max_infer_fps", 8.0))

        a = self.cfg.get("alert") or {}
        self._alert_kw = dict(
            trend_window=a.get("trend_window", 5),
            trend_min_conf=a.get("trend_min_conf", 0.6),
            high_window=a.get("high_window", 3),
            high_conf=a.get("high_conf", 0.8),
            off_miss=a.get("off_miss", 5),
        )

        self.streams: Dict[str, CombCameraStream] = {}
        self.engines: Dict[str, CombAlertEngine] = {}
        self.clips: Dict[str, CombEventClip] = {}   # ghi video mỗi lần báo đứt
        self._last_seq: Dict[str, int] = {}
        self._frame_count: Dict[str, int] = {}
        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._was_paused = False
        self._reset_seq = 0        # seq yêu cầu tắt cảnh báo đã xử lý

    # ── Khởi động ─────────────────────────────────────────────────────────────
    def start(self):
        cams = [c for c in (self.cfg.get("cameras") or []) if c.get("enabled")]
        if not cams:
            logger.info("Comb: không có camera lược nào được bật → bỏ qua.")
            return False

        for c in cams:
            cam_id = c.get("id", "comb")
            if not str(c.get("mvs_serial") or "").strip():
                logger.error(f"Comb: camera '{cam_id}' thiếu mvs_serial → bỏ qua "
                             "(không được lấy camera đầu tiên).")
                continue
            st = CombCameraStream(c, self.sdk_path)
            self.streams[cam_id] = st
            self.engines[cam_id] = CombAlertEngine(**self._alert_kw)
            self.clips[cam_id] = CombEventClip(cam_id, self.cfg)
            self._last_seq[cam_id] = -1
            self._frame_count[cam_id] = 0
            st._report_status(connected=False, alarm_active=False)
            st.start()

        if not self.streams:
            return False

        self._thread = threading.Thread(target=self._detector_guard, daemon=True,
                                        name="CombDetector")
        self._thread.start()
        logger.info(f"Comb pipeline đã khởi động với {len(self.streams)} camera: "
                    f"{list(self.streams)}")
        any_clip = next(iter(self.clips.values()), None)
        if any_clip is not None and any_clip.enabled:
            logger.info(f"Comb: ghi clip báo đứt vào '{os.path.dirname(any_clip.dir)}' "
                        f"(tối thiểu {any_clip.min_s:.0f}s, pre-roll {any_clip.pre_s:.0f}s, "
                        f"tối đa {any_clip.max_s:.0f}s, scale {any_clip.scale})")
        else:
            logger.info("Comb: KHÔNG ghi clip báo đứt (comb.event_clip = false).")
        return True

    def stop(self):
        self.stop_event.set()
        for st in self.streams.values():
            try:
                st.stop()
            except Exception:
                pass
        # Đóng mọi clip đang ghi để file .mp4 không bị hỏng header.
        self._close_clips("STOP")
        # Nhả mọi reason alarm của camera lược để không giữ CH1/CH3 khi tắt.
        self._clear_all_alarms()

    # ── Relay ────────────────────────────────────────────────────────────────
    def _set_alarm(self, cam_id: str, on: bool):
        if self.hub is None:
            return
        try:
            for ch in self.alarm_outputs:
                self.hub.set_reason(ch, f"alarm:{cam_id}", on)
        except Exception as exc:
            logger.warning(f"[{cam_id}] ghi relay lỗi: {exc}")

    def _clear_all_alarms(self):
        for cam_id in list(self.streams):
            self._set_alarm(cam_id, False)

    # ── Clip ─────────────────────────────────────────────────────────────────
    def _close_clips(self, why: str):
        for clip in self.clips.values():
            try:
                clip.close(why)
            except Exception:
                pass

    # ── Luồng detector ───────────────────────────────────────────────────────
    def _detector_guard(self):
        while not self.stop_event.is_set():
            try:
                self._detector()
            except Exception as exc:
                logger.error(f"Comb detector lỗi: {exc} → khởi động lại sau 5s",
                             exc_info=True)
                self._model = None
                self._clear_all_alarms()
                self._close_clips("detector lỗi")
                time.sleep(5.0)

    def _load_model(self):
        """Nạp model TRONG luồng detector (TensorRT: warmup + inference phải cùng
        luồng, nếu không engine sẽ treo sau ~2s)."""
        from ultralytics import YOLO
        path = self.cfg.get("model", "./models/yarnbreakcomb.engine")
        task = self.cfg.get("task", "detect")
        logger.info(f"Comb: nạp model {path} (task={task}, imgsz={self.imgsz})")
        model = YOLO(path, task=task)
        # Warmup ngay trong luồng này
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=self.imgsz, conf=self.conf, verbose=False)
        logger.info("Comb: model đã nạp + warmup xong.")
        self._model = model

    def _detector(self):
        if self._model is None:
            self._load_model()

        min_interval = 1.0 / self.max_infer_fps if self.max_infer_fps > 0 else 0.0
        last_infer = 0.0
        cam_ids = list(self.streams)
        rr = 0

        while not self.stop_event.is_set():
            # PAUSE: dừng detect, xoá cảnh báo, khi RESUME phát hiện lại từ đầu.
            if self.pause_event is not None and self.pause_event.is_set():
                if not self._was_paused:
                    logger.info("Comb: PAUSE → xoá cảnh báo, dừng nhận diện.")
                    for e in self.engines.values():
                        e.reset()
                    self._clear_all_alarms()
                    # PAUSE không còn frame nào tới nữa → chốt file clip ngay,
                    # đừng để writer treo cho tới lúc RESUME.
                    self._close_clips("PAUSE")
                    for cid in cam_ids:
                        self._report(cid, alarm_active=False)
                    self._was_paused = True
                time.sleep(0.2)
                continue
            self._was_paused = False

            # Nút "TẮT CẢNH BÁO" trên web → nhả CH1/CH3 và bắt phát hiện lại
            # từ đầu (giống camera trục).
            self._check_reset_request()

            # Giãn nhịp tổng cho cả 2 camera lược (nhường GPU cho camera trục)
            now = time.monotonic()
            if min_interval and (now - last_infer) < min_interval:
                time.sleep(min(0.02, min_interval - (now - last_infer)))
                continue

            # Round-robin: chọn camera kế tiếp CÓ frame mới
            picked = None
            for _ in range(len(cam_ids)):
                cid = cam_ids[rr % len(cam_ids)]
                rr += 1
                st = self.streams[cid]
                frame, seq = st.read()
                if frame is not None and seq != self._last_seq[cid]:
                    picked = (cid, st, frame, seq)
                    break
            if picked is None:
                time.sleep(0.02)
                continue

            cam_id, st, frame, seq = picked
            self._last_seq[cam_id] = seq

            # detect_every: bỏ bớt frame cho nhẹ GPU
            self._frame_count[cam_id] += 1
            every = max(1, int(st.cfg.get("detect_every") or 1))
            if self._frame_count[cam_id] % every != 0:
                continue

            last_infer = time.monotonic()
            self._process(cam_id, st, frame)

    def _check_reset_request(self):
        """Web bấm TẮT CẢNH BÁO → state_manager.bump_alarm_reset() tăng seq."""
        try:
            from yarnbreak.state_manager import state_manager
            seq = state_manager.alarm_reset_seq
        except Exception:
            return
        if seq == self._reset_seq:
            return
        self._reset_seq = seq
        logger.info("Comb: nhận yêu cầu TẮT CẢNH BÁO từ web → nhả relay, dò lại từ đầu.")
        for e in self.engines.values():
            e.reset()
        self._clear_all_alarms()
        for cid in self.streams:
            self._report(cid, alarm_active=False)

    # ── Xử lý 1 frame ────────────────────────────────────────────────────────
    def _process(self, cam_id: str, st: CombCameraStream, frame):
        h, w = frame.shape[:2]
        roi_poly = self._roi_poly(st.cfg, w, h)

        # Crop theo bounding rect của ROI trước khi predict (giảm pixel cho GPU)
        if roi_poly is not None:
            x1, y1, bw, bh = cv2.boundingRect(roi_poly)
            x1 = max(0, x1); y1 = max(0, y1)
            crop = frame[y1:y1 + bh, x1:x1 + bw]
        else:
            x1 = y1 = 0
            crop = frame
        if crop.size == 0:
            return

        t0 = time.perf_counter()
        res = self._model.predict(crop, imgsz=self.imgsz, conf=self.conf,
                                  max_det=self.max_det, verbose=False)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        targets = self._parse(res, x1, y1, roi_poly)
        top_conf = targets[0]["conf"] if targets else None
        alert, just_on, just_off = self.engines[cam_id].update(top_conf)

        # Ghi clip video: nạp MỌI frame đã detect (kể cả lúc chưa báo — để có
        # pre-roll). Clip tự mở file khi just_on và tự đóng khi đủ thời lượng.
        clip = self.clips.get(cam_id)
        if clip is not None:
            clip.feed(frame, alert, just_on,
                      draw=lambda img, s: self._annotate(img, targets, roi_poly,
                                                         alert, scale=s))

        if just_on:
            logger.warning(f"⚠  ĐỨT SỢI – LƯỢC [{cam_id}] conf={top_conf:.3f} "
                           f"→ bật CH{self.alarm_outputs}")
            self._set_alarm(cam_id, True)
            video = clip.current_path if clip is not None else None
            self._log_event(cam_id, st, frame, targets, top_conf, video)
        elif just_off:
            logger.info(f"✓  Hết báo đứt [{cam_id}]")
            self._set_alarm(cam_id, False)

        self._report(cam_id, connected=st.connected, infer_ms=round(infer_ms, 1),
                     alarm_active=alert, name=st.name)

        # Live view: chỉ vẽ + encode khi CÓ người xem (tiết kiệm CPU/nhiệt)
        try:
            from yarnbreak.live_view import get_stream
            ls = get_stream(cam_id)
            if ls.want_frame():
                ls.publish(self._annotate(frame.copy(), targets, roi_poly, alert))
        except Exception:
            pass

    def _roi_poly(self, cam_cfg: dict, w: int, h: int):
        """ROI đa giác theo toạ độ CHUẨN HOÁ 0..1 → pixel. None nếu tắt."""
        if not cam_cfg.get("roi_enabled"):
            return None
        pts = cam_cfg.get("roi") or []
        if len(pts) < 3:
            return None
        arr = np.array([[int(px * w), int(py * h)] for px, py in pts], dtype=np.int32)
        return arr

    def _parse(self, res, off_x: int, off_y: int, roi_poly) -> List[dict]:
        """Lấy TẤT CẢ box hợp lệ, sort conf giảm dần, lọc theo đa giác ROI."""
        out: List[dict] = []
        if not res:
            return out
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        try:
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
        except Exception:
            return out

        order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        for i in order:
            conf = float(confs[i])
            bx1, by1, bx2, by2 = (int(v) for v in xyxy[i])
            bx1 += off_x; bx2 += off_x
            by1 += off_y; by2 += off_y
            if roi_poly is not None:
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                if cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) < 0:
                    continue
            out.append({"conf": conf, "box": (bx1, by1, bx2, by2)})
            if len(out) >= self.max_det:
                break
        return out

    def _annotate(self, img, targets, roi_poly, alert: bool, scale: float = 1.0):
        """Vẽ ROI + box + viền báo đứt. `scale` < 1 dùng khi vẽ lên frame ĐÃ HẠ
        CỠ (clip video): mọi toạ độ/độ dày/cỡ chữ nhân theo scale, có chặn dưới
        để chữ không nhỏ đến mức không đọc được."""
        s = scale if scale and scale > 0 else 1.0
        th = lambda base, lo=1: max(lo, int(round(base * s)))   # noqa: E731
        if roi_poly is not None:
            poly = roi_poly if s == 1.0 else (roi_poly * s).astype(np.int32)
            cv2.polylines(img, [poly], True, (0, 200, 255), th(2))
        for t in targets:
            if t["conf"] <= self.draw_conf_min:
                continue
            bx1, by1, bx2, by2 = [int(v * s) for v in t["box"]]
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), th(3))
            cv2.putText(img, f"YB {t['conf']:.2f}", (bx1, max(th(24), by1 - th(8))),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, 0.9 * s), (0, 0, 255),
                        th(2), cv2.LINE_AA)
        if alert:
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), th(10, 2))
            # Chữ vẽ đè lên ảnh dùng TIẾNG ANH, giống badge của camera trục
            # ("YARN BREAK DETECTED"): font Hershey của OpenCV không vẽ được dấu
            # tiếng Việt, viết không dấu thì nhìn như sai chính tả.
            txt = "YARN BREAK - COMB"
            fs = max(0.7, 1.4 * s)
            cv2.rectangle(img, (th(12), th(12)),
                          (th(12) + int(round(30 * fs / 1.4)) * len(txt), th(78, 34)),
                          (0, 0, 200), -1)
            cv2.putText(img, txt, (th(24), th(60, 27)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th(3), cv2.LINE_AA)
        return img

    def _report(self, cam_id: str, **fields):
        try:
            from yarnbreak.state_manager import state_manager
            state_manager.set_camera_status(cam_id, **fields)
        except Exception:
            pass

    def _log_event(self, cam_id, st, frame, targets, top_conf, video_path=None):
        """Ghi DB + lưu ảnh cảnh báo. Lỗi ở đây không được làm dừng pipeline."""
        try:
            import os
            from datetime import datetime
            from yarnbreak.alarm_logger import alarm_logger

            snap_dir = os.path.join("outputs", "snapshots", cam_id)
            os.makedirs(snap_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(snap_dir, f"{cam_id}_{ts}.jpg")
            img = self._annotate(frame.copy(), targets, None, True)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])

            from yarnbreak.state_manager import state_manager
            alarm_logger.log_alarm(
                defect_type="YarnBreakComb",
                confidence=float(top_conf or 0.0),
                camera_id=cam_id,
                image_path=path,
                machine_status=state_manager.state.value,
            )
            if video_path:
                logger.info(f"[{cam_id}] clip báo đứt: {video_path}")
            self._report(cam_id, last_alarm={
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conf": round(float(top_conf or 0.0), 2),
                "image_path": path,
                "video_path": video_path or "",
            })
        except Exception as exc:
            logger.warning(f"[{cam_id}] ghi event lỗi: {exc}")
