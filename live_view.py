"""
live_view.py — Luồng xem camera trực tiếp (MJPEG) cho web UI.

Detector (main.py) đẩy frame ĐÃ vẽ bounding box + viền ROI vào đây khi có người
đang xem; web_server phát lại dưới dạng multipart/x-mixed-replace để hiển thị
trong thẻ <img>. Mục đích: người vận hành căn chỉnh vị trí camera sao cho vùng
trục (ROI) nằm đúng chỗ.

Thiết kế:
  • Chỉ mã hoá JPEG khi CÓ viewer (viewers > 0) và đủ thời gian giãn cách
    (rate-limit) → không tốn CPU/nhiệt khi không ai xem.
  • Frame gốc từ camera (5MP) được thu nhỏ về max_width trước khi mã hoá.
  • Generator tự thoát sau idle_timeout giây không có frame mới (pipeline dừng)
    để giải phóng thread trong threadpool của server.
"""

import threading
import time

import cv2


class LiveStream:
    def __init__(self, max_fps: float = 12.0, max_width: int = 1024,
                 jpeg_quality: int = 70):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._gen = 0                 # tăng mỗi frame mới → đánh thức viewer
        self._viewers = 0
        self._last_pub = 0.0
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._max_width = max_width
        self._jpeg_quality = int(jpeg_quality)

    # ── Đếm người xem ────────────────────────────────────────────────────────
    def add_viewer(self):
        with self._lock:
            self._viewers += 1

    def remove_viewer(self):
        with self._lock:
            if self._viewers > 0:
                self._viewers -= 1

    def active(self) -> bool:
        with self._lock:
            return self._viewers > 0

    def want_frame(self) -> bool:
        """True nếu đang có người xem VÀ đã đủ giãn cách để publish frame mới.
        Gọi ở main loop để tránh copy/encode frame khi không cần."""
        now = time.monotonic()
        with self._lock:
            if self._viewers <= 0:
                return False
            return (now - self._last_pub) >= self._min_interval

    # ── Nhà sản xuất (detector) ──────────────────────────────────────────────
    def publish(self, frame_bgr):
        """Thu nhỏ + mã hoá JPEG frame (BGR) rồi phát cho các viewer."""
        try:
            h, w = frame_bgr.shape[:2]
            if self._max_width and w > self._max_width:
                scale = self._max_width / float(w)
                frame_bgr = cv2.resize(
                    frame_bgr, (self._max_width, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA
                )
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                return
            data = buf.tobytes()
        except Exception:
            return

        with self._cond:
            self._jpeg = data
            self._gen += 1
            self._last_pub = time.monotonic()
            self._cond.notify_all()

    # ── Người tiêu thụ (web_server) ──────────────────────────────────────────
    def frames(self, idle_timeout: float = 5.0):
        """Generator sinh các chunk MJPEG. Tự thoát nếu không có frame mới trong
        idle_timeout giây (ví dụ pipeline đã dừng)."""
        boundary = b"--frame\r\n"
        last_gen = -1
        idle = 0.0
        self.add_viewer()
        try:
            while True:
                data = None
                with self._cond:
                    got = self._cond.wait_for(
                        lambda: self._gen != last_gen, timeout=1.0
                    )
                    if got:
                        last_gen = self._gen
                        data = self._jpeg
                if data is None:
                    idle += 1.0
                    if idle >= idle_timeout:
                        break
                    continue
                idle = 0.0
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )
        finally:
            self.remove_viewer()


# Singleton dùng chung giữa main.py (detector) và web_server.py
live_stream = LiveStream()

# ── [multicam] Nhiều camera: mỗi cam_id một LiveStream ───────────────────────
# "cam_main" trả về đúng singleton ở trên để web/API cũ không vỡ.
_streams_lock = threading.Lock()
_streams: dict = {"cam_main": live_stream}


def get_stream(cam_id: str = "cam_main") -> LiveStream:
    """Trả về LiveStream của camera; tự tạo nếu chưa có."""
    cam_id = cam_id or "cam_main"
    with _streams_lock:
        st = _streams.get(cam_id)
        if st is None:
            # Camera lược ảnh rất lớn (12MP) → giảm bớt để web nhẹ.
            st = LiveStream(max_fps=10.0, max_width=960, jpeg_quality=65)
            _streams[cam_id] = st
        return st


def stream_ids() -> list:
    with _streams_lock:
        return list(_streams.keys())
