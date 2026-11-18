"""Event CSV log, annotated video writer and pre-roll event clips.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from yarnbreak.config import AppConfig

logger = logging.getLogger("YarnBreak")


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
