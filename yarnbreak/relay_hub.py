#!/usr/bin/env python3
"""
RelayHub — tiến trình DUY NHẤT sở hữu cổng RS485 (/dev/ttyACM0) cho cả hệ
đa camera (camera cũ "trục" + các camera "lược").

Vì sao cần hub:
  • Board Waveshare 8DI/8DO chỉ cho MỘT tiến trình mở cổng serial cùng lúc.
  • Nhiều nguồn cùng muốn điều khiển một kênh DO (ví dụ CH1 đỏ bị cả camera cũ
    lẫn camera lược kích; CH3 vàng bị paused + watchdog + alarm comb kích).

Mô hình "reason" (lý do):
  • Mỗi kênh DO = OR của tập các "lý do" đang bật.
      hub.set_reason("CH1", "alarm:cam_main", True)   # camera cũ báo đứt
      hub.set_reason("CH3", "alarm:comb_1", True)     # camera lược báo đứt
      hub.set_reason("CH3", "paused", True)           # đang tạm dừng
    → CH3 thực tế = OR(paused, watchdog:cam_main, alarm:comb_1, alarm:comb_2).
  • Chỉ ghi Modbus khi giá trị tổng hợp của kênh THAY ĐỔI (chống spam bus).
  • Toàn bộ giao dịch dùng chung một lock RS485 (đọc DI nút bấm cũng qua hub).
  • Tự reconnect do lớp transport (ModbusRelay) tự lo trong set_relay().

Hub làm việc ở KHÔNG GIAN LOGIC (reason bật = kênh "ON"). Việc một kênh ON ứng
với ĐÓNG (close) hay MỞ (open) tiếp điểm do `active_close` quyết định:
  • active_close=True  (mặc định, giống relay_active_close hiện tại):
        kênh ON  → set_relay(ch, True)  (CLOSE)
        kênh OFF → set_relay(ch, False) (OPEN)
  • active_close=False: đảo lại.

Kênh trạng thái CH8 cũng chạy qua hub nhưng dùng một reason duy nhất
("status") nên hành vi là "ai ghi sau thắng", giữ nguyên logic cũ.

CH7 là kênh "đi theo có trễ" của CH1 (xem set_delayed_follow): CH1 đóng liên
tục đủ 7 giây thì CH7 bắt đầu NHÁY theo chu kỳ 1 s (0,5 s đóng / 0,5 s mở),
CH1 nhả thì CH7 nhả ngay.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("YarnBreak.RelayHub")

ChannelLike = Union[int, str]


def _norm_ch(channel: ChannelLike) -> int:
    """Chuẩn hoá 'CH1' / 'ch3' / 3 / '3' → số kênh 1-based (int)."""
    if isinstance(channel, int):
        return channel
    s = str(channel).strip().upper()
    if s.startswith("CH"):
        s = s[2:]
    return int(s)


class RelayHub:
    """Bộ điều phối relay dùng chung, an toàn luồng, theo mô hình reason-OR."""

    def __init__(self, transport, active_close: bool = True,
                 managed_channels: Optional[List[int]] = None):
        """
        transport        : đối tượng kiểu ModbusRelay (hoặc FakeRelay khi test).
        active_close      : True nếu kênh ON ↔ CLOSE tiếp điểm (giống cấu hình hiện tại).
        managed_channels  : danh sách kênh hub quản lý; None = 1..8.
        """
        self._t = transport
        self._active_close = active_close
        self._lock = threading.RLock()          # bảo vệ reasons + bus RS485
        self._reasons: Dict[int, Set[str]] = {}  # ch → tập reason đang bật
        self._state: Dict[int, Optional[bool]] = {}  # ch → giá trị LOGIC đã ghi
        # Kênh "gương": ngoài reason của chính nó, còn BẬT khi có BẤT KỲ reason
        # nào mang tiền tố này trên BẤT KỲ kênh nào. Dùng cho CH2 (đèn xanh):
        # mọi camera báo đứt — kể cả camera lược vốn chỉ ghi CH1/CH3 — đều phải
        # tắt đèn xanh. Nhờ vậy comb.alarm_outputs vẫn giữ nguyên [1, 3].
        self._mirrors: Dict[int, str] = {}       # ch → tiền tố reason
        # Kênh "đi theo có trễ" (CH7 theo CH1): mỗi luật là một luồng canh giá
        # trị LOGIC của kênh nguồn, có thể NHÁY sau khi kích. Xem
        # set_delayed_follow().
        self._followers: Set[Tuple[int, int, str]] = set()
        self._follow_threads: List[threading.Thread] = []
        self._stop_followers = threading.Event()
        chans = managed_channels if managed_channels is not None else list(range(1, 9))
        for ch in chans:
            self._reasons[ch] = set()
            self._state[ch] = None              # None = chưa ghi lần nào

    # ── Vòng đời ──────────────────────────────────────────────────────────────
    def open(self) -> bool:
        with self._lock:
            ok = self._t.open()
        if ok:
            logger.info("RelayHub: cổng relay đã mở, đồng bộ trạng thái ban đầu.")
        return ok

    @property
    def is_open(self) -> bool:
        return getattr(self._t, "is_open", False)

    def close(self):
        """Đóng cổng serial. KHÔNG tự ý đổi trạng thái kênh ở đây — caller
        (main.close()) tự đặt trạng thái 'dừng hẳn' qua set_reason trước khi gọi."""
        self.stop_followers()
        with self._lock:
            try:
                self._t.close()
            except Exception as exc:
                logger.warning(f"RelayHub close error: {exc}")

    # ── Tính & ghi một kênh ─────────────────────────────────────────────────
    def set_mirror(self, channel: ChannelLike, prefix: str) -> None:
        """Khai báo kênh "gương": kênh này BẬT khi chính nó có reason, HOẶC khi
        có bất kỳ reason nào mang `prefix` trên bất kỳ kênh nào."""
        ch = _norm_ch(channel)
        with self._lock:
            self._reasons.setdefault(ch, set())
            self._state.setdefault(ch, None)
            self._mirrors[ch] = prefix
            self._write(ch)

    def _prefix_anywhere(self, prefix: str) -> bool:
        return any(any(r.startswith(prefix) for r in s)
                   for s in self._reasons.values())

    def _logical(self, ch: int) -> bool:
        if bool(self._reasons.get(ch)):
            return True
        prefix = self._mirrors.get(ch)
        return bool(prefix) and self._prefix_anywhere(prefix)

    def _write_mirrors(self, exclude: Optional[int] = None) -> None:
        """Ghi lại các kênh gương (gọi sau mỗi lần reason thay đổi ở kênh khác)."""
        for ch in self._mirrors:
            if ch != exclude:
                self._write(ch)

    def _write(self, ch: int, force: bool = False) -> bool:
        """Ghi kênh ch xuống transport nếu giá trị logic thay đổi (hoặc force)."""
        desired = self._logical(ch)
        if not force and self._state.get(ch) == desired:
            return True
        physical = desired if self._active_close else (not desired)
        ok = self._t.set_relay(ch, physical)
        if ok:
            self._state[ch] = desired
            logger.info(
                f"RelayHub CH{ch} → {'ON' if desired else 'OFF'} "
                f"(reasons={sorted(self._reasons.get(ch, set())) or '∅'})"
            )
        else:
            # Ghi thất bại: KHÔNG cập nhật _state để lần sau thử lại.
            logger.warning(f"RelayHub CH{ch} ghi thất bại (desired={desired})")
        return ok

    # ── API công khai ─────────────────────────────────────────────────────────
    def set_reason(self, channel: ChannelLike, reason: str, active: bool) -> bool:
        """Bật/tắt một 'lý do' cho kênh, rồi ghi lại nếu kênh đổi giá trị."""
        ch = _norm_ch(channel)
        with self._lock:
            if ch not in self._reasons:
                self._reasons[ch] = set()
                self._state[ch] = None
            s = self._reasons[ch]
            if active:
                s.add(reason)
            else:
                s.discard(reason)
            ok = self._write(ch)
            self._write_mirrors(exclude=ch)
            return ok

    def clear_reason_prefix(self, prefix: str,
                            channel: Optional[ChannelLike] = None) -> None:
        """Xoá mọi reason bắt đầu bằng `prefix`. Ghi lại các kênh bị ảnh hưởng.

        channel=None → áp cho TẤT CẢ kênh (vd xoá 'alarm:' khi force_off).
        channel=<ch> → chỉ kênh đó (vd PAUSE chỉ nhả CH1, KHÔNG chạm CH2 để
                       tránh nháy đèn xanh trong lúc chờ _sync_pause bật CH2)."""
        with self._lock:
            if channel is None:
                targets = list(self._reasons.keys())
            else:
                targets = [_norm_ch(channel)]
            for ch in targets:
                s = self._reasons.get(ch)
                if not s:
                    continue
                hit = {r for r in s if r.startswith(prefix)}
                if hit:
                    s -= hit
                    self._write(ch)
            self._write_mirrors()

    def set_status(self, channel: ChannelLike, on: bool) -> bool:
        """Kênh trạng thái 'ai ghi sau thắng' (CH7/CH8): dùng 1 reason 'status'."""
        return self.set_reason(channel, "status", on)

    # ── Kênh "đi theo có trễ" (CH7 theo CH1) ─────────────────────────────────
    def set_delayed_follow(self, source: ChannelLike, target: ChannelLike,
                           delay_s: float, reason: str = "follow",
                           poll_interval: float = 0.1,
                           blink_on_s: float = 0.0,
                           blink_off_s: float = 0.0) -> None:
        """Khai báo: kênh `target` BẬT sau khi kênh `source` đã ON LIÊN TỤC
        `delay_s` giây, rồi GIỮ (hoặc NHÁY) cho tới khi `source` OFF thì TẮT NGAY.

        blink_on_s/blink_off_s > 0 → sau khi kích, `target` nháy liên tục
        (`blink_on_s` giây đóng rồi `blink_off_s` giây mở). Cả hai = 0 → giữ
        đóng liên tục như trước.

        Dùng cho CH7 theo CH1: sợi đứt làm CH1 đóng, báo đủ 7 giây thì CH7 bắt
        đầu nháy chu kỳ 1 s (0,5 s đóng / 0,5 s mở), tới khi CH1 mở thì CH7 mở.
        Follower chỉ nhìn giá trị LOGIC tổng hợp của kênh nguồn nên MỌI nguồn
        kích CH1 đều tính (camera trục, camera lược, hay bất kỳ reason nào
        khác) — không phải sửa chỗ gọi nào.

        Nếu `source` ON ngắn hơn `delay_s` rồi nhả thì bộ đếm reset và `target`
        không bao giờ đóng (đúng nghĩa "báo được 7 giây").
        """
        src = _norm_ch(source)
        tgt = _norm_ch(target)
        with self._lock:
            key = (src, tgt, reason)
            if key in self._followers:
                return
            self._followers.add(key)
            self._reasons.setdefault(tgt, set())
            self._state.setdefault(tgt, None)
            # Trạng thái xuất phát: follower chưa kích (cũng ghi idle cho kênh).
            self.set_reason(tgt, reason, False)
        th = threading.Thread(
            target=self._follow_loop,
            args=(src, tgt, float(delay_s), reason, float(poll_interval),
                  float(blink_on_s), float(blink_off_s)),
            daemon=True, name=f"RelayFollow{src}->{tgt}",
        )
        with self._lock:
            self._follow_threads.append(th)
        th.start()
        mode = (f"nháy {blink_on_s:.2f}s/{blink_off_s:.2f}s"
                if (blink_on_s > 0 and blink_off_s > 0) else "giữ đóng")
        logger.info(
            f"RelayHub: CH{tgt} đi theo CH{src} với trễ {delay_s:.1f}s, "
            f"sau đó {mode} (reason='{reason}')"
        )

    def stop_followers(self) -> None:
        """Dừng mọi luồng follower (gọi trước khi đóng cổng để CH7 không bị
        ghi lại sau khi caller đã đặt trạng thái 'dừng hẳn')."""
        self._stop_followers.set()
        with self._lock:
            threads = list(self._follow_threads)
            self._follow_threads.clear()
            self._followers.clear()
        for th in threads:
            if th.is_alive():
                th.join(timeout=1.0)

    def _follow_loop(self, src: int, tgt: int, delay_s: float, reason: str,
                     poll_interval: float, blink_on_s: float = 0.0,
                     blink_off_s: float = 0.0) -> None:
        blink = blink_on_s > 0.0 and blink_off_s > 0.0
        on_since: Optional[float] = None   # mốc kênh nguồn bắt đầu ON liên tục
        asserted = False                   # đã qua ngưỡng trễ chưa
        out = False                        # giá trị đang ghi ra kênh đích
        next_toggle = 0.0                  # mốc đảo trạng thái kế (khi nháy)
        while not self._stop_followers.is_set():
            wait = poll_interval
            try:
                src_on = self.logical_state(src)
                now = time.monotonic()
                if src_on:
                    if on_since is None:
                        on_since = now
                    if not asserted and (now - on_since) >= delay_s:
                        if self.set_reason(tgt, reason, True):
                            asserted = True
                            out = True
                            next_toggle = now + blink_on_s
                            logger.info(
                                f"RelayHub CH{tgt} ĐÓNG: CH{src} đã ON liên tục "
                                f"{delay_s:.1f}s"
                                + (f", bắt đầu nháy {blink_on_s:.2f}s/"
                                   f"{blink_off_s:.2f}s" if blink else "")
                            )
                        # ghi lỗi → giữ asserted=False để vòng sau thử lại
                    elif asserted and blink and now >= next_toggle:
                        if self.set_reason(tgt, reason, not out):
                            out = not out
                            next_toggle = now + (blink_on_s if out else blink_off_s)
                        # ghi lỗi → next_toggle chưa đổi nên vòng sau thử lại
                    if asserted and blink:
                        # ngủ vừa đủ tới mốc đảo kế (nhưng vẫn quét kênh nguồn
                        # ít nhất mỗi poll_interval để nhả CH1 là tắt ngay).
                        wait = max(0.01, min(poll_interval, next_toggle - now))
                else:
                    on_since = None
                    if asserted or out:
                        if self.set_reason(tgt, reason, False):
                            asserted = False
                            out = False
                            logger.info(f"RelayHub CH{tgt} MỞ: CH{src} đã OFF")
            except Exception as exc:
                logger.warning(f"RelayHub follower CH{src}→CH{tgt} lỗi: {exc}")
            self._stop_followers.wait(wait)

    def sync_all(self) -> None:
        """Ép ghi lại toàn bộ kênh theo trạng thái logic hiện tại (sau reconnect
        hoặc lúc khởi động)."""
        with self._lock:
            for ch in self._reasons:
                self._write(ch, force=True)

    # ── Đọc (dùng chung lock RS485) ─────────────────────────────────────────
    def read_inputs(self, count: int = 8) -> Optional[List[bool]]:
        with self._lock:
            return self._t.read_inputs(count)

    def read_relays(self, count: int = 8) -> Optional[List[bool]]:
        with self._lock:
            return self._t.read_relays(count)

    # ── Nội quan (debug/test) ──────────────────────────────────────────────
    def reasons(self, channel: ChannelLike) -> Set[str]:
        with self._lock:
            return set(self._reasons.get(_norm_ch(channel), set()))

    def logical_state(self, channel: ChannelLike) -> bool:
        with self._lock:
            return self._logical(_norm_ch(channel))


class FakeRelay:
    """Transport giả lập ModbusRelay cho unit test — không cần phần cứng.

    Ghi lại trạng thái các kênh DO và cho phép bơm trạng thái DI đầu vào để
    kiểm thử vòng đọc nút bấm."""

    def __init__(self, n_channels: int = 8):
        self._states: List[bool] = [False] * n_channels   # trạng thái CLOSE/OPEN vật lý
        self._inputs: List[bool] = [False] * n_channels    # DI đầu vào mô phỏng
        self._open = False
        self.write_log: List[tuple] = []                   # lịch sử (ch, physical_on)
        self._lock = threading.Lock()

    def open(self) -> bool:
        self._open = True
        return True

    def close(self):
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def set_relay(self, channel: int, on: bool) -> bool:
        with self._lock:
            self._states[channel - 1] = on
            self.write_log.append((channel, on))
        return True

    def set_relays(self, channels: List[int], values: List[bool]) -> bool:
        for ch, v in zip(channels, values):
            self.set_relay(ch, v)
        return True

    def read_inputs(self, count: int = 8) -> Optional[List[bool]]:
        return list(self._inputs[:count])

    def read_relays(self, count: int = 8) -> Optional[List[bool]]:
        return list(self._states[:count])

    # ── Tiện ích cho test ──────────────────────────────────────────────────
    def set_input(self, channel: int, value: bool):
        self._inputs[channel - 1] = value

    def physical(self, channel: int) -> bool:
        """Trạng thái vật lý (CLOSE=True) của kênh."""
        return self._states[channel - 1]
