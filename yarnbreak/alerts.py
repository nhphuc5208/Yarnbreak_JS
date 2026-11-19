"""Alarm output: relay board, tower lights, push buttons.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
import sys


from yarnbreak.config import AppConfig

logger = logging.getLogger("YarnBreak")


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
            from yarnbreak.modbus_relay import ModbusRelay
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
        from yarnbreak.relay_hub import RelayHub
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
            from yarnbreak.relay_hub import RelayHub
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
