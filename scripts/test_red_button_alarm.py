#!/usr/bin/env python3
"""
Kịch bản test OFFLINE: nút đỏ (DI1) dừng chương trình + tắt alarm, và trạng thái
relay khi PAUSE = DO1 tắt, DO2 bật, DO3 bật.

Không cần camera/model/phần cứng — dùng FakeRelay trong RAM để kiểm chứng logic
thật của AlertNotifier + state_manager mà bản vá dựa vào, và mô phỏng _sync_pause.

Chạy:
    ./venv/bin/python scripts/test_red_button_alarm.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yarnbreak.config import AppConfig              # noqa: E402
from yarnbreak.alerts import AlertNotifier          # noqa: E402
from yarnbreak.state_manager import state_manager, SystemState  # noqa: E402


# Kênh (1-based) theo config: DO1=CH1 (còi/dừng máy), DO2=CH2 (đèn báo), DO3=CH3 (đèn tháp)
CH_STOP = 1     # relay_channel
CH_LIGHT = 2    # relay_alarm_light_channel
CH_TOWER = 3


class FakeRelay:
    """Ghi trạng thái 8 coil trong RAM, mô phỏng API ModbusRelay."""
    def __init__(self):
        self.coils = [False] * 8      # False = OPEN (tắt), True = CLOSE (bật)

    def set_relay(self, channel: int, on: bool) -> bool:
        self.coils[channel - 1] = on
        return True

    def read_relays(self, count: int = 8):
        return list(self.coils[:count])

    def read_inputs(self, count: int = 8):
        return [False] * count

    @property
    def is_open(self):
        return True

    def close(self):
        pass

    def CH(self, ch: int) -> bool:
        return self.coils[ch - 1]


_failures = []


def check(cond: bool, msg: str):
    print(f"{'  ✓ PASS' if cond else '  ✗ FAIL'}  {msg}")
    if not cond:
        _failures.append(msg)


def build_notifier():
    cfg = AppConfig()
    cfg.relay_enabled = False           # không mở serial thật
    notifier = AlertNotifier(cfg)
    fake = FakeRelay()
    notifier._relay = fake
    return notifier, fake


def sync_pause_apply(notifier, paused: bool):
    """Mô phỏng main.py:_sync_pause khi có chuyển trạng thái paused:
    CH8/CH7 = paused, CH2 = paused or alarm_active, DO3 (CH3) = PAUSED/RUNNING."""
    r = notifier._relay
    r.set_relay(8, paused)
    r.set_relay(7, paused)
    r.set_relay(CH_LIGHT, paused or state_manager.alarm_active)
    notifier.set_tower_lights("PAUSED" if paused else "RUNNING")


def red_button_pause(notifier):
    """Mô phỏng handler nút đỏ (main.py:_di_poll_loop) + _sync_pause sau đó."""
    state_manager.pause_pipeline()               # xoá alarm_active
    notifier.release_stop_output("red button")   # CHỈ nhả CH1
    sync_pause_apply(notifier, paused=True)      # _sync_pause bật CH2/CH3


def check_paused_state(fake, tag=""):
    check(fake.CH(CH_STOP) is False, f"{tag}DO1 (CH1) TẮT")
    check(fake.CH(CH_LIGHT) is True, f"{tag}DO2 (CH2) BẬT")
    check(fake.CH(CH_TOWER) is True, f"{tag}DO3 (CH3) BẬT")


# ── Kịch bản ──────────────────────────────────────────────────────────────────
def scenario_pause_during_alarm():
    print("\n[1] Bấm NÚT ĐỎ khi ĐANG alarm → DO1 tắt, DO2 bật, DO3 bật")
    notifier, fake = build_notifier()
    state_manager.start_pipeline()
    notifier.on_alert(100, 0.9)
    state_manager.trigger_alarm(1, "YarnBreak", 0.9)
    check(fake.CH(CH_STOP) is True, "Alarm nổ → DO1 (CH1) BẬT")
    check(fake.CH(CH_LIGHT) is True, "Alarm nổ → DO2 (CH2) BẬT")

    red_button_pause(notifier)
    check_paused_state(fake)
    check(state_manager.alarm_active is False, "alarm_active = False (đã tắt alarm)")
    check(state_manager.state == SystemState.PAUSED, "Trạng thái = PAUSED")
    state_manager.stop_pipeline()


def scenario_pause_no_alarm():
    print("\n[2] Bấm NÚT ĐỎ khi KHÔNG alarm → vẫn DO1 tắt, DO2 bật, DO3 bật")
    notifier, fake = build_notifier()
    state_manager.start_pipeline()
    check(fake.CH(CH_STOP) is False, "Đang chạy, không alarm → DO1 tắt")

    red_button_pause(notifier)
    check_paused_state(fake)
    state_manager.stop_pipeline()


def scenario_resume():
    print("\n[3] Bấm NÚT XANH (resume) → DO2 tắt, DO3 tắt (về RUNNING)")
    notifier, fake = build_notifier()
    state_manager.start_pipeline()
    notifier.on_alert(120, 0.9)
    state_manager.trigger_alarm(2, "YarnBreak", 0.9)
    red_button_pause(notifier)
    check_paused_state(fake, "Đang PAUSED: ")

    state_manager.resume_pipeline()
    sync_pause_apply(notifier, paused=False)     # _sync_pause khi bỏ pause
    check(fake.CH(CH_LIGHT) is False, "Resume → DO2 (CH2) TẮT")
    check(fake.CH(CH_TOWER) is False, "Resume → DO3 (CH3) TẮT")
    check(state_manager.state == SystemState.RUNNING, "Trạng thái = RUNNING")
    state_manager.stop_pipeline()


class FakeDetector:
    """Mô phỏng detector _run(): break còn → cạnh lên đóng CH1; vào pause →
    release_stop_output nhả CH1 và không báo lại."""
    def __init__(self, notifier, pause_evt, break_present=True):
        self.notifier = notifier
        self.pause = pause_evt
        self.break_present = break_present
        self._alert = False
        self._was_paused = False

    def tick(self):
        if self.pause.is_set():
            if not self._was_paused:
                self._alert = False
                self.notifier.release_stop_output("paused")
                self._was_paused = True
            return
        self._was_paused = False
        if self.break_present and not self._alert:
            self._alert = True
            if not self.pause.is_set():
                self.notifier.on_alert(999, 0.9)


def scenario_standalone_persistent_break():
    print("\n[4] STANDALONE + sợi VẪN đứt: detector phải pause, DO1 KHÔNG bật lại")
    notifier, fake = build_notifier()
    state_manager.start_pipeline()
    user_pause_evt = threading.Event()
    combined_pause_evt = threading.Event()
    det = FakeDetector(notifier, combined_pause_evt, break_present=True)

    det.tick()
    state_manager.trigger_alarm(7, "YarnBreak", 0.9)
    check(fake.CH(CH_STOP) is True, "Sợi đứt → DO1 BẬT")

    # Nút đỏ: pause_pipeline + set event local + nhả CH1 + _sync_pause
    state_manager.pause_pipeline()
    user_pause_evt.set()
    combined_pause_evt.set()
    notifier.release_stop_output("red button")
    sync_pause_apply(notifier, paused=True)
    check_paused_state(fake, "Ngay sau nút đỏ: ")

    for _ in range(5):
        det.tick()
    check(fake.CH(CH_STOP) is False, "Sau nhiều vòng, sợi vẫn đứt → DO1 VẪN TẮT (đã pause)")
    check(fake.CH(CH_LIGHT) is True, "DO2 vẫn BẬT trong lúc pause")
    state_manager.stop_pipeline()


def main():
    print("=" * 64)
    print("  TEST: Nút đỏ → PAUSE = DO1 tắt, DO2 bật, DO3 bật (+ tắt alarm)")
    print("=" * 64)

    scenario_pause_during_alarm()
    scenario_pause_no_alarm()
    scenario_resume()
    scenario_standalone_persistent_break()

    state_manager.stop_pipeline()

    print("\n" + "=" * 64)
    if _failures:
        print(f"  KẾT QUẢ: ✗ {len(_failures)} kiểm tra THẤT BẠI")
        for m in _failures:
            print(f"    - {m}")
        print("=" * 64)
        sys.exit(1)
    print("  KẾT QUẢ: ✓ TẤT CẢ PASS")
    print("=" * 64)
    sys.exit(0)


if __name__ == "__main__":
    main()
