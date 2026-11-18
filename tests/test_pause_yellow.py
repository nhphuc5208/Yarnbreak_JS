"""Test den VANG khi PAUSE tren RelayHub + FakeRelay (khong can board that).

Mo phong dung thu tu ghi cua main.py:
  _sync_pause:  CH2 'status:paused'  ->  set_tower_lights  ->  CH3 'status:paused'
  watchdog:     CH3 'status:tower'   (co the ghi de bat ky luc nao)

Chay: python test_pause_yellow.py <duong-dan-du-an>
"""
import sys

proj = sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")
sys.path.insert(0, proj)

from relay_hub import RelayHub, FakeRelay

fails = []

GREEN_CH = 2    # relay_alarm_light_channel: ON -> den xanh TAT (dau NC)
YELLOW_CH = 3   # relay_pause_yellow_channel


def check(label, cond):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        fails.append(label)


def new_hub():
    t = FakeRelay(8)
    hub = RelayHub(t, active_close=True)
    hub.set_reason(1, "alarm:cam_main", False)
    hub.set_reason(GREEN_CH, "alarm:cam_main", False)
    hub.set_mirror(GREEN_CH, "alarm:")
    hub.set_reason(YELLOW_CH, "status:tower", False)
    hub.set_reason(YELLOW_CH, "status:paused", False)
    return hub, t


def do_pause(hub, paused: bool):
    """Dung THU TU nhu _sync_pause trong main.py sau khi sua."""
    hub.set_status(8, paused)
    hub.set_reason(GREEN_CH, "alarm:cam_main", False)
    hub.set_reason(GREEN_CH, "status:paused", paused)
    hub.set_reason(3, "status:tower", paused)            # set_tower_lights
    hub.set_reason(YELLOW_CH, "status:paused", paused)   # moi them


print("1) Bam PAUSE")
hub, t = new_hub()
do_pause(hub, True)
check("CH2 ON  (den xanh tat)", t.physical(GREEN_CH) is True)
check("CH3 ON  (den vang sang)", t.physical(YELLOW_CH) is True)
check("CH1 OFF (khong kich may)", t.physical(1) is False)
check("CH8 ON  (den trang thai)", t.physical(8) is True)

print("2) Watchdog den thap ghi de 'RUNNING' trong luc dang PAUSE")
hub.set_reason(3, "status:tower", False)                 # set_tower_lights("RUNNING")
check("CH3 VAN ON (reason PAUSE giu den vang)", t.physical(YELLOW_CH) is True)
check("reason con lai dung", hub.reasons(YELLOW_CH) == {"status:paused"})

print("3) Bam RESUME")
do_pause(hub, False)
check("CH3 OFF (het tam dung)", t.physical(YELLOW_CH) is False)
check("CH2 OFF (den xanh sang lai)", t.physical(GREEN_CH) is False)
check("CH8 OFF", t.physical(8) is False)

print("4) Camera luoc bao dut TRONG LUC dang PAUSE -> CH3 van ON sau resume-less")
hub, t = new_hub()
do_pause(hub, True)
hub.set_reason(3, "alarm:comb_1", True)
do_pause(hub, False)                                     # resume nhung luoc con bao
check("CH3 ON (con alarm camera luoc)", t.physical(YELLOW_CH) is True)
hub.set_reason(3, "alarm:comb_1", False)
check("CH3 OFF sau khi het alarm luoc", t.physical(YELLOW_CH) is False)

print("5) PAUSE khi dang co alarm -> den xanh tat, den vang sang")
hub, t = new_hub()
hub.set_reason(1, "alarm:cam_main", True)
hub.set_reason(GREEN_CH, "alarm:cam_main", True)
do_pause(hub, True)
# _sync_pause KHONG tu nha CH1 (viec do do release_stop_output lam) nen o day
# CH1 van ON — dung nhu hanh vi hien tai cua main.py.
check("CH1 van ON (do release_stop_output moi nha)", t.physical(1) is True)
check("CH3 ON (den vang)", t.physical(YELLOW_CH) is True)

print("6) Dung chuong trinh (close) -> nha reason PAUSE cua den vang")
hub, t = new_hub()
do_pause(hub, True)
hub.clear_reason_prefix("alarm:")
hub.set_reason(GREEN_CH, "status:paused", False)
hub.set_reason(GREEN_CH, "status:stopped", True)
hub.set_reason(3, "status:tower", False)                 # set_tower_lights("OFF")
hub.set_reason(YELLOW_CH, "status:paused", False)        # moi them trong close()
check("CH3 OFF sau khi dung", t.physical(YELLOW_CH) is False)
check("CH2 ON  (den xanh tat khi dung)", t.physical(GREEN_CH) is True)

print()
print("KET QUA:", "TAT CA PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
