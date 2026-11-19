"""Relay truth table — the contract with the real machine.

Ported from the comb repo's scripts/test_relay_hub.py (a print-and-exit script)
into pytest. Runs on FakeRelay, no RS485 board needed.

The hub is wired exactly like production:
  * AlertNotifier.__init__  -> CH2 mirrors every "alarm:*" reason
                               (green light OFF whenever ANY camera alarms)
  * beam camera alarm       -> reason "alarm:cam_main" on CH1 + CH2
  * comb camera alarm       -> reason "alarm:<cam_id>" on comb.alarm_outputs = [1, 3]
  * pause                   -> clear "alarm:*", CH2 + CH3 reason "status:paused"
  * beam heartbeat watchdog -> CH3 reason "watchdog:cam_main"
  * stop                    -> clear "alarm:*", CH2 reason "status:stopped"

CH7 follows CH1 with a delay, then blinks. Those tests drive the follower loop
with a simulated clock, so they are deterministic and take no wall time.
"""
import pytest

from yarnbreak import relay_hub as rh
from yarnbreak.relay_hub import FakeRelay, RelayHub

RED, GREEN_OFF, YELLOW, FOLLOW = 1, 2, 3, 7
COMB_OUTPUTS = [1, 3]          # config.yaml: comb.alarm_outputs


# ── production-equivalent helpers ─────────────────────────────────────────────
def make_hub():
    board = FakeRelay(8)
    hub = RelayHub(board, active_close=True)
    hub.open()
    hub.set_reason(RED, "alarm:cam_main", False)
    hub.set_reason(GREEN_OFF, "alarm:cam_main", False)
    hub.set_mirror(GREEN_OFF, "alarm:")
    hub.sync_all()
    return hub, board


def beam_alarm(hub, on):
    hub.set_reason(RED, "alarm:cam_main", on)
    hub.set_reason(GREEN_OFF, "alarm:cam_main", on)


def comb_alarm(hub, cam_id, on):
    for ch in COMB_OUTPUTS:                     # CombPipeline._set_alarm
        hub.set_reason(ch, f"alarm:{cam_id}", on)


def pause(hub):
    hub.clear_reason_prefix("alarm:")
    hub.set_reason(GREEN_OFF, "status:paused", True)
    hub.set_reason(YELLOW, "status:paused", True)


def resume(hub):
    hub.set_reason(GREEN_OFF, "status:paused", False)
    hub.set_reason(YELLOW, "status:paused", False)


def stop(hub):
    hub.clear_reason_prefix("alarm:")
    hub.set_reason(YELLOW, "status:paused", False)
    hub.set_reason(YELLOW, "watchdog:cam_main", False)
    hub.set_reason(GREEN_OFF, "status:paused", False)
    hub.set_reason(GREEN_OFF, "status:stopped", True)


def outputs(board):
    return tuple(int(board.physical(ch)) for ch in (RED, GREEN_OFF, YELLOW))


# ── the 7-state truth table ───────────────────────────────────────────────────
def _running(hub):
    pass


def _beam(hub):
    beam_alarm(hub, True)


def _comb(hub):
    comb_alarm(hub, "comb_1", True)


def _both(hub):
    beam_alarm(hub, True)
    comb_alarm(hub, "comb_2", True)


def _paused(hub):
    beam_alarm(hub, True)
    comb_alarm(hub, "comb_1", True)
    pause(hub)


def _watchdog(hub):
    hub.set_reason(YELLOW, "watchdog:cam_main", True)


def _stopped(hub):
    comb_alarm(hub, "comb_1", True)
    hub.set_reason(YELLOW, "watchdog:cam_main", True)
    stop(hub)


TRUTH_TABLE = [
    #  state                        setup        CH1 CH2 CH3
    ("running, no fault",           _running,   (0, 0, 0)),
    ("beam camera alarm",           _beam,      (1, 1, 0)),
    ("comb camera alarm",           _comb,      (1, 1, 1)),
    ("beam + comb alarm",           _both,      (1, 1, 1)),
    ("paused (red button / web)",   _paused,    (0, 1, 1)),
    ("beam heartbeat lost",         _watchdog,  (0, 0, 1)),
    ("stopped",                     _stopped,   (0, 1, 0)),
]


@pytest.mark.parametrize("label,setup,expected", TRUTH_TABLE, ids=[r[0] for r in TRUTH_TABLE])
def test_truth_table(label, setup, expected):
    hub, board = make_hub()
    setup(hub)
    assert outputs(board) == expected


def test_comb_clears_while_beam_still_alarms():
    hub, board = make_hub()
    beam_alarm(hub, True)
    comb_alarm(hub, "comb_1", True)
    comb_alarm(hub, "comb_1", False)
    assert outputs(board) == (1, 1, 0)


def test_beam_clears_while_comb_still_alarms():
    hub, board = make_hub()
    beam_alarm(hub, True)
    comb_alarm(hub, "comb_1", True)
    beam_alarm(hub, False)
    assert outputs(board) == (1, 1, 1)


def test_one_comb_camera_cannot_clear_the_other():
    hub, board = make_hub()
    comb_alarm(hub, "comb_1", True)
    comb_alarm(hub, "comb_2", True)
    comb_alarm(hub, "comb_1", False)
    assert outputs(board) == (1, 1, 1)


def test_resume_without_fault_returns_to_idle():
    hub, board = make_hub()
    comb_alarm(hub, "comb_1", True)
    pause(hub)
    resume(hub)
    assert outputs(board) == (0, 0, 0)


def test_unchanged_value_is_not_rewritten():
    hub, board = make_hub()
    stop(hub)
    before = len(board.write_log)
    hub.set_reason(GREEN_OFF, "status:stopped", True)
    assert len(board.write_log) == before


# ── integration: red button sequence as wired in pipeline.py / alerts.py ─────
def test_red_button_pause_resume_sequence():
    hub, board = make_hub()
    alarm_reason = "alarm:cam_main"            # AlertNotifier._alarm_reason

    def set_tower_lights(state):               # AlertNotifier.set_tower_lights
        hub.set_reason(YELLOW, "status:tower", state in ("STOPPED", "PAUSED"))

    def release_stop_output():                 # AlertNotifier.release_stop_output
        hub.clear_reason_prefix("alarm:", channel=RED)

    def sync_pause(paused, alarm_active):      # _sync_pause in pipeline.main()
        hub.set_reason(GREEN_OFF, alarm_reason, alarm_active)
        hub.set_reason(GREEN_OFF, "status:paused", paused)
        set_tower_lights("PAUSED" if paused else "RUNNING")

    set_tower_lights("RUNNING")
    assert outputs(board) == (0, 0, 0)

    beam_alarm(hub, True)
    assert outputs(board) == (1, 1, 0)

    release_stop_output()                      # CH1 released, CH2 must not flicker off
    assert outputs(board) == (0, 1, 0)

    sync_pause(paused=True, alarm_active=False)
    assert outputs(board) == (0, 1, 1)

    sync_pause(paused=False, alarm_active=False)
    assert outputs(board) == (0, 0, 0)         # no reason left hanging

    comb_alarm(hub, "comb_1", True)
    set_tower_lights("RUNNING")                # tower RUNNING must not clear comb yellow
    assert outputs(board) == (1, 1, 1)
    comb_alarm(hub, "comb_1", False)
    assert outputs(board) == (0, 0, 0)


# ── CH7: delayed follow + blink, simulated clock ──────────────────────────────
class SimClock:
    """Stands in for both time.monotonic and the hub's stop Event.

    The follower loop calls `stop.wait(dt)`; we advance simulated time by dt,
    fire any scheduled actions, and record CH7 — no real sleeping."""

    def __init__(self, board, end, actions):
        self.t = 0.0
        self.board = board
        self.end = end
        self.actions = sorted(actions, key=lambda a: a[0])
        self.trace = []

    def monotonic(self):
        return self.t

    def is_set(self):
        return self.t >= self.end

    def set(self):
        self.end = self.t

    def wait(self, dt):
        self.t = round(self.t + dt, 6)
        while self.actions and self.actions[0][0] <= self.t:
            self.actions.pop(0)[1]()
        self.trace.append((self.t, self.board.physical(FOLLOW)))
        return self.is_set()


def run_follower(monkeypatch, actions, end, delay=7.0, on=0.5, off=0.5, poll=0.1):
    hub, board = make_hub()
    clock = SimClock(board, end, actions(hub))
    while clock.actions and clock.actions[0][0] <= 0.0:      # actions at t=0 happen before the loop starts
        clock.actions.pop(0)[1]()
    monkeypatch.setattr(rh.time, "monotonic", clock.monotonic)
    hub._stop_followers = clock
    hub.set_reason(FOLLOW, "follow:CH1", False)
    hub._follow_loop(RED, FOLLOW, delay, "follow:CH1", poll, on, off)
    return clock.trace


def first_close(trace):
    return next((t for t, v in trace if v), None)


def test_ch7_ignores_short_alarm(monkeypatch):
    trace = run_follower(monkeypatch, lambda hub: [
        (1.0, lambda: beam_alarm(hub, True)),
        (7.5, lambda: beam_alarm(hub, False)),     # held 6.5 s < 7 s
    ], end=20.0)
    assert first_close(trace) is None


def test_ch7_closes_after_delay_then_blinks_and_releases(monkeypatch):
    trace = run_follower(monkeypatch, lambda hub: [
        (1.0, lambda: beam_alarm(hub, True)),
        (14.0, lambda: beam_alarm(hub, False)),
    ], end=20.0)

    t_close = first_close(trace)
    assert t_close == pytest.approx(8.0, abs=0.11)            # 1.0 + 7 s delay

    # Blinking: 0.5 s on / 0.5 s off while CH1 stays closed.
    window = [(t, v) for t, v in trace if t_close <= t < 14.0]
    toggles = sum(1 for (_, a), (_, b) in zip(window, window[1:]) if a != b)
    cycles = toggles / 2
    assert 5 <= cycles <= 6                                    # ~6 s of blinking

    # Released within one poll interval and never closes again.
    after = [(t, v) for t, v in trace if t >= 14.1]
    assert after and not any(v for _, v in after)


def test_ch7_follows_comb_alarm_too(monkeypatch):
    trace = run_follower(monkeypatch, lambda hub: [
        (0.5, lambda: comb_alarm(hub, "comb_2", True)),
    ], end=10.0)
    assert first_close(trace) == pytest.approx(7.5, abs=0.11)


def test_ch7_hold_mode_without_blink(monkeypatch):
    trace = run_follower(monkeypatch, lambda hub: [
        (0.0, lambda: beam_alarm(hub, True)),
    ], end=12.0, on=0.0, off=0.0)
    t_close = first_close(trace)
    assert t_close == pytest.approx(7.0, abs=0.11)
    assert all(v for t, v in trace if t >= t_close)
