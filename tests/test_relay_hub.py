"""Tower-light behaviour of RelayHub on PAUSE, using FakeRelay (no real board).

Mirrors the write order used by main.py:
  _sync_pause: CH2 'status:paused' -> set_tower_lights -> CH3 'status:paused'
  watchdog:    CH3 'status:tower' (may overwrite at any time)
"""
import pytest

from relay_hub import FakeRelay, RelayHub

RED_CH = 1      # stop-machine output
GREEN_CH = 2    # ON -> green light OFF (wired NC)
YELLOW_CH = 3   # paused / comb-camera alarm
STATUS_CH = 8


@pytest.fixture
def hub_and_board():
    board = FakeRelay(8)
    hub = RelayHub(board, active_close=True)
    hub.set_reason(RED_CH, "alarm:cam_main", False)
    hub.set_reason(GREEN_CH, "alarm:cam_main", False)
    hub.set_mirror(GREEN_CH, "alarm:")
    hub.set_reason(YELLOW_CH, "status:tower", False)
    hub.set_reason(YELLOW_CH, "status:paused", False)
    return hub, board


def set_paused(hub, paused: bool):
    """Same order as _sync_pause in main.py."""
    hub.set_status(STATUS_CH, paused)
    hub.set_reason(GREEN_CH, "alarm:cam_main", False)
    hub.set_reason(GREEN_CH, "status:paused", paused)
    hub.set_reason(YELLOW_CH, "status:tower", paused)    # set_tower_lights
    hub.set_reason(YELLOW_CH, "status:paused", paused)


def test_pause_turns_on_yellow_and_status(hub_and_board):
    hub, board = hub_and_board
    set_paused(hub, True)
    assert board.physical(GREEN_CH) is True
    assert board.physical(YELLOW_CH) is True
    assert board.physical(RED_CH) is False
    assert board.physical(STATUS_CH) is True


def test_watchdog_cannot_clear_yellow_while_paused(hub_and_board):
    hub, board = hub_and_board
    set_paused(hub, True)
    hub.set_reason(YELLOW_CH, "status:tower", False)     # watchdog writes RUNNING
    assert board.physical(YELLOW_CH) is True
    assert hub.reasons(YELLOW_CH) == {"status:paused"}


def test_resume_restores_lights(hub_and_board):
    hub, board = hub_and_board
    set_paused(hub, True)
    set_paused(hub, False)
    assert board.physical(YELLOW_CH) is False
    assert board.physical(GREEN_CH) is False
    assert board.physical(STATUS_CH) is False


def test_comb_alarm_keeps_yellow_after_resume(hub_and_board):
    hub, board = hub_and_board
    set_paused(hub, True)
    hub.set_reason(YELLOW_CH, "alarm:comb_1", True)
    set_paused(hub, False)
    assert board.physical(YELLOW_CH) is True
    hub.set_reason(YELLOW_CH, "alarm:comb_1", False)
    assert board.physical(YELLOW_CH) is False


def test_pause_during_alarm_keeps_red_latched(hub_and_board):
    hub, board = hub_and_board
    hub.set_reason(RED_CH, "alarm:cam_main", True)
    hub.set_reason(GREEN_CH, "alarm:cam_main", True)
    set_paused(hub, True)
    # _sync_pause does not release CH1; release_stop_output does that.
    assert board.physical(RED_CH) is True
    assert board.physical(YELLOW_CH) is True


def test_stop_releases_pause_reason(hub_and_board):
    hub, board = hub_and_board
    set_paused(hub, True)
    hub.clear_reason_prefix("alarm:")
    hub.set_reason(GREEN_CH, "status:paused", False)
    hub.set_reason(GREEN_CH, "status:stopped", True)
    hub.set_reason(YELLOW_CH, "status:tower", False)     # set_tower_lights("OFF")
    hub.set_reason(YELLOW_CH, "status:paused", False)
    assert board.physical(YELLOW_CH) is False
    assert board.physical(GREEN_CH) is True
