"""CombEventClip timing rules, tested with a fake clock.

No camera or Jetson needed: frames are generated with numpy.
"""
import os

import cv2
import numpy as np
import pytest

import comb_pipeline as cp

FPS = 4.0
DT = 1.0 / FPS
FRAME = np.full((200, 640, 3), 40, dtype=np.uint8)


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cp.time, "monotonic", lambda: now["t"])
    return now


@pytest.fixture
def cfg(tmp_path):
    return {
        "event_clip": True,
        "event_clip_dir": str(tmp_path),
        "event_buffer_scale": 0.5,
        "event_pre_seconds": 3.0,
        "event_min_seconds": 30.0,
        "event_post_seconds": 3.0,
        "event_max_seconds": 120.0,
        "event_clip_fps": 0,
    }


def run(clip, clock, seconds, alarm_on, just_on=False):
    """Feed `seconds` worth of frames; only the first one carries `just_on`."""
    for i in range(max(1, int(seconds * FPS))):
        clock["t"] += DT
        clip.feed(FRAME, alarm_on, just_on and i == 0)


def clips(cfg):
    d = os.path.join(cfg["event_clip_dir"], "comb_1")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def test_short_alarm_still_records_minimum_30s(clock, cfg):
    clip = cp.CombEventClip("comb_1", cfg)

    run(clip, clock, 10, alarm_on=False)
    assert clips(cfg) == [], "nothing is written before an alarm"
    assert len(clip._buf) <= int(3 * FPS) + 1, "pre-roll buffer is capped at 3 s"

    run(clip, clock, DT, alarm_on=True, just_on=True)
    assert len(clips(cfg)) == 1
    assert clip.current_path is not None

    run(clip, clock, 2, alarm_on=True)
    run(clip, clock, 20, alarm_on=False)
    assert clip.current_path is not None, "still recording before 30 s"

    run(clip, clock, 9, alarm_on=False)
    assert clip.current_path is None, "file closed after ~31 s"

    cap = cv2.VideoCapture(os.path.join(cfg["event_clip_dir"], "comb_1", clips(cfg)[0]))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    assert fps and n_frames / fps >= 30.0
    assert size == (320, 100), "clip is scaled by event_buffer_scale"


def test_long_alarm_is_capped_at_max_seconds(clock, cfg):
    clip = cp.CombEventClip("comb_1", cfg)
    run(clip, clock, DT, alarm_on=True, just_on=True)
    run(clip, clock, 130, alarm_on=True)
    assert clip.current_path is None
    assert len(clips(cfg)) == 1


def test_re_alarm_while_closing_keeps_same_file(clock, cfg):
    clip = cp.CombEventClip("comb_1", cfg)
    run(clip, clock, DT, alarm_on=True, just_on=True)
    run(clip, clock, 5, alarm_on=False)
    run(clip, clock, 5, alarm_on=True)
    run(clip, clock, 25, alarm_on=False)
    assert len(clips(cfg)) == 1


def test_disabled_event_clip_writes_nothing(clock, cfg):
    cfg["event_clip"] = False
    clip = cp.CombEventClip("comb_1", cfg)
    run(clip, clock, DT, alarm_on=True, just_on=True)
    run(clip, clock, 40, alarm_on=True)
    assert clips(cfg) == []
