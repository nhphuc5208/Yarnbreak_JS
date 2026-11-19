"""Comb alarm rules (CombAlertEngine) — pure logic, no camera or model.

Engine defaults:
  Rule A (trend): 5 consecutive hits, first > 0.6, strictly increasing
  Rule B (high) : 3 consecutive hits, all > 0.8
  Off           : 5 consecutive misses
A single miss clears the window. config.yaml overrides these per site, so the
last tests check that the shipped config is what the pipeline actually uses.
"""
from pathlib import Path

import pytest
import yaml

from yarnbreak.comb_pipeline import CombAlertEngine, CombPipeline

ROOT = Path(__file__).resolve().parent.parent
MISS = None


def feed(engine, seq):
    """Return the alert state after each update."""
    return [engine.update(c)[0] for c in seq]


# ── Rule A: rising trend ──────────────────────────────────────────────────────
def test_rule_a_fires_on_fifth_rising_hit():
    # Values stay <= 0.8 so rule B cannot be the one firing.
    assert feed(CombAlertEngine(), [0.61, 0.65, 0.70, 0.75, 0.78]) == [False] * 4 + [True]


def test_rule_a_needs_first_hit_above_threshold():
    assert not any(feed(CombAlertEngine(), [0.60, 0.65, 0.70, 0.75, 0.78]))


def test_rule_a_needs_strictly_increasing():
    assert not any(feed(CombAlertEngine(), [0.61, 0.65, 0.65, 0.70, 0.75]))


def test_rule_a_window_is_reset_by_a_miss():
    seq = [0.61, 0.65, 0.70, MISS, 0.72, 0.74]
    assert not any(feed(CombAlertEngine(), seq))


# ── Rule B: high confidence ───────────────────────────────────────────────────
def test_rule_b_fires_on_third_high_hit():
    assert feed(CombAlertEngine(), [0.85, 0.90, 0.81]) == [False, False, True]


def test_rule_b_threshold_is_strict():
    assert not any(feed(CombAlertEngine(), [0.80, 0.80, 0.80]))


def test_rule_b_needs_consecutive_hits():
    assert not any(feed(CombAlertEngine(), [0.9, 0.9, MISS, 0.9, 0.9]))


def test_rule_b_uses_last_hits_only():
    assert feed(CombAlertEngine(), [0.5, 0.9, 0.9, 0.9])[-1] is True


# ── switching off ─────────────────────────────────────────────────────────────
def test_alarm_turns_off_after_five_misses_not_four():
    e = CombAlertEngine()
    feed(e, [0.9, 0.9, 0.9])
    assert feed(e, [MISS] * 4) == [True] * 4
    assert e.update(MISS)[0] is False


def test_hits_keep_alarm_latched():
    e = CombAlertEngine()
    feed(e, [0.9, 0.9, 0.9])
    assert all(feed(e, [0.1, MISS, MISS, 0.2, MISS, MISS, MISS, MISS]))


def test_edge_flags():
    e = CombAlertEngine()
    flags = [e.update(c) for c in [0.9, 0.9, 0.9, 0.9] + [MISS] * 5]
    assert [f[1] for f in flags].count(True) == 1      # just_on once
    assert [f[2] for f in flags].count(True) == 1      # just_off once
    assert flags[2] == (True, True, False)
    assert flags[-1] == (False, False, True)


def test_reset_clears_state():
    e = CombAlertEngine()
    feed(e, [0.9, 0.9, 0.9])
    e.reset()
    assert e.alert is False
    assert e.update(0.9)[0] is False                    # window starts from scratch


# ── config.yaml is the single source of truth ─────────────────────────────────
@pytest.fixture(scope="module")
def comb_cfg():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["comb"]


def test_pipeline_uses_alert_section_from_config(comb_cfg):
    kw = CombPipeline(comb_cfg, sdk_path="")._alert_kw
    for key, value in comb_cfg["alert"].items():
        assert kw[key] == value


def test_shipped_config_rule(comb_cfg):
    """Behaviour with the thresholds actually deployed (config.yaml)."""
    a = comb_cfg["alert"]
    e = CombAlertEngine(**CombPipeline(comb_cfg, sdk_path="")._alert_kw)
    above = a["high_conf"] + 0.01
    states = feed(e, [above] * a["high_window"])
    assert states == [False] * (a["high_window"] - 1) + [True]


def test_comb_alarm_outputs_red_and_yellow(comb_cfg):
    assert CombPipeline(comb_cfg, sdk_path="").alarm_outputs == [1, 3]
