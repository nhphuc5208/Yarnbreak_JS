"""FPS meter and Jetson thermal guard.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Optional


from yarnbreak.alerts import AlertNotifier
from yarnbreak.config import AppConfig

logger = logging.getLogger("YarnBreak")


# ══════════════════════════════════════════════════════════════════════════════
#  FPS METER
# ══════════════════════════════════════════════════════════════════════════════

class FPSMeter:
    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self._ema = 0.0
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        raw = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now
        self._ema = self.alpha * raw + (1 - self.alpha) * self._ema if self._ema > 0 else raw
        return self._ema


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD RENDERER
#  Carries all drawing logic from the original Windows code, fully refactored
#  into a class to eliminate global state.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  THERMAL GUARD  (CPU/GPU over-temperature pause)
# ══════════════════════════════════════════════════════════════════════════════

class ThermalGuard:
    """Background monitor that pauses the pipeline when the SoC runs too hot.

    Reads the CPU and GPU thermal zones from sysfs. When either reaches
    cfg.temp_limit_c, it sets `pause_evt` — the grabber and detector then idle,
    dropping GPU/CPU load so the chip cools — waits cfg.cooldown_s, clears the
    event so work resumes, and keeps monitoring (re-pausing if still hot).

    Structure/principle of the detection pipeline is untouched: this only gates
    when the existing threads are allowed to run.
    """

    THERMAL_DIR = "/sys/devices/virtual/thermal"

    def __init__(self, cfg: AppConfig, pause_evt: threading.Event, stop: threading.Event,
                 notifier: "Optional[AlertNotifier]" = None):
        self.cfg = cfg
        self.pause = pause_evt
        self.stop = stop
        self._notifier = notifier
        self._thread: Optional[threading.Thread] = None
        self._cpu_path = self._find_zone("cpu-thermal")
        self._gpu_path = self._find_zone("gpu-thermal")

    def _find_zone(self, ztype: str) -> Optional[str]:
        for zd in sorted(glob.glob(os.path.join(self.THERMAL_DIR, "thermal_zone*"))):
            try:
                with open(os.path.join(zd, "type")) as f:
                    if f.read().strip() == ztype:
                        return os.path.join(zd, "temp")
            except OSError:
                continue
        return None

    @staticmethod
    def _read_c(path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        try:
            with open(path) as f:
                s = f.read().strip()
            return int(s) / 1000.0 if s else None
        except (OSError, ValueError):
            return None

    def available(self) -> bool:
        return self._cpu_path is not None or self._gpu_path is not None

    def start(self):
        if not self.cfg.thermal_guard:
            return
        if not self.available():
            logger.warning("Thermal guard requested but no CPU/GPU thermal zone found — disabled.")
            return
        logger.info(
            f"Thermal guard ON: limit={self.cfg.temp_limit_c:.0f}°C  "
            f"cooldown={self.cfg.cooldown_s/60:.0f}min  check every {self.cfg.thermal_check_s:.0f}s"
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="ThermalGuard")
        self._thread.start()

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _interruptible_sleep(self, seconds: float):
        """Sleep in 1 s steps so a stop request ends the wait promptly."""
        end = time.perf_counter() + seconds
        while not self.stop.is_set() and time.perf_counter() < end:
            time.sleep(min(1.0, end - time.perf_counter()))

    def _run(self):
        limit = self.cfg.temp_limit_c
        while not self.stop.is_set():
            cpu = self._read_c(self._cpu_path)
            gpu = self._read_c(self._gpu_path)
            hot = max([t for t in (cpu, gpu) if t is not None], default=0.0)

            if hot >= limit:
                logger.warning(
                    f"🔥 OVER-TEMP: cpu={cpu}°C gpu={gpu}°C ≥ {limit:.0f}°C — "
                    f"pausing pipeline for {self.cfg.cooldown_s/60:.0f} min to cool down."
                )
                self.pause.set()
                # Force the alert output OFF so a latched siren/lamp can't stay
                # asserted through the cooldown. The detector clears its alert
                # latch on pause, so a still-present break re-triggers on resume.
                if self._notifier is not None:
                    # Nhả CH1; CH2/CH3 do _sync_pause bật theo trạng thái paused.
                    self._notifier.release_stop_output("thermal pause")
                self._interruptible_sleep(self.cfg.cooldown_s)
                self.pause.clear()
                if not self.stop.is_set():
                    c2, g2 = self._read_c(self._cpu_path), self._read_c(self._gpu_path)
                    logger.info(f"✓ Cooldown done — resuming. cpu={c2}°C gpu={g2}°C")
                continue

            self._interruptible_sleep(self.cfg.thermal_check_s)
