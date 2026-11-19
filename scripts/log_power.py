#!/usr/bin/env python3
"""Log INA3221 power/current data to CSV over time.

Usage:
    python3 scripts/log_power.py                     # log every 1s to outputs/logs/
    python3 scripts/log_power.py --interval 0.5      # 500ms interval
    python3 scripts/log_power.py --out /tmp/power.csv
    python3 scripts/log_power.py --stdout            # print only, no file
"""
import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
CHANNELS = [
    ("VDD_IN",         1),
    ("VDD_CPU_GPU_CV", 2),
    ("VDD_SOC",        3),
]


def find_ina3221() -> Path:
    for d in HWMON.iterdir():
        name_file = d / "name"
        if name_file.exists() and name_file.read_text().strip() == "ina3221":
            return d
    raise RuntimeError("INA3221 hwmon not found under /sys/class/hwmon")


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def sample(hwmon: Path) -> dict:
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
    for label, ch in CHANNELS:
        v_mv  = read_int(hwmon / f"in{ch}_input")   # millivolts
        i_ma  = read_int(hwmon / f"curr{ch}_input") # milliamps
        v = v_mv / 1000 if v_mv is not None else None
        i = i_ma / 1000 if i_ma is not None else None
        p = round(v * i, 3) if (v and i) else None
        row[f"{label}_V"]  = round(v, 3) if v else None
        row[f"{label}_A"]  = round(i, 3) if i else None
        row[f"{label}_W"]  = p
    return row


FIELDS = ["time"] + [
    f"{lbl}_{unit}"
    for lbl, _ in CHANNELS
    for unit in ("V", "A", "W")
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=1.0, help="Sample interval in seconds (default 1.0)")
    p.add_argument("--out",      type=str,   default="",  help="Output CSV path (default: outputs/logs/power_<timestamp>.csv)")
    p.add_argument("--stdout",   action="store_true",     help="Print to stdout only, no file")
    args = p.parse_args()

    hwmon = find_ina3221()

    if args.stdout:
        out_path = None
    else:
        if args.out:
            out_path = Path(args.out)
        else:
            log_dir = Path(__file__).resolve().parent.parent / "outputs" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = log_dir / f"power_{ts}.csv"

    f = open(out_path, "w", newline="") if out_path else None
    writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
    if writer:
        writer.writeheader()
        f.flush()
        print(f"Logging to: {out_path}")

    print(f"{'Time':<23}  {'VDD_IN':>8}  {'CPU_GPU':>8}  {'SOC':>7}")
    print(f"{'':23}  {'V / A / W':>8}  {'V / A / W':>8}  {'V/A/W':>7}")
    print("-" * 70)

    stop = False
    def _sig(*_): nonlocal stop; stop = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        while not stop:
            row = sample(hwmon)
            if writer:
                writer.writerow(row)
                f.flush()

            vdd_in = f"{row['VDD_IN_V']}V {row['VDD_IN_A']}A {row['VDD_IN_W']}W"
            cpu    = f"{row['VDD_CPU_GPU_CV_V']}V {row['VDD_CPU_GPU_CV_A']}A {row['VDD_CPU_GPU_CV_W']}W"
            soc    = f"{row['VDD_SOC_V']}V {row['VDD_SOC_A']}A {row['VDD_SOC_W']}W"
            print(f"{row['time']}  {vdd_in:>28}  {cpu:>28}  {soc:>22}", flush=True)

            time.sleep(args.interval)
    finally:
        if f:
            f.close()
        if out_path and out_path.exists():
            print(f"\nSaved: {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
