#!/usr/bin/env python3
"""Ghi log nhiet do / dien ap / dong dien cua Jetson Orin Nano ra file CSV.

Doc truc tiep tu sysfs:
  - Nhiet do: /sys/devices/virtual/thermal/thermal_zone*  (don vi: milli-degC)
  - Dien ap/dong dien: INA3221 hwmon  (in*_input mV, curr*_input mA)

Cach dung:
  python3 scripts/log_jetson_stats.py                       # log moi 1s ra jetson_stats.csv
  python3 scripts/log_jetson_stats.py -i 0.5 -o out.csv     # chu ky 0.5s, file out.csv
  python3 scripts/log_jetson_stats.py -d 60                 # chay 60 giay roi dung
  Ctrl-C de dung bat ky luc nao.
"""
import argparse
import glob
import os
import sys
import time
from datetime import datetime

THERMAL_DIR = "/sys/devices/virtual/thermal"


def read_int(path):
    """Doc 1 file sysfs tra ve int, hoac None neu loi/rong."""
    try:
        with open(path) as f:
            s = f.read().strip()
        return int(s) if s else None
    except (OSError, ValueError, TypeError):
        return None


def discover_thermal_zones():
    """Tra ve list (ten_zone, duong_dan_temp) cho cac zone co the doc duoc."""
    zones = []
    for zd in sorted(glob.glob(os.path.join(THERMAL_DIR, "thermal_zone*"))):
        ztype = read_text(os.path.join(zd, "type"))
        if ztype is None:
            continue
        temp_path = os.path.join(zd, "temp")
        if read_int(temp_path) is not None:  # bo qua zone rong (cv*-thermal)
            zones.append((ztype, temp_path))
    return zones


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, TypeError):
        return None


def find_ina3221():
    """Tim hwmon co name=ina3221, tra ve list (label, in_path, curr_path)."""
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if read_text(os.path.join(hw, "name")) == "ina3221":
            chans = []
            for i in range(1, 8):
                in_path = os.path.join(hw, f"in{i}_input")
                curr_path = os.path.join(hw, f"curr{i}_input")
                label_path = os.path.join(hw, f"in{i}_label")
                # chi giu rail that: co ca dien ap, dong dien va nhan ten.
                # bo cac kenh "sum of shunt voltages" (in4..in7 khong co curr/label)
                if not (os.path.exists(in_path) and os.path.exists(curr_path)
                        and os.path.exists(label_path)):
                    continue
                label = read_text(label_path) or f"ch{i}"
                chans.append((label, in_path, curr_path))
            return chans
    return []


def build_header(zones, chans):
    cols = ["timestamp", "elapsed_s"]
    for ztype, _ in zones:
        cols.append(f"temp_{ztype}_C")
    for label, _, _ in chans:
        cols.append(f"{label}_mV")
        cols.append(f"{label}_mA")
        cols.append(f"{label}_mW")
    if chans:
        cols.append("power_total_W")  # tong cong suat board (Watt)
    return cols


def sample(zones, chans, t0):
    row = [datetime.now().isoformat(timespec="milliseconds"),
           f"{time.time() - t0:.3f}"]
    for _, temp_path in zones:
        t = read_int(temp_path)
        row.append("" if t is None else f"{t / 1000.0:.3f}")
    rail_mw = {}
    for label, in_path, curr_path in chans:
        mv = read_int(in_path)
        ma = read_int(curr_path)
        row.append("" if mv is None else str(mv))
        row.append("" if ma is None else str(ma))
        if mv is None or ma is None:
            row.append("")
        else:
            mw = mv * ma / 1000.0
            rail_mw[label] = mw
            row.append(f"{mw:.1f}")  # mW
    if chans:
        # tong cong suat: uu tien rail VDD_IN (tong nguon vao board),
        # neu khong co thi cong cac rail lai
        if "VDD_IN" in rail_mw:
            total_mw = rail_mw["VDD_IN"]
        elif rail_mw:
            total_mw = sum(rail_mw.values())
        else:
            total_mw = None
        row.append("" if total_mw is None else f"{total_mw / 1000.0:.3f}")  # W
    return row


def main():
    ap = argparse.ArgumentParser(description="Log nhiet do/dien ap/dong dien Jetson ra CSV")
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="Chu ky lay mau (giay), mac dinh 1.0")
    ap.add_argument("-o", "--output", default="jetson_stats.csv",
                    help="File CSV xuat ra, mac dinh jetson_stats.csv")
    ap.add_argument("-d", "--duration", type=float, default=None,
                    help="Tong thoi gian chay (giay); mac dinh chay vo han")
    args = ap.parse_args()

    zones = discover_thermal_zones()
    chans = find_ina3221()
    if not zones and not chans:
        sys.exit("Khong tim thay sensor nao (thermal/ina3221).")

    header = build_header(zones, chans)
    t0 = time.time()
    n = 0

    print(f"Ghi log -> {args.output} | chu ky {args.interval}s | "
          f"{len(zones)} thermal zone, {len(chans)} kenh INA. Ctrl-C de dung.")

    with open(args.output, "w", buffering=1) as f:
        f.write(",".join(header) + "\n")
        try:
            while True:
                row = sample(zones, chans, t0)
                f.write(",".join(row) + "\n")
                # chong mat dien dot ngot: ep ghi vat ly xuong o dia ngay
                f.flush()
                os.fsync(f.fileno())
                n += 1
                if args.duration is not None and (time.time() - t0) >= args.duration:
                    break
                # ngu phan con lai cua chu ky de bu thoi gian doc sensor
                next_t = t0 + n * args.interval
                sleep_s = next_t - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            pass

    print(f"\nDa ghi {n} dong vao {args.output} ({time.time() - t0:.1f}s).")


if __name__ == "__main__":
    main()
