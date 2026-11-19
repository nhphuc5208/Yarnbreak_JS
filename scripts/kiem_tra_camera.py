#!/usr/bin/env python3
"""kiem_tra_camera.py — kiem tra camera luoc comb_2 (MV-CU060-10GC) tren Jetson.

Dung CHINH lop CombCameraStream cua du an (khong viet lai duong doc rieng) nen
ket qua phan anh dung cai chuong trinh that se lam: ROI phan cung, exposure,
gain, packet size, do fps.

Dung:
    sudo systemctl stop yarnbreak.service      # nha camera truoc
    export MVCAM_COMMON_RUNENV=/opt/MVS/lib
    export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
    source venv/bin/activate
    python3 scripts/kiem_tra_camera.py                  # comb_2, 15s
    python3 scripts/kiem_tra_camera.py --cam comb_1
    python3 scripts/kiem_tra_camera.py --giay 30 --luu /tmp/comb2.jpg
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Chay duoc tu bat ky thu muc nao: tu suy ra goc du an (thu muc cha cua
# scripts/) roi them vao sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--cam", default="comb_2", help="id camera trong config.yaml")
    ap.add_argument("--sdk", default="",
                    help="duong dan MvImport (de trong = lay mvs_sdk_path trong config)")
    ap.add_argument("--giay", type=float, default=15.0, help="thoi gian do (giay)")
    ap.add_argument("--luu", default="", help="luu 1 anh JPEG ra duong dan nay")
    ap.add_argument("--packet", type=int, default=-1,
                    help="ep packet size (0 = de SDK tu chon). Dung khi cong mang "
                         "chua bat jumbo frame; -1 = lay theo config")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import cv2  # noqa: E402
    import yaml  # noqa: E402
    from yarnbreak import mvs_sdk  # noqa: E402
    from yarnbreak.comb_pipeline import CombCameraStream  # noqa: E402

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sdk_path = args.sdk or cfg.get(
        "mvs_sdk_path", "/opt/MVS/Samples/aarch64/Python/MvImport")
    cams = {c.get("id"): c for c in (cfg.get("comb", {}).get("cameras") or [])}
    if args.cam not in cams:
        print(f"LOI: khong co camera id '{args.cam}' trong {args.config}. "
              f"Dang co: {list(cams)}")
        return 2
    cam_cfg = dict(cams[args.cam])
    if args.packet >= 0:
        cam_cfg["packet_size"] = args.packet

    print("=" * 72)
    print(f"Camera          : {args.cam} ({cam_cfg.get('name')})")
    print(f"  mvs_serial    : {cam_cfg.get('mvs_serial')}")
    print(f"  hw_roi        : {cam_cfg.get('hw_roi')}")
    print(f"  exposure/gain : {cam_cfg.get('exposure_us')} us / {cam_cfg.get('gain')} dB")
    print(f"  fps / packet  : {cam_cfg.get('fps')} / {cam_cfg.get('packet_size')}")
    print("=" * 72)

    # 1. SDK co thay camera nao khong
    mvs_sdk.load_sdk(sdk_path)
    mvs_sdk.sdk_init()
    found = mvs_sdk.enumerate_cameras()
    print("\n--- Camera SDK dang thay ---")
    for c in found:
        print(f"  [{c['tlayer']}] model={c['model']:16s} serial={c['serial']:12s} ip={c['ip']}")
    if not found:
        print("  (KHONG thay camera nao — kiem tra day mang / adapter / IP cong mang)")

    want = str(cam_cfg.get("mvs_serial") or "").strip()
    if want and want not in [c["serial"] for c in found]:
        print(f"\nLOI: khong thay serial {want}. Dung tiep.")
        return 3

    # 2. Mo that bang lop cua du an
    print(f"\n--- Mo camera va do {args.giay:.0f}s ---")
    st = CombCameraStream(cam_cfg, sdk_path)
    st.start()

    t0 = time.time()
    last_seq, n, frame = -1, 0, None
    while time.time() - t0 < args.giay:
        f, seq = st.read()
        if f is not None and seq != last_seq:
            last_seq, n, frame = seq, n + 1, f
        time.sleep(0.02)
    do_fps = st.fps           # fps do boi chinh _tick_fps() cua chuong trinh
    connected = st.connected
    st.stop()

    print("\n--- Ket qua ---")
    print(f"  connected            : {connected}")
    print(f"  so frame nhan duoc   : {n} trong {args.giay:.0f}s "
          f"(~{n / max(args.giay, 1e-6):.1f} fps do o day)")
    print(f"  st.fps (_tick_fps)   : {do_fps:.1f}  <- so hien tren web")
    if frame is not None:
        h, w = frame.shape[:2]
        sang = float(frame.mean())
        if sang < 5:
            nhan = "GAN NHU DEN HOAN TOAN"
        elif sang < 40:
            nhan = "TOI - tang gain/exposure"
        elif sang > 200:
            nhan = "CHAY - giam gain/exposure"
        else:
            nhan = "on"
        print(f"  kich thuoc frame     : {w}x{h}")
        print(f"  do sang trung binh   : {sang:.1f} / 255 ({nhan})")
        if sang < 5:
            print(f"  >> Anh den nhung VAN co frame ve: gan nhu chac chan do PACKET SIZE "
                  f"({cam_cfg.get('packet_size')}) LON HON MTU cua cong mang.")
            print(f"     GigE Vision gui payload bang UDP: goi lon hon MTU bi rot sach "
                  f"ma KHONG bao loi -> frame rong.")
            print(f"     Cach xu ly: bat jumbo frame (MTU 9000) cho cong mang "
                  f"(scripts/mang_comb2.sh),")
            print(f"     hoac chay lai script nay voi  --packet 0  de kiem chung.")
        if args.luu:
            cv2.imwrite(args.luu, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  da luu anh           : {args.luu}")
    else:
        print("  KHONG lay duoc frame nao")

    hw = cam_cfg.get("hw_roi")
    if frame is not None and hw:
        if (frame.shape[1], frame.shape[0]) != (int(hw[2]), int(hw[3])):
            print(f"  CANH BAO: frame {frame.shape[1]}x{frame.shape[0]} khac hw_roi "
                  f"{hw[2]}x{hw[3]} — camera tu choi ROI (kiem tra buoc Width 8 / Height 2)")
    if frame is not None and cam_cfg.get("fps") and do_fps:
        dat = do_fps / float(cam_cfg["fps"])
        if dat < 0.8:
            print(f"  CANH BAO: chi dat {dat * 100:.0f}% fps dat trong config — "
                  f"kiem tra MTU/packet size, bang thong USB->GigE, hoac exposure qua dai")

    mvs_sdk.sdk_finalize(force=True)
    return 0 if frame is not None else 1


if __name__ == "__main__":
    sys.exit(main())
