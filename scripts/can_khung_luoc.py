#!/usr/bin/env python3
"""can_khung_luoc.py — can hw_roi cua camera luoc sao cho KHUNG LUOC chiem tron
khung hinh, giong camera luoc 1 (comb_1).

Vi sao phai can:
  - Model yarnbreakcomb duoc train tren anh ma khung luoc CHIEM GAN HET khung
    hinh (comb_1: 4024x1012, ti le ~4:1). Neu de camera nhin ca nha xuong, soi
    luoc chi con vai pixel -> bo sot hoac bao nham.
  - Cat bang hw_roi (ROI PHAN CUNG) tot hon cat bang phan mem: camera chi truyen
    phan anh can -> nhe bang thong GigE, nhe GPU, fps cao hon.
  - Muon cat them hinh KHONG phai chu nhat thi dung them `roi:` (da giac toa do
    0..1 tinh TRONG hw_roi). comb_1 de da giac full vi hw_roi da om dung luoc.

Cach dung (tren Jetson, da dung yarnbreak.service de nha camera):
  1) Chup anh TOAN CAM BIEN co luoi toa do:
        python3 scripts/can_khung_luoc.py --cam comb_2 --xem
     -> mo /tmp/comb_2_full.jpg, doc toa do goc trai-tren va phai-duoi cua
        khung luoc (don vi: pixel tren cam bien day du).

  2) Thu vung vua doc:
        python3 ... --cam comb_2 --vung 0,700,3072,1500
     -> in ra hw_roi hop le (da lam tron theo buoc cua camera) + luu anh
        /tmp/comb_2_roi.jpg de kiem tra khung da om dung luoc chua.

  3) Vua y thi ghi vao config.yaml:
        python3 ... --cam comb_2 --vung 0,700,3072,1500 --ghi
     (tu dong backup config.yaml.bak truoc khi sua)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from ctypes import byref, memset, sizeof

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--cam", default="comb_2")
    ap.add_argument("--sdk", default="",
                    help="duong dan MvImport (de trong = lay mvs_sdk_path trong config)")
    ap.add_argument("--xem", action="store_true",
                    help="chup 1 anh TOAN CAM BIEN kem luoi toa do")
    ap.add_argument("--vung", default="",
                    help="x1,y1,x2,y2 (pixel tren cam bien day du) cua khung luoc")
    ap.add_argument("--ghi", action="store_true",
                    help="ghi hw_roi vua tinh vao config.yaml")
    ap.add_argument("--ti-le", default="comb_1", dest="ti_le",
                    help="ep khung ve dung ti le nay: 'comb_1' (mac dinh, lay tu "
                         "hw_roi cua comb_1 trong config), mot so nhu 3.976, hoac "
                         "'off' de giu nguyen vung da nhap")
    ap.add_argument("--thumuc", default="/tmp", help="noi luu anh")
    args = ap.parse_args()

    if not args.xem and not args.vung:
        ap.error("chon --xem (chup toan cam bien) hoac --vung x1,y1,x2,y2")

    import cv2
    import numpy as np
    import yaml
    from yarnbreak import mvs_sdk

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sdk_path = args.sdk or cfg.get(
        "mvs_sdk_path", "/opt/MVS/Samples/aarch64/Python/MvImport")
    cams = {c.get("id"): c for c in (cfg.get("comb", {}).get("cameras") or [])}
    if args.cam not in cams:
        print(f"LOI: khong co camera '{args.cam}'. Dang co: {list(cams)}")
        return 2
    cam_cfg = cams[args.cam]

    # Ti le khung can giu (rong/cao). Mac dinh = dung ti le cua camera luoc 1 vi
    # model yarnbreakcomb train tren khung do; khung vuong hon lam sai dang soi.
    ti_le = 0.0
    if str(args.ti_le).lower() not in ("off", "0", "none", ""):
        if str(args.ti_le) == "comb_1":
            hw1 = (cams.get("comb_1") or {}).get("hw_roi")
            if hw1 and len(hw1) == 4 and int(hw1[3]):
                ti_le = float(hw1[2]) / float(hw1[3])
            else:
                ti_le = 3.976
        else:
            try:
                ti_le = float(args.ti_le)
            except ValueError:
                print(f"LOI: --ti-le khong hieu '{args.ti_le}'")
                return 2

    serial = str(cam_cfg.get("mvs_serial") or "").strip()
    if not serial:
        print(f"LOI: camera '{args.cam}' chua co mvs_serial trong config.")
        return 2

    mvs = mvs_sdk.load_sdk(sdk_path)
    mvs_sdk.sdk_init()
    info = mvs_sdk.find_by_serial(serial)
    cam = mvs.MvCamera()
    if cam.MV_CC_CreateHandle(info) != 0:
        print("LOI: CreateHandle")
        return 3
    ret = cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"LOI: OpenDevice 0x{ret:x} — camera dang bi chuong trinh khac giu? "
              "(sudo systemctl stop yarnbreak.service)")
        cam.MV_CC_DestroyHandle()
        return 3

    def ri(key):
        v = mvs.MVCC_INTVALUE_EX()
        memset(byref(v), 0, sizeof(v))
        if cam.MV_CC_GetIntValueEx(key, v) != 0:
            return None
        return int(v.nCurValue), int(v.nMin), int(v.nMax), max(1, int(v.nInc))

    sensor_w = ri("WidthMax")[0]
    sensor_h = ri("HeightMax")[0]
    w_inc = ri("Width")[3]
    h_inc = ri("Height")[3]
    ox_inc = ri("OffsetX")[3]
    oy_inc = ri("OffsetY")[3]
    print(f"Cam bien {sensor_w}x{sensor_h} | buoc Width={w_inc} Height={h_inc} "
          f"OffsetX={ox_inc} OffsetY={oy_inc}")

    def apply_roi(ox, oy, rw, rh):
        cam.MV_CC_SetIntValue("OffsetX", 0)
        cam.MV_CC_SetIntValue("OffsetY", 0)
        cam.MV_CC_SetIntValue("Width", int(rw))
        cam.MV_CC_SetIntValue("Height", int(rh))
        cam.MV_CC_SetIntValue("OffsetX", int(ox))
        cam.MV_CC_SetIntValue("OffsetY", int(oy))

    def grab():
        """Tra ve 1 frame BGR (bo vai frame dau cho on dinh phoi sang)."""
        cam.MV_CC_SetEnumValue("TriggerMode", mvs.MV_TRIGGER_MODE_OFF)
        if cam_cfg.get("exposure_us"):
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            cam.MV_CC_SetFloatValue("ExposureTime", float(cam_cfg["exposure_us"]))
        if cam_cfg.get("gain") is not None:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            cam.MV_CC_SetFloatValue("Gain", float(cam_cfg["gain"]))
        cam.MV_CC_SetImageNodeNum(2)
        cam.MV_CC_SetGrabStrategy(mvs.MV_GrabStrategy_LatestImagesOnly)
        if cam.MV_CC_StartGrabbing() != 0:
            return None
        out = None
        st_frame = mvs.MV_FRAME_OUT()
        memset(byref(st_frame), 0, sizeof(st_frame))
        try:
            for _ in range(6):
                if cam.MV_CC_GetImageBuffer(st_frame, 2000) != 0:
                    continue
                fi = st_frame.stFrameInfo
                w, h = fi.nWidth, fi.nHeight
                need = w * h * 3
                buf = (mvs.c_ubyte * need)()
                p = mvs.MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(p), 0, sizeof(p))
                p.nWidth, p.nHeight = w, h
                p.pSrcData = st_frame.pBufAddr
                p.nSrcDataLen = fi.nFrameLen
                p.enSrcPixelType = fi.enPixelType
                p.enDstPixelType = mvs.PixelType_Gvsp_BGR8_Packed
                p.pDstBuffer = buf
                p.nDstBufferSize = need
                ok = cam.MV_CC_ConvertPixelTypeEx(p) == 0
                cam.MV_CC_FreeImageBuffer(st_frame)
                if ok:
                    out = np.frombuffer(buf, dtype=np.uint8,
                                        count=need).reshape(h, w, 3).copy()
        finally:
            cam.MV_CC_StopGrabbing()
        return out

    rc = 0
    try:
        if args.xem:
            apply_roi(0, 0, sensor_w, sensor_h)
            img = grab()
            if img is None:
                print("LOI: khong lay duoc frame")
                return 4
            # Luoi 10% + nhan toa do pixel de doc toa do khung luoc.
            for k in range(1, 10):
                x = sensor_w * k // 10
                y = sensor_h * k // 10
                cv2.line(img, (x, 0), (x, sensor_h), (0, 255, 255), 2)
                cv2.line(img, (0, y), (sensor_w, y), (0, 255, 255), 2)
                cv2.putText(img, str(x), (x + 6, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 3)
                cv2.putText(img, str(y), (8, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 3)
            path = os.path.join(args.thumuc, f"{args.cam}_full.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"\nDa luu anh toan cam bien: {path}")
            print("Mo anh, doc toa do goc TRAI-TREN va PHAI-DUOI cua KHUNG LUOC,")
            print(f"roi chay lai voi:  --cam {args.cam} --vung x1,y1,x2,y2")
            return 0

        # --- che do --vung ---
        try:
            x1, y1, x2, y2 = (int(v) for v in args.vung.replace(" ", "").split(","))
        except Exception:
            print("LOI: --vung phai co dang x1,y1,x2,y2")
            return 2
        x1, x2 = sorted((max(0, x1), min(sensor_w, x2)))
        y1, y2 = sorted((max(0, y1), min(sensor_h, y2)))

        # Lam tron theo buoc cua camera; NO RONG ra ngoai de khong cat mat luoc.
        ox = (x1 // ox_inc) * ox_inc
        oy = (y1 // oy_inc) * oy_inc
        rw = -(-(x2 - ox) // w_inc) * w_inc
        rh = -(-(y2 - oy) // h_inc) * h_inc
        rw = min(rw, ((sensor_w - ox) // w_inc) * w_inc)
        rh = min(rh, ((sensor_h - oy) // h_inc) * h_inc)

        if ti_le > 0:
            # Giu DUNG ti le: tinh lai chieu cao theo be ngang, can giua theo
            # tam vung nguoi dung nhap (chi doi dai len/xuong, khong lam vuong).
            cy = (y1 + y2) / 2.0
            rh2 = int(round(rw / ti_le))
            rh2 = max(h_inc, (rh2 // h_inc) * h_inc)
            if rh2 > sensor_h:                       # qua cao -> thu be ngang
                rh2 = (sensor_h // h_inc) * h_inc
                rw = max(w_inc, (int(rh2 * ti_le) // w_inc) * w_inc)
            oy2 = int(cy - rh2 / 2.0)
            oy2 = max(0, min(sensor_h - rh2, oy2))
            oy2 = (oy2 // oy_inc) * oy_inc
            if ox + rw > sensor_w:
                ox = max(0, ((sensor_w - rw) // ox_inc) * ox_inc)
            if (oy, rh) != (oy2, rh2):
                print(f"EP TI LE {ti_le:.3f}:1 (cua comb_1): "
                      f"[{ox}, {oy}, {rw}, {rh}] -> [{ox}, {oy2}, {rw}, {rh2}]")
            oy, rh = oy2, rh2

        print(f"\nVung yeu cau : x {x1}..{x2}  y {y1}..{y2}")
        print(f"hw_roi hop le: [{ox}, {oy}, {rw}, {rh}]   ({rw}x{rh}, ti le "
              f"{rw / max(rh, 1):.2f}:1)")
        if rw * rh > 0:
            print(f"Giam tai     : con {rw * rh * 100.0 / (sensor_w * sensor_h):.0f}% "
                  f"so pixel so voi toan cam bien")
        if ti_le > 0:
            print(f"Ti le dich   : {ti_le:.3f}:1 (giong camera luoc 1) — dung "
                  f"--ti-le off neu muon giu nguyen vung da nhap")
        elif rw / max(rh, 1) < 2.0:
            print("LUU Y: comb_1 dang o ti le ~4:1. Khung qua vuong so voi anh da "
                  "train -> nen cat det hon theo dung be ngang cua luoc.")

        apply_roi(ox, oy, rw, rh)
        img = grab()
        if img is None:
            print("LOI: camera tu choi ROI nay hoac khong lay duoc frame")
            return 4
        path = os.path.join(args.thumuc, f"{args.cam}_roi.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Anh kiem tra : {path}  (kich thuoc thuc {img.shape[1]}x{img.shape[0]})")
        print(f"Do sang TB   : {float(img.mean()):.1f}/255")

        if args.ghi:
            rc = ghi_config(args.config, args.cam, [ox, oy, rw, rh])
        else:
            print("\nDan vao config.yaml (trong khoi cua camera nay):")
            print(f"      hw_roi: [{ox}, {oy}, {rw}, {rh}]")
            print("Hoac chay lai kem --ghi de tu sua config.yaml.")
    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        mvs_sdk.sdk_finalize(force=True)
    return rc


def ghi_config(path: str, cam_id: str, roi) -> int:
    """Thay DUNG dong hw_roi trong khoi cua camera cam_id, giu nguyen comment."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    i_cam = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*-\s*id:\s*"?' + re.escape(cam_id) + r'"?\s*$', ln):
            i_cam = i
            break
    if i_cam is None:
        print(f"LOI: khong tim thay '- id: {cam_id}' trong {path}")
        return 2

    i_roi = None
    for i in range(i_cam + 1, len(lines)):
        if re.match(r'^\s*-\s*id:', lines[i]):      # sang camera khac
            break
        if re.match(r'^\s*hw_roi:', lines[i]):
            i_roi = i
            break
    if i_roi is None:
        print(f"LOI: khong tim thay dong hw_roi trong khoi {cam_id}")
        return 2

    indent = re.match(r'^(\s*)', lines[i_roi]).group(1)
    cu = lines[i_roi].rstrip("\n")
    moi = f"{indent}hw_roi: [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}]\n"

    with open(path + ".bak", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    lines[i_roi] = moi
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"\nDa sua {path} (backup: {path}.bak)")
    print(f"  cu : {cu.strip()}")
    print(f"  moi: {moi.strip()}")
    print("Khoi dong lai: sudo systemctl restart yarnbreak.service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
