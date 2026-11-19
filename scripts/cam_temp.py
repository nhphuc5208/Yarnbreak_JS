#!/usr/bin/env python3
"""Đọc nhiệt độ camera Hikrobot MVS qua GenICam feature DeviceTemperature.
Chạy khi KHÔNG có main.py đang giữ camera (MVS mở exclusive).
    python3 scripts/cam_temp.py
"""
import sys, os

sys.path.append(os.environ.get("MVS_SDK_PATH",
                                "/opt/MVS/Samples/aarch64/Python/MvImport"))
from ctypes import cast, POINTER, byref
from MvCameraControl_class import *  # noqa
import MvCameraControl_class as mvs


def dev_serial(info):
    if info.nTLayerType == mvs.MV_GIGE_DEVICE:
        chars = info.SpecialInfo.stGigEInfo.chSerialNumber
    else:
        chars = info.SpecialInfo.stUsb3VInfo.chSerialNumber
    return "".join(chr(c) for c in chars if c)


def main():
    mvs.MvCamera.MV_CC_Initialize()
    dl = mvs.MV_CC_DEVICE_INFO_LIST()
    ret = mvs.MvCamera.MV_CC_EnumDevices(mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE, dl)
    if ret != 0 or dl.nDeviceNum == 0:
        print("Khong tim thay camera MVS (dang bi main.py giu? ret=%s)" % ret)
        return 1

    info = cast(dl.pDeviceInfo[0], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
    serial = dev_serial(info)
    cam = mvs.MvCamera()
    if cam.MV_CC_CreateHandle(info) != 0:
        print("CreateHandle failed"); return 1
    if cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0) != 0:
        print("OpenDevice failed — camera dang bi chuong trinh khac giu?"); return 1

    print(f"Camera serial = {serial}")
    # Thu tung nguon nhiet neu ho tro selector; neu khong co selector thi doc thang.
    read_any = False
    for sel in ("Sensor", "Mainboard", "Junction", ""):
        if sel:
            if cam.MV_CC_SetEnumValueByString("DeviceTemperatureSelector", sel) != 0:
                continue
        val = mvs.MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("DeviceTemperature", val) == 0:
            label = sel or "Device"
            print(f"  {label:10s}: {val.fCurValue:.1f} °C")
            read_any = True
    if not read_any:
        print("  Camera khong expose feature DeviceTemperature.")

    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    mvs.MvCamera.MV_CC_Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
