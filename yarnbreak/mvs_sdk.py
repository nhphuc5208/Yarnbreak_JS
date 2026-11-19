#!/usr/bin/env python3
"""
mvs_sdk.py — lớp dùng chung Hikrobot MVS SDK cho NHIỀU camera trong MỘT tiến trình.

Nguyên tắc:
  • Import SDK theo `mvs_sdk_path` (Jetson: /opt/MVS/Samples/aarch64/Python/MvImport).
  • Gọi MV_CC_Initialize() ĐÚNG MỘT LẦN cho cả tiến trình (đếm tham chiếu).
  • Gọi MV_CC_Finalize() CHỈ khi tắt chương trình — KHÔNG gọi khi một camera
    reconnect/stop (sẽ phá hỏng các camera khác đang mở).
  • Luôn chọn camera THEO SERIAL (bắt buộc). Serial rỗng → báo lỗi rõ ràng,
    tuyệt đối không "lấy camera đầu tiên".
  • Phát hiện SDK có hàm mới (Initialize / ConvertPixelTypeEx / MVCC_INTVALUE_EX)
    hay không; nếu thiếu thì để caller fallback sang API cũ.

Module này KHÔNG mở/không cấu hình camera — chỉ lo import SDK, init/finalize,
và dò/chọn thiết bị theo serial. Việc mở, set ROI/exposure… do grabber làm.
"""
from __future__ import annotations

import logging
import sys
import threading
from ctypes import POINTER, cast
from typing import List, Optional

logger = logging.getLogger("YarnBreak.MvsSdk")

# ── Trạng thái module ─────────────────────────────────────────────────────────
_lock = threading.Lock()
_mvs = None                 # module MvCameraControl_class đã import
_init_count = 0             # đếm tham chiếu init
_sdk_path: Optional[str] = None

# Cờ năng lực (điền sau khi import)
HAS_INITIALIZE = False
HAS_CONVERT_EX = False
HAS_INTVALUE_EX = False


def load_sdk(mvs_sdk_path: str):
    """Import module SDK từ đường dẫn cho trước (idempotent). Trả về module."""
    global _mvs, _sdk_path, HAS_INITIALIZE, HAS_CONVERT_EX, HAS_INTVALUE_EX
    with _lock:
        if _mvs is not None:
            return _mvs
        if mvs_sdk_path and mvs_sdk_path not in sys.path:
            sys.path.append(mvs_sdk_path)
        _sdk_path = mvs_sdk_path
        import MvCameraControl_class as mvs  # noqa: E402
        _mvs = mvs
        HAS_INITIALIZE = hasattr(mvs.MvCamera, "MV_CC_Initialize")
        HAS_CONVERT_EX = hasattr(mvs.MvCamera, "MV_CC_ConvertPixelTypeEx")
        HAS_INTVALUE_EX = hasattr(mvs, "MVCC_INTVALUE_EX")
        logger.info(
            "MVS SDK nạp từ %s | Initialize=%s ConvertPixelTypeEx=%s INTVALUE_EX=%s",
            mvs_sdk_path, HAS_INITIALIZE, HAS_CONVERT_EX, HAS_INTVALUE_EX,
        )
        return _mvs


def get_sdk():
    """Trả về module SDK đã nạp (phải gọi load_sdk trước)."""
    if _mvs is None:
        raise RuntimeError("MVS SDK chưa được nạp — gọi load_sdk(path) trước.")
    return _mvs


def sdk_init() -> None:
    """Khởi tạo SDK cho tiến trình (đếm tham chiếu). An toàn gọi nhiều lần.
    Chỉ thực sự gọi MV_CC_Initialize ở lần đầu, nếu SDK có hàm đó."""
    global _init_count
    mvs = get_sdk()
    with _lock:
        if _init_count == 0 and HAS_INITIALIZE:
            ret = mvs.MvCamera.MV_CC_Initialize()
            if ret != 0:
                raise RuntimeError(f"MV_CC_Initialize lỗi 0x{ret:x}")
            logger.info("MV_CC_Initialize() OK")
        _init_count += 1


def sdk_finalize(force: bool = False) -> None:
    """Giảm đếm tham chiếu; chỉ gọi MV_CC_Finalize khi về 0 (hoặc force=True
    lúc tắt hẳn chương trình)."""
    global _init_count
    mvs = get_sdk()
    with _lock:
        if _init_count > 0:
            _init_count -= 1
        if (force or _init_count == 0) and HAS_INITIALIZE:
            try:
                mvs.MvCamera.MV_CC_Finalize()
                logger.info("MV_CC_Finalize() OK")
            except Exception as exc:
                logger.warning(f"MV_CC_Finalize lỗi: {exc}")
            _init_count = 0


# ── Dò / chọn thiết bị ──────────────────────────────────────────────────────
def _ip_to_str(v: int) -> str:
    return "%d.%d.%d.%d" % ((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)


def _decode(buf) -> str:
    try:
        return bytes(buf).split(b"\x00")[0].decode(errors="ignore")
    except Exception:
        return ""


def enumerate_cameras() -> List[dict]:
    """Liệt kê mọi camera GigE + USB. Trả về list dict:
    {index, tlayer, model, serial, ip, info(raw)}."""
    mvs = get_sdk()
    dev_list = mvs.MV_CC_DEVICE_INFO_LIST()
    tlayer = mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE
    ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
    if ret != 0:
        raise RuntimeError(f"MV_CC_EnumDevices lỗi 0x{ret:x}")
    out = []
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == mvs.MV_GIGE_DEVICE:
            g = info.SpecialInfo.stGigEInfo
            out.append({
                "index": i, "tlayer": "GIGE",
                "model": _decode(g.chModelName),
                "serial": _decode(g.chSerialNumber),
                "ip": _ip_to_str(g.nCurrentIp),
                "info": info,
            })
        elif info.nTLayerType == mvs.MV_USB_DEVICE:
            u = info.SpecialInfo.stUsb3VInfo
            out.append({
                "index": i, "tlayer": "USB",
                "model": _decode(u.chModelName),
                "serial": _decode(u.chSerialNumber),
                "ip": "",
                "info": info,
            })
    return out


def find_by_serial(serial: str):
    """Tìm device info theo serial (bắt buộc). Trả về đối tượng MV_CC_DEVICE_INFO.
    Serial rỗng → lỗi. Không tìm thấy → lỗi (có liệt kê serial đang có)."""
    if not serial or not str(serial).strip():
        raise ValueError(
            "mvs_serial rỗng — PHẢI chỉ định serial camera, không được lấy "
            "camera đầu tiên."
        )
    serial = str(serial).strip()
    cams = enumerate_cameras()
    for c in cams:
        if c["serial"] == serial:
            logger.info("Chọn camera serial=%s model=%s ip=%s (%s)",
                        c["serial"], c["model"], c["ip"], c["tlayer"])
            return c["info"]
    have = ", ".join(f"{c['serial']}({c['model']})" for c in cams) or "(không có camera)"
    raise RuntimeError(f"Không tìm thấy camera serial '{serial}'. Đang thấy: {have}")


if __name__ == "__main__":
    # Tiện ích dòng lệnh: liệt kê camera. python3 mvs_sdk.py [sdk_path]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/MVS/Samples/aarch64/Python/MvImport"
    load_sdk(path)
    for c in enumerate_cameras():
        print(f"[{c['tlayer']}] model={c['model']} serial={c['serial']} ip={c['ip']}")
