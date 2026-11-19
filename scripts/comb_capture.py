#!/usr/bin/env python3
"""
comb_capture.py — Công cụ CHỤP ẢNH MẪU từ camera LƯỢC (Hikrobot MVS).

TÁCH BIỆT HOÀN TOÀN với chương trình gốc: không import main.py / comb_pipeline.py /
web_server.py / state_manager.py, không nạp model AI, không điều khiển relay,
không ghi vào database hay log cảnh báo. Chỉ làm MỘT việc: mở camera lược, cho
người vận hành xem trực tiếp trên web và BẤM CHỤP để thu ảnh huấn luyện.

Chỉ dùng chung 2 thứ với chương trình gốc (đều ở chế độ CHỈ ĐỌC):
  • mvs_sdk.py  — lớp nạp SDK Hikrobot + dò camera theo serial.
  • config.yaml — đọc danh sách camera lược (serial, exposure, gain, hw_roi…)
                  để điền sẵn vào form. KHÔNG bao giờ ghi lại file này.

⚠ CAMERA MỞ Ở CHẾ ĐỘ ĐỘC QUYỀN (MV_ACCESS_Exclusive):
  phải DỪNG chương trình gốc (main.py / dịch vụ yarnbreak) trước khi chạy công
  cụ này, nếu không MV_CC_OpenDevice sẽ báo lỗi vì camera đang bị giữ.
      sudo systemctl stop yarnbreak

Ảnh lưu ra là ẢNH GỐC (nguyên độ phân giải cảm biến, KHÔNG vẽ box/ROI, không
đóng dấu thời gian lên ảnh) → nạp trực tiếp vào bộ dữ liệu để gán nhãn.

Cách chạy:
    export MVCAM_COMMON_RUNENV=/opt/MVS/lib
    export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
    python3 scripts/comb_capture.py                      # → http://<ip-jetson>:8010
    python3 scripts/comb_capture.py --out /mnt/ssd/mau   # đổi nơi lưu ảnh
    python3 scripts/comb_capture.py --port 8011
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Script nằm trong scripts/ → thêm thư mục gốc dự án vào sys.path để import package yarnbreak.
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("CombCapture")

# Giới hạn luồng của OpenCV: máy này chỉ encode JPEG, không cần trải hết nhân CPU.
cv2.setNumThreads(2)


# ══════════════════════════════════════════════════════════════════════════════
#  ĐỌC CONFIG (chỉ để điền sẵn form — không ghi)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_SDK_PATH = "/opt/MVS/Samples/aarch64/Python/MvImport"


def load_cfg(path: str) -> dict:
    """Đọc config.yaml của chương trình gốc. Thiếu file cũng không sao."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Không đọc được %s (%s) → dùng giá trị mặc định.", path, exc)
        return {}


def camera_presets(cfg: dict) -> List[dict]:
    """Lấy danh sách camera lược trong config → preset cho form.

    Lấy CẢ camera đang enabled: false, vì lúc đi lấy mẫu người dùng có thể muốn
    chụp từ camera chưa bật trong dây chuyền.
    """
    out = []
    for cam in (cfg.get("comb") or {}).get("cameras") or []:
        out.append({
            "id": cam.get("id") or "comb",
            "name": cam.get("name") or cam.get("id") or "comb",
            "serial": str(cam.get("mvs_serial") or "").strip(),
            "enabled": bool(cam.get("enabled")),
            "exposure_us": float(cam.get("exposure_us") or 17000),
            "gain": float(cam.get("gain") if cam.get("gain") is not None else 10),
            "fps": float(cam.get("fps") or 8),
            "binning": int(cam.get("binning") or 1),
            "packet_size": int(cam.get("packet_size") or 9000),
            "hw_roi": list(cam.get("hw_roi")) if cam.get("hw_roi") else None,
            "video": cam.get("video") or "",
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  GRABBER CAMERA MVS (bản gọn, chỉ cho việc lấy mẫu)
# ══════════════════════════════════════════════════════════════════════════════
class MvsGrabber:
    """Đọc frame từ 1 camera Hikrobot, luôn giữ frame MỚI NHẤT.

    Khác bản trong comb_pipeline.py ở hai điểm cố ý:
      • KHÔNG tự kết nối lại vô hạn — công cụ lấy mẫu cần báo lỗi ngay cho người
        dùng (ví dụ camera đang bị chương trình gốc giữ) thay vì lặng lẽ retry.
      • Cho phép ĐỔI exposure / gain / fps lúc đang chạy; thay đổi được áp dụng
        trong CHÍNH luồng đọc (SDK không bảo đảm an toàn khi gọi từ luồng khác).
    """

    GET_TIMEOUT_MS = 1000

    def __init__(self, serial: str, settings: dict, sdk_path: str):
        self.serial = str(serial).strip()
        self.settings = dict(settings)
        self._sdk_path = sdk_path

        self.mvs = None
        self.cam = None
        self._dev_info = None

        self.connected = False
        self.error: str = ""
        self.info: Dict = {}
        self.limits: Dict = {}

        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._grabbed = 0
        self._fps_marks = deque(maxlen=30)

        self._lock = threading.Lock()
        self._pending: Dict[str, float] = {}       # tham số chờ áp dụng
        self._pending_lock = threading.Lock()

        self._running = False
        self._ready = threading.Event()            # mở xong (thành công/thất bại)
        self._thread: Optional[threading.Thread] = None

        self._cvt_buf = None
        self._cvt_buf_size = 0
        self._cvt_view = None

    # ── Vòng đời ────────────────────────────────────────────────────────────
    def start(self, timeout: float = 20.0) -> bool:
        """Mở camera và bắt đầu đọc. Trả về True nếu mở được."""
        self._running = True
        self._ready.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name=f"Grab-{self.serial}")
        self._thread.start()
        self._ready.wait(timeout)
        return self.connected

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._close_device()

    def read(self):
        """(frame, seq) — frame là array riêng, an toàn dùng ở luồng khác."""
        with self._lock:
            return self._frame, self._seq

    def stats(self) -> dict:
        with self._lock:
            seq, ts = self._seq, self._ts
        marks = list(self._fps_marks)
        fps = 0.0
        if len(marks) >= 2 and marks[-1] > marks[0]:
            fps = (len(marks) - 1) / (marks[-1] - marks[0])
        return {
            "connected": self.connected,
            "error": self.error,
            "serial": self.serial,
            "frames": seq,
            "age_s": round(time.monotonic() - ts, 2) if ts else None,
            "fps": round(fps, 1),
            "info": self.info,
            "limits": self.limits,
        }

    # ── Đổi tham số lúc đang chạy ───────────────────────────────────────────
    def queue_params(self, **kw):
        """Ghi nhận tham số mới; luồng đọc sẽ áp dụng ở vòng kế tiếp."""
        with self._pending_lock:
            for k, v in kw.items():
                if v is not None:
                    self._pending[k] = float(v)

    def _apply_pending(self):
        with self._pending_lock:
            if not self._pending:
                return
            todo, self._pending = self._pending, {}
        cam = self.cam
        if cam is None:
            return
        try:
            if "exposure_us" in todo:
                cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
                cam.MV_CC_SetFloatValue("ExposureTime", todo["exposure_us"])
                self.settings["exposure_us"] = todo["exposure_us"]
            if "gain" in todo:
                cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
                cam.MV_CC_SetFloatValue("Gain", todo["gain"])
                self.settings["gain"] = todo["gain"]
            if "fps" in todo:
                cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                cam.MV_CC_SetFloatValue("AcquisitionFrameRate", todo["fps"])
                self.settings["fps"] = todo["fps"]
            self._read_info()
        except Exception as exc:
            logger.warning("Áp dụng tham số lỗi: %s", exc)

    # ── Mở / cấu hình ───────────────────────────────────────────────────────
    def _open_device(self):
        from yarnbreak import mvs_sdk
        self.mvs = mvs_sdk.load_sdk(self._sdk_path)
        mvs_sdk.sdk_init()
        mvs = self.mvs

        self._dev_info = mvs_sdk.find_by_serial(self.serial)

        cam = mvs.MvCamera()
        ret = cam.MV_CC_CreateHandle(self._dev_info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle lỗi 0x{ret:x}")
        ret = cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"MV_CC_OpenDevice lỗi 0x{ret:x} — camera đang bị tiến trình khác "
                "giữ. Hãy dừng chương trình gốc (sudo systemctl stop yarnbreak) "
                "hoặc đóng MVS Client rồi thử lại."
            )
        self.cam = cam
        self._apply_settings()

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            self._close_device()
            raise RuntimeError(f"MV_CC_StartGrabbing lỗi 0x{ret:x}")

        self._read_info()
        self._read_limits()
        self.connected = True
        self.error = ""
        logger.info("Đã mở camera %s — %sx%s", self.serial,
                    self.info.get("width"), self.info.get("height"))

    def _apply_settings(self):
        mvs, cam, c = self.mvs, self.cam, self.settings

        def warn(label, ret):
            if ret != 0:
                logger.warning("set %s lỗi 0x%x", label, ret)

        # GigE: packet size không được vượt MTU thật của cổng mạng, nếu không
        # frame bị rách (ảnh đen kèm vệt ngang) mà SDK KHÔNG báo lỗi.
        if self._dev_info.nTLayerType in (mvs.MV_GIGE_DEVICE,
                                          getattr(mvs, "MV_GENTL_GIGE_DEVICE", -1)):
            want = int(c.get("packet_size") or 0)
            if want <= 0:
                try:
                    want = int(cam.MV_CC_GetOptimalPacketSize())
                except Exception:
                    want = 1500
            mtu = self._nic_mtu()
            if mtu and want > mtu:
                logger.warning("packet_size %s > MTU %s → hạ về %s", want, mtu, mtu)
                want = mtu
            iv = mvs.MVCC_INTVALUE_EX()
            mvs.memset(mvs.byref(iv), 0, mvs.sizeof(iv))
            if cam.MV_CC_GetIntValueEx("GevSCPSPacketSize", iv) == 0:
                inc = int(iv.nInc) or 1
                want = max(min(int(want), int(iv.nMax)), int(iv.nMin))
                want = (want // inc) * inc
            if want > 0 and cam.MV_CC_SetIntValue("GevSCPSPacketSize", want) != 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)

        warn("TriggerMode=Off",
             cam.MV_CC_SetEnumValue("TriggerMode", mvs.MV_TRIGGER_MODE_OFF))

        binning = int(c.get("binning") or 1)
        if binning > 1:
            warn("BinningHorizontal", cam.MV_CC_SetIntValue("BinningHorizontal", binning))
            warn("BinningVertical", cam.MV_CC_SetIntValue("BinningVertical", binning))

        # ROI cảm biến: thu nhỏ Width/Height TRƯỚC rồi mới set Offset, ngược lại
        # camera từ chối vì offset+width vượt kích thước hiện tại.
        hw_roi = c.get("hw_roi")
        warn("OffsetX=0", cam.MV_CC_SetIntValue("OffsetX", 0))
        warn("OffsetY=0", cam.MV_CC_SetIntValue("OffsetY", 0))
        if hw_roi:
            ox, oy, rw, rh = [int(v) for v in hw_roi]
            warn("Width", cam.MV_CC_SetIntValue("Width", rw))
            warn("Height", cam.MV_CC_SetIntValue("Height", rh))
            warn("OffsetX", cam.MV_CC_SetIntValue("OffsetX", ox))
            warn("OffsetY", cam.MV_CC_SetIntValue("OffsetY", oy))
        else:
            # Full sensor: nới Width/Height về max của node.
            for key in ("Width", "Height"):
                iv = mvs.MVCC_INTVALUE_EX()
                mvs.memset(mvs.byref(iv), 0, mvs.sizeof(iv))
                if cam.MV_CC_GetIntValueEx(key, iv) == 0:
                    warn(key, cam.MV_CC_SetIntValue(key, int(iv.nMax)))

        if c.get("exposure_us"):
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            warn("ExposureTime",
                 cam.MV_CC_SetFloatValue("ExposureTime", float(c["exposure_us"])))
        if c.get("gain") is not None:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            warn("Gain", cam.MV_CC_SetFloatValue("Gain", float(c["gain"])))
        if c.get("fps"):
            warn("AcquisitionFrameRateEnable",
                 cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True))
            warn("AcquisitionFrameRate",
                 cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(c["fps"])))

        cam.MV_CC_SetImageNodeNum(2)
        cam.MV_CC_SetGrabStrategy(mvs.MV_GrabStrategy_LatestImagesOnly)

    def _nic_mtu(self) -> Optional[int]:
        try:
            import subprocess
            mvs = self.mvs
            if self._dev_info.nTLayerType != mvs.MV_GIGE_DEVICE:
                return None
            v = self._dev_info.SpecialInfo.stGigEInfo.nCurrentIp
            ip = "%d.%d.%d.%d" % ((v >> 24) & 0xFF, (v >> 16) & 0xFF,
                                  (v >> 8) & 0xFF, v & 0xFF)
            out = subprocess.run(["ip", "route", "get", ip], capture_output=True,
                                 text=True, timeout=3).stdout.split()
            dev = out[out.index("dev") + 1]
            with open(f"/sys/class/net/{dev}/mtu") as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _rd_int(self, key: str) -> int:
        mvs, cam = self.mvs, self.cam
        v = mvs.MVCC_INTVALUE()
        mvs.memset(mvs.byref(v), 0, mvs.sizeof(v))
        return int(v.nCurValue) if cam.MV_CC_GetIntValue(key, v) == 0 else 0

    def _rd_float(self, key: str):
        mvs, cam = self.mvs, self.cam
        v = mvs.MVCC_FLOATVALUE()
        mvs.memset(mvs.byref(v), 0, mvs.sizeof(v))
        if cam.MV_CC_GetFloatValue(key, v) != 0:
            return None
        return {"cur": float(v.fCurValue), "min": float(v.fMin),
                "max": float(v.fMax)}

    def _read_info(self):
        try:
            exp = self._rd_float("ExposureTime") or {}
            gain = self._rd_float("Gain") or {}
            rate = self._rd_float("ResultingFrameRate") or {}
            self.info = {
                "width": self._rd_int("Width"),
                "height": self._rd_int("Height"),
                "offset_x": self._rd_int("OffsetX"),
                "offset_y": self._rd_int("OffsetY"),
                "packet_size": self._rd_int("GevSCPSPacketSize"),
                "exposure_us": round(exp.get("cur", 0.0), 1),
                "gain": round(gain.get("cur", 0.0), 2),
                "resulting_fps": round(rate.get("cur", 0.0), 1),
            }
        except Exception as exc:
            logger.warning("Đọc thông tin camera lỗi: %s", exc)

    def _read_limits(self):
        try:
            exp = self._rd_float("ExposureTime") or {}
            gain = self._rd_float("Gain") or {}
            self.limits = {
                "exposure_us": [round(exp.get("min", 10.0)),
                                round(min(exp.get("max", 500000.0), 500000.0))],
                "gain": [round(gain.get("min", 0.0), 1),
                         round(gain.get("max", 24.0), 1)],
            }
        except Exception:
            self.limits = {}

    def _close_device(self):
        if self.cam is not None:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception as exc:
                logger.warning("Đóng camera lỗi: %s", exc)
            self.cam = None
        # Công cụ này là tiến trình RIÊNG và chỉ mở 1 camera → gọi Finalize để
        # nhả hẳn SDK, tránh giữ cổng mạng khi chương trình gốc khởi động lại.
        try:
            from yarnbreak import mvs_sdk
            mvs_sdk.sdk_finalize(force=True)
        except Exception:
            pass
        self.connected = False

    # ── Chuyển pixel ────────────────────────────────────────────────────────
    def _to_bgr(self, st_frame):
        """(ảnh BGR, borrowed) — borrowed=True nghĩa là trỏ vào buffer dùng chung."""
        mvs = self.mvs
        fi = st_frame.stFrameInfo
        w, h = fi.nWidth, fi.nHeight

        if fi.enPixelType == mvs.PixelType_Gvsp_Mono8:
            ptr = mvs.cast(st_frame.pBufAddr, mvs.POINTER(mvs.c_ubyte * (w * h)))
            gray = np.frombuffer(ptr.contents, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), False

        if fi.enPixelType == mvs.PixelType_Gvsp_BGR8_Packed:
            ptr = mvs.cast(st_frame.pBufAddr, mvs.POINTER(mvs.c_ubyte * (w * h * 3)))
            return np.frombuffer(ptr.contents, dtype=np.uint8).reshape(h, w, 3), True

        # BayerRG8: để SDK convert — tên pattern GenICam lệch 1 pixel so với
        # OpenCV nên cv2.cvtColor(COLOR_BAYER_*) rất dễ ra sai màu.
        need = w * h * 3
        if self._cvt_buf is None or self._cvt_buf_size < need:
            self._cvt_buf = (mvs.c_ubyte * need)()
            self._cvt_buf_size = need
            self._cvt_view = None

        param = mvs.MV_CC_PIXEL_CONVERT_PARAM_EX()
        mvs.memset(mvs.byref(param), 0, mvs.sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.pSrcData = st_frame.pBufAddr
        param.nSrcDataLen = fi.nFrameLen
        param.enSrcPixelType = fi.enPixelType
        param.enDstPixelType = mvs.PixelType_Gvsp_BGR8_Packed
        param.pDstBuffer = self._cvt_buf
        param.nDstBufferSize = self._cvt_buf_size

        ret = self.cam.MV_CC_ConvertPixelTypeEx(param)
        if ret != 0:
            raise RuntimeError(f"MV_CC_ConvertPixelTypeEx lỗi 0x{ret:x}")

        if self._cvt_view is None or self._cvt_view.shape != (h, w, 3):
            self._cvt_view = np.frombuffer(self._cvt_buf, dtype=np.uint8,
                                           count=need).reshape(h, w, 3)
        return self._cvt_view, True

    # ── Vòng đọc ────────────────────────────────────────────────────────────
    def _reader(self):
        try:
            self._open_device()
        except Exception as exc:
            self.error = str(exc)
            self.connected = False
            logger.error("Mở camera thất bại: %s", exc)
            self._ready.set()
            return
        finally:
            self._ready.set()

        mvs = self.mvs
        st_frame = mvs.MV_FRAME_OUT()
        mvs.memset(mvs.byref(st_frame), 0, mvs.sizeof(st_frame))
        fail_streak = 0

        while self._running:
            self._apply_pending()
            ret = self.cam.MV_CC_GetImageBuffer(st_frame, self.GET_TIMEOUT_MS)
            if ret != 0:
                fail_streak += 1
                if fail_streak == 5:
                    self.error = (f"Không nhận được frame (0x{ret:x}) — kiểm tra "
                                  "cáp mạng / nguồn camera.")
                continue

            try:
                bgr, borrowed = self._to_bgr(st_frame)
                frame = bgr.copy() if borrowed else bgr
            except Exception as exc:
                logger.warning("Chuyển frame lỗi: %s", exc)
                frame = None
            finally:
                self.cam.MV_CC_FreeImageBuffer(st_frame)

            if frame is None:
                continue

            fail_streak = 0
            self.error = ""
            self._fps_marks.append(time.monotonic())
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._ts = time.monotonic()

        self._close_device()


class VideoGrabber:
    """Đọc frame từ file video — dùng để thử giao diện khi chưa cắm camera."""

    def __init__(self, path: str):
        self.path = path
        self.serial = f"video:{os.path.basename(path)}"
        self.connected = False
        self.error = ""
        self.info: Dict = {}
        self.limits: Dict = {}
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 10.0) -> bool:
        if not os.path.isfile(self.path):
            self.error = f"Không thấy file video: {self.path}"
            return False
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error = f"Không mở được video: {self.path}"
            return False
        self._cap = cap
        self.info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "resulting_fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1),
        }
        self.connected = True
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name="Grab-video")
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        try:
            self._cap.release()
        except Exception:
            pass
        self.connected = False

    def read(self):
        with self._lock:
            return self._frame, self._seq

    def stats(self) -> dict:
        with self._lock:
            seq, ts = self._seq, self._ts
        return {"connected": self.connected, "error": self.error,
                "serial": self.serial, "frames": seq,
                "age_s": round(time.monotonic() - ts, 2) if ts else None,
                "fps": self.info.get("resulting_fps", 0), "info": self.info,
                "limits": {}}

    def queue_params(self, **kw):
        pass

    def _reader(self):
        fps = self.info.get("resulting_fps") or 25.0
        delay = 1.0 / max(fps, 1.0)
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._ts = time.monotonic()
            time.sleep(delay)


# ══════════════════════════════════════════════════════════════════════════════
#  PHIÊN LẤY MẪU: lưu ảnh, đếm, chụp tự động, chống ảnh trùng
# ══════════════════════════════════════════════════════════════════════════════
IMG_EXT = {"jpg": ".jpg", "png": ".png"}


class CaptureSession:
    """Quản lý việc ghi ảnh mẫu ra đĩa.

    Cấu trúc thư mục:  <out_dir>/<nhãn>/<prefix>_<YYYYmmdd-HHMMSS-mmm>.jpg
    Nhãn là tên thư mục con (ví dụ "dut_soi", "binh_thuong") → gán nhãn ngay
    lúc chụp, khỏi phải phân loại lại sau.
    """

    def __init__(self, out_dir: str):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.label = "chua_phan_loai"
        self.prefix = "comb"
        self.fmt = "jpg"
        self.quality = 95
        self.dedup_thresh = 0.0        # 0 = tắt kiểm tra ảnh gần trùng

        self.session_count = 0
        self.skipped_dup = 0
        self.last_error = ""
        self.recent = deque(maxlen=24)

        self._last_small = None        # ảnh thu nhỏ của mẫu vừa lưu (so trùng)
        self._lock = threading.Lock()

        # Chụp tự động
        self.auto_on = False
        self.auto_interval = 2.0
        self.auto_limit = 0            # 0 = không giới hạn
        self.auto_done = 0
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()

    # ── Thiết lập ───────────────────────────────────────────────────────────
    def configure(self, label=None, prefix=None, fmt=None, quality=None,
                  dedup=None):
        if label is not None:
            self.label = safe_name(label) or "chua_phan_loai"
        if prefix is not None:
            self.prefix = safe_name(prefix) or "comb"
        if fmt is not None and fmt in IMG_EXT:
            self.fmt = fmt
        if quality is not None:
            self.quality = int(max(50, min(100, int(quality))))
        if dedup is not None:
            self.dedup_thresh = float(max(0.0, min(60.0, float(dedup))))

    def label_dir(self, label: Optional[str] = None) -> str:
        return os.path.join(self.out_dir, label or self.label)

    # ── Ghi ảnh ─────────────────────────────────────────────────────────────
    def _is_duplicate(self, frame: np.ndarray) -> bool:
        """So ảnh với mẫu vừa lưu: giống quá thì bỏ, tránh thu 500 ảnh y nhau."""
        if self.dedup_thresh <= 0:
            return False
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (128, 128),
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        if self._last_small is None:
            self._last_small = small
            return False
        diff = float(np.mean(np.abs(small - self._last_small)))
        if diff < self.dedup_thresh:
            return True
        self._last_small = small
        return False

    def save(self, frame: np.ndarray, note: str = "manual") -> Optional[dict]:
        """Ghi 1 frame ra đĩa. Trả về bản ghi, hoặc None nếu bỏ qua/lỗi."""
        if frame is None:
            self.last_error = "Chưa có frame từ camera."
            return None
        with self._lock:
            if self._is_duplicate(frame):
                self.skipped_dup += 1
                return None

            folder = self.label_dir()
            try:
                os.makedirs(folder, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-") + \
                    f"{int(time.time() * 1000) % 1000:03d}"
                name = f"{self.prefix}_{stamp}{IMG_EXT[self.fmt]}"
                path = os.path.join(folder, name)
                if self.fmt == "jpg":
                    params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                else:
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                if not cv2.imwrite(path, frame, params):
                    raise RuntimeError("cv2.imwrite trả về False")
            except Exception as exc:
                self.last_error = f"Ghi ảnh lỗi: {exc}"
                logger.error(self.last_error)
                return None

            h, w = frame.shape[:2]
            rec = {
                "name": name,
                "rel": os.path.relpath(path, self.out_dir).replace("\\", "/"),
                "label": self.label,
                "w": w, "h": h,
                "bytes": os.path.getsize(path),
                "time": datetime.now().strftime("%H:%M:%S"),
                "note": note,
            }
            self.session_count += 1
            self.last_error = ""
            self.recent.appendleft(rec)
            return rec

    def delete(self, rel: str) -> bool:
        """Xoá 1 ảnh đã chụp (chỉ trong out_dir — chặn đường dẫn đi ra ngoài)."""
        path = self.resolve(rel)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        with self._lock:
            self.recent = deque(
                [r for r in self.recent if r["rel"] != rel], maxlen=24)
            self.session_count = max(0, self.session_count - 1)
        return True

    def resolve(self, rel: str) -> str:
        """Đổi đường dẫn tương đối → tuyệt đối, bắt buộc nằm trong out_dir."""
        path = os.path.abspath(os.path.join(self.out_dir, rel))
        try:
            inside = os.path.commonpath([path, self.out_dir]) == self.out_dir
        except ValueError:          # khác ổ đĩa (Windows) → chắc chắn ở ngoài
            inside = False
        if not inside:
            raise HTTPException(400, "Đường dẫn không hợp lệ.")
        return path

    # ── Chụp tự động ────────────────────────────────────────────────────────
    def start_auto(self, frame_getter, interval: float, limit: int):
        self.stop_auto()
        self.auto_interval = max(0.2, float(interval))
        self.auto_limit = max(0, int(limit))
        self.auto_done = 0
        self._auto_stop.clear()
        self.auto_on = True
        self._auto_thread = threading.Thread(
            target=self._auto_loop, args=(frame_getter,), daemon=True,
            name="AutoCapture")
        self._auto_thread.start()

    def stop_auto(self):
        self.auto_on = False
        self._auto_stop.set()
        if self._auto_thread is not None:
            self._auto_thread.join(timeout=3.0)
            self._auto_thread = None

    def _auto_loop(self, frame_getter):
        while not self._auto_stop.is_set():
            frame, _ = frame_getter()
            if frame is not None:
                self.save(frame, note="auto")
                self.auto_done += 1
            if self.auto_limit and self.auto_done >= self.auto_limit:
                break
            if self._auto_stop.wait(self.auto_interval):
                break
        self.auto_on = False

    # ── Thống kê ────────────────────────────────────────────────────────────
    def labels(self) -> List[dict]:
        """Danh sách thư mục nhãn + số ảnh trong mỗi thư mục."""
        out = []
        try:
            for name in sorted(os.listdir(self.out_dir)):
                d = os.path.join(self.out_dir, name)
                if not os.path.isdir(d):
                    continue
                n = sum(1 for f in os.listdir(d)
                        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
                out.append({"label": name, "count": n})
        except Exception as exc:
            logger.warning("Đọc thư mục mẫu lỗi: %s", exc)
        return out

    def disk(self) -> dict:
        try:
            du = shutil.disk_usage(self.out_dir)
            return {"free_gb": round(du.free / 1e9, 1),
                    "total_gb": round(du.total / 1e9, 1)}
        except Exception:
            return {}


def safe_name(s: str) -> str:
    """Giữ lại chữ/số/gạch — tên thư mục và tên file phải an toàn trên mọi OS."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(s).strip()]
    return "".join(keep).strip("_")[:48]


# ══════════════════════════════════════════════════════════════════════════════
#  TRẠNG THÁI TOÀN CỤC + ỨNG DỤNG WEB
# ══════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, cfg_path: str, out_dir: str):
        self.cfg_path = cfg_path
        self.cfg = load_cfg(cfg_path)
        self.sdk_path = self.cfg.get("mvs_sdk_path") or DEFAULT_SDK_PATH
        self.presets = camera_presets(self.cfg)
        self.session = CaptureSession(out_dir)
        self.grabber = None
        self.lock = threading.Lock()

    def frame(self):
        g = self.grabber
        return g.read() if g is not None else (None, 0)

    def open_mvs(self, serial: str, settings: dict):
        with self.lock:
            self.close()
            g = MvsGrabber(serial, settings, self.sdk_path)
            ok = g.start()
            self.grabber = g
            if not ok:
                raise HTTPException(400, g.error or "Không mở được camera.")

    def open_video(self, path: str):
        with self.lock:
            self.close()
            g = VideoGrabber(path)
            ok = g.start()
            self.grabber = g
            if not ok:
                raise HTTPException(400, g.error or "Không mở được video.")

    def close(self):
        self.session.stop_auto()
        if self.grabber is not None:
            try:
                self.grabber.stop()
            except Exception as exc:
                logger.warning("Đóng grabber lỗi: %s", exc)
            self.grabber = None


APP: App = None          # gán trong main()
api = FastAPI(title="Comb Capture", docs_url=None, redoc_url=None)


# ── Models ───────────────────────────────────────────────────────────────────
class OpenReq(BaseModel):
    serial: str = ""
    source: str = "mvs"               # mvs | video
    video: str = ""
    exposure_us: Optional[float] = None
    gain: Optional[float] = None
    fps: Optional[float] = None
    binning: int = 1
    packet_size: int = 9000
    use_hw_roi: bool = True
    hw_roi: Optional[List[int]] = None


class ParamReq(BaseModel):
    exposure_us: Optional[float] = None
    gain: Optional[float] = None
    fps: Optional[float] = None


class SetupReq(BaseModel):
    label: Optional[str] = None
    prefix: Optional[str] = None
    fmt: Optional[str] = None
    quality: Optional[int] = None
    dedup: Optional[float] = None


class ShotReq(BaseModel):
    count: int = 1
    gap_ms: int = 150


class AutoReq(BaseModel):
    on: bool = False
    interval: float = 2.0
    limit: int = 0


# ── Endpoints ────────────────────────────────────────────────────────────────
@api.get("/", response_class=HTMLResponse)
def page():
    return HTML


@api.get("/api/presets")
def api_presets():
    return {"cameras": APP.presets, "sdk_path": APP.sdk_path,
            "out_dir": APP.session.out_dir, "config": APP.cfg_path}


@api.get("/api/scan")
def api_scan():
    """Dò mọi camera Hikrobot đang thấy trên mạng/USB."""
    try:
        from yarnbreak import mvs_sdk
        mvs_sdk.load_sdk(APP.sdk_path)
        mvs_sdk.sdk_init()
        try:
            cams = [{"serial": c["serial"], "model": c["model"],
                     "ip": c["ip"], "tlayer": c["tlayer"]}
                    for c in mvs_sdk.enumerate_cameras()]
        finally:
            mvs_sdk.sdk_finalize()
        return {"ok": True, "cameras": cams}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=200)


@api.post("/api/open")
def api_open(req: OpenReq):
    if req.source == "video":
        APP.open_video(req.video)
    else:
        settings = {
            "exposure_us": req.exposure_us,
            "gain": req.gain,
            "fps": req.fps,
            "binning": req.binning,
            "packet_size": req.packet_size,
            "hw_roi": req.hw_roi if (req.use_hw_roi and req.hw_roi) else None,
        }
        APP.open_mvs(req.serial, settings)
    return api_status()


@api.post("/api/close")
def api_close():
    APP.close()
    return {"ok": True}


@api.get("/api/status")
def api_status():
    g = APP.grabber
    s = APP.session
    return {
        "camera": g.stats() if g is not None else {"connected": False},
        "session": {
            "label": s.label, "prefix": s.prefix, "fmt": s.fmt,
            "quality": s.quality, "dedup": s.dedup_thresh,
            "count": s.session_count, "skipped_dup": s.skipped_dup,
            "last_error": s.last_error,
            "auto_on": s.auto_on, "auto_interval": s.auto_interval,
            "auto_limit": s.auto_limit, "auto_done": s.auto_done,
            "out_dir": s.out_dir,
        },
        "labels": s.labels(),
        "disk": s.disk(),
        "recent": list(s.recent),
    }


@api.post("/api/setup")
def api_setup(req: SetupReq):
    APP.session.configure(label=req.label, prefix=req.prefix, fmt=req.fmt,
                          quality=req.quality, dedup=req.dedup)
    return api_status()


@api.post("/api/params")
def api_params(req: ParamReq):
    g = APP.grabber
    if g is None:
        raise HTTPException(400, "Chưa mở camera.")
    g.queue_params(exposure_us=req.exposure_us, gain=req.gain, fps=req.fps)
    return {"ok": True}


@api.post("/api/shot")
def api_shot(req: ShotReq):
    g = APP.grabber
    if g is None or not g.connected:
        raise HTTPException(400, "Chưa mở camera.")
    saved, dup = [], 0
    n = max(1, min(50, int(req.count)))
    last_seq = -1
    for i in range(n):
        # Chờ frame MỚI để loạt ảnh không lặp lại cùng một frame.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            frame, seq = g.read()
            if frame is not None and seq != last_seq:
                last_seq = seq
                break
            time.sleep(0.01)
        else:
            frame, seq = g.read()
        rec = APP.session.save(frame, note="manual")
        if rec:
            saved.append(rec)
        else:
            dup += 1
        if i < n - 1:
            time.sleep(max(0, req.gap_ms) / 1000.0)
    return {"ok": True, "saved": saved, "skipped": dup,
            "error": APP.session.last_error}


@api.post("/api/auto")
def api_auto(req: AutoReq):
    g = APP.grabber
    if req.on:
        if g is None or not g.connected:
            raise HTTPException(400, "Chưa mở camera.")
        APP.session.start_auto(APP.frame, req.interval, req.limit)
    else:
        APP.session.stop_auto()
    return api_status()


@api.post("/api/delete")
def api_delete(rel: str = Query(...)):
    ok = APP.session.delete(rel)
    return {"ok": ok}


@api.get("/thumb")
def thumb(rel: str = Query(...), w: int = 240):
    path = APP.session.resolve(rel)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(404, "Không đọc được ảnh.")
    h0, w0 = img.shape[:2]
    w = max(64, min(1200, int(w)))
    if w0 > w:
        img = cv2.resize(img, (w, max(1, int(h0 * w / w0))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Mã hoá ảnh lỗi.")
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


@api.get("/full")
def full(rel: str = Query(...)):
    """Tải ảnh gốc về máy (để kiểm tra chi tiết sợi trước khi gán nhãn)."""
    path = APP.session.resolve(rel)
    if not os.path.isfile(path):
        raise HTTPException(404, "Không thấy file.")
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return StreamingResponse(iter([data]), media_type=mime)


@api.get("/stream")
def stream(width: int = 1100, fps: float = 10.0, grid: int = 0):
    """MJPEG xem trực tiếp. Chỉ phục vụ việc căn khung — ảnh LƯU vẫn là ảnh gốc."""
    max_w = max(320, min(1920, int(width)))
    interval = 1.0 / max(1.0, min(20.0, float(fps)))
    show_grid = bool(grid)

    def gen():
        last_seq = -1
        idle = 0.0
        while True:
            frame, seq = APP.frame()
            if frame is None or seq == last_seq:
                time.sleep(0.05)
                idle += 0.05
                if idle > 30.0:
                    break
                continue
            idle = 0.0
            last_seq = seq
            h0, w0 = frame.shape[:2]
            if w0 > max_w:
                img = cv2.resize(frame, (max_w, max(1, int(h0 * max_w / w0))),
                                 interpolation=cv2.INTER_AREA)
            else:
                img = frame.copy()
            if show_grid:
                _draw_grid(img)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                data = buf.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            time.sleep(interval)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def _draw_grid(img):
    """Lưới 1/3 + tâm ảnh — giúp căn camera cho giống lúc chạy thật."""
    h, w = img.shape[:2]
    c = (0, 220, 255)
    for i in (1, 2):
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), c, 1)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), c, 1)
    cv2.drawMarker(img, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 24, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  GIAO DIỆN (nhúng sẵn — không dùng templates/ của chương trình gốc)
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chụp mẫu camera lược</title>
<style>
:root{
  --bg:#f8fafc; --card:#ffffff; --border:#e2e8f0; --text:#0f172a; --muted:#64748b;
  --primary:#2563eb; --primary-h:#1d4ed8; --success:#10b981; --success-h:#059669;
  --danger:#ef4444; --danger-h:#dc2626; --warning:#f59e0b; --metric:#f1f5f9;
  --font:'Be Vietnam Pro','Segoe UI',Roboto,'Noto Sans',Arial,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:15px;line-height:1.5}
.wrap{max-width:1500px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:20px;font-weight:700}
h2{font-size:15px;font-weight:700;margin-bottom:10px}
.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 380px;gap:14px;align-items:start}
@media(max-width:1100px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:14px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.row+.row{margin-top:8px}
label{font-size:13px;color:var(--muted);display:block;margin-bottom:3px}
input,select{font-family:var(--font);font-size:14px;padding:7px 9px;border:1px solid var(--border);
  border-radius:8px;background:#fff;color:var(--text);width:100%}
input[type=checkbox]{width:auto}
.f{flex:1;min-width:92px}
button{font-family:var(--font);font-size:14px;font-weight:600;padding:9px 14px;border:0;
  border-radius:9px;background:var(--metric);color:var(--text);cursor:pointer}
button:hover{filter:brightness(.97)}
button:disabled{opacity:.5;cursor:not-allowed}
.b-primary{background:var(--primary);color:#fff}.b-primary:hover{background:var(--primary-h)}
.b-success{background:var(--success);color:#fff}.b-success:hover{background:var(--success-h)}
.b-danger{background:var(--danger);color:#fff}.b-danger:hover{background:var(--danger-h)}
.b-big{font-size:17px;padding:16px;width:100%}
#view{position:relative;background:#0b1220;border-radius:10px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;min-height:260px}
#view img{width:100%;display:block}
#view .ph{color:#94a3b8;padding:60px 20px;text-align:center;font-size:14px}
.flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none}
.flash.on{animation:fl .32s ease-out}
@keyframes fl{from{opacity:.85}to{opacity:0}}
.pill{display:inline-flex;align-items:center;gap:6px;background:var(--metric);
  border-radius:999px;padding:4px 11px;font-size:13px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.dot.ok{background:var(--success)}.dot.bad{background:var(--danger)}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));gap:8px}
.metric{background:var(--metric);border-radius:9px;padding:8px 10px}
.metric .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.3px}
.metric .v{font-size:17px;font-weight:700}
.note{background:#fffbeb;border:1px solid #fde68a;color:#78350f;border-radius:9px;
  padding:9px 11px;font-size:13px}
.err{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;border-radius:9px;
  padding:9px 11px;font-size:13px;margin-top:8px;word-break:break-word}
.hint{font-size:12px;color:var(--muted);margin-top:6px}
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:8px}
.shot{position:relative;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:#000}
.shot img{width:100%;height:76px;object-fit:cover;display:block}
.shot .cap{font-size:10px;padding:3px 5px;background:var(--card);color:var(--muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.shot .x{position:absolute;top:3px;right:3px;background:rgba(239,68,68,.92);color:#fff;
  border:0;border-radius:6px;padding:1px 6px;font-size:12px;cursor:pointer}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px}
.kbd{background:var(--metric);border:1px solid var(--border);border-radius:5px;
  padding:1px 6px;font-size:12px;font-family:monospace}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Chụp mẫu huấn luyện — camera lược</h1>
    <span class="pill"><span class="dot" id="dot"></span><span id="st">Chưa mở camera</span></span>
    <span class="sub" id="sub"></span>
  </header>

  <div class="note">
    Camera mở ở chế độ <b>độc quyền</b>: hãy <b>dừng chương trình gốc</b>
    (<code>sudo systemctl stop yarnbreak</code>) trước khi mở camera ở đây.
    Ảnh lưu là <b>ảnh gốc nguyên độ phân giải</b>, không vẽ khung, không đóng dấu.
  </div>

  <div class="grid" style="margin-top:14px">
    <!-- ══ CỘT TRÁI: xem trực tiếp + nút chụp ══ -->
    <div>
      <div class="card">
        <div id="view">
          <div class="ph" id="ph">Chọn camera ở cột bên phải rồi bấm <b>MỞ CAMERA</b>.</div>
          <div class="flash" id="flash"></div>
        </div>
        <div class="row" style="margin-top:10px">
          <label style="margin:0"><input type="checkbox" id="grid"> Lưới căn khung</label>
          <div style="flex:1"></div>
          <span class="sub" id="res"></span>
        </div>
      </div>

      <div class="card">
        <h2>Chụp mẫu</h2>
        <button class="b-success b-big" id="shot" disabled>
          📸 CHỤP &nbsp;<span class="kbd">Space</span>
        </button>
        <div class="row" style="margin-top:10px">
          <div class="f"><label>Số ảnh mỗi lần bấm</label>
            <input type="number" id="count" value="1" min="1" max="50"></div>
          <div class="f"><label>Giãn cách trong loạt (ms)</label>
            <input type="number" id="gap" value="150" min="0" max="5000" step="50"></div>
        </div>

        <div class="row" style="margin-top:12px;border-top:1px solid var(--border);padding-top:12px">
          <div class="f"><label>Chụp tự động mỗi (giây)</label>
            <input type="number" id="interval" value="2" min="0.2" max="600" step="0.2"></div>
          <div class="f"><label>Dừng sau (ảnh, 0 = không giới hạn)</label>
            <input type="number" id="limit" value="0" min="0" max="100000"></div>
          <div style="align-self:end">
            <button class="b-primary" id="auto" disabled>▶ Bật tự động</button>
          </div>
        </div>
        <div class="hint" id="autoinfo"></div>
        <div class="err" id="shoterr" style="display:none"></div>
      </div>

      <div class="card">
        <h2>Ảnh vừa chụp <span class="sub" id="recentn"></span></h2>
        <div class="shots" id="shots"></div>
        <div class="hint" id="noshots">Chưa có ảnh nào trong phiên này.</div>
      </div>
    </div>

    <!-- ══ CỘT PHẢI: điều khiển ══ -->
    <div>
      <div class="card">
        <h2>Camera</h2>
        <div class="row">
          <div class="f" style="min-width:200px"><label>Nguồn ảnh</label>
            <select id="preset"></select></div>
        </div>
        <div class="row" id="rowserial">
          <div class="f" style="min-width:160px"><label>Serial</label>
            <input id="serial" placeholder="VD: DB1438128"></div>
          <div style="align-self:end"><button id="scan">Dò camera</button></div>
        </div>
        <div class="row" id="rowvideo" style="display:none">
          <div class="f" style="min-width:200px"><label>File video</label>
            <input id="video" placeholder="./videos/YBC1.mp4"></div>
        </div>
        <div class="row">
          <div class="f"><label>Exposure (µs)</label>
            <input type="number" id="exp" value="17000" min="10" step="500"></div>
          <div class="f"><label>Gain (dB)</label>
            <input type="number" id="gain" value="10" min="0" step="0.5"></div>
          <div class="f"><label>FPS</label>
            <input type="number" id="fps" value="8" min="1" step="1"></div>
        </div>
        <div class="row">
          <div class="f"><label>Binning</label>
            <input type="number" id="binning" value="1" min="1" max="4"></div>
          <div class="f"><label>Packet size</label>
            <input type="number" id="packet" value="9000" min="1500" step="500"></div>
        </div>
        <div class="row">
          <label style="margin:0"><input type="checkbox" id="usroi" checked>
            Dùng ROI phần cứng <span id="roitxt" class="sub"></span></label>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="b-primary" id="open" style="flex:1">MỞ CAMERA</button>
          <button class="b-danger" id="closecam" disabled>Đóng</button>
        </div>
        <div class="hint">Đổi Exposure / Gain / FPS khi đang mở sẽ áp dụng ngay —
          nên thu mẫu ở nhiều mức sáng khác nhau để model học khoẻ hơn.</div>
        <div class="row" style="margin-top:6px">
          <button id="apply" disabled style="flex:1">Áp dụng Exposure/Gain/FPS</button>
        </div>
        <div class="err" id="camerr" style="display:none"></div>
      </div>

      <div class="card">
        <h2>Nơi lưu &amp; nhãn</h2>
        <div class="row">
          <div class="f" style="min-width:190px"><label>Nhãn (thư mục con)</label>
            <input id="label" value="chua_phan_loai"></div>
        </div>
        <div class="row">
          <div class="f"><label>Tiền tố tên file</label><input id="prefix" value="comb"></div>
          <div class="f"><label>Định dạng</label>
            <select id="fmt"><option value="jpg">JPG</option><option value="png">PNG (không nén mất mát)</option></select></div>
        </div>
        <div class="row">
          <div class="f"><label>Chất lượng JPG</label>
            <input type="number" id="quality" value="95" min="50" max="100"></div>
          <div class="f"><label>Bỏ ảnh gần trùng (0 = tắt)</label>
            <input type="number" id="dedup" value="0" min="0" max="60" step="0.5"></div>
        </div>
        <div class="hint">Ngưỡng trùng là mức chênh sáng trung bình so với ảnh vừa
          lưu; 1–3 là đủ để loại các frame y nhau khi máy đứng im.</div>
        <div class="row" style="margin-top:10px">
          <button class="b-primary" id="savesetup" style="flex:1">Lưu thiết lập</button>
        </div>
        <div class="hint" id="outdir" style="margin-top:8px"></div>
      </div>

      <div class="card">
        <h2>Thống kê</h2>
        <div class="metrics">
          <div class="metric"><div class="k">Phiên này</div><div class="v" id="m-count">0</div></div>
          <div class="metric"><div class="k">Bỏ vì trùng</div><div class="v" id="m-dup">0</div></div>
          <div class="metric"><div class="k">FPS camera</div><div class="v" id="m-fps">–</div></div>
          <div class="metric"><div class="k">Đĩa trống</div><div class="v" id="m-disk">–</div></div>
        </div>
        <table style="margin-top:12px">
          <thead><tr><th>Nhãn</th><th style="text-align:right">Số ảnh trên đĩa</th></tr></thead>
          <tbody id="labels"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let presets = [], opened = false, autoOn = false, busy = false;

async function jget(u){ const r = await fetch(u); if(!r.ok) throw new Error((await r.json()).detail || r.status); return r.json(); }
async function jpost(u, body){
  const r = await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'},
    body: body===undefined ? null : JSON.stringify(body)});
  const d = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(d.detail || ('HTTP '+r.status));
  return d;
}
function showErr(el, msg){ const e=$(el); e.style.display = msg?'block':'none'; e.textContent = msg||''; }

// ── Nạp preset camera từ config.yaml của chương trình gốc ──────────────────
async function loadPresets(){
  const d = await jget('/api/presets');
  presets = d.cameras;
  const sel = $('preset');
  sel.innerHTML = '';
  presets.forEach((c,i)=>{
    const o = document.createElement('option');
    o.value = 'p'+i;
    o.textContent = `${c.id} — ${c.name}` + (c.serial? ` (${c.serial})` : ' (chưa có serial)')
      + (c.enabled? '' : ' [tắt trong config]');
    sel.appendChild(o);
  });
  let o = document.createElement('option'); o.value='manual'; o.textContent='↳ Nhập serial thủ công'; sel.appendChild(o);
  o = document.createElement('option'); o.value='video'; o.textContent='↳ Thử bằng file video'; sel.appendChild(o);
  $('outdir').innerHTML = `Lưu vào: <code>${d.out_dir}</code><br>Config đọc từ: <code>${d.config}</code>`;
  applyPreset();
}

function applyPreset(){
  const v = $('preset').value;
  const isVideo = v === 'video';
  $('rowvideo').style.display = isVideo ? 'flex' : 'none';
  $('rowserial').style.display = isVideo ? 'none' : 'flex';
  if(v.startsWith('p')){
    const c = presets[+v.slice(1)];
    $('serial').value = c.serial; $('exp').value = c.exposure_us;
    $('gain').value = c.gain; $('fps').value = c.fps;
    $('binning').value = c.binning; $('packet').value = c.packet_size;
    $('video').value = c.video || '';
    $('prefix').value = c.id;
    $('roitxt').textContent = c.hw_roi ? '[' + c.hw_roi.join(', ') + ']' : '(config không có ROI → dùng full cảm biến)';
    $('usroi').disabled = !c.hw_roi;
    window._roi = c.hw_roi;
  } else {
    $('roitxt').textContent = '(full cảm biến)';
    $('usroi').disabled = true; window._roi = null;
  }
}

// ── Mở / đóng camera ───────────────────────────────────────────────────────
async function openCam(){
  showErr('camerr',''); busy = true; $('open').disabled = true; $('open').textContent='Đang mở…';
  try{
    const isVideo = $('preset').value === 'video';
    await jpost('/api/open', {
      source: isVideo ? 'video' : 'mvs',
      video: $('video').value.trim(),
      serial: $('serial').value.trim(),
      exposure_us: +$('exp').value, gain: +$('gain').value, fps: +$('fps').value,
      binning: +$('binning').value, packet_size: +$('packet').value,
      use_hw_roi: $('usroi').checked, hw_roi: window._roi || null
    });
    startStream();
  }catch(e){ showErr('camerr', e.message); }
  finally{ busy=false; $('open').disabled=false; $('open').textContent='MỞ CAMERA'; refresh(); }
}

function startStream(){
  const g = $('grid').checked ? 1 : 0;
  $('view').querySelectorAll('img').forEach(n=>n.remove());
  const img = document.createElement('img');
  img.src = '/stream?width=1100&fps=10&grid=' + g + '&t=' + Date.now();
  $('view').insertBefore(img, $('flash'));
  $('ph').style.display = 'none';
}
function stopStream(){
  $('view').querySelectorAll('img').forEach(n=>{ n.src=''; n.remove(); });
  $('ph').style.display = 'block';
}

async function closeCam(){
  await jpost('/api/close'); stopStream(); refresh();
}

// ── Chụp ───────────────────────────────────────────────────────────────────
async function shoot(){
  if(busy || !opened) return;
  busy = true; $('shot').disabled = true;
  showErr('shoterr','');
  try{
    const d = await jpost('/api/shot', {count:+$('count').value, gap_ms:+$('gap').value});
    $('flash').classList.remove('on'); void $('flash').offsetWidth; $('flash').classList.add('on');
    if(d.skipped) showErr('shoterr', `Bỏ qua ${d.skipped} ảnh (gần trùng ảnh trước hoặc lỗi ghi).` + (d.error? ' ' + d.error : ''));
    else if(d.error) showErr('shoterr', d.error);
  }catch(e){ showErr('shoterr', e.message); }
  finally{ busy=false; $('shot').disabled = !opened; refresh(); }
}

async function toggleAuto(){
  try{
    await jpost('/api/auto', {on: !autoOn, interval:+$('interval').value, limit:+$('limit').value});
  }catch(e){ showErr('shoterr', e.message); }
  refresh();
}

async function saveSetup(){
  await jpost('/api/setup', {label:$('label').value, prefix:$('prefix').value,
    fmt:$('fmt').value, quality:+$('quality').value, dedup:+$('dedup').value});
  refresh();
}

async function applyParams(){
  try{
    await jpost('/api/params', {exposure_us:+$('exp').value, gain:+$('gain').value, fps:+$('fps').value});
  }catch(e){ showErr('camerr', e.message); }
}

async function scan(){
  $('scan').disabled = true; $('scan').textContent = 'Đang dò…';
  try{
    const d = await jget('/api/scan');
    if(!d.ok){ showErr('camerr', d.error); }
    else if(!d.cameras.length){ showErr('camerr','Không thấy camera nào. Kiểm tra cáp mạng / nguồn / IP.'); }
    else {
      showErr('camerr','');
      $('serial').value = d.cameras[0].serial;
      $('preset').value = 'manual'; applyPreset();
      alert('Camera tìm thấy:\n' + d.cameras.map(c=>`${c.serial}  ${c.model}  ${c.ip||c.tlayer}`).join('\n'));
    }
  }catch(e){ showErr('camerr', e.message); }
  finally{ $('scan').disabled=false; $('scan').textContent='Dò camera'; }
}

async function delShot(rel){
  await jpost('/api/delete?rel=' + encodeURIComponent(rel));
  refresh();
}

// ── Làm mới trạng thái ─────────────────────────────────────────────────────
async function refresh(){
  let d;
  try{ d = await jget('/api/status'); }catch(e){ return; }
  const cam = d.camera || {}, s = d.session || {};
  opened = !!cam.connected; autoOn = !!s.auto_on;

  $('dot').className = 'dot ' + (opened ? 'ok' : (cam.error ? 'bad' : ''));
  $('st').textContent = opened ? 'Đang mở' : (cam.error ? 'Lỗi' : 'Chưa mở camera');
  const i = cam.info || {};
  const bits = [];
  if(opened){
    bits.push(cam.serial, `${i.width}×${i.height}`);
    if(i.exposure_us != null) bits.push(`exp ${i.exposure_us}µs`);
    if(i.gain != null) bits.push(`gain ${i.gain}dB`);
    bits.push(`${cam.fps} fps`);
  }
  $('sub').textContent = bits.join(' · ');
  $('res').textContent = opened ? `Ảnh gốc ${i.width}×${i.height} px` : '';
  if(cam.error) showErr('camerr', cam.error);

  // Tải lại trang khi camera đã mở sẵn → tự gắn lại luồng xem, khỏi bấm lại.
  if(opened && !$('view').querySelector('img')) startStream();
  if(!opened) stopStream();

  $('shot').disabled = !opened || busy;
  $('auto').disabled = !opened;
  $('apply').disabled = !opened;
  $('closecam').disabled = !opened;
  $('auto').textContent = autoOn ? '■ Dừng tự động' : '▶ Bật tự động';
  $('auto').className = autoOn ? 'b-danger' : 'b-primary';
  $('autoinfo').textContent = autoOn
    ? `Đang chụp tự động mỗi ${s.auto_interval}s — đã chụp ${s.auto_done}` +
      (s.auto_limit ? ` / ${s.auto_limit}` : ' ảnh')
    : '';

  if(!window._setupSynced){
    window._setupSynced = true;
    $('label').value = s.label; $('prefix').value = s.prefix;
    $('fmt').value = s.fmt; $('quality').value = s.quality; $('dedup').value = s.dedup;
    $('interval').value = s.auto_interval; $('limit').value = s.auto_limit;
  }

  $('m-count').textContent = s.count;
  $('m-dup').textContent = s.skipped_dup;
  $('m-fps').textContent = opened ? cam.fps : '–';
  $('m-disk').textContent = d.disk && d.disk.free_gb!=null ? d.disk.free_gb + ' GB' : '–';

  $('labels').innerHTML = (d.labels||[]).map(l =>
    `<tr><td>${l.label}</td><td style="text-align:right">${l.count}</td></tr>`).join('')
    || '<tr><td colspan="2" class="sub">Chưa có thư mục mẫu nào.</td></tr>';

  const rec = d.recent || [];
  $('recentn').textContent = rec.length ? `(${rec.length} ảnh gần nhất)` : '';
  $('noshots').style.display = rec.length ? 'none' : 'block';
  $('shots').innerHTML = rec.map(r => `
    <div class="shot">
      <a href="/full?rel=${encodeURIComponent(r.rel)}" target="_blank" title="Mở ảnh gốc">
        <img src="/thumb?rel=${encodeURIComponent(r.rel)}&w=220" loading="lazy"></a>
      <button class="x" onclick="delShot('${r.rel}')" title="Xoá ảnh này">✕</button>
      <div class="cap" title="${r.rel}">${r.time} · ${(r.bytes/1048576).toFixed(1)}MB</div>
    </div>`).join('');
}

// ── Gán sự kiện ────────────────────────────────────────────────────────────
$('preset').onchange = applyPreset;
$('open').onclick = openCam;
$('closecam').onclick = closeCam;
$('scan').onclick = scan;
$('shot').onclick = shoot;
$('auto').onclick = toggleAuto;
$('savesetup').onclick = saveSetup;
$('apply').onclick = applyParams;
$('grid').onchange = () => { if(opened) startStream(); };
document.addEventListener('keydown', e => {
  if(e.code === 'Space' && !['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)){
    e.preventDefault(); shoot();
  }
});

loadPresets().then(refresh);
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
def main():
    global APP
    ap = argparse.ArgumentParser(
        description="Chụp ảnh mẫu huấn luyện từ camera lược (độc lập main.py)")
    ap.add_argument("--config", default="config.yaml",
                    help="config.yaml của chương trình gốc (chỉ để đọc preset)")
    ap.add_argument("--out", default="./dataset_comb",
                    help="thư mục lưu ảnh mẫu")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    APP = App(args.config, args.out)
    logger.info("Ảnh mẫu sẽ lưu vào: %s", APP.session.out_dir)
    logger.info("Mở giao diện: http://<ip-may>:%d", args.port)
    try:
        uvicorn.run(api, host=args.host, port=args.port, log_level="warning")
    finally:
        APP.close()


if __name__ == "__main__":
    main()
