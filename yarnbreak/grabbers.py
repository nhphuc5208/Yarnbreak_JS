"""Frame sources: video file, RTSP IP camera, Hikrobot MVS industrial camera.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from yarnbreak.config import AppConfig

logger = logging.getLogger("YarnBreak")


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME GRABBER
# ══════════════════════════════════════════════════════════════════════════════

class FrameGrabber:
    """Base class – subclasses implement _run()."""

    def __init__(self, cfg: AppConfig, frame_q: queue.Queue, stop: threading.Event,
                 display_q: Optional[queue.Queue] = None,
                 pause: Optional[threading.Event] = None,
                 on_camera_connected: Optional[callable] = None):
        self.cfg = cfg
        self.frame_q = frame_q
        self.display_q = display_q
        self.stop = stop
        self.pause = pause
        self.on_camera_connected = on_camera_connected
        self.source_fps: float = 30.0
        self._thread: Optional[threading.Thread] = None

    def _wait_if_paused(self) -> bool:
        """Block while the thermal guard holds the pause. Returns True if a pause
        actually happened (caller should resync any frame-pacing clock)."""
        paused = False
        while self.pause is not None and self.pause.is_set() and not self.stop.is_set():
            paused = True
            time.sleep(0.5)
        return paused

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="FrameGrabber")
        self._thread.start()

    def join(self, timeout: float = 3.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _push(self, frame: np.ndarray):
        """Drop oldest frame, push newest (keep queue at maxsize=1)."""
        try:
            self.frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            pass
        if self.display_q is not None:
            try:
                self.display_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.display_q.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self):
        raise NotImplementedError


# ─── Video file ───────────────────────────────────────────────────────────────

class VideoFileGrabber(FrameGrabber):
    def _run(self):
        path = self.cfg.video
        if not os.path.exists(path):
            logger.error(f"Video file not found: {path}")
            self.stop.set()
            return

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {path}")
            self.stop.set()
            return

        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Video: {path}  FPS={self.source_fps:.2f}  frames={total}  "
            f"pace={self.cfg.pace_video}  loop={self.cfg.loop_video}"
        )
        if self.on_camera_connected:
            self.on_camera_connected(True)

        frame_period = 1.0 / max(self.source_fps, 1.0)
        next_t = time.perf_counter()

        try:
            while not self.stop.is_set():
                # Thermal pause: stop decoding so the CPU cools; resync clock after.
                if self._wait_if_paused():
                    next_t = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.cfg.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_t = time.perf_counter()
                        continue
                    logger.info("End of video.")
                    self.stop.set()
                    break

                self._push(frame)

                if self.cfg.pace_video:
                    # Sleep until next frame's real-time slot
                    next_t += frame_period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        # We're behind schedule; resync to avoid drift
                        next_t = time.perf_counter()
        finally:
            cap.release()
            if self.on_camera_connected:
                self.on_camera_connected(False)


# ─── IP Camera (RTSP) ─────────────────────────────────────────────────────────

def build_rtsp_gstreamer_pipeline(rtsp_url: str, codec: str = "h264", latency: int = 200) -> str:
    """
    Hardware-accelerated GStreamer pipeline for Jetson RTSP capture.
    codec   : "h264" | "h265" – choose based on your camera stream.
    latency : rtspsrc latency in milliseconds (lower = less delay, more jitter).
    Requires: OpenCV built with GStreamer support (check cv2.getBuildInformation()).
    """
    codec = (codec or "h264").lower()
    if codec == "h265":
        depay_parse = "rtph265depay ! h265parse"
    else:
        depay_parse = "rtph264depay ! h264parse"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} "
        f"drop-on-latency=true buffer-mode=0 "
        f"! {depay_parse} "
        "! nvv4l2decoder "
        "! nvvidconv "
        "! video/x-raw,format=BGRx "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=1 max-buffers=1 sync=false"
    )


# Backward-compatible alias for the previous private name.
_gst_rtsp_pipeline = build_rtsp_gstreamer_pipeline


class IPCamGrabber(FrameGrabber):
    def _run(self):
        url = self.cfg.rtsp
        delay = self.cfg.reconnect_delay
        fail_count = 0

        while not self.stop.is_set():
            cap = self._open(url)

            if cap is None or not cap.isOpened():
                fail_count += 1
                if fail_count == 1 or fail_count % 10 == 0:
                    logger.warning(f"[Attempt {fail_count}] Cannot connect to {url}. Retry in {delay}s")
                time.sleep(delay)
                continue

            self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            logger.info(f"Camera connected: {url}  FPS={self.source_fps:.1f}")
            if self.on_camera_connected:
                self.on_camera_connected(True)
            fail_count = 0

            try:
                consec_fail = 0
                while not self.stop.is_set():
                    # Thermal pause: drop the live stream (buffers flush) while cooling.
                    self._wait_if_paused()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consec_fail += 1
                        if consec_fail >= 5:
                            logger.warning("Camera stream interrupted. Reconnecting…")
                            break
                        continue
                    consec_fail = 0
                    self._push(frame)
            finally:
                cap.release()
                if self.on_camera_connected:
                    self.on_camera_connected(False)

            if not self.stop.is_set():
                logger.info(f"Reconnecting in {delay}s…")
                time.sleep(delay)

    def _open(self, url: str) -> Optional[cv2.VideoCapture]:
        # Codec order: preferred from config first, the other as fallback.
        preferred = (self.cfg.codec or "h264").lower()
        codec_order = [preferred] + [c for c in ("h264", "h265") if c != preferred]
        latency = int(self.cfg.latency_ms or 200)

        logger.info(f"Connecting RTSP… codec_pref={preferred} latency={latency}ms")

        # 1) Try GStreamer hardware pipeline (Jetson)
        for codec in codec_order:
            pipe = build_rtsp_gstreamer_pipeline(url, codec=codec, latency=latency)
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info(f"Connected: GStreamer pipeline ({codec}).")
                return cap

        # 2) OpenCV software RTSP fallback
        logger.warning("GStreamer pipeline unavailable – falling back to cv2 RTSP…")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set timeouts to avoid blocking forever (OpenCV 4.5.4+)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass
        if cap.isOpened():
            logger.info("Connected: cv2 RTSP fallback.")
            return cap

        return None


# ─── Hikrobot MVS industrial camera (GigE / USB3 Vision) ──────────────────────

class MVSCameraGrabber(FrameGrabber):
    """Grab frames directly from a Hikrobot camera via the MVS SDK.

    Tested with MV-CS050-10GC (GigE, 2448×2048 BayerRG8). Each raw frame is
    converted to BGR8 on the host with MV_CC_ConvertPixelType so the rest of the
    pipeline sees the same np.ndarray (H, W, 3) BGR frames as the video/RTSP paths.
    """

    def _import_sdk(self):
        """Import the MVS Python bindings lazily so the module loads on machines
        without the SDK installed. Returns the MvImport module namespace."""
        sdk_path = self.cfg.mvs_sdk_path
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
        import MvCameraControl_class as mvs  # noqa: E402
        return mvs

    def _run(self):
        try:
            mvs = self._import_sdk()
        except Exception as e:
            logger.error(f"Cannot import MVS SDK (mvs_sdk_path={self.cfg.mvs_sdk_path}): {e}")
            self.stop.set()
            return

        delay = self.cfg.reconnect_delay
        fail_count = 0

        while not self.stop.is_set():
            cam = self._open_camera(mvs)
            if cam is None:
                fail_count += 1
                if fail_count == 1 or fail_count % 10 == 0:
                    logger.warning(f"[Attempt {fail_count}] No Hikrobot camera. Retry in {delay}s")
                time.sleep(delay)
                continue
            fail_count = 0
            if self.on_camera_connected:
                self.on_camera_connected(True)

            try:
                self._grab_loop(mvs, cam)
            except Exception as e:
                logger.warning(f"MVS grab loop error: {e}. Reconnecting…")
            finally:
                self._close_camera(cam)
                if self.on_camera_connected:
                    self.on_camera_connected(False)

            if not self.stop.is_set():
                logger.info(f"Reconnecting MVS camera in {delay}s…")
                time.sleep(delay)

    def _open_camera(self, mvs):
        """Enumerate, pick the camera (by serial if configured), open and configure it."""
        from ctypes import cast, POINTER, byref, c_bool

        dev_list = mvs.MV_CC_DEVICE_INFO_LIST()
        tlayer = mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE
        ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
        if ret != 0 or dev_list.nDeviceNum == 0:
            return None

        # Select device: matching serial if requested, else the first one.
        want = (self.cfg.mvs_serial or "").strip()
        chosen = None
        for i in range(dev_list.nDeviceNum):
            info = cast(dev_list.pDeviceInfo[i], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            serial = self._dev_serial(mvs, info)
            if not want or serial == want:
                chosen = (info, serial)
                if not want:
                    break
                if serial == want:
                    break
        if chosen is None:
            logger.warning(f"MVS: serial '{want}' not found among {dev_list.nDeviceNum} device(s)")
            return None

        info, serial = chosen
        cam = mvs.MvCamera()
        if cam.MV_CC_CreateHandle(info) != 0:
            return None
        if cam.MV_CC_OpenDevice(mvs.MV_ACCESS_Exclusive, 0) != 0:
            logger.warning("MVS: OpenDevice failed (camera busy or wrong subnet?)")
            cam.MV_CC_DestroyHandle()
            return None

        # GigE: use an optimal jumbo packet size for max throughput.
        if info.nTLayerType == mvs.MV_GIGE_DEVICE:
            ps = self.cfg.mvs_packet_size or cam.MV_CC_GetOptimalPacketSize()
            if ps > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", ps)

        # Continuous, free-running acquisition (no hardware trigger).
        cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        cam.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")

        # Optional manual exposure / gain (otherwise leave the camera's own auto).
        if self.cfg.mvs_exposure_us and self.cfg.mvs_exposure_us > 0:
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            cam.MV_CC_SetFloatValue("ExposureTime", float(self.cfg.mvs_exposure_us))
        if self.cfg.mvs_gain is not None and self.cfg.mvs_gain >= 0:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
            cam.MV_CC_SetFloatValue("Gain", float(self.cfg.mvs_gain))

        # Frame-rate cap. >0 caps AcquisitionFrameRate (lighter GPU/thermal load);
        # <=0 explicitly DISABLES the cap so the camera free-runs at its max. The
        # disable is important because the enable flag is stored in the camera and
        # would otherwise stick at a previous run's value.
        if self.cfg.mvs_fps and self.cfg.mvs_fps > 0:
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.cfg.mvs_fps))
        else:
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

        # Report resulting frame rate / resolution.
        fr = mvs.MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue("ResultingFrameRate", fr) == 0 and fr.fCurValue > 0:
            self.source_fps = float(fr.fCurValue)
        w = mvs.MVCC_INTVALUE(); h = mvs.MVCC_INTVALUE()
        cam.MV_CC_GetIntValue("Width", w); cam.MV_CC_GetIntValue("Height", h)

        if cam.MV_CC_StartGrabbing() != 0:
            logger.warning("MVS: StartGrabbing failed")
            cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
            return None

        logger.info(f"MVS camera connected: serial={serial}  "
                    f"{w.nCurValue}x{h.nCurValue} @ {self.source_fps:.1f}fps")
        return cam

    @staticmethod
    def _dev_serial(mvs, info) -> str:
        if info.nTLayerType == mvs.MV_GIGE_DEVICE:
            chars = info.SpecialInfo.stGigEInfo.chSerialNumber
        else:
            chars = info.SpecialInfo.stUsb3VInfo.chSerialNumber
        return "".join(chr(c) for c in chars if c)

    def _grab_loop(self, mvs, cam):
        from ctypes import byref, memset, sizeof, cast, POINTER, c_ubyte

        frame_out = mvs.MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        dst_buf = None
        dst_size = 0
        consec_fail = 0

        while not self.stop.is_set():
            # Thermal pause: stop the stream so the sensor/SoC load drops, then resume.
            if self.pause is not None and self.pause.is_set():
                cam.MV_CC_StopGrabbing()
                self._wait_if_paused()
                if self.stop.is_set():
                    break
                cam.MV_CC_StartGrabbing()
                consec_fail = 0
                continue

            ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
            if ret != 0:
                consec_fail += 1
                if consec_fail >= 5:
                    raise RuntimeError(f"GetImageBuffer failed (0x{ret & 0xffffffff:X})")
                continue
            consec_fail = 0

            try:
                fi = frame_out.stFrameInfo
                need = fi.nWidth * fi.nHeight * 3
                if dst_buf is None or dst_size < need:
                    dst_buf = (c_ubyte * need)()
                    dst_size = need

                cvt = mvs.MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(cvt), 0, sizeof(cvt))
                cvt.nWidth = fi.nWidth
                cvt.nHeight = fi.nHeight
                cvt.pSrcData = frame_out.pBufAddr
                cvt.nSrcDataLen = fi.nFrameLen
                cvt.enSrcPixelType = fi.enPixelType
                cvt.enDstPixelType = mvs.PixelType_Gvsp_BGR8_Packed
                cvt.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
                cvt.nDstBufferSize = dst_size

                if cam.MV_CC_ConvertPixelType(cvt) != 0:
                    continue

                # BGR8 packed → contiguous (H, W, 3) BGR frame for the pipeline.
                frame = np.frombuffer(dst_buf, dtype=np.uint8,
                                      count=need).reshape(fi.nHeight, fi.nWidth, 3).copy()
            finally:
                cam.MV_CC_FreeImageBuffer(frame_out)

            self._push(frame)

    def _close_camera(self, cam):
        try:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass


def make_grabber(cfg: AppConfig, fq: queue.Queue, stop: threading.Event,
                 display_q: Optional[queue.Queue] = None,
                 pause: Optional[threading.Event] = None,
                 on_camera_connected: Optional[callable] = None) -> FrameGrabber:
    if cfg.source == "ipcam":
        if not cfg.rtsp:
            raise ValueError("--rtsp URL is required when --source ipcam")
        return IPCamGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)
    if cfg.source == "mvs":
        return MVSCameraGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)
    # source == "video"
    if not cfg.video:
        raise ValueError("--video PATH is required when --source video")
    if not Path(cfg.video).is_file():
        raise FileNotFoundError(f"Video file not found: {cfg.video}")
    return VideoFileGrabber(cfg, fq, stop, display_q, pause, on_camera_connected)
