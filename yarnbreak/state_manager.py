import os
import threading
import time
from enum import Enum
from typing import Dict, Optional


class SystemState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ALARM = "ALARM"
    ERROR = "ERROR"


class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.lock = threading.Lock()
        self.state = SystemState.STOPPED
        
        # Control events
        self.stop_event = threading.Event()
        self.user_pause_event = threading.Event()
        self.reset_alarm_event = threading.Event()
        
        # Runtime metrics
        self.fps = 0.0
        self.infer_ms = 0.0
        self.uptime_start = 0.0
        self.camera_connected = False
        self.model_loaded = False
        self.alarm_active = False
        self.last_heartbeat = time.monotonic()

        # [multicam] Trạng thái theo từng camera: cam_id → dict(connected, fps,
        # infer_ms, alarm_active, last_alarm, name). alarm_active tổng của hệ =
        # OR(alarm_active camera trục, các camera lược).
        self.cameras: Dict[str, Dict] = {}
        # [multicam] Bộ đếm yêu cầu tắt cảnh báo. KHÔNG dùng chung
        # reset_alarm_event với camera trục vì detector của nó xoá event ngay sau
        # khi đọc → camera lược sẽ hụt. Mỗi bên nhớ seq lần cuối đã xử lý.
        self.alarm_reset_seq = 0

        # Last alarm details
        self.last_alarm_id = None
        self.last_alarm_time = None
        self.last_alarm_type = None
        self.last_alarm_conf = 0.0
        self.last_alarm_img = None

    def start_pipeline(self):
        with self.lock:
            self.stop_event.clear()
            self.user_pause_event.clear()
            self.reset_alarm_event.clear()
            self.uptime_start = time.time()
            self.state = SystemState.RUNNING
            self.alarm_active = False

    def stop_pipeline(self):
        with self.lock:
            self.stop_event.set()
            self.state = SystemState.STOPPED
            self.fps = 0.0
            self.infer_ms = 0.0
            self.uptime_start = 0.0
            self.camera_connected = False
            self.model_loaded = False
            self.alarm_active = False

    def pause_pipeline(self):
        with self.lock:
            self.user_pause_event.set()
            if self.state == SystemState.RUNNING or self.state == SystemState.ALARM:
                self.state = SystemState.PAUSED
            # Tạm dừng cũng TẮT alarm: xoá cờ alarm để khi resume không quay lại
            # ALARM cũ. Việc nhả relay CH1 do detector làm khi vào pause
            # (release_stop_output) — KHÔNG set reset_alarm_event ở đây vì nó khiến
            # detector force_off cả CH2, đua với _sync_pause làm DO2 tắt oan.
            self.alarm_active = False

    def resume_pipeline(self):
        with self.lock:
            self.user_pause_event.clear()
            if self.state == SystemState.PAUSED:
                if self.alarm_active:
                    self.state = SystemState.ALARM
                else:
                    self.state = SystemState.RUNNING

    def trigger_alarm(self, alarm_id: int, defect_type: str, confidence: float):
        with self.lock:
            self.alarm_active = True
            self.last_alarm_id = alarm_id
            self.last_alarm_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_alarm_type = defect_type
            self.last_alarm_conf = confidence
            if self.state == SystemState.RUNNING:
                self.state = SystemState.ALARM

    def clear_alarm(self):
        with self.lock:
            self.alarm_active = False
            if self.state == SystemState.ALARM:
                self.state = SystemState.RUNNING

    def set_error(self):
        with self.lock:
            self.state = SystemState.ERROR

    def update_metrics(self, fps: float, infer_ms: float):
        with self.lock:
            self.fps = fps
            self.infer_ms = infer_ms

    def set_camera_connected(self, connected: bool):
        with self.lock:
            self.camera_connected = connected

    # ── [multicam] Trạng thái từng camera ────────────────────────────────────
    def set_camera_status(self, cam_id: str, **fields):
        """Cập nhật/khởi tạo trạng thái của một camera (chỉ các field truyền vào)."""
        with self.lock:
            cur = self.cameras.setdefault(cam_id, {
                "id": cam_id, "name": cam_id, "connected": False,
                "fps": 0.0, "infer_ms": 0.0, "alarm_active": False,
                "last_alarm": None,
            })
            cur.update(fields)

    def bump_alarm_reset(self) -> int:
        """Ghi nhận một yêu cầu TẮT CẢNH BÁO cho mọi camera (web gọi)."""
        with self.lock:
            self.alarm_reset_seq += 1
            for c in self.cameras.values():
                c["alarm_active"] = False
            return self.alarm_reset_seq

    def any_camera_alarm(self) -> bool:
        """True nếu BẤT KỲ camera nào đang báo đứt (gồm cả camera trục)."""
        with self.lock:
            if self.alarm_active:
                return True
            return any(c.get("alarm_active") for c in self.cameras.values())

    def set_model_loaded(self, loaded: bool):
        with self.lock:
            self.model_loaded = loaded

    def set_last_alarm_image(self, alarm_id: int, img_path: str):
        with self.lock:
            if self.last_alarm_id == alarm_id:
                self.last_alarm_img = img_path

    def _read_temp(self, ztype: str) -> Optional[float]:
        try:
            if not hasattr(self, "_thermal_paths"):
                self._thermal_paths = {}
            
            if ztype not in self._thermal_paths:
                import glob
                for zd in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
                    with open(os.path.join(zd, "type")) as f:
                        if f.read().strip() == ztype:
                            self._thermal_paths[ztype] = os.path.join(zd, "temp")
                            break
            
            path = self._thermal_paths.get(ztype)
            if path and os.path.exists(path):
                with open(path) as f:
                    return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            pass
        return None

    def get_status(self) -> Dict:
        # Đọc ngoài lock
        cpu_temp = self._read_temp("cpu-thermal")
        gpu_temp = self._read_temp("gpu-thermal")

        with self.lock:
            uptime = 0.0
            if self.uptime_start > 0:
                uptime = time.time() - self.uptime_start

            return {
                "state": self.state.value,
                "running": self.state != SystemState.STOPPED,
                "paused": self.user_pause_event.is_set(),
                "alarm_active": self.alarm_active,
                "camera_connected": self.camera_connected,
                "model_loaded": self.model_loaded,
                "fps": round(self.fps, 1),
                "infer_ms": round(self.infer_ms, 1),
                "uptime": int(uptime),
                "cpu_temp": cpu_temp,
                "gpu_temp": gpu_temp,
                # [multicam] Thêm mới, KHÔNG đổi field cũ để web hiện tại không vỡ.
                "cameras": {k: dict(v) for k, v in self.cameras.items()},
                "any_alarm": self.alarm_active or any(
                    c.get("alarm_active") for c in self.cameras.values()
                ),
                "last_alarm": {
                    "id": self.last_alarm_id,
                    "time": self.last_alarm_time,
                    "type": self.last_alarm_type,
                    "confidence": round(self.last_alarm_conf, 2),
                    "image_path": self.last_alarm_img
                } if self.last_alarm_time else None
            }

# Singleton instance
state_manager = StateManager()
