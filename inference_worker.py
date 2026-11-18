import os
import sys
import threading
import traceback
import logging
import time
import yaml
from typing import Optional

from state_manager import state_manager
from alarm_logger import alarm_logger
from main import main

logger = logging.getLogger("YarnBreak.Worker")

def _load_relay_di_config():
    try:
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "enabled": data.get("relay_di_enabled", False),
                "port": data.get("relay_port", "/dev/ttyACM0"),
                "baud": data.get("relay_baud", 9600),
                "slave": data.get("relay_slave", 1),
                "red": data.get("relay_di_red", 1),
                "green": data.get("relay_di_green", 2),
                "interval": data.get("relay_di_poll_interval", 0.2),
                "mode": data.get("relay_di_control_mode", "pause_resume"),
                "status_enabled": data.get("relay_status_enabled", False),
                "status_channel": data.get("relay_status_channel", 8),
                "status2_enabled": data.get("relay_status2_enabled", False),
                "status2_channel": data.get("relay_status2_channel", 7),
                "alarm_light_channel": data.get("relay_alarm_light_channel", 2)
            }
    except Exception as e:
        logger.error(f"Failed to load config for DI watcher: {e}")
    return {"enabled": False}

def _load_shift_count_config():
    """Cấu hình đếm số lần báo theo ca (đọc mỗi lần báo → chỉnh không cần restart)."""
    try:
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "enabled": data.get("shift_count_enabled", True),
                "min_avg_conf": float(data.get("shift_count_avg_conf", 0.8)),
                "debounce_s": float(data.get("shift_count_debounce_s", 600.0)),
            }
    except Exception as e:
        logger.error(f"Failed to load shift-count config: {e}")
    return {"enabled": True, "min_avg_conf": 0.8, "debounce_s": 600.0}

class InferenceWorker:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._di_watcher_thread: Optional[threading.Thread] = None
        self._di_watcher_stop = threading.Event()
        self._start_di_watcher()

    def _start_di_watcher(self):
        def _watcher_loop():
            logger.info("Background Modbus DI Watcher thread started.")
            relay_obj = None
            last_green = False
            
            while not self._di_watcher_stop.is_set():
                cfg = _load_relay_di_config()
                if not cfg.get("enabled"):
                    if relay_obj is not None:
                        relay_obj.close()
                        relay_obj = None
                    time.sleep(2.0)
                    continue
                
                # We only poll the DI from this thread if the pipeline is STOPPED
                from state_manager import SystemState
                if state_manager.state == SystemState.STOPPED:
                    if relay_obj is None:
                        from modbus_relay import ModbusRelay
                        relay_obj = ModbusRelay(
                            port=cfg["port"],
                            baudrate=cfg["baud"],
                            slave=cfg["slave"]
                        )
                        if not relay_obj.open():
                            relay_obj = None
                            time.sleep(2.0)
                            continue
                        
                        # Set CH8 -> True (ON), CH7 -> False (OFF) when stopped
                        if cfg.get("status_enabled"):
                            relay_obj.set_relay(cfg["status_channel"], True)
                        if cfg.get("status2_enabled"):
                            relay_obj.set_relay(cfg["status2_channel"], False)
                        
                        # Dừng hẳn: CH3 (DO3, đèn tháp) TẮT; CH2 (DO2) BẬT để TẮT
                        # đèn xanh (đèn xanh đấu vào tiếp điểm NC của DO2).
                        try:
                            relay_obj.set_relay(3, False)
                            relay_obj.set_relay(cfg.get("alarm_light_channel", 2), True)
                        except Exception as e:
                            logger.error(f"Failed to set stopped-state relays in background DI watcher: {e}")
                    
                    try:
                        states = relay_obj.read_inputs()
                        if states is not None:
                            green_idx = cfg["green"] - 1
                            if 0 <= green_idx < len(states):
                                green_state = states[green_idx]
                                if green_state and not last_green:
                                    logger.info("🟢 Modbus GREEN button pressed while STOPPED! Starting pipeline...")
                                    alarm_logger.log_audit("Physical Button", "START", "SUCCESS", "Khởi động chương trình (Nút Xanh)")
                                    # Close relay connection first to release the port for main.py
                                    relay_obj.close()
                                    relay_obj = None
                                    self.start()
                                    # Wait a bit for start to complete and state to transition
                                    time.sleep(1.0)
                                    last_green = False
                                    continue
                                last_green = green_state
                    except Exception as e:
                        logger.error(f"Error in background DI poll: {e}")
                        if relay_obj is not None:
                            relay_obj.close()
                            relay_obj = None
                else:
                    # Pipeline is running, release port if we have it open
                    if relay_obj is not None:
                        relay_obj.close()
                        relay_obj = None
                
                time.sleep(cfg.get("interval", 0.2))
                
        self._di_watcher_thread = threading.Thread(target=_watcher_loop, daemon=True, name="BgDIWatcher")
        self._di_watcher_thread.start()

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("Inference worker is already running.")
            return False

        state_manager.start_pipeline()
        
        self._thread = threading.Thread(
            target=self._run_inference,
            daemon=True,
            name="InferenceWorkerThread"
        )
        self._thread.start()
        logger.info("Inference worker thread started.")
        return True

    def stop(self):
        if not self._thread or not self._thread.is_alive():
            logger.warning("Inference worker is not running.")
            return False

        logger.info("Stopping inference worker...")
        state_manager.stop_pipeline()
        self._thread.join(timeout=5.0)
        logger.info("Inference worker stopped.")
        return True

    def _run_inference(self):
        # Setup callbacks
        def on_camera_connected(connected: bool):
            state_manager.set_camera_connected(connected)
            logger.info(f"Camera status update: connected={connected}")

        def on_model_loaded():
            state_manager.set_model_loaded(True)
            logger.info("Model loaded successfully.")

        def on_fps_update(fps: float, infer_ms: float):
            state_manager.update_metrics(fps, infer_ms)

        def on_alarm_trigger(frame_idx: int, conf: float, avg_conf: float = 0.0):
            # Create a database entry (PENDING state)
            camera_id = "Hikrobot_MVS" if os.path.exists("/opt/MVS") else "Default_Cam"
            alarm_id = alarm_logger.log_alarm(
                defect_type="YarnBreak",
                confidence=conf,
                camera_id=camera_id,
                image_path=None,
                machine_status=state_manager.state.value
            )
            state_manager.trigger_alarm(alarm_id, "YarnBreak", conf)
            logger.warning(f"Alarm triggered! ID={alarm_id}, Confidence={conf:.2f}, avg10={avg_conf:.2f}")

            # Đếm theo ca: chỉ +1 khi TB conf 10 detect > ngưỡng VÀ đã qua debounce.
            try:
                sc = _load_shift_count_config()
                if sc["enabled"] and avg_conf > sc["min_avg_conf"]:
                    summary = alarm_logger.record_shift_event(conf, sc["debounce_s"])
                    if summary is not None:
                        now_s = summary["now"]
                        logger.info(f"Shift-count +1 → Now shift (Ca {now_s['label']}) "
                                    f"= {now_s['count']} (avg10={avg_conf:.2f})")
                    else:
                        logger.info(f"Shift-count bỏ qua (debounce 10 phút, avg10={avg_conf:.2f})")
            except Exception as e:
                logger.error(f"Shift-count update failed: {e}")

        def on_alarm_clear(frame_idx: int):
            state_manager.clear_alarm()
            logger.info("Alarm cleared.")

        def on_snapshot_saved(frame_idx: int, img_path: str):
            # Update the latest pending alarm log with the actual snapshot image path
            alarm_id = state_manager.last_alarm_id
            if alarm_id:
                # Save relative path or absolute path
                rel_path = os.path.relpath(img_path, os.getcwd())
                # Replace with web path format (forward slashes)
                rel_path = rel_path.replace("\\", "/")
                alarm_logger.update_image_path(alarm_id, rel_path)
                state_manager.set_last_alarm_image(alarm_id, rel_path)
                logger.info(f"Alarm ID={alarm_id} snapshot updated: {rel_path}")

        # Temporarily mock sys.argv to run main() using our config.yaml file
        # We always enforce --headless in production/production-web to prevent window display threads locking.
        orig_argv = sys.argv
        sys.argv = [
            "main.py",
            "--config", "config.yaml",
            "--headless"
        ]

        try:
            main(
                stop_evt=state_manager.stop_event,
                user_pause_evt=state_manager.user_pause_event,
                reset_alarm_evt=state_manager.reset_alarm_event,
                on_camera_connected=on_camera_connected,
                on_model_loaded=on_model_loaded,
                on_fps_update=on_fps_update,
                on_alarm_trigger=on_alarm_trigger,
                on_alarm_clear=on_alarm_clear,
                on_snapshot_saved=on_snapshot_saved
            )
        except Exception as e:
            logger.error(f"Error in inference thread: {e}")
            traceback.print_exc()
            state_manager.set_error()
        finally:
            sys.argv = orig_argv
            state_manager.stop_pipeline()
            logger.info("Inference thread execution finished.")

# Singleton instance
inference_worker = InferenceWorker()
