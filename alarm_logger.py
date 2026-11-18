import csv
import io
import os
import sqlite3
import time
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Ranh giới ca sản xuất (giờ địa phương):
#   Ca A: 06:00 → 18:00 cùng ngày
#   Ca B: 18:00 → 06:00 ngày hôm sau
SHIFT_DAY_START_HOUR = 6     # ngày sản xuất bắt đầu lúc 06:00
SHIFT_A_END_HOUR = 18        # A→B đổi ca lúc 18:00
_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _current_shift(now: Optional[datetime] = None) -> Tuple[datetime, datetime, str]:
    """Khung [start, end) và nhãn (A/B) của ca ĐANG CHẠY tại thời điểm now.
    Ca A = 06:00→18:00 cùng ngày; Ca B = 18:00→06:00 hôm sau.
    """
    now = now or datetime.now()
    h = now.hour
    today = now.date()
    if SHIFT_DAY_START_HOUR <= h < SHIFT_A_END_HOUR:          # 06:00–18:00 → Ca A
        start = datetime.combine(today, dtime(SHIFT_DAY_START_HOUR, 0))
        end = datetime.combine(today, dtime(SHIFT_A_END_HOUR, 0))
        label = "A"
    elif h >= SHIFT_A_END_HOUR:                               # 18:00–24:00 → Ca B (đêm nay)
        start = datetime.combine(today, dtime(SHIFT_A_END_HOUR, 0))
        end = datetime.combine(today, dtime(SHIFT_DAY_START_HOUR, 0)) + timedelta(days=1)
        label = "B"
    else:                                                     # 00:00–06:00 → Ca B (bắt đầu tối qua)
        start = datetime.combine(today, dtime(SHIFT_A_END_HOUR, 0)) - timedelta(days=1)
        end = datetime.combine(today, dtime(SHIFT_DAY_START_HOUR, 0))
        label = "B"
    return start, end, label


def _shift_windows(now: Optional[datetime] = None) -> Dict:
    """Trả về ca ĐANG CHẠY (now) và ca LIỀN TRƯỚC (last). Các ca luân phiên A/B mỗi
    12 giờ; ca trước = lùi đúng 12 giờ so với mốc bắt đầu ca hiện tại.
    """
    now = now or datetime.now()
    cur_start, cur_end, cur_label = _current_shift(now)
    last_end = cur_start
    last_start = cur_start - timedelta(hours=12)
    last_label = "B" if cur_label == "A" else "A"
    return {
        "now": (cur_start, cur_end, cur_label),
        "last": (last_start, last_end, last_label),
    }

# Load env variables
load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "./outputs/alarms.db")

class AlarmLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alarm_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    image_path TEXT,
                    machine_status TEXT NOT NULL,
                    ack_status TEXT NOT NULL,
                    note TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                )
            """)
            # Mỗi dòng = 1 lần báo sợi đứt ĐÃ ĐƯỢC TÍNH vào bộ đếm ca (đã qua cổng
            # conf TB > ngưỡng và qua debounce 10 phút). Tách khỏi alarm_logs để bộ
            # đếm ca không bị ảnh hưởng bởi các lần báo yếu / báo lại trong 10 phút.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shift_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    shift TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alarm_ts ON alarm_logs(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shift_ts ON shift_events(timestamp)"
            )
            conn.commit()

    def log_alarm(self, defect_type: str, confidence: float, camera_id: str,
                  image_path: Optional[str], machine_status: str) -> int:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alarm_logs (timestamp, defect_type, confidence, camera_id, image_path, machine_status, ack_status, note)
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', '')
                """,
                (timestamp, defect_type, confidence, camera_id, image_path, machine_status)
            )
            conn.commit()
            return cursor.lastrowid

    def update_image_path(self, alarm_id: int, image_path: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE alarm_logs SET image_path = ? WHERE id = ?",
                (image_path, alarm_id)
            )
            conn.commit()

    def acknowledge_alarm(self, alarm_id: int, note: str = "Acknowledged via Web API") -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE alarm_logs SET ack_status = 'ACKNOWLEDGED', note = ? WHERE id = ? AND ack_status = 'PENDING'",
                (note, alarm_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def acknowledge_all_pending(self, note: str = "Acknowledge all pending") -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE alarm_logs SET ack_status = 'ACKNOWLEDGED', note = ? WHERE ack_status = 'PENDING'",
                (note,)
            )
            conn.commit()
            return cursor.rowcount

    def get_alarms(self, limit: int = 50, offset: int = 0,
                   from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM alarm_logs WHERE 1=1"
        params = []
        
        if from_date:
            query += " AND timestamp >= ?"
            params.append(from_date)
        if to_date:
            query += " AND timestamp <= ?"
            params.append(to_date)
            
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def export_csv(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
        query = "SELECT * FROM alarm_logs WHERE 1=1"
        params = []
        
        if from_date:
            query += " AND timestamp >= ?"
            params.append(from_date)
        if to_date:
            query += " AND timestamp <= ?"
            params.append(to_date)
            
        query += " ORDER BY id DESC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["ID", "Timestamp", "Defect Type", "Confidence", "Camera ID", "Image Path", "Machine Status", "ACK Status", "Note"])
        
        # Write data rows
        for row in rows:
            writer.writerow([
                row["id"],
                row["timestamp"],
                row["defect_type"],
                row["confidence"],
                row["camera_id"],
                row["image_path"] or "",
                row["machine_status"],
                row["ack_status"],
                row["note"] or ""
            ])

        return output.getvalue()

    # ── Đếm số lần báo sợi đứt theo CA ────────────────────────────────────────
    def record_shift_event(self, confidence: float, debounce_s: float = 600.0) -> Optional[Dict]:
        """Ghi 1 lần báo ĐÃ TÍNH vào bộ đếm ca, ÁP DỤNG DEBOUNCE.

        Bên gọi phải tự kiểm tra điều kiện conf TB > ngưỡng TRƯỚC khi gọi hàm này.
        Ở đây chỉ lo debounce: nếu cách lần ĐÃ TÍNH gần nhất < debounce_s giây thì
        BỎ QUA (báo→tắt→báo lại trong 10 phút không tính). Đọc mốc từ DB nên đúng
        cả khi chương trình vừa khởi động lại. Trả về summary nếu tính, None nếu bỏ.
        """
        now = datetime.now()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT timestamp FROM shift_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                try:
                    last = datetime.strptime(row["timestamp"], _TS_FMT)
                    if (now - last).total_seconds() < debounce_s:
                        return None    # còn trong cửa debounce → không tính
                except ValueError:
                    pass
            _, _, cur_label = _current_shift(now)
            conn.execute(
                "INSERT INTO shift_events (timestamp, shift, confidence) VALUES (?, ?, ?)",
                (now.strftime(_TS_FMT), cur_label, round(float(confidence), 4))
            )
            conn.commit()
        return self.get_shift_summary(now)

    def get_shift_summary(self, now: Optional[datetime] = None) -> Dict:
        """Số lần báo ĐÃ TÍNH của ca ĐANG CHẠY (now) và ca LIỀN TRƯỚC (last)."""
        win = _shift_windows(now)

        def _count(rng: Tuple[datetime, datetime, str]) -> int:
            with self._get_connection() as conn:
                r = conn.execute(
                    "SELECT COUNT(*) AS c FROM shift_events WHERE timestamp >= ? AND timestamp < ?",
                    (rng[0].strftime(_TS_FMT), rng[1].strftime(_TS_FMT))
                ).fetchone()
                return int(r["c"]) if r else 0

        def _pack(rng: Tuple[datetime, datetime, str]) -> Dict:
            return {
                "label": rng[2],                       # A/B (thông tin phụ)
                "count": _count(rng),
                "start": rng[0].strftime(_TS_FMT),
                "end": rng[1].strftime(_TS_FMT),
            }

        return {
            "now": _pack(win["now"]),
            "last": _pack(win["last"]),
        }

    def log_audit(self, username: str, action: str, status: str, details: str = ""):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (timestamp, username, action, status, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, username, action, status, details)
            )
            conn.commit()

    def get_audit_logs(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]

    def export_audit_csv(self) -> str:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC"
            ).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)

        # Write headers
        writer.writerow(["ID", "Timestamp", "Username", "Action", "Status", "Details"])

        # Write data rows
        for row in rows:
            writer.writerow([
                row["id"],
                row["timestamp"],
                row["username"],
                row["action"],
                row["status"],
                row["details"] or ""
            ])

        return output.getvalue()

# Singleton instance
alarm_logger = AlarmLogger()
