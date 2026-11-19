import os
import secrets
import uvicorn
import subprocess
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Header, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from yarnbreak.auth import load_accounts, verify_password
from yarnbreak.state_manager import state_manager
from yarnbreak.alarm_logger import alarm_logger
from yarnbreak.inference_worker import inference_worker
from yarnbreak.live_view import get_stream

# Load config
load_dotenv()

HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8000"))

# --- Accounts (username -> {password_hash, role}) --------------------------
# Đọc từ .env dưới dạng hash PBKDF2 (xem auth.py). Không có mật khẩu mặc định:
# tài khoản nào chưa cấu hình trong .env thì bị vô hiệu hoá.
ACCOUNTS = load_accounts()

# In-memory session store: session_token -> {"username", "role"}
# Mất khi restart server; client sẽ nhận 401 và đăng nhập lại.
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()

SNAPSHOT_DIR = os.getenv("ALARM_IMAGE_DIR", "./outputs/snapshots")

# Ensure static and templates dirs exist
os.makedirs("./static", exist_ok=True)
os.makedirs("./templates", exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Initialize app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("AUTO_START_AI", "true").lower() == "true":
        inference_worker.start()
    yield

app = FastAPI(title="YarnBreak Control System", version="2.0.0",
              lifespan=lifespan)

# CORS: the web UI is served from this same server, so no cross-origin access
# is needed by default. Set CORS_ORIGINS="http://host1:port,http://host2" in
# .env only if another front-end must call this API.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,  # auth uses a Bearer header, not cookies
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

# Mount static and snapshots
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

# Templates
templates = Jinja2Templates(directory="templates")


# --- Authentication helpers -------------------------------------------------
def _create_session(username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    with SESSIONS_LOCK:
        SESSIONS[token] = {"username": username, "role": role}
    return token


def _destroy_session(token: str):
    with SESSIONS_LOCK:
        SESSIONS.pop(token, None)


async def get_current_user(authorization: str = Header(None)) -> dict:
    """Xác thực Bearer <session_token> -> trả {'username', 'role'}."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme, must be Bearer"
        )

    with SESSIONS_LOCK:
        session = SESSIONS.get(token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn"
        )
    return {"token": token, **session}


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Chỉ cho phép tài khoản admin."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Chỉ tài khoản admin mới được phép truy cập chức năng này"
        )
    return user


def _validate_token(token: str) -> bool:
    """Kiểm tra session token còn hợp lệ. Dùng cho endpoint <img> stream — nơi
    trình duyệt không gửi được header Authorization nên token đi qua query param."""
    if not token:
        return False
    with SESSIONS_LOCK:
        return token in SESSIONS


# --- Web UI Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# --- Authentication Endpoints ---
@app.post("/api/login")
async def login(data: dict):
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    account = ACCOUNTS.get(username)
    if not account or not verify_password(password, account["password_hash"]):
        alarm_logger.log_audit(
            username or "(unknown)", "LOGIN", "FAILED", "Sai tên đăng nhập hoặc mật khẩu"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu"
        )

    role = account["role"]
    token = _create_session(username, role)
    alarm_logger.log_audit(username, "LOGIN", "SUCCESS", f"Đăng nhập (role={role})")
    return {"success": True, "token": token, "username": username, "role": role}


@app.get("/api/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/logout")
async def logout(user: dict = Depends(get_current_user)):
    alarm_logger.log_audit(user["username"], "LOGOUT", "SUCCESS", "Đăng xuất")
    _destroy_session(user["token"])
    return {"success": True}


# --- Control API Routes ---
@app.get("/api/status")
async def get_status(user: dict = Depends(get_current_user)):
    return state_manager.get_status()


@app.get("/api/shift-counts")
async def get_shift_counts(user: dict = Depends(get_current_user)):
    """Số lần báo sợi đứt ĐÃ TÍNH theo ca của ngày sản xuất hiện tại (Ca A / Ca B)."""
    return alarm_logger.get_shift_summary()


@app.get("/api/live/mjpeg")
async def live_mjpeg(token: str = Query(None), cam: str = Query("cam_main"),
                     authorization: str = Header(None)):
    """Luồng xem camera trực tiếp (MJPEG) có viền ROI để căn chỉnh vị trí camera.
    Xác thực qua query ?token= (thẻ <img> không gửi được header) hoặc Bearer.
    [multicam] ?cam=<cam_id> chọn camera; mặc định "cam_main" để API cũ không vỡ."""
    tok = token
    if not tok and authorization:
        try:
            scheme, t = authorization.split()
            if scheme.lower() == "bearer":
                tok = t
        except ValueError:
            tok = None

    if not _validate_token(tok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn"
        )

    return StreamingResponse(
        get_stream(cam).frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/start")
async def start_ai(user: dict = Depends(get_current_user)):
    success = inference_worker.start()
    alarm_logger.log_audit(
        user["username"], "START", "SUCCESS" if success else "FAILED",
        "Bật chương trình nhận diện AI"
    )
    return {"success": success, "status": state_manager.get_status()}


@app.post("/api/stop")
async def stop_ai(data: dict = None, user: dict = Depends(get_current_user)):
    # DỪNG chương trình yêu cầu nhập lại mật khẩu của tài khoản đang đăng nhập.
    password = (data or {}).get("password") or ""
    account = ACCOUNTS.get(user["username"])
    if not account or not verify_password(password, account["password_hash"]):
        alarm_logger.log_audit(
            user["username"], "STOP", "DENIED", "Nhập sai mật khẩu khi dừng chương trình"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sai mật khẩu — không thể dừng chương trình"
        )

    success = inference_worker.stop()
    alarm_logger.log_audit(
        user["username"], "STOP", "SUCCESS" if success else "FAILED",
        "Dừng chương trình nhận diện (đã xác nhận mật khẩu)"
    )
    return {"success": success, "status": state_manager.get_status()}


@app.post("/api/pause")
async def pause_ai(user: dict = Depends(get_current_user)):
    state_manager.pause_pipeline()
    alarm_logger.log_audit(user["username"], "PAUSE", "SUCCESS", "Tạm dừng nhận diện")
    return {"success": True, "status": state_manager.get_status()}


@app.post("/api/resume")
async def resume_ai(user: dict = Depends(get_current_user)):
    state_manager.resume_pipeline()
    alarm_logger.log_audit(user["username"], "RESUME", "SUCCESS", "Tiếp tục nhận diện")
    return {"success": True, "status": state_manager.get_status()}


@app.post("/api/reset-alarm")
async def reset_alarm(user: dict = Depends(get_current_user)):
    # 1. Acknowledge inside the database
    alarm_id = state_manager.last_alarm_id
    note = f"Reset thủ công qua web UI bởi {user['username']}"
    if alarm_id:
        alarm_logger.acknowledge_alarm(alarm_id, note)
    else:
        # Fallback to acknowledge all pending alarms if somehow last_alarm_id is not tracked
        alarm_logger.acknowledge_all_pending(note + " (Acknowledge All)")

    # 2. Trigger hardware relay/còi tắt via reset alarm event
    state_manager.reset_alarm_event.set()

    # 3. Reset StateManager alarm states
    state_manager.clear_alarm()

    # 4. [multicam] Báo cho các camera lược tắt cảnh báo (bộ đếm riêng, không
    #    tranh chấp reset_alarm_event vốn bị detector camera trục xoá ngay).
    state_manager.bump_alarm_reset()

    alarm_logger.log_audit(user["username"], "RESET_ALARM", "SUCCESS", "Tắt cảnh báo")
    return {"success": True, "status": state_manager.get_status()}


@app.get("/api/alarms")
async def get_alarms(
    limit: int = 50,
    offset: int = 0,
    from_date: str = Query(None),
    to_date: str = Query(None),
    user: dict = Depends(get_current_user)
):
    alarms = alarm_logger.get_alarms(limit, offset, from_date, to_date)
    return {"alarms": alarms}


@app.get("/api/alarms/export")
async def export_alarms(
    from_date: str = Query(None),
    to_date: str = Query(None),
    user: dict = Depends(get_current_user)
):
    csv_data = alarm_logger.export_csv(from_date, to_date)

    # Return as download response
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=yarnbreak_alarms.csv"}
    )


# --- Audit log (Lịch sử truy cập & hành động) — CHỈ admin ---
@app.get("/api/audit")
async def get_audit(
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(require_admin)
):
    logs = alarm_logger.get_audit_logs(limit, offset)
    return {"audit": logs}


@app.get("/api/audit/export")
async def export_audit(user: dict = Depends(require_admin)):
    csv_data = alarm_logger.export_audit_csv()
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=yarnbreak_audit.csv"}
    )


# --- Shutdown Endpoint ---
@app.post("/api/shutdown")
async def shutdown_system(user: dict = Depends(get_current_user)):
    # Bắt buộc phải DỪNG chương trình nhận diện trước khi được tắt máy.
    if state_manager.get_status()["running"]:
        alarm_logger.log_audit(
            user["username"], "SHUTDOWN", "DENIED",
            "Chưa dừng chương trình nhận diện"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Vui lòng nhấn STOP để dừng chương trình trước khi tắt máy Jetson."
        )

    alarm_logger.log_audit(user["username"], "SHUTDOWN", "SUCCESS", "Tắt nguồn máy Jetson")

    # Trigger an orderly OS shutdown in a background thread:
    # dừng sạch từng tiến trình còn sót lại rồi mới tắt hẳn.
    def perform_shutdown():
        time.sleep(2.0)  # Wait 2 seconds for server to respond to client

        # 1. Bảo đảm luồng nhận diện AI (camera/relay) đã dừng và dọn dẹp xong.
        #    (chương trình đã được STOP trước đó; gọi lại để chắc chắn không còn sót)
        try:
            inference_worker.stop()
        except Exception:
            pass

        # 2. Đẩy mọi dữ liệu xuống đĩa (DB cảnh báo, ảnh snapshot) trước khi tắt.
        try:
            subprocess.run(["sync"], timeout=10)
        except Exception:
            pass

        # 3. Tắt nguồn. systemd sẽ dừng nốt service một cách có trật tự
        #    (gửi SIGTERM, chờ TimeoutStopSec rồi mới kết thúc) trước khi power off.
        subprocess.run(["sudo", "shutdown", "-h", "now"])

    threading.Thread(target=perform_shutdown, daemon=True).start()
    return {"success": True, "message": "System is shutting down..."}


# --- Reset (reboot) Endpoint ---
@app.post("/api/reset-jetson")
async def reset_system(user: dict = Depends(get_current_user)):
    # Bắt buộc phải DỪNG chương trình nhận diện trước khi được khởi động lại.
    if state_manager.get_status()["running"]:
        alarm_logger.log_audit(
            user["username"], "RESET", "DENIED",
            "Chưa dừng chương trình nhận diện"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Vui lòng nhấn STOP để dừng chương trình trước khi khởi động lại máy Jetson."
        )

    alarm_logger.log_audit(user["username"], "RESET", "SUCCESS", "Khởi động lại máy Jetson")

    def perform_reset():
        time.sleep(2.0)  # Chờ server trả lời client trước khi khởi động lại

        # 1. Dừng sạch luồng nhận diện AI (camera/relay) dù đang chạy hay đã dừng.
        try:
            inference_worker.stop()
        except Exception:
            pass

        # 2. Đẩy mọi dữ liệu xuống đĩa (DB cảnh báo, ảnh snapshot) trước khi reboot.
        try:
            subprocess.run(["sync"], timeout=10)
        except Exception:
            pass

        # 3. Khởi động lại. systemd sẽ dừng nốt service một cách có trật tự
        #    (gửi SIGTERM, chờ TimeoutStopSec) trước khi reboot.
        subprocess.run(["sudo", "shutdown", "-r", "now"])

    threading.Thread(target=perform_reset, daemon=True).start()
    return {"success": True, "message": "System is rebooting..."}


if __name__ == "__main__":
    uvicorn.run("web_server:app", host=HOST, port=PORT, reload=False)
