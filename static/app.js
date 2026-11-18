// YarnBreak Web Control App

const API_BASE = '/api';
let pollingTimer = null;
let shiftTimer = null;
let token = localStorage.getItem('yarnbreak_token');
let currentRole = localStorage.getItem('yarnbreak_role');
let currentUser = localStorage.getItem('yarnbreak_user');

// DOM Elements
const loginOverlay = document.getElementById('login-overlay');
const usernameInput = document.getElementById('username-input');
const passwordInput = document.getElementById('password-input');
const loginBtn = document.getElementById('login-btn');
const loginError = document.getElementById('login-error');

// Audit (Access & Action History) — admin only
const auditSection = document.getElementById('audit-section');
const auditLogsBody = document.getElementById('audit-logs-body');
const exportAuditBtn = document.getElementById('export-audit-btn');
const auditToggle = document.getElementById('audit-toggle');
const auditCount = document.getElementById('audit-count');

const systemStatusDot = document.getElementById('system-status-dot');
const systemStateLabel = document.getElementById('system-state-label');

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const pauseBtn = document.getElementById('pause-btn');
const resumeBtn = document.getElementById('resume-btn');
const resetAlarmBtn = document.getElementById('reset-alarm-btn');
const shutdownBtn = document.getElementById('shutdown-btn');
const shutdownHint = document.getElementById('shutdown-hint');
const resetBtn = document.getElementById('reset-btn');

const fpsVal = document.getElementById('fps-val');
const inferVal = document.getElementById('infer-val');
const cameraVal = document.getElementById('camera-val');
const modelVal = document.getElementById('model-val');
const cpuTempVal = document.getElementById('cpu-temp-val');
const gpuTempVal = document.getElementById('gpu-temp-val');
const uptimeVal = document.getElementById('uptime-val');

let lastTempUpdateTime = 0;

const activeAlarmSection = document.getElementById('active-alarm-section');
const activeAlarmType = document.getElementById('active-alarm-type');
const activeAlarmConf = document.getElementById('active-alarm-conf');
const activeAlarmTime = document.getElementById('active-alarm-time');
const activeAlarmImg = document.getElementById('active-alarm-img');

const alarmLogsBody = document.getElementById('alarm-logs-body');
const exportCsvBtn = document.getElementById('export-csv-btn');
const logoutBtn = document.getElementById('logout-btn');

// Shift yarn-break counters (Now shift / Last shift)
const shiftNowCount = document.getElementById('shift-now-count');
const shiftLastCount = document.getElementById('shift-last-count');
const shiftNowDate = document.getElementById('shift-now-date');
const shiftLastDate = document.getElementById('shift-last-date');
const shiftNowHours = document.getElementById('shift-now-hours');
const shiftLastHours = document.getElementById('shift-last-hours');

// Image Modal Elements
const imageModal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const closeModal = document.querySelector('.close-modal');

// Live Camera Modal Elements
const liveModal = document.getElementById('live-modal');
const liveClose = document.getElementById('live-close');
const liveImg = document.getElementById('live-img');
const liveStatus = document.getElementById('live-status');

// Hộp thoại xác nhận trong trang (thay confirm()/prompt() của trình duyệt)
const dlgModal = document.getElementById('confirm-dialog');
const dlgTitle = document.getElementById('dialog-title');
const dlgMsg = document.getElementById('dialog-msg');
const dlgPassField = document.getElementById('dialog-pass-field');
const dlgPassLabel = document.getElementById('dialog-pass-label');
const dlgPass = document.getElementById('dialog-pass');
const dlgPassToggle = document.getElementById('dialog-pass-toggle');
const dlgError = document.getElementById('dialog-error');
const dlgCancel = document.getElementById('dialog-cancel');
const dlgOk = document.getElementById('dialog-ok');

// Trạng thái hệ thống mới nhất (cập nhật mỗi lần poll) — dùng cho live view.
let latestStatus = null;
let liveHideTimer = null;

// Init application
document.addEventListener('DOMContentLoaded', () => {
    if (!token) {
        showLogin();
    } else {
        // Có token cũ trong localStorage: xác nhận phiên còn hợp lệ và lấy lại role.
        initSession();
    }

    // Event listeners
    loginBtn.addEventListener('click', handleLogin);
    usernameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') passwordInput.focus();
    });
    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleLogin();
    });

    startBtn.addEventListener('click', async () => {
        const ok = await askConfirm({
            title: t('dlg.titleStart'),
            message: t('confirm.start'),
            okClass: 'success',
        });
        if (ok) sendControl('start');
    });
    stopBtn.addEventListener('click', async () => {
        // DỪNG chương trình yêu cầu nhập lại mật khẩu tài khoản đang đăng nhập.
        // Nhập trong hộp thoại của trang (type="password") nên không hiện rõ
        // mật khẩu như prompt() của trình duyệt trước đây.
        const res = await askConfirm({
            title: t('dlg.titleStop'),
            message: t('confirm.stop'),
            password: true,
            okClass: 'danger',
        });
        if (!res) return;
        sendControl('stop', { password: res.password });
    });
    pauseBtn.addEventListener('click', async () => {
        const ok = await askConfirm({
            title: t('dlg.titlePause'),
            message: t('confirm.pause'),
            okClass: 'warning',
        });
        if (ok) sendControl('pause');
    });
    resumeBtn.addEventListener('click', async () => {
        const ok = await askConfirm({
            title: t('dlg.titleResume'),
            message: t('confirm.resume'),
            okClass: 'primary',
        });
        if (ok) sendControl('resume');
    });
    resetAlarmBtn.addEventListener('click', () => sendControl('reset-alarm'));

    shutdownBtn.addEventListener('click', async () => {
        if (shutdownBtn.disabled) {
            showToast(t('toast.shutdownBlocked'), "warning");
            return;
        }
        // Tắt nguồn: xác nhận HAI bước, giữ nguyên như trước.
        const step1 = await askConfirm({
            title: t('dlg.titleShutdown'),
            message: t('confirm.shutdown1'),
            okClass: 'danger',
        });
        if (!step1) return;
        const step2 = await askConfirm({
            title: t('dlg.titleShutdownFinal'),
            message: t('confirm.shutdown2'),
            okClass: 'danger',
        });
        if (step2) sendControl('shutdown');
    });

    resetBtn.addEventListener('click', async () => {
        if (resetBtn.disabled) {
            showToast(t('toast.resetBlocked'), "warning");
            return;
        }
        const step1 = await askConfirm({
            title: t('dlg.titleReset'),
            message: t('confirm.reset1'),
            okClass: 'danger',
        });
        if (!step1) return;
        const step2 = await askConfirm({
            title: t('dlg.titleResetFinal'),
            message: t('confirm.reset2'),
            okClass: 'danger',
        });
        if (step2) sendControl('reset-jetson');
    });

    // Hộp thoại xác nhận: nút, Enter, Escape, hiện/ẩn mật khẩu
    dlgOk.addEventListener('click', submitDialog);
    dlgCancel.addEventListener('click', () => closeDialog(null));
    dlgModal.addEventListener('click', (e) => {
        if (e.target === dlgModal) closeDialog(null);
    });
    dlgPass.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitDialog();
    });
    dlgPassToggle.addEventListener('click', () => {
        const hidden = dlgPass.type === 'password';
        dlgPass.type = hidden ? 'text' : 'password';
        dlgPassToggle.textContent = hidden ? t('dlg.hide') : t('dlg.show');
        dlgPass.focus();
    });

    exportCsvBtn.addEventListener('click', exportCSV);
    if (exportAuditBtn) {
        exportAuditBtn.addEventListener('click', (e) => {
            e.stopPropagation();   // đừng để bấm "Xuất CSV" lại đóng/mở mục
            exportAuditCSV();
        });
    }
    // Các mục gập được (mặc định đóng): lịch sử thao tác và nguồn thiết bị
    setupCollapsible('audit-toggle', 'audit-section');
    setupCollapsible('power-toggle', 'power-section');
    logoutBtn.addEventListener('click', logout);
    
    // Modal events
    closeModal.addEventListener('click', () => imageModal.style.display = "none");
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal || e.target === closeModal) {
            imageModal.style.display = "none";
        }
    });
    
    activeAlarmImg.addEventListener('click', () => {
        if (activeAlarmImg.src) {
            openImageModal(activeAlarmImg.src);
        }
    });

    // Live camera view
    if (liveClose) liveClose.addEventListener('click', closeLiveView);
    if (liveModal) {
        liveModal.addEventListener('click', (e) => {
            if (e.target === liveModal) closeLiveView();
        });
    }
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (dlgModal && dlgModal.classList.contains('open')) {
            closeDialog(null);
            return;
        }
        if (liveModal && liveModal.style.display === 'block') {
            closeLiveView();
        }
    });
    if (liveImg) {
        // Chromium phát sự kiện load cho mỗi frame của luồng MJPEG → có frame là ẩn báo.
        liveImg.addEventListener('load', () => {
            if (liveStatus) liveStatus.style.display = 'none';
        });
        liveImg.addEventListener('error', () => {
            if (liveStatus) {
                liveStatus.style.display = 'block';
                liveStatus.textContent = t('live.error');
            }
        });
    }

    // Khi đổi ngôn ngữ: vẽ lại bảng log động (View Frame / None / no-data...)
    document.addEventListener('langchange', () => {
        if (token) {
            fetchAlarmLogs();
            if (currentRole === 'admin') fetchAuditLogs();
        }
    });
});

// Authentication logic
function showLogin() {
    loginOverlay.style.display = 'flex';
}

function hideLogin() {
    loginOverlay.style.display = 'none';
}

async function handleLogin() {
    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    if (!username) {
        showError(t('login.userEmpty'));
        return;
    }
    if (!password) {
        showError(t('login.passEmpty'));
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        if (response.ok) {
            const data = await response.json();
            token = data.token;
            currentRole = data.role;
            currentUser = data.username;
            localStorage.setItem('yarnbreak_token', token);
            localStorage.setItem('yarnbreak_role', currentRole);
            localStorage.setItem('yarnbreak_user', currentUser);
            passwordInput.value = '';
            startSession();
        } else {
            const data = await response.json();
            showError(data.detail || t('login.failed'));
        }
    } catch (err) {
        showError(t('login.network'));
        console.error(err);
    }
}

// Xác nhận phiên đang lưu trong localStorage còn hợp lệ, lấy lại role rồi vào app.
async function initSession() {
    const data = await apiFetch('me');
    if (!data) return;   // apiFetch tự logout nếu 401/403
    currentRole = data.role;
    currentUser = data.username;
    localStorage.setItem('yarnbreak_role', currentRole);
    localStorage.setItem('yarnbreak_user', currentUser);
    startSession();
}

// Vào trạng thái đã đăng nhập: ẩn login, áp role, bắt đầu polling + tải logs.
function startSession() {
    hideLogin();
    applyRole(currentRole);
    startPolling();
    fetchAlarmLogs();
}

// Phân quyền giao diện: chỉ admin thấy card "Lịch sử truy cập & hành động".
function applyRole(role) {
    const isAdmin = role === 'admin';
    if (auditSection) auditSection.style.display = isAdmin ? 'block' : 'none';
    if (isAdmin) fetchAuditLogs();
}

function showError(msg) {
    loginError.textContent = msg;
    setTimeout(() => { loginError.textContent = ''; }, 3000);
}

function logout() {
    // Báo server xóa phiên (best-effort), rồi dọn localStorage và tải lại.
    if (token) {
        fetch(`${API_BASE}/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        }).catch(() => {});
    }
    localStorage.removeItem('yarnbreak_token');
    localStorage.removeItem('yarnbreak_role');
    localStorage.removeItem('yarnbreak_user');
    token = null;
    currentRole = null;
    currentUser = null;
    stopPolling();
    closeLiveView();
    window.location.reload();
}

// API Helpers
async function apiFetch(endpoint, method = 'GET', body = null) {
    const headers = {
        'Authorization': `Bearer ${token}`
    };
    
    if (body) {
        headers['Content-Type'] = 'application/json';
    }
    
    const options = {
        method,
        headers
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`${API_BASE}/${endpoint}`, options);
        // 401 = phiên hỏng/hết hạn -> đăng xuất. 403 = bị từ chối quyền hoặc
        // sai mật khẩu khi STOP -> KHÔNG đăng xuất, chỉ báo lỗi.
        if (response.status === 401) {
            logout();
            return null;
        }
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "API response error");
        }
        return await response.json();
    } catch (err) {
        console.error(`API Error (${endpoint}):`, err);
        // Các poll NỀN (status, shift-counts, alarms) lỗi thoáng qua thì im lặng —
        // không bung toast "Lỗi kết nối" spam mỗi vài giây; poll sau sẽ tự phục hồi.
        if (endpoint !== 'status' && endpoint !== 'shift-counts' && !endpoint.startsWith('alarms')) {
            showToast(t('toast.connErr', { endpoint: endpoint, message: err.message }), 'error');
        }
        return null;
    }
}

// Control actions
async function sendControl(action, body = null) {
    const actionLabel = action.toUpperCase();
    showToast(t('toast.sending', { action: actionLabel }), 'info');
    const data = await apiFetch(action, 'POST', body);
    if (data) {
        if (data.success !== false) {
            showToast(t('toast.success', { action: actionLabel }), 'success');
            if (action === 'shutdown') {
                showToast(t('toast.shuttingDown'), "warning");
                stopPolling();
            }
            if (action === 'reset-jetson') {
                showToast(t('toast.rebooting'), "warning");
                stopPolling();
            }
        } else {
            showToast(t('toast.failed', { action: actionLabel }), 'error');
        }
        if (data.status) {
            updateUI(data.status);
            fetchAlarmLogs();
        }
        if (currentRole === 'admin') fetchAuditLogs();
    } else {
        showToast(t('toast.noResponse', { action: actionLabel }), 'error');
    }
}

// Polling status
function startPolling() {
    stopPolling();
    pollStatus(); // initial poll
    pollingTimer = setInterval(pollStatus, 1000);
    // Bộ đếm theo ca thay đổi chậm (debounce 10 phút) → 5s/lần là đủ.
    fetchShiftCounts();
    shiftTimer = setInterval(fetchShiftCounts, 5000);
}

function stopPolling() {
    if (pollingTimer) {
        clearInterval(pollingTimer);
        pollingTimer = null;
    }
    if (shiftTimer) {
        clearInterval(shiftTimer);
        shiftTimer = null;
    }
}

async function pollStatus() {
    const statusData = await apiFetch('status');
    if (statusData) {
        updateUI(statusData);
    }
}

// Số lần báo sợi đứt theo ca (Ca A / Ca B) của ngày sản xuất hiện tại.
async function fetchShiftCounts() {
    const data = await apiFetch('shift-counts');
    if (data) renderShiftCounts(data);
}

// "YYYY-MM-DD HH:MM:SS" | "YYYY-MM-DD" -> {d, m, y} (số nguyên) hoặc null.
function parseYmd(ts) {
    if (!ts) return null;
    const m = String(ts).match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!m) return null;
    return { y: +m[1], m: +m[2], d: +m[3] };
}
const pad2 = (n) => String(n).padStart(2, '0');
const fmtDMY = (o) => o ? `${pad2(o.d)}/${pad2(o.m)}/${o.y}` : '--/--/----';
const fmtDM = (o) => o ? `${pad2(o.d)}/${pad2(o.m)}` : '--/--';

// Giờ "HH:MM" từ chuỗi "YYYY-MM-DD HH:MM:SS".
function hhmm(ts) {
    const m = String(ts || '').match(/\s(\d{2}):(\d{2})/);
    return m ? `${m[1]}:${m[2]}` : '--:--';
}

// Khoảng ngày của 1 ca: cùng ngày → "19/08/2026"; vắt đêm → "19/08 → 20/08/2026".
function shiftDateText(sh) {
    const s = parseYmd(sh.start), e = parseYmd(sh.end);
    if (!s || !e) return '--/--/----';
    // end là mốc loại trừ (06:00 hôm sau) → nếu khác ngày thì đây là ca vắt đêm.
    if (s.d === e.d && s.m === e.m && s.y === e.y) return fmtDMY(s);
    return `${fmtDM(s)} → ${fmtDMY(e)}`;
}

function fillShiftPanel(sh, countEl, dateEl, hoursEl) {
    if (!sh) return;
    if (countEl) countEl.textContent = sh.count;
    if (dateEl) dateEl.textContent = shiftDateText(sh);
    if (hoursEl) hoursEl.textContent = `${hhmm(sh.start)} – ${hhmm(sh.end)}`;
}

function renderShiftCounts(data) {
    fillShiftPanel(data.now, shiftNowCount, shiftNowDate, shiftNowHours);
    fillShiftPanel(data.last, shiftLastCount, shiftLastDate, shiftLastHours);
}

// Update DOM elements
function updateUI(status) {
    latestStatus = status;
    // 1. System state & indicators
    const state = status.state;
    systemStateLabel.textContent = state;
    
    // Màu đèn + badge theo class trong style.css (không gán inline style nữa:
    // trước đây màu STOPPED là #252e3b, màu của theme tối còn sót lại).
    systemStatusDot.className = 'status-indicator ' + state.toLowerCase();
    systemStateLabel.className = 'system-state-badge ' + state.toLowerCase();

    // 2. Metrics values
    fpsVal.textContent = status.fps.toFixed(1);
    inferVal.textContent = `${status.infer_ms.toFixed(0)} ms`;
    
    // Camera status
    if (status.camera_connected) {
        cameraVal.textContent = 'ONLINE';
        cameraVal.className = 'value connected';
    } else {
        cameraVal.textContent = status.running ? 'CONNECTING' : 'OFFLINE';
        cameraVal.className = status.running ? 'value warning' : 'value disconnected';
    }
    
    // Model status
    if (status.model_loaded) {
        modelVal.textContent = 'LOADED';
        modelVal.className = 'value connected';
    } else {
        modelVal.textContent = status.running ? 'LOADING' : 'UNLOADED';
        modelVal.className = 'value disconnected';
    }
    
    // Uptime formatter
    uptimeVal.textContent = formatUptime(status.uptime);
    
    // 2.5 Update CPU/GPU temps every 30 seconds
    const now = Date.now();
    if (now - lastTempUpdateTime >= 30000 || lastTempUpdateTime === 0) {
        if (status.cpu_temp !== undefined && status.cpu_temp !== null) {
            cpuTempVal.textContent = `${status.cpu_temp.toFixed(1)} °C`;
        } else {
            cpuTempVal.textContent = '-- °C';
        }
        
        if (status.gpu_temp !== undefined && status.gpu_temp !== null) {
            gpuTempVal.textContent = `${status.gpu_temp.toFixed(1)} °C`;
        } else {
            gpuTempVal.textContent = '-- °C';
        }
        lastTempUpdateTime = now;
    }
    
    // 3. Control button visibility/disabled
    if (status.running) {
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-flex';
        
        if (status.paused) {
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = 'inline-flex';
        } else {
            pauseBtn.style.display = 'inline-flex';
            resumeBtn.style.display = 'none';
        }
    } else {
        startBtn.style.display = 'inline-flex';
        stopBtn.style.display = 'none';
        pauseBtn.style.display = 'none';
        resumeBtn.style.display = 'none';
    }

    // Shutdown chỉ cho phép khi chương trình đã DỪNG hoàn toàn
    if (status.running) {
        shutdownBtn.disabled = true;
        shutdownBtn.classList.add('disabled');
        if (shutdownHint) shutdownHint.style.display = 'block';
    } else {
        shutdownBtn.disabled = false;
        shutdownBtn.classList.remove('disabled');
        if (shutdownHint) shutdownHint.style.display = 'none';
    }

    // Reset (khởi động lại) cũng chỉ cho phép khi chương trình đã DỪNG hoàn toàn
    if (status.running) {
        resetBtn.disabled = true;
        resetBtn.classList.add('disabled');
    } else {
        resetBtn.disabled = false;
        resetBtn.classList.remove('disabled');
    }
    
    // Reset alarm button
    // [multicam] any_alarm = OR của MỌI camera (trục + lược). Trước đây chỉ xét
    // alarm_active của camera trục nên camera lược báo đứt thì nút không hiện.
    const anyAlarm = (status.any_alarm !== undefined)
                   ? status.any_alarm : status.alarm_active;
    if (anyAlarm) {
        resetAlarmBtn.style.display = 'inline-flex';
    } else {
        resetAlarmBtn.style.display = 'none';
    }

    // 4. Last/Active Alarm Card — hiện ảnh chụp của camera ĐANG báo đứt
    const alarmInfo = pickActiveAlarm(status);
    if (alarmInfo) {
        activeAlarmSection.style.display = 'block';
        activeAlarmType.textContent = alarmInfo.type;
        activeAlarmConf.textContent = Number(alarmInfo.confidence || 0).toFixed(2);
        activeAlarmTime.textContent = alarmInfo.time || '-';

        if (alarmInfo.image_path) {
            activeAlarmImg.src = `/${alarmInfo.image_path}`;
            activeAlarmImg.style.display = 'block';
        } else {
            activeAlarmImg.src = '';
            activeAlarmImg.style.display = 'none';
        }
    } else {
        activeAlarmSection.style.display = 'none';
    }

    // [multicam] Thẻ trạng thái từng camera (trục + lược)
    renderCameras(status);
}

// Fetch recent database alarm logs
async function fetchAlarmLogs() {
    const data = await apiFetch('alarms?limit=10');
    if (data && data.alarms) {
        renderLogs(data.alarms);
    }
}

// Format logs to HTML table
function renderLogs(alarms) {
    if (alarms.length === 0) {
        alarmLogsBody.innerHTML = `
            <tr>
                <td colspan="4" class="no-data">${t('logs.noData')}</td>
            </tr>
        `;
        return;
    }

    alarmLogsBody.innerHTML = alarms.map(alarm => {
        const imageCell = alarm.image_path
            ? `<button class="thumbnail-btn" onclick="openImageModal('/${alarm.image_path}')">${t('logs.viewFrame')}</button>`
            : `<span class="text-muted">${t('logs.none')}</span>`;
            
        const statusBadge = alarm.ack_status === 'PENDING'
            ? '<span class="log-status-badge pending">PENDING</span>'
            : '<span class="log-status-badge acknowledged">ACK</span>';
            
        return `
            <tr>
                <td>${alarm.timestamp.substring(11)}<br><small class="text-muted">${alarm.timestamp.substring(0, 10)}</small></td>
                <td><strong>${alarm.confidence.toFixed(2)}</strong></td>
                <td>${statusBadge}</td>
                <td>${imageCell}</td>
            </tr>
        `;
    }).join('');
}

// Trigger CSV export and download
async function exportCSV() {
    try {
        const response = await fetch(`${API_BASE}/alarms/export`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        if (!response.ok) throw new Error("CSV Export failed");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `yarnbreak_alarms_${timeString()}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        console.error(err);
        showToast(t('logs.exportFail'), 'error');
    }
}

// ── Lịch sử truy cập & hành động (audit) — chỉ admin ──────────────────────
async function fetchAuditLogs() {
    const data = await apiFetch('audit?limit=50');
    if (data && data.audit) {
        renderAuditLogs(data.audit);
    }
}

// Mở/đóng một thẻ gập được (lịch sử thao tác, nguồn thiết bị...)
function setupCollapsible(headerId, sectionId) {
    const header = document.getElementById(headerId);
    const section = document.getElementById(sectionId);
    if (!header || !section) return;

    const toggle = () => {
        const collapsed = section.classList.toggle('collapsed');
        header.setAttribute('aria-expanded', String(!collapsed));
    };

    header.addEventListener('click', toggle);
    header.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
        }
    });
}

function renderAuditLogs(logs) {
    if (auditCount) auditCount.textContent = (logs && logs.length) ? logs.length : 0;

    if (!logs || logs.length === 0) {
        auditLogsBody.innerHTML = `
            <tr>
                <td colspan="5" class="no-data">${t('audit.noData')}</td>
            </tr>
        `;
        return;
    }

    const escapeHtml = (s) => String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    auditLogsBody.innerHTML = logs.map(log => {
        const statusClass = log.status === 'SUCCESS' ? 'acknowledged' : 'pending';
        const statusBadge = `<span class="log-status-badge ${statusClass}">${escapeHtml(log.status)}</span>`;
        return `
            <tr>
                <td>${escapeHtml(log.timestamp.substring(11))}<br><small class="text-muted">${escapeHtml(log.timestamp.substring(0, 10))}</small></td>
                <td><strong>${escapeHtml(log.username)}</strong></td>
                <td>${escapeHtml(log.action)}</td>
                <td>${statusBadge}</td>
                <td><small>${escapeHtml(log.details)}</small></td>
            </tr>
        `;
    }).join('');
}

async function exportAuditCSV() {
    try {
        const response = await fetch(`${API_BASE}/audit/export`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        if (!response.ok) throw new Error("Audit CSV Export failed");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `yarnbreak_audit_${timeString()}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        console.error(err);
        showToast(t('logs.exportFail'), 'error');
    }
}

// UI helper: Uptime conversion
function formatUptime(seconds) {
    if (!seconds) return '00:00:00';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return [
        hrs.toString().padStart(2, '0'),
        mins.toString().padStart(2, '0'),
        secs.toString().padStart(2, '0')
    ].join(':');
}

// UI helper: Get compact timestamp for filenames
function timeString() {
    const d = new Date();
    return d.toISOString().replace(/T/, '_').replace(/\..+/, '').replace(/:/g, '');
}

// ── Live camera view (MJPEG có viền ROI để căn chỉnh camera) ───────────────
function openLiveView(camId) {
    const cam = camId || 'cam_main';
    if (!liveModal || !liveImg) return;

    liveModal.style.display = 'block';
    if (liveStatus) liveStatus.style.display = 'block';

    // Hệ thống phải đang chạy thì camera mới có frame để phát.
    if (!latestStatus || !latestStatus.running) {
        liveImg.removeAttribute('src');
        if (liveStatus) liveStatus.textContent = t('live.needStart');
        return;
    }

    if (liveStatus) liveStatus.textContent = t('live.connecting');
    // token qua query vì thẻ <img> không gửi được header Authorization.
    // Thêm timestamp để tránh cache và ép mở kết nối mới.
    liveImg.src = `${API_BASE}/live/mjpeg?token=${encodeURIComponent(token)}&cam=${encodeURIComponent(cam)}&_=${Date.now()}`;

    // Dự phòng: nếu 4s không nhận được frame (sự kiện load), vẫn nhắc người dùng.
    if (liveHideTimer) clearTimeout(liveHideTimer);
    liveHideTimer = setTimeout(() => {
        if (liveStatus && liveStatus.style.display !== 'none') {
            liveStatus.textContent = t('live.noSignal');
        }
    }, 4000);
}

function closeLiveView() {
    if (!liveModal) return;
    liveModal.style.display = 'none';
    if (liveHideTimer) { clearTimeout(liveHideTimer); liveHideTimer = null; }
    // Ngắt luồng MJPEG: xoá src để trình duyệt đóng kết nối, giải phóng thread server.
    if (liveImg) liveImg.removeAttribute('src');
}

// Open modal containing fullscreen warning image
function openImageModal(imgUrl) {
    imageModal.style.display = "block";
    modalImg.src = imgUrl;
}

// Export openImageModal globally so log table rows inline onclick events can reach it
window.openImageModal = openImageModal;

// Toast notification helper
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${message}</span>
        <span style="cursor:pointer;margin-left:15px;font-weight:bold;font-size:16px;" onclick="this.parentElement.remove()">&times;</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 4000);
}


// ── [multicam] Thẻ trạng thái camera theo vị trí ─────────────────────────────
// Dữ liệu từ status.cameras (state_manager.set_camera_status). Camera trục được
// dựng từ các field cũ (camera_connected/fps/infer_ms/alarm_active) để thẻ luôn
// hiện đủ cả 2 vị trí dù pipeline lược chưa chạy.
function renderCameras(status) {
    const section = document.getElementById('cameras-section');
    const grid = document.getElementById('cameras-grid');
    if (!section || !grid) return;

    const cams = [];
    // Camera trục (vị trí cũ)
    cams.push({
        id: 'cam_main',
        label: t('cams.axis'),
        connected: !!status.camera_connected,
        fps: status.fps,
        infer_ms: status.infer_ms,
        alarm: !!status.alarm_active,
    });
    // Các camera lược
    const extra = status.cameras || {};
    Object.keys(extra).sort().forEach((id) => {
        if (id === 'cam_main') return;
        const c = extra[id] || {};
        cams.push({
            id: id,
            label: camLabel(t('cams.comb'), c.name || id),
            connected: !!c.connected,
            fps: c.fps,
            infer_ms: c.infer_ms,
            alarm: !!c.alarm_active,
        });
    });

    // Chỉ hiện card khi có nhiều hơn 1 camera (tránh làm rối UI cũ)
    section.style.display = cams.length > 1 ? 'block' : 'none';
    if (cams.length <= 1) return;

    grid.innerHTML = cams.map((c) => {
        const stateTxt = c.alarm ? t('cams.break')
                       : (c.connected ? t('cams.ok') : t('cams.offline'));
        const stateCls = c.alarm ? 'alarm' : (c.connected ? 'ok' : 'off');
        const infer = (c.infer_ms === undefined || c.infer_ms === null)
                    ? '--' : `${Number(c.infer_ms).toFixed(0)} ms`;
        // fps = nhịp frame THỰC TẾ của camera đó (camera lược đo ở luồng đọc).
        const fps = (c.fps === undefined || c.fps === null)
                  ? '--' : Number(c.fps).toFixed(1);
        return `
            <div class="cam-tile ${stateCls}">
                <div class="cam-tile-head">
                    <span class="cam-name">${escapeHtml(c.label)}</span>
                </div>
                <div class="cam-state-big ${stateCls}">${escapeHtml(stateTxt)}</div>
                <div class="cam-tile-meta">${t('cams.fps')}: ${fps} · ${t('cams.infer')}: ${infer}</div>
                <button class="cam-live-btn" data-cam="${escapeHtml(c.id)}">
                    ${escapeHtml(t('cams.live'))}
                </button>
            </div>`;
    }).join('');

    // Mỗi thẻ có nút xem trực tiếp riêng → mở MJPEG của ĐÚNG camera đó.
    grid.querySelectorAll('.cam-live-btn').forEach((b) => {
        b.addEventListener('click', () => openLiveView(b.dataset.cam));
    });
}

// Nhãn thẻ camera = vị trí (dịch theo ngôn ngữ) + tên trong config.yaml.
// Tên trong config có thể đã chứa sẵn vị trí (ví dụ "Lược - CU120") → khi đó
// giữ nguyên tên, không thêm tiền tố nữa để không ra "Lược · Lược - CU120".
function camLabel(position, name) {
    const nm = String(name == null ? '' : name).trim();
    if (!nm) return position;
    const norm = (s) => String(s).toLowerCase().trim();
    return norm(nm).startsWith(norm(position)) ? nm : `${position} · ${nm}`;
}

function escapeHtml(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[ch]);
}


// ── [multicam] Chọn cảnh báo đang hoạt động để hiện ảnh chụp ────────────────
// Ưu tiên camera trục (giữ nguyên hành vi cũ). Nếu trục không báo mà camera
// lược đang báo thì lấy cảnh báo của camera lược đó.
// Lưu ý: camera lược lưu {time, conf, image_path}, camera trục lưu
// {time, type, confidence, image_path} → chuẩn hoá về một dạng.
function pickActiveAlarm(status) {
    if (status.alarm_active && status.last_alarm) {
        return {
            type: status.last_alarm.type,
            confidence: status.last_alarm.confidence,
            time: status.last_alarm.time,
            image_path: status.last_alarm.image_path,
        };
    }
    const cams = status.cameras || {};
    for (const id of Object.keys(cams).sort()) {
        const c = cams[id];
        if (c && c.alarm_active && c.last_alarm) {
            return {
                type: `YarnBreakComb (${c.name || id})`,
                confidence: c.last_alarm.conf,
                time: c.last_alarm.time,
                image_path: c.last_alarm.image_path,
            };
        }
    }
    return null;
}


// ── Hộp thoại xác nhận trong trang ──────────────────────────────────────────
// Thay confirm()/prompt() của trình duyệt vì:
//   • confirm()/prompt() không theo ngôn ngữ đang chọn, chữ nhỏ, khó bấm trên
//     tablet ngoài xưởng, và Chrome cho phép người dùng chặn vĩnh viễn;
//   • prompt() HIỆN RÕ mật khẩu khi nhập — hộp thoại này dùng
//     <input type="password">, có nút Hiện/Ẩn do người dùng tự chọn.
// Trả về Promise:
//   • null           → người dùng huỷ
//   • true           → đồng ý (hộp thoại không có mật khẩu)
//   • {password}     → đồng ý (hộp thoại có mật khẩu)
let dlgResolve = null;      // resolve của hộp thoại đang mở
let dlgNeedPass = false;

function askConfirm(opts) {
    const o = opts || {};
    return new Promise((resolve) => {
        // Nếu còn hộp thoại đang mở (trường hợp bấm liên tiếp) thì huỷ nó trước.
        if (dlgResolve) {
            const prev = dlgResolve;
            dlgResolve = null;
            prev(null);
        }
        dlgResolve = resolve;
        dlgNeedPass = !!o.password;

        dlgTitle.textContent = o.title || t('dlg.titleDefault');
        dlgMsg.textContent = o.message || '';
        dlgError.textContent = '';
        dlgCancel.textContent = t('dlg.cancel');
        dlgOk.textContent = o.okLabel || t('dlg.confirm');
        dlgOk.className = 'btn ' + (o.okClass || 'primary');

        dlgPassLabel.textContent = t('dlg.passLabel');
        dlgPass.value = '';
        dlgPass.type = 'password';
        dlgPassToggle.textContent = t('dlg.show');
        dlgPassField.classList.toggle('show', dlgNeedPass);

        dlgModal.classList.add('open');
        // Đợi hộp thoại hiện xong rồi mới focus (Chromium trên Jetson).
        setTimeout(() => (dlgNeedPass ? dlgPass : dlgOk).focus(), 30);
    });
}

function closeDialog(result) {
    const resolve = dlgResolve;
    dlgResolve = null;
    dlgModal.classList.remove('open');
    dlgPass.value = '';
    dlgPass.type = 'password';
    if (resolve) resolve(result);
}

function submitDialog() {
    if (!dlgResolve) return;
    if (dlgNeedPass) {
        const pwd = dlgPass.value;
        if (!pwd.trim()) {
            dlgError.textContent = t('login.passEmpty');
            dlgPass.focus();
            return;                      // giữ hộp thoại mở để nhập lại
        }
        closeDialog({ password: pwd });
        return;
    }
    closeDialog(true);
}
