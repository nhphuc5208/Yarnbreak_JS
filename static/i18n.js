// YarnBreak Web Console - Internationalization (i18n)
// ----------------------------------------------------
// Hỗ trợ 3 ngôn ngữ: Tiếng Việt (vi), Tiếng Hàn (ko), English (en).
// Nguyên tắc: các thuật ngữ kỹ thuật khó dịch được GIỮ NGUYÊN tiếng Anh
//   (FPS, AI, CSV, CPU, GPU, Jetson, API Token, GPIO, LAN, ms, °C,
//    ONLINE/OFFLINE, LOADED/UNLOADED, RUNNING/STOPPED/PAUSED/ALARM, PENDING/ACK...).
// 한국어 안내: FPS·AI·CSV·CPU·GPU·Jetson 등 기술 용어는 영어 그대로 표기합니다.

const I18N = {
    vi: {
        // Login
        "login.title": "Xác thực hệ thống",
        "login.desc": "Nhập tên đăng nhập và mật khẩu để truy cập bảng điều khiển YarnBreak trên Jetson.",
        "login.userPlaceholder": "Tên đăng nhập...",
        "login.passPlaceholder": "Mật khẩu...",
        "login.unlock": "Mở khóa bảng điều khiển",
        "login.userEmpty": "Vui lòng nhập tên đăng nhập",
        "login.passEmpty": "Vui lòng nhập mật khẩu",
        "login.failed": "Xác thực thất bại",
        "login.network": "Lỗi kết nối mạng",

        // Lịch sử truy cập & hành động (chỉ admin)
        "audit.title": "Lịch sử truy cập & hành động",
        "audit.export": "Xuất CSV",
        "audit.colTime": "Thời gian",
        "audit.colUser": "Người dùng",
        "audit.colAction": "Hành động",
        "audit.colStatus": "Trạng thái",
        "audit.colDetail": "Chi tiết",
        "audit.noData": "Chưa có lịch sử nào",

        // Xác nhận mật khẩu khi DỪNG
        "stop.passwordPrompt": "Nhập lại MẬT KHẨU của bạn để xác nhận DỪNG chương trình:",
        "stop.passwordWrong": "Sai mật khẩu — không thể dừng chương trình.",

        // Header / brand
        "brand.sub": "Jetson Console",

        // Controls
        "controls.title": "Bảng điều khiển",
        "btn.start": "BẬT AI",
        "btn.stop": "DỪNG HỆ THỐNG",
        "btn.pause": "TẠM DỪNG INFER",
        "btn.resume": "TIẾP TỤC INFER",
        "btn.resetAlarm": "TẮT CẢNH BÁO",

        // Hộp thoại xác nhận trong trang
        "dlg.confirm": "Xác nhận",
        "dlg.cancel": "Hủy",
        "dlg.show": "Hiện",
        "dlg.hide": "Ẩn",
        "dlg.passLabel": "Mật khẩu của bạn",
        "dlg.titleDefault": "Xác nhận",
        "dlg.titleStart": "Bật nhận diện AI",
        "dlg.titleStop": "Dừng hệ thống",
        "dlg.titlePause": "Tạm dừng nhận diện",
        "dlg.titleResume": "Tiếp tục nhận diện",
        "dlg.titleShutdown": "Tắt máy Jetson",
        "dlg.titleShutdownFinal": "Xác nhận lần cuối",
        "dlg.titleReset": "Khởi động lại Jetson",
        "dlg.titleResetFinal": "Xác nhận lần cuối",

        // Nguồn thiết bị
        "power.title": "Nguồn thiết bị",
        "power.desc": "Hai lệnh dưới đây tắt hoặc khởi động lại máy Jetson. Chỉ dùng khi đã DỪNG chương trình.",

        // Live camera
        "live.title": "Camera trực tiếp — Căn chỉnh vùng trục (ROI)",
        "live.hint": "Chỉnh camera sao cho sợi trục nằm trong khung viền ROI màu xanh lá.",
        "live.connecting": "Đang kết nối camera...",
        "live.needStart": "Hãy BẬT hệ thống (START AI) để xem camera trực tiếp.",
        "live.noSignal": "Chưa nhận được tín hiệu — kiểm tra camera đã kết nối chưa.",
        "live.error": "Lỗi luồng camera. Hãy đóng và thử lại.",

        // Metrics
        "metrics.title": "Thông số hệ thống",
        "metrics.fps": "FPS",
        "metrics.infer": "Inference",
        "metrics.camera": "Camera",
        "cams.title": "Camera theo vị trí",
        "cams.axis": "Trục",
        "cams.comb": "Lược",
        "cams.break": "ĐỨT SỢI",
        "cams.ok": "Bình thường",
        "cams.offline": "Mất kết nối",
        "cams.live": "Xem trực tiếp",
        "cams.infer": "Nhận diện",
        "cams.fps": "FPS",
        "metrics.model": "Model",
        "metrics.cpuTemp": "Nhiệt độ CPU",
        "metrics.gpuTemp": "Nhiệt độ GPU",
        "metrics.uptime": "Thời gian chạy",

        // Shift yarn-break count
        "shift.title": "Số lần báo sợi đứt theo ca",
        "shift.now": "Ca hiện tại",
        "shift.last": "Ca trước",
        "shift.running": "Đang chạy",
        "shift.times": "lần",

        // Active alarm
        "alarm.title": "Thông tin cảnh báo hiện tại",
        "alarm.type": "Loại lỗi:",
        "alarm.confidence": "Độ tin cậy:",
        "alarm.detectedAt": "Phát hiện lúc:",
        "alarm.noSnapshot": "Chưa có ảnh cảnh báo",

        // Logs
        "logs.title": "Cảnh báo gần đây",
        "logs.export": "Xuất CSV",
        "logs.colTime": "Thời gian",
        "logs.colConfidence": "Độ tin cậy",
        "logs.colStatus": "Trạng thái",
        "logs.colImage": "Hình ảnh",
        "logs.noData": "Chưa có cảnh báo nào",
        "logs.viewFrame": "Xem ảnh",
        "logs.none": "Không có",
        "logs.exportFail": "Xuất dữ liệu CSV thất bại",

        // Shutdown / footer
        "shutdown.btn": "TẮT MÁY JETSON",
        "shutdown.hint": "Phải nhấn STOP để dừng chương trình trước khi tắt máy.",
        "reset.btn": "KHỞI ĐỘNG LẠI JETSON",
        "footer.info": "Jetson Orin Nano Production Console • LAN Mode",
        "footer.lock": "Khóa bảng điều khiển",

        // Confirms
        "confirm.start": "Bạn có chắc chắn muốn BẮT ĐẦU chương trình nhận diện AI?",
        "confirm.stop": "Bạn có chắc chắn muốn DỪNG HOÀN TOÀN hệ thống?",
        "confirm.pause": "Bạn có chắc chắn muốn TẠM DỪNG nhận diện?",
        "confirm.resume": "Bạn có chắc chắn muốn TIẾP TỤC nhận diện?",
        "confirm.shutdown1": "CẢNH BÁO: Bạn có chắc chắn muốn TẮT HOÀN TOÀN máy tính Jetson?\nHệ thống đã dừng nhận diện AI và sẽ tắt nguồn thiết bị.",
        "confirm.shutdown2": "XÁC NHẬN LẦN CUỐI: Tắt nguồn máy tính Jetson?",
        "confirm.reset1": "CẢNH BÁO: Bạn có chắc chắn muốn KHỞI ĐỘNG LẠI máy tính Jetson?\nHệ thống sẽ tự đóng mọi tiến trình và reset thiết bị.",
        "confirm.reset2": "XÁC NHẬN LẦN CUỐI: Khởi động lại máy tính Jetson?",

        // Toasts
        "toast.shutdownBlocked": "Phải nhấn STOP để dừng chương trình trước khi tắt máy Jetson.",
        "toast.resetBlocked": "Phải nhấn STOP để dừng chương trình trước khi khởi động lại máy Jetson.",
        "toast.sending": "Đang gửi lệnh {action}...",
        "toast.success": "Lệnh {action} thực hiện thành công!",
        "toast.failed": "Lệnh {action} thất bại!",
        "toast.noResponse": "Không nhận được phản hồi từ Jetson cho lệnh {action}",
        "toast.connErr": "Lỗi kết nối ({endpoint}): {message}",
        "toast.shuttingDown": "Máy tính Jetson đang tắt nguồn. Kết nối sẽ bị ngắt!",
        "toast.rebooting": "Máy tính Jetson đang khởi động lại. Kết nối sẽ bị ngắt tạm thời!",
    },

    ko: {
        // Login
        "login.title": "시스템 인증",
        "login.desc": "Jetson YarnBreak 제어판에 접속하려면 아이디와 비밀번호를 입력하세요.",
        "login.userPlaceholder": "아이디...",
        "login.passPlaceholder": "비밀번호...",
        "login.unlock": "제어판 잠금 해제",
        "login.userEmpty": "아이디를 입력하세요",
        "login.passEmpty": "비밀번호를 입력하세요",
        "login.failed": "인증에 실패했습니다",
        "login.network": "네트워크 연결 오류",

        // 접속 및 작업 이력 (admin 전용)
        "audit.title": "접속 및 작업 이력",
        "audit.export": "CSV 내보내기",
        "audit.colTime": "시간",
        "audit.colUser": "사용자",
        "audit.colAction": "작업",
        "audit.colStatus": "상태",
        "audit.colDetail": "상세",
        "audit.noData": "기록된 이력이 없습니다",

        // 정지 시 비밀번호 확인
        "stop.passwordPrompt": "프로그램 정지를 확인하려면 비밀번호를 다시 입력하세요:",
        "stop.passwordWrong": "비밀번호가 틀렸습니다 — 프로그램을 정지할 수 없습니다.",

        // Header / brand
        "brand.sub": "Jetson Console",

        // Controls
        "controls.title": "제어 대시보드",
        "btn.start": "AI 시작",
        "btn.stop": "시스템 정지",
        "btn.pause": "Inference 일시정지",
        "btn.resume": "Inference 재개",
        "btn.resetAlarm": "알람 해제",

        // 페이지 내 확인 대화상자
        "dlg.confirm": "확인",
        "dlg.cancel": "취소",
        "dlg.show": "표시",
        "dlg.hide": "숨기기",
        "dlg.passLabel": "비밀번호",
        "dlg.titleDefault": "확인",
        "dlg.titleStart": "AI 감지 시작",
        "dlg.titleStop": "시스템 정지",
        "dlg.titlePause": "감지 일시정지",
        "dlg.titleResume": "감지 재개",
        "dlg.titleShutdown": "Jetson 전원 끄기",
        "dlg.titleShutdownFinal": "최종 확인",
        "dlg.titleReset": "Jetson 재시작",
        "dlg.titleResetFinal": "최종 확인",

        // 장비 전원
        "power.title": "장비 전원",
        "power.desc": "아래 두 명령은 Jetson 장비의 전원을 끄거나 재시작합니다. 프로그램을 정지한 뒤에만 사용하세요.",

        // Live camera
        "live.title": "실시간 카메라 — ROI 정렬",
        "live.hint": "실이 초록색 ROI 외곽선 안에 들어오도록 카메라를 조정하세요.",
        "live.connecting": "카메라 연결 중...",
        "live.needStart": "실시간 카메라를 보려면 시스템을 시작(START AI)하세요.",
        "live.noSignal": "신호가 없습니다 — 카메라 연결 상태를 확인하세요.",
        "live.error": "카메라 스트림 오류입니다. 닫고 다시 시도하세요.",

        // Metrics
        "metrics.title": "시스템 텔레메트리",
        "metrics.fps": "FPS",
        "metrics.infer": "Inference",
        "metrics.camera": "Camera",
        "cams.title": "위치별 카메라",
        "cams.axis": "축",
        "cams.comb": "콤",
        "cams.break": "실 끊김",
        "cams.ok": "정상",
        "cams.offline": "연결 끊김",
        "cams.live": "실시간 보기",
        "cams.infer": "추론",
        "cams.fps": "FPS",
        "metrics.model": "Model",
        "metrics.cpuTemp": "CPU 온도",
        "metrics.gpuTemp": "GPU 온도",
        "metrics.uptime": "가동 시간",

        // Shift yarn-break count
        "shift.title": "교대조별 실 끊김 알람 횟수",
        "shift.now": "현재 교대",
        "shift.last": "이전 교대",
        "shift.running": "진행 중",
        "shift.times": "회",

        // Active alarm
        "alarm.title": "현재 알람 정보",
        "alarm.type": "불량 유형:",
        "alarm.confidence": "신뢰도:",
        "alarm.detectedAt": "감지 시각:",
        "alarm.noSnapshot": "알람 스냅샷 없음",

        // Logs
        "logs.title": "최근 감지 내역",
        "logs.export": "CSV 내보내기",
        "logs.colTime": "시간",
        "logs.colConfidence": "신뢰도",
        "logs.colStatus": "상태",
        "logs.colImage": "이미지",
        "logs.noData": "기록된 알람이 없습니다",
        "logs.viewFrame": "프레임 보기",
        "logs.none": "없음",
        "logs.exportFail": "CSV 데이터 내보내기에 실패했습니다",

        // Shutdown / footer
        "shutdown.btn": "JETSON 전원 끄기",
        "shutdown.hint": "전원을 끄기 전에 반드시 STOP을 눌러 프로그램을 정지하세요.",
        "reset.btn": "JETSON 재시작",
        "footer.info": "Jetson Orin Nano Production Console • LAN Mode",
        "footer.lock": "제어판 잠금",

        // Confirms
        "confirm.start": "AI 감지 프로그램을 시작하시겠습니까?",
        "confirm.stop": "시스템을 완전히 정지하시겠습니까?",
        "confirm.pause": "감지를 일시정지하시겠습니까?",
        "confirm.resume": "감지를 재개하시겠습니까?",
        "confirm.shutdown1": "경고: Jetson 장비의 전원을 완전히 끄시겠습니까?\nAI 감지가 정지되었으며 장비 전원이 꺼집니다.",
        "confirm.shutdown2": "최종 확인: Jetson 전원을 끄시겠습니까?",
        "confirm.reset1": "경고: Jetson 장비를 재시작하시겠습니까?\n모든 프로세스가 자동으로 종료되고 장비가 재시작됩니다.",
        "confirm.reset2": "최종 확인: Jetson 장비를 재시작하시겠습니까?",

        // Toasts
        "toast.shutdownBlocked": "전원을 끄기 전에 STOP을 눌러 프로그램을 정지해야 합니다.",
        "toast.resetBlocked": "재시작하기 전에 STOP을 눌러 프로그램을 정지해야 합니다.",
        "toast.sending": "{action} 명령 전송 중...",
        "toast.success": "{action} 명령이 성공했습니다!",
        "toast.failed": "{action} 명령이 실패했습니다!",
        "toast.noResponse": "{action} 명령에 대한 Jetson의 응답이 없습니다",
        "toast.connErr": "연결 오류 ({endpoint}): {message}",
        "toast.shuttingDown": "Jetson 장비의 전원을 끄는 중입니다. 연결이 끊어집니다!",
        "toast.rebooting": "Jetson 장비를 재시작하는 중입니다. 연결이 일시적으로 끊어집니다!",
    },

    en: {
        // Login
        "login.title": "System Authentication",
        "login.desc": "Enter your username and password to access the Jetson YarnBreak Control Panel.",
        "login.userPlaceholder": "Username...",
        "login.passPlaceholder": "Password...",
        "login.unlock": "Unlock Panel",
        "login.userEmpty": "Please enter a username",
        "login.passEmpty": "Please enter a password",
        "login.failed": "Authentication failed",
        "login.network": "Network connection error",

        // Access & Action History (admin only)
        "audit.title": "Access & Action History",
        "audit.export": "Export CSV",
        "audit.colTime": "Time",
        "audit.colUser": "User",
        "audit.colAction": "Action",
        "audit.colStatus": "Status",
        "audit.colDetail": "Details",
        "audit.noData": "No history records",

        // Password confirmation on STOP
        "stop.passwordPrompt": "Re-enter your PASSWORD to confirm STOPPING the program:",
        "stop.passwordWrong": "Wrong password — cannot stop the program.",

        // Header / brand
        "brand.sub": "Jetson Console",

        // Controls
        "controls.title": "Control Dashboard",
        "btn.start": "START AI",
        "btn.stop": "STOP SYSTEM",
        "btn.pause": "PAUSE INFER",
        "btn.resume": "RESUME INFER",
        "btn.resetAlarm": "RESET ALARM",

        // In-page confirmation dialog
        "dlg.confirm": "Confirm",
        "dlg.cancel": "Cancel",
        "dlg.show": "Show",
        "dlg.hide": "Hide",
        "dlg.passLabel": "Your password",
        "dlg.titleDefault": "Confirm",
        "dlg.titleStart": "Start AI detection",
        "dlg.titleStop": "Stop the system",
        "dlg.titlePause": "Pause detection",
        "dlg.titleResume": "Resume detection",
        "dlg.titleShutdown": "Shut down the Jetson",
        "dlg.titleShutdownFinal": "Final confirmation",
        "dlg.titleReset": "Reboot the Jetson",
        "dlg.titleResetFinal": "Final confirmation",

        // Device power
        "power.title": "Device Power",
        "power.desc": "These two commands power off or reboot the Jetson. Use them only after the program has stopped.",

        // Live camera
        "live.title": "Live Camera — ROI Alignment",
        "live.hint": "Adjust the camera so the yarn axis falls inside the green ROI outline.",
        "live.connecting": "Connecting to camera...",
        "live.needStart": "Start the system (START AI) to view the live camera.",
        "live.noSignal": "No signal yet — check that the camera is connected.",
        "live.error": "Camera stream error. Please close and try again.",

        // Metrics
        "metrics.title": "System Telemetry",
        "metrics.fps": "FPS",
        "metrics.infer": "Inference",
        "metrics.camera": "Camera",
        "cams.title": "Cameras by Position",
        "cams.axis": "Axis",
        "cams.comb": "Comb",
        "cams.break": "YARN BREAK",
        "cams.ok": "Normal",
        "cams.offline": "Offline",
        "cams.live": "Live view",
        "cams.infer": "Inference",
        "cams.fps": "FPS",
        "metrics.model": "Model",
        "metrics.cpuTemp": "CPU Temp",
        "metrics.gpuTemp": "GPU Temp",
        "metrics.uptime": "Uptime",

        // Shift yarn-break count
        "shift.title": "Yarn-Break Count by Shift",
        "shift.now": "Now Shift",
        "shift.last": "Last Shift",
        "shift.running": "Running",
        "shift.times": "times",

        // Active alarm
        "alarm.title": "Active Alarm Info",
        "alarm.type": "Defect Type:",
        "alarm.confidence": "Confidence:",
        "alarm.detectedAt": "Detected At:",
        "alarm.noSnapshot": "No Alert Snapshot Available",

        // Logs
        "logs.title": "Recent Detections",
        "logs.export": "Export CSV",
        "logs.colTime": "Time",
        "logs.colConfidence": "Confidence",
        "logs.colStatus": "Status",
        "logs.colImage": "Image",
        "logs.noData": "No alarm logs recorded",
        "logs.viewFrame": "View Frame",
        "logs.none": "None",
        "logs.exportFail": "Failed to export alarms CSV data",

        // Shutdown / footer
        "shutdown.btn": "SHUTDOWN JETSON",
        "shutdown.hint": "You must press STOP to stop the program before shutting down.",
        "reset.btn": "RESET JETSON",
        "footer.info": "Jetson Orin Nano Production Console • LAN Mode",
        "footer.lock": "Lock Panel",

        // Confirms
        "confirm.start": "Are you sure you want to START the AI detection program?",
        "confirm.stop": "Are you sure you want to COMPLETELY STOP the system?",
        "confirm.pause": "Are you sure you want to PAUSE detection?",
        "confirm.resume": "Are you sure you want to RESUME detection?",
        "confirm.shutdown1": "WARNING: Are you sure you want to COMPLETELY shut down the Jetson?\nAI detection has stopped and the device will power off.",
        "confirm.shutdown2": "FINAL CONFIRMATION: Power off the Jetson device?",
        "confirm.reset1": "WARNING: Are you sure you want to RESTART the Jetson?\nAll processes will be closed automatically and the device will reboot.",
        "confirm.reset2": "FINAL CONFIRMATION: Reboot the Jetson device?",

        // Toasts
        "toast.shutdownBlocked": "You must press STOP to stop the program before shutting down the Jetson.",
        "toast.resetBlocked": "You must press STOP to stop the program before rebooting the Jetson.",
        "toast.sending": "Sending {action} command...",
        "toast.success": "{action} command executed successfully!",
        "toast.failed": "{action} command failed!",
        "toast.noResponse": "No response received from Jetson for the {action} command",
        "toast.connErr": "Connection error ({endpoint}): {message}",
        "toast.shuttingDown": "The Jetson device is powering off. Connection will be lost!",
        "toast.rebooting": "The Jetson device is rebooting. Connection will be lost temporarily!",
    },
};

const LANG_NAMES = { vi: "Tiếng Việt", ko: "한국어", en: "English" };
const DEFAULT_LANG = "vi";

let currentLang = localStorage.getItem("yarnbreak_lang") || DEFAULT_LANG;
if (!I18N[currentLang]) currentLang = DEFAULT_LANG;

// t(key, params): trả về chuỗi đã dịch, thay thế {placeholder} bằng params.
function t(key, params) {
    const table = I18N[currentLang] || I18N[DEFAULT_LANG];
    let str = (table && table[key]) || (I18N[DEFAULT_LANG][key]) || key;
    if (params) {
        for (const p in params) {
            str = str.replace(new RegExp("\\{" + p + "\\}", "g"), params[p]);
        }
    }
    return str;
}

// Áp dụng bản dịch cho toàn bộ phần tử tĩnh có data-i18n*.
function applyTranslations() {
    document.documentElement.lang = currentLang;

    document.querySelectorAll("[data-i18n]").forEach((el) => {
        el.textContent = t(el.getAttribute("data-i18n"));
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
        el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
    });
    document.querySelectorAll("[data-i18n-alt]").forEach((el) => {
        el.setAttribute("alt", t(el.getAttribute("data-i18n-alt")));
    });

    // Đồng bộ tất cả ô chọn ngôn ngữ (header + login)
    document.querySelectorAll(".lang-select").forEach((sel) => {
        sel.value = currentLang;
    });

    // Cho phép các module khác (app.js) cập nhật lại phần động sau khi đổi ngôn ngữ
    document.dispatchEvent(new CustomEvent("langchange", { detail: { lang: currentLang } }));
}

function setLang(lang) {
    if (!I18N[lang]) return;
    currentLang = lang;
    localStorage.setItem("yarnbreak_lang", lang);
    applyTranslations();
}

// Khởi tạo ngay khi DOM sẵn sàng (chạy trước app.js logic nhờ thứ tự <script>).
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".lang-select").forEach((sel) => {
        sel.addEventListener("change", (e) => setLang(e.target.value));
    });
    applyTranslations();
});

// Export cho app.js
window.t = t;
window.setLang = setLang;
window.applyTranslations = applyTranslations;
window.getLang = () => currentLang;
