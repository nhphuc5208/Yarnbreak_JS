# Tổng hợp lệnh chạy chương trình YarnBreak (Jetson)

Thư mục dự án: `~/Documents/yarnbreak_jetson`

> ⚠️ **Chỉ MỘT tiến trình được giữ camera tại một thời điểm.** Nếu chạy bị báo camera bận
> → còn `yarnbreak.service` hoặc tiến trình khác đang chạy. Phải dừng nó trước.

---

## 1. Cách chính thức — qua systemd service (khuyến nghị)

Đây là cách hệ thống dùng để **tự chạy khi khởi động máy** (headless + web UI kiosk).
Service chạy `web_server.py`, web UI ở cổng **8000**.

```bash
sudo systemctl start yarnbreak.service     # chạy ngay
sudo systemctl stop yarnbreak.service      # dừng (khi boot vẫn chạy lại)
sudo systemctl restart yarnbreak.service   # khởi động lại
systemctl status yarnbreak.service         # xem trạng thái
sudo systemctl enable yarnbreak.service    # BẬT tự chạy khi khởi động máy
sudo systemctl disable yarnbreak.service   # TẮT tự chạy khi khởi động máy
journalctl -u yarnbreak.service -f         # xem log realtime
```

---

## 2. Chạy thủ công bằng script .sh

Đứng tại thư mục dự án. **Phải `sudo systemctl stop yarnbreak.service` trước** để nhả camera.

| Script | Mục đích |
|--------|----------|
| `./scripts/chay_web.sh`   | Chạy **web server** (giống service, cổng 8000) |
| `./scripts/chay.sh`       | Chạy detector **có cửa sổ dashboard** (main.py → yarnbreak/pipeline.py) |
| `./scripts/chay_nen.sh`   | Chạy **nền/headless 24/7**, tự khởi động lại khi crash, log vào `outputs/logs/` |
| `./scripts/open_web_ui.sh`| Mở Chromium kiosk toàn màn hình vào `http://localhost:8000` |
| `./scripts/chay_chup_mau.sh` | Chụp **ảnh mẫu** từ camera lược (web cổng 8010) — phải dừng service trước |
| `./scripts/run_jetson_log.sh` | Ghi nhiệt độ / tải Jetson mỗi 5 s ra file CSV |

---

## 3. Chạy trực tiếp bằng Python (để debug)

```bash
cd ~/Documents/yarnbreak_jetson
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:$LD_LIBRARY_PATH
source venv/bin/activate

python3 main.py --config config.yaml              # có cửa sổ camera
python3 main.py --config config.yaml --headless   # không cửa sổ
python3 web_server.py                             # web server
```

---

## Cách dừng nhanh

- Service:            `sudo systemctl stop yarnbreak.service`
- Script terminal:    `Ctrl+C` (hoặc đóng terminal với `chay_nen.sh`)
- Web kiosk Chromium: `Alt+F4` hoặc `pkill chromium`

---

## Kiểm tra nhanh

```bash
systemctl is-active yarnbreak.service      # active / inactive
systemctl is-enabled yarnbreak.service     # enabled  = tự chạy khi boot
ps aux | grep -E "web_server|main.py" | grep -v grep   # xem tiến trình đang chạy
```

---

## Công cụ camera lược

```bash
python3 scripts/kiem_tra_camera.py --cam comb_2 --giay 20 --luu /tmp/comb2.jpg  # đo fps, ROI, độ sáng
python3 scripts/kiem_tra_camera.py --cam comb_2 --packet 0                      # ảnh đen? kiểm tra MTU
python3 scripts/can_khung_luoc.py --cam comb_2 --xem                             # căn hw_roi ôm khung lược
sudo bash scripts/mang_comb2.sh eth1                                             # IP tĩnh + MTU 9000
```

> Từ bản hợp nhất, mọi script `.sh` nằm trong `scripts/` (tên file giữ nguyên).
> Shortcut desktop / autostart cũ trỏ vào thư mục gốc cần sửa lại đường dẫn.
