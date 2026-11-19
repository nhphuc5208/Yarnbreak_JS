# Changelog

## [Unreleased] — merge of Yarnbreak_COMB_JS into Yarnbreak_JS

### Changed
- All runtime modules moved into the `yarnbreak/` package (`relay_hub`,
  `modbus_relay`, `comb_pipeline`, `mvs_sdk`, `state_manager`, `alarm_logger`,
  `live_view`, `inference_worker`, `auth`). The detection loop moved from
  `main.py` to `yarnbreak/pipeline.py`; `main.py` is now a thin CLI wrapper, so
  `python3 main.py --config config.yaml` works as before.
- systemd unit moved to `deploy/yarnbreak.service` (content unchanged).
- Launch scripts (`chay.sh`, `chay_nen.sh`, `chay_web.sh`, `chay_chup_mau.sh`,
  `open_web_ui.sh`, `run_jetson_log.sh`) now live in `scripts/` with the same
  file names. **Desktop shortcuts / GNOME autostart entries that point to the
  repo root must be updated** (e.g. `~/.config/autostart/yarnbreak-web-ui.desktop`).
- CI: one workflow (`ci.yml`) running ruff, pytest and the i18n check.

### Added (from Yarnbreak_COMB_JS)
- Operator tools: `can_khung_luoc.py`, `kiem_tra_camera.py`, `mang_comb2.sh`,
  `comb_capture.py`, `log_jetson_stats.py`, `log_power.py`, `cam_temp.py`,
  `run_ipcam.sh`, `test_red_button_alarm.py`, `watch_alarm_test.sh`.
- `docs/LENH_CHAY.md` operator guide (Vietnamese).
- Tests: full relay truth table (ported from the comb repo's script), comb
  alarm rules, i18n duplicate keys, static import check.

### Removed
- Unused imports and a never-read `prev_alert` variable (no behaviour change).

### Config
- **No key in `config.yaml` was renamed, added or removed.**

### Web login (already in Yarnbreak_JS, new for sites running the comb repo)
- Passwords are PBKDF2 hashes (`ADMIN_PASSWORD_HASH`). An old plain
  `ADMIN_PASSWORD` in `.env` is still accepted (with a warning).
- There is no built-in default password any more: an account with no password
  in `.env` is disabled.
- CORS is off by default (`CORS_ORIGINS` in `.env` to enable).
