# Migration: Yarnbreak_COMB_JS → Yarnbreak_JS

`Yarnbreak_JS` is now the only repository. Everything `Yarnbreak_COMB_JS`
(commit `f45dfaa`, the last one with all operator tools) had that `Yarnbreak_JS`
lacked was brought over; nothing that runs in production was dropped.

## 1. Duplication removed

Measured over `.py .js .html .css .sh`, ignoring blank and comment lines; a line
counts as duplicated when it is part of a run of ≥ 6 identical lines that
appears in more than one place.

| | Files | Code lines | Duplicated lines |
|---|---:|---:|---:|
| Before (JS `a5d1c50` + COMB `f45dfaa`) | 69 | 20,652 | **15,983** (77 %) |
| After (merged repo) | 55 | 11,152 | **300** (2.7 %) |

The overlap was much larger than the ~1,400 lines first estimated
(`comb_pipeline.py` 1,043 + `relay_hub.py` 380):

- **Identical files** in both repos: `comb_pipeline.py`, `relay_hub.py`,
  `alarm_logger.py`, `inference_worker.py`, `modbus_relay.py`,
  `state_manager.py`, `mvs_sdk.py`, `live_view.py`, `scripts/export_tensorrt.py`,
  `scripts/benchmark.py` — 3,098 Python lines — plus the whole web UI
  (`app.js`, `i18n.js`, `style.css`, `index.html`, 3,019 lines).
- **COMB `main.py` (3,253 lines)** is the pre-refactor version of JS's
  `main.py` + `yarnbreak/` package. An AST comparison of all 31 classes and
  functions found **0 differing lines** — it held no unique feature.

The 300 lines still flagged are mostly the Hikrobot open/configure sequence in
`scripts/comb_capture.py`, which deliberately keeps its own copy (exclusive
access, no relay, different buffering) so the capture tool stays independent of
the production pipeline. Sharing it would mean editing `comb_pipeline.py`'s
camera code, which was out of scope for a no-behaviour-change merge.

## 2. Where each file went

| From | To | Note |
|---|---|---|
| `relay_hub.py`, `modbus_relay.py` | `yarnbreak/` | moved verbatim |
| `comb_pipeline.py`, `mvs_sdk.py` | `yarnbreak/` | moved verbatim |
| `state_manager.py`, `alarm_logger.py`, `live_view.py` | `yarnbreak/` | moved verbatim |
| `auth.py`, `inference_worker.py` | `yarnbreak/` | moved verbatim |
| JS `main.py` (loop) | `yarnbreak/pipeline.py` | `main.py` is now a 15-line CLI wrapper |
| COMB `main.py` | — | identical to JS package, dropped |
| `yarnbreak.service` | `deploy/` | content unchanged |
| COMB `chay*.sh`, `open_web_ui.sh`, `run_jetson_log.sh` | `scripts/` | same names; `cd` to repo root |
| COMB `comb_capture.py`, `log_jetson_stats.py` | `scripts/` | same names; imports updated |
| COMB `scripts/*` tools | `scripts/` | same names; imports updated |
| COMB `scripts/test_relay_hub.py` | `tests/test_relay_truth_table.py` | rewritten as pytest |
| COMB `docs/LENH_CHAY.md` | `docs/LENH_CHAY.md` | paths updated |
| COMB `reference/comb_forcamera.py` | — | Windows prototype, never ran in production; kept in COMB history |
| COMB `web_server.py` | — | older copy; JS version kept (see §3) |

Import paths changed from `from relay_hub import …` to
`from yarnbreak.relay_hub import …` and so on. `tests/test_imports.py` fails the
build if anything still imports a module from its old location.

## 3. Feature source per area

| Area | Kept from | Why |
|---|---|---|
| Detection pipeline, comb pipeline, relay, SDK, state, DB, web UI | identical in both | — |
| Package layout, CI, pytest suite, MIT license | JS | |
| Web login (PBKDF2, no default password, CORS off) | JS | more secure; still reads old plain `ADMIN_PASSWORD` |
| `config.yaml` | JS | same keys as COMB; serials left blank for the public repo |
| Operator scripts, camera tools, operator guide | COMB | not in JS |
| Comb README images | COMB | |

## 4. Discrepancies found and how they were resolved

1. **Relay truth table in the COMB README was wrong for CH2.** It said a comb
   alarm leaves CH2 at 0 (green light on). The production code
   (`yarnbreak/alerts.py`, comment "QUYẾT ĐỊNH ĐÃ CHỐT") mirrors every
   `alarm:*` reason onto CH2, so the green light goes **off** for comb alarms
   too. The code was not changed; the tests and README now describe the code.
2. **Comb alarm thresholds.** The engine defaults are rule B = 3 hits > 0.8, but
   the shipped `config.yaml` sets `high_window: 4`, `high_conf: 0.5`. Both are
   tested; the config values are what runs.
3. **Beam (roller) station in config.** Comb cameras are already a config list.
   The roller camera is still a single set of top-level keys, and the loop in
   `yarnbreak/pipeline.py` runs exactly one. Making it a list is a pipeline
   change that needs hardware testing, so it was left for a separate change.

## 5. Check on the Jetson before deploying

Run on the real machine (none of this can be tested without the hardware):

- [ ] `git pull`, then `sudo systemctl restart yarnbreak` and
      `journalctl -u yarnbreak -f`: no `ImportError` / `ModuleNotFoundError`.
- [ ] `.env` still logs in. If it relied on the old built-in default password
      (no `ADMIN_PASSWORD` line), set one first — accounts without a password
      are now disabled. Optional: switch to `ADMIN_PASSWORD_HASH` via
      `python3 scripts/hash_password.py`.
- [ ] Web UI shows all 3 camera cards with FPS and inference ms; live view of
      each camera opens.
- [ ] Log shows `[comb_1] … ResultingFrameRate=8.0` and `[comb_2] …` and
      `Comb pipeline đã khởi động với 2 camera`.
- [ ] Relay, comb alarm: CH1 + CH3 on, **green light off**.
- [ ] Relay, roller alarm: CH1 + CH2 on, yellow off.
- [ ] CH1 held ≥ 7 s → CH7 blinks; CH1 released → CH7 off.
- [ ] Red button: CH1 releases, yellow on (paused). Green button: back to idle.
- [ ] Unplug one comb camera for 1 min: roller camera keeps running, yellow does
      not light, comb camera reconnects.
- [ ] Update shortcuts / autostart that pointed to `chay*.sh` or
      `open_web_ui.sh` in the repo root → now `scripts/…`
      (e.g. `~/.config/autostart/yarnbreak-web-ui.desktop`).
- [ ] If the systemd unit was installed from the old path, nothing changes
      (`web_server.py` is still at the root); re-copy from `deploy/` only if you
      edit it.
- [ ] Tools still work: `python3 scripts/kiem_tra_camera.py --cam comb_2`,
      `./scripts/chay_chup_mau.sh`, `./scripts/run_jetson_log.sh`.
- [ ] `python3 scripts/test_red_button_alarm.py` with the relay board connected.
- [ ] Watch temperature for the first hours (throttle 75 °C, pause 80 °C).
