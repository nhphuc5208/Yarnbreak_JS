# Contributing

This code drives a real machine 24/7: a wrong relay output can stop production
or, worse, fail to stop it. Changes go through the gate below.

## Quality gate (enforced by CI on every push / PR)

```bash
ruff check .                 # syntax errors + pyflakes
pytest                       # hardware-free: fake relay, simulated clock
node tests/check_i18n.js .   # i18n keys used by the UI exist in vi / ko / en
```

All three must pass before merging.

## Rules

1. **Relay or alarm logic changes start with the test.**
   Changing `yarnbreak/relay_hub.py`, the relay wiring in `yarnbreak/alerts.py` /
   `yarnbreak/comb_pipeline.py`, or the alarm rules (`CombAlertEngine`,
   `AlertNotifier`) requires updating `tests/test_relay_truth_table.py` or
   `tests/test_comb_alert.py` **first**, in the same PR, so the new expected
   behaviour is written down before the code changes. The relay truth table in
   the README is the contract with the machine.
2. **`config.yaml` is the single source of truth.** Thresholds, channels, camera
   serials, ROIs and timings belong in config, not in code. Adding or removing a
   camera must not require a new Python file. Never rename a key that is used on
   site; if a rename is unavoidable, keep reading the old key and note it in
   `CHANGELOG.md`.
3. **All runtime logic lives in the `yarnbreak/` package.** `main.py` and
   `web_server.py` are entry points only; `scripts/` holds operator tools.
   Script file names are used by maintenance staff — do not rename them.
4. **i18n:** every new UI string gets a key in all three languages (vi, ko, en).
5. **Never commit** model weights (`*.pt`, `*.onnx`, `*.engine`), videos,
   factory dataset images, `.env` or real passwords.
6. **Commits:** small and meaningful, [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`).
7. Anything that cannot be tested on a PC (camera, GPU, relay board) must be
   listed in the PR description with how it was checked on the Jetson.
