#!/usr/bin/env bash
# Wrapper: chay log_jetson_stats.py, record moi 5 giay 1 lan.
# File CSV dat ten theo thoi gian de moi lan chay khong ghi de len nhau.
cd "$(dirname "$0")/.." || exit 1   # scripts/ -> project root

OUT="jetson_stats_$(date +%Y%m%d_%H%M%S).csv"
echo "==> Ghi log moi 5s vao: $(pwd)/$OUT"
echo "==> Nhan Ctrl-C de dung."
echo

python3 scripts/log_jetson_stats.py -i 5 -o "$OUT"

echo
echo "==> Da dung. File luu tai: $(pwd)/$OUT"
echo "Nhan Enter de dong cua so..."
read -r
