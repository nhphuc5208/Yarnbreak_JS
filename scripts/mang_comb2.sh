#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# mang_comb2.sh — dat IP tinh + MTU 9000 cho cong mang noi camera luoc comb_2
#                 (Hikrobot MV-CU060-10GC, serial DB2796591).
#
# Camera DA duoc ghi IP TINH trong flash: 192.168.3.101/24, gw 192.168.3.1.
# Cong mang cua Jetson phai nam cung dai -> 192.168.3.100/24, MTU 9000.
# comb_1 (CU120) dung dai 192.168.2.x -> KHAC subnet, dung thiet ke, dung sua.
#
# Dung:  sudo bash scripts/mang_comb2.sh            # liet ke cong
#        sudo bash scripts/mang_comb2.sh eth1       # ap dung
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CAM_IP="192.168.3.101"
HOST_IP="192.168.3.100"
PREFIX="24"
MTU="9000"
CON="comb2"

IFACE="${1:-}"

if [ -z "$IFACE" ]; then
  echo "Chua chon cong mang. Cac cong ethernet dang co:"
  echo
  nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status \
    | awk -F: '$2=="ethernet"{printf "  %-12s trang thai=%-12s connection=%s\n",$1,$3,$4}'
  echo
  echo "Cong cua comb_1 (dai 192.168.2.x) hien la:"
  ip -4 -br addr show | grep -E '192\.168\.2\.' || echo "  (khong thay)"
  echo
  echo "Chay lai voi ten cong, vi du:  sudo bash $0 eth1"
  exit 1
fi

if ! ip link show "$IFACE" >/dev/null 2>&1; then
  echo "LOI: khong co cong mang '$IFACE'." >&2
  exit 2
fi

# Khong de nguoi dung lo tay de len cong cua comb_1.
if ip -4 addr show dev "$IFACE" | grep -q '192\.168\.2\.'; then
  echo "LOI: '$IFACE' dang o dai 192.168.2.x = cong cua comb_1. Chon cong KHAC." >&2
  exit 2
fi

echo "=== Cau hinh $IFACE -> $HOST_IP/$PREFIX, MTU $MTU"

if nmcli -t -f NAME connection show | grep -qx "$CON"; then
  nmcli connection modify "$CON" \
    connection.interface-name "$IFACE" \
    ipv4.method manual ipv4.addresses "$HOST_IP/$PREFIX" \
    ipv4.gateway "" ipv4.never-default yes ipv6.method ignore \
    802-3-ethernet.mtu "$MTU"
else
  nmcli connection add type ethernet con-name "$CON" ifname "$IFACE" \
    ipv4.method manual ipv4.addresses "$HOST_IP/$PREFIX" \
    ipv4.never-default yes ipv6.method ignore \
    802-3-ethernet.mtu "$MTU"
fi
nmcli connection up "$CON"

# ipv4.never-default: cong camera KHONG duoc thanh default route, neu khong
# Jetson se mat duong ra internet/LAN.

sleep 2
echo
echo "=== Ket qua"
ip -4 -br addr show dev "$IFACE"
echo -n "MTU thuc te: "; cat "/sys/class/net/$IFACE/mtu"

echo
echo "=== Ping camera $CAM_IP (goi 8972 byte de thu jumbo frame)"
if ping -c 2 -W 2 "$CAM_IP" >/dev/null 2>&1; then
  echo "  ping thuong: OK"
else
  echo "  ping thuong: KHONG TOI -> kiem tra day mang / adapter USB->GigE"
fi
# -M do = cam phan manh; 8972 = 9000 - 20 (IP) - 8 (ICMP)
if ping -c 2 -W 2 -M do -s 8972 "$CAM_IP" >/dev/null 2>&1; then
  echo "  jumbo 9000: OK -> packet_size 9000 trong config.yaml se chay dung"
else
  echo "  jumbo 9000: KHONG QUA -> MTU chua an toan, chuong trinh se tu ha packet size"
  echo "              (kiem tra ca MTU cua switch/adapter neu co)"
fi
