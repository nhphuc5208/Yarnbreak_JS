#!/usr/bin/env python3
"""
Modbus RTU relay control for the Waveshare "Modbus RTU Relay (D) 8DI/8DO"
board, driven over a Waveshare USB-TO-RS485 (B) adapter.

Protocol (Waveshare relay family — standard Modbus RTU):
  • Close one relay  : func 0x05  addr=CH-1  data=0xFF00
  • Open one relay   : func 0x05  addr=CH-1  data=0x0000
  • Read relay status: func 0x01  addr=0x0000  qty=8
  CH1 → coil address 0x0000, CH2 → 0x0001, … CH8 → 0x0007.
  Default slave address 0x01, 9600 8N1.

This module depends only on pyserial (no pymodbus) so the framing/CRC is
explicit and version-stable.

CLI test (after the user account is in the `dialout` group):
    python3 modbus_relay.py --port /dev/ttyACM0 --on  1   # close CH1
    python3 modbus_relay.py --port /dev/ttyACM0 --off 1   # open  CH1
    python3 modbus_relay.py --port /dev/ttyACM0 --status  # read 8 relays
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

logger = logging.getLogger("YarnBreak.Relay")


def _crc16(data: bytes) -> bytes:
    """Modbus RTU CRC-16 (poly 0xA001), returned little-endian (lo, hi)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF))


class ModbusRelay:
    """Minimal Modbus RTU master for one Waveshare relay board."""

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 9600,
                 slave: int = 1, timeout: float = 0.3):
        self.port = port
        self.baudrate = baudrate
        self.slave = slave
        self.timeout = timeout
        self._ser = None
        self._lock = threading.Lock()  # serialise the half-duplex RS485 bus

    def open(self) -> bool:
        try:
            import serial  # pyserial
        except ImportError as exc:
            logger.error(f"Relay requested but pyserial is missing: {exc}")
            return False
        try:
            self._ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except Exception as exc:
            logger.error(f"Relay serial open failed on {self.port}: {exc}")
            self._ser = None
            return False
        logger.info(
            f"Relay ready: {self.port} @ {self.baudrate} 8N1 slave={self.slave}"
        )
        return True

    def close(self):
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── Low-level transaction ────────────────────────────────────────────────
    def _txn(self, pdu: bytes, expected_len: int) -> Optional[bytes]:
        """Send slave+PDU+CRC, read a fixed-length reply, validate CRC."""
        if self._ser is None:
            return None
        frame = bytes((self.slave,)) + pdu
        frame += _crc16(frame)
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write(frame)
                self._ser.flush()
                reply = self._ser.read(expected_len)
            except Exception as exc:
                logger.warning(f"Relay txn I/O error: {exc}")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                return None
        if len(reply) < expected_len:
            logger.warning(
                f"Relay txn short reply: got {len(reply)}/{expected_len} bytes "
                f"({reply.hex()})"
            )
            return None
        if _crc16(reply[:-2]) != reply[-2:]:
            logger.warning(f"Relay txn CRC mismatch: {reply.hex()}")
            return None
        return reply

    # ── Public API ───────────────────────────────────────────────────────────
    def set_relay(self, channel: int, on: bool) -> bool:
        """Close (on=True) or open (on=False) a 1-based relay channel."""
        if self._ser is None:
            logger.info("Relay: connection lost, attempting reconnect...")
            if not self.open():
                logger.warning(f"Relay CH{channel} skipped — reconnect failed")
                return False
        addr = channel - 1
        value = 0xFF00 if on else 0x0000
        # func 0x05 write-single-coil; the board echoes the request back.
        pdu = bytes((0x05, (addr >> 8) & 0xFF, addr & 0xFF,
                     (value >> 8) & 0xFF, value & 0xFF))
        reply = self._txn(pdu, expected_len=8)
        ok = reply is not None
        if ok:
            logger.info(f"Relay CH{channel} → {'CLOSE' if on else 'OPEN'}")
        else:
            logger.warning(
                f"Relay CH{channel} {'CLOSE' if on else 'OPEN'} — no/invalid ack"
            )
        return ok

    def set_relays(self, channels: List[int], values: List[bool]) -> bool:
        """Set multiple relay channels (1-based) atomically using function 0x0F (Write Multiple Coils).
        Falls back to writing sequentially using set_relay if the transaction fails.
        """
        if not channels:
            return True
        if len(channels) != len(values):
            logger.error("set_relays: channels and values list length mismatch")
            return False

        # First, try to write using function 0x0F if they are contiguous
        sorted_indices = sorted(range(len(channels)), key=lambda k: channels[k])
        sorted_ch = [channels[i] for i in sorted_indices]
        is_contiguous = len(sorted_ch) > 0 and all(sorted_ch[i] == sorted_ch[i-1] + 1 for i in range(1, len(sorted_ch)))

        if is_contiguous:
            start_ch = sorted_ch[0]
            start_addr = start_ch - 1
            quantity = len(sorted_ch)
            byte_count = (quantity + 7) // 8
            
            # Map sorted values to byte stream
            val_byte = 0
            for i, idx in enumerate(sorted_indices):
                if values[idx]:
                    val_byte |= (1 << i)
            
            pdu = bytes((0x0F, (start_addr >> 8) & 0xFF, start_addr & 0xFF,
                         (quantity >> 8) & 0xFF, quantity & 0xFF,
                         byte_count, val_byte))
            
            # Expected reply: slave (1) + func (1) + start_addr (2) + quantity (2) + CRC (2) = 8 bytes
            reply = self._txn(pdu, expected_len=8)
            if reply is not None and reply[1] == 0x0F:
                logger.debug(f"Relays {channels} set to {values} via Modbus 0x0F")
                return True
            else:
                logger.warning("Multiple coils write (0x0F) failed or unsupported. Falling back to single write...")

        # Fallback to writing sequentially
        success = True
        for ch, val in zip(channels, values):
            if not self.set_relay(ch, val):
                success = False
        return success

    def read_relays(self, count: int = 8) -> Optional[List[bool]]:
        """Read coil (relay) states. func 0x01 read-coils from address 0."""
        pdu = bytes((0x01, 0x00, 0x00, (count >> 8) & 0xFF, count & 0xFF))
        nbytes = (count + 7) // 8
        reply = self._txn(pdu, expected_len=3 + nbytes + 2)
        if reply is None or reply[1] != 0x01:
            return None
        data = reply[3:3 + nbytes]
        return [bool((data[i // 8] >> (i % 8)) & 1) for i in range(count)]

    def read_inputs(self, count: int = 8) -> Optional[List[bool]]:
        """Read digital input states. func 0x02 read-inputs from address 0."""
        pdu = bytes((0x02, 0x00, 0x00, (count >> 8) & 0xFF, count & 0xFF))
        nbytes = (count + 7) // 8
        reply = self._txn(pdu, expected_len=3 + nbytes + 2)
        if reply is None or reply[1] != 0x02:
            return None
        data = reply[3:3 + nbytes]
        return [bool((data[i // 8] >> (i % 8)) & 1) for i in range(count)]




# ── Standalone CLI tester ─────────────────────────────────────────────────────
def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Waveshare Modbus RTU relay tester")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", type=int, metavar="CH", help="close channel CH (1-8)")
    g.add_argument("--off", type=int, metavar="CH", help="open channel CH (1-8)")
    g.add_argument("--status", action="store_true", help="read 8 relay states")
    g.add_argument("--inputs", action="store_true", help="read 8 digital input states")
    g.add_argument("--blink", type=int, metavar="CH", help="close→1s→open channel CH")
    args = p.parse_args()

    r = ModbusRelay(port=args.port, baudrate=args.baud, slave=args.slave)
    if not r.open():
        raise SystemExit(1)
    try:
        if args.status:
            print("relays:", r.read_relays())
        elif args.inputs:
            print("inputs:", r.read_inputs())
        elif args.on is not None:
            r.set_relay(args.on, True)
        elif args.off is not None:
            r.set_relay(args.off, False)
        elif args.blink is not None:
            r.set_relay(args.blink, True)
            time.sleep(1.0)
            r.set_relay(args.blink, False)
    finally:
        r.close()


if __name__ == "__main__":
    _main()
