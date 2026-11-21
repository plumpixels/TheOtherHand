"""ESP8266 link: transport abstraction, frame protocol, health tracking, simulator.

Wire protocol (one frame per line, ``\\n`` terminated) -- see README for the firmware contract::

    laptop -> ESP   M,<seq>,<us0>,...,<us5>   servo pulse widths in microseconds (ch 0-5)
                    H,<seq>                    heartbeat (also the firmware watchdog feed)
                    E,<seq>                    ESTOP: freeze outputs, ignore M until C
                    C,<seq>                    clear ESTOP
    ESP -> laptop   R,<fw>                     ready after boot
                    A,<seq>[,info]             ack of H (and optionally M)
                    W                          firmware watchdog tripped

``frame_format="json"`` sends the same frames as {"c":"M","s":1,"p":[...]}.
The laptop owns calibration: it converts angles to microseconds so the firmware stays dumb.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from .config import LinkSettings, NUM_SERVOS

log = logging.getLogger("esp")


# --------------------------------------------------------------------------- #
# Frames                                                                       #
# --------------------------------------------------------------------------- #
def angles_to_pulses(angles: Sequence[float], cals: Sequence) -> list:
    """Angle (0..180 deg) -> pulse width in us, clamped to each servo's min/max angle."""
    out = []
    for a, c in zip(angles, cals):
        a = min(c.max_angle, max(c.min_angle, a))
        out.append(int(round(c.min_pulse + (c.max_pulse - c.min_pulse) * a / 180.0)))
    return out


def encode_move(seq: int, pulses: Sequence[int], fmt: str = "csv") -> bytes:
    if fmt == "json":
        return (json.dumps({"c": "M", "s": seq, "p": list(pulses)}, separators=(",", ":")) + "\n").encode()
    return f"M,{seq},{','.join(str(p) for p in pulses)}\n".encode()


def encode_cmd(code: str, seq: int, fmt: str = "csv") -> bytes:
    if fmt == "json":
        return (json.dumps({"c": code, "s": seq}, separators=(",", ":")) + "\n").encode()
    return f"{code},{seq}\n".encode()


# --------------------------------------------------------------------------- #
# Transports                                                                   #
# --------------------------------------------------------------------------- #
class Transport(ABC):
    name = "transport"

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write(self, data: bytes) -> None: ...

    @abstractmethod
    def readline(self) -> Optional[bytes]:
        """Return one stripped line, or None if nothing arrived within ~50 ms."""


class SerialTransport(Transport):
    def __init__(self, port: str = "auto", baud: int = 115200):
        self.port, self.baud = port, baud
        self._ser = None
        self._buf = b""
        self.name = f"serial:{port}"

    @staticmethod
    def find_port() -> str:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        # Raspberry Pi RP2040/RP2350 (Pico) first, then CH340, CP210x, FTDI (typical ESP8266 boards).
        for vids in ({0x2E8A}, {0x1A86, 0x10C4, 0x0403}):
            for p in ports:
                if p.vid in vids:
                    return p.device
        if ports:
            return ports[0].device
        raise ConnectionError("no serial ports found")

    def open(self) -> None:
        import serial
        port = self.find_port() if self.port == "auto" else self.port
        self._ser = serial.Serial(port, self.baud, timeout=0.05, write_timeout=0.2)
        self._buf = b""
        self.name = f"serial:{port}"

    def close(self) -> None:
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def write(self, data: bytes) -> None:
        if not self._ser:
            raise ConnectionError("serial closed")
        self._ser.write(data)

    def readline(self) -> Optional[bytes]:
        if not self._ser:
            raise ConnectionError("serial closed")
        if b"\n" not in self._buf:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                self._buf += chunk
        if b"\n" in self._buf:
            line, _, self._buf = self._buf.partition(b"\n")
            return line.strip() or None
        if len(self._buf) > 512:
            self._buf = b""
        return None


class UdpTransport(Transport):
    """Optional Wi-Fi path: the ESP8266 listens on a UDP port and replies to the sender."""

    def __init__(self, host: str, port: int = 4210):
        self.host, self.port = host, port
        self._sock: Optional[socket.socket] = None
        self._pending: list = []
        self.name = f"udp:{host}:{port}"

    def open(self) -> None:
        if not self.host:
            raise ConnectionError("udp_host not configured")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.05)
        self._sock.connect((self.host, self.port))

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def write(self, data: bytes) -> None:
        if not self._sock:
            raise ConnectionError("udp closed")
        self._sock.send(data)

    def readline(self) -> Optional[bytes]:
        if not self._sock:
            raise ConnectionError("udp closed")
        if not self._pending:
            try:
                data = self._sock.recv(512)
            except (socket.timeout, ConnectionRefusedError):
                return None
            self._pending = [l.strip() for l in data.split(b"\n") if l.strip()]
        return self._pending.pop(0) if self._pending else None


class SimTransport(Transport):
    """In-process fake ESP8266 + PCA9685.  Parses the real wire protocol and acks like firmware would."""
    name = "sim"

    def __init__(self):
        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._lock = threading.Lock()
        self.is_open = False
        self.offline_until = 0.0
        self.pulses = [1500] * NUM_SERVOS
        self.frozen = False
        self.move_frames = 0
        self.estops = 0

    def set_offline(self, seconds: float) -> None:
        self.offline_until = time.monotonic() + seconds

    def _offline(self) -> bool:
        return time.monotonic() < self.offline_until

    def open(self) -> None:
        if self._offline():
            raise ConnectionError("sim ESP offline")
        self.is_open = True
        self._q.put(b"R,SIM1")

    def close(self) -> None:
        self.is_open = False

    def write(self, data: bytes) -> None:
        if self._offline() or not self.is_open:
            raise ConnectionError("sim ESP offline")
        for raw in data.splitlines():
            self._handle(raw.strip())

    def _handle(self, raw: bytes) -> None:
        if not raw:
            return
        try:
            if raw.startswith(b"{"):
                msg = json.loads(raw)
                code, seq, vals = msg["c"], int(msg.get("s", 0)), msg.get("p", [])
            else:
                parts = raw.decode().split(",")
                code, seq, vals = parts[0], int(parts[1]), [int(v) for v in parts[2:]]
        except (ValueError, KeyError, IndexError):
            self._q.put(b"W,bad_frame")
            return
        with self._lock:
            if code == "M":
                if not self.frozen and len(vals) == NUM_SERVOS:
                    self.pulses = list(vals)
                    self.move_frames += 1
            elif code == "E":
                self.frozen = True
                self.estops += 1
            elif code == "C":
                self.frozen = False
        if code in ("H", "E", "C"):
            self._q.put(f"A,{seq},{'FROZEN' if self.frozen else 'OK'}".encode())

    def readline(self) -> Optional[bytes]:
        if self._offline():
            raise ConnectionError("sim ESP offline")
        try:
            return self._q.get(timeout=0.05)
        except queue.Empty:
            return None


# --------------------------------------------------------------------------- #
# Link                                                                         #
# --------------------------------------------------------------------------- #
class EspLink:
    """Owns the transport.  A background thread does connect/reconnect, reads, heartbeats and health."""

    def __init__(self, factory: Callable[[], Transport], settings: LinkSettings, cals: Sequence,
                 loop=None, on_change: Optional[Callable[[bool], None]] = None):
        self._factory, self.s, self._cals = factory, settings, list(cals)
        self._loop, self.on_change = loop, on_change
        self._lock = threading.RLock()
        self._t: Optional[Transport] = None
        self._connected = False
        self._frozen = False
        self._seq = 0
        self._last_pulses: Optional[list] = None
        self._last_rx = 0.0
        self._opened_at = 0.0
        self._last_hb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.tx_frames = 0
        self.last_rx_line = ""
        self.last_error = ""
        self.transport_name = ""

    # -- public ------------------------------------------------------------- #
    @property
    def connected(self) -> bool:
        return self._connected

    def set_calibration(self, cals: Sequence) -> None:
        self._cals = list(cals)
        self._last_pulses = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="esp-link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._drop("shutdown", notify=False)

    def status(self) -> dict:
        age = time.monotonic() - self._last_rx if self._last_rx else None
        return {"connected": self._connected, "transport": self.transport_name, "frozen": self._frozen,
                "tx_frames": self.tx_frames, "rx_age_s": None if age is None else round(age, 2),
                "last_rx": self.last_rx_line, "last_error": self.last_error}

    def send_angles(self, angles: Sequence[float]) -> bool:
        """Called by the motion loop every time the pose changes.  Drops silently while disconnected/frozen."""
        if self._frozen or not self._connected:
            return False
        pulses = angles_to_pulses(angles, self._cals)
        if pulses == self._last_pulses:
            return True
        ok = self._write(encode_move(self._next_seq(), pulses, self.s.frame_format))
        if ok:
            self._last_pulses = pulses
        return ok

    def send_estop(self) -> None:
        """Highest priority path: write E immediately (3x for lossy links), bypassing any queueing."""
        self._frozen = True
        for _ in range(3):
            self._write(encode_cmd("E", self._next_seq(), self.s.frame_format))

    def send_clear(self) -> bool:
        self._frozen = False
        self._last_pulses = None
        return self._write(encode_cmd("C", self._next_seq(), self.s.frame_format))

    # -- internals ----------------------------------------------------------- #
    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    def _write(self, data: bytes) -> bool:
        with self._lock:
            if self._t is None:
                return False
            try:
                self._t.write(data)
                self.tx_frames += 1
                return True
            except Exception as exc:
                self._drop(f"write failed: {exc}")
                return False

    def _set_connected(self, flag: bool) -> None:
        if flag == self._connected:
            return
        self._connected = flag
        log.info("ESP %s (%s)", "CONNECTED" if flag else "DISCONNECTED", self.transport_name)
        if self.on_change and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.on_change, flag)

    def _drop(self, reason: str, notify: bool = True) -> None:
        with self._lock:
            if self._t is not None:
                try:
                    self._t.close()
                except Exception:
                    pass
                self._t = None
            if reason:
                self.last_error = reason
            self._last_pulses = None
        if notify:
            self._set_connected(False)
        else:
            self._connected = False

    def _run(self) -> None:
        backoff = self.s.reconnect_min_s
        while not self._stop.is_set():
            if self._t is None:
                try:
                    t = self._factory()
                    t.open()
                    with self._lock:
                        self._t = t
                    self.transport_name = t.name
                    self._opened_at = time.monotonic()
                    self._last_rx = 0.0
                    self._last_hb = 0.0
                    backoff = self.s.reconnect_min_s
                    if not self.s.require_ack:
                        self._last_rx = time.monotonic()
                        self._set_connected(True)
                except Exception as exc:
                    self.last_error = f"open failed: {exc}"
                    log.debug("link open failed: %s", exc)
                    self._stop.wait(backoff)
                    backoff = min(self.s.reconnect_max_s, backoff * 1.7)
                    continue
            t = self._t
            now = time.monotonic()
            try:
                line = t.readline() if t else None
            except Exception as exc:
                self._drop(f"read failed: {exc}")
                continue
            if line:
                self._last_rx = time.monotonic()
                self.last_rx_line = line.decode(errors="replace")[:80]
                if not self._connected:
                    self._set_connected(True)
            if self._t is not None and now - self._last_hb >= self.s.heartbeat_s:
                self._last_hb = now
                self._write(encode_cmd("H", self._next_seq(), self.s.frame_format))
            if self._t is not None:
                if self._connected and self.s.require_ack and now - self._last_rx > self.s.rx_timeout_s:
                    self._drop("no reply from ESP (timeout)")
                elif not self._connected and now - self._opened_at > self.s.connect_grace_s:
                    self._drop("ESP did not answer within the connect grace period")
