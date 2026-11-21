"""Entry point:  python -m laptop_brain --sim   |   python -m laptop_brain --serial auto --token SECRET"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .app import run_forever
from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="laptop_brain", description="theOtherHand laptop host")
    p.add_argument("--sim", action="store_true", help="simulated ESP8266, vision and voice; separate database; no hardware needed")
    p.add_argument("--config", help="JSON file overriding config.py defaults")
    p.add_argument("--serial", metavar="PORT", help="ESP8266 serial port (COM5, /dev/ttyUSB0) or 'auto'")
    p.add_argument("--baud", type=int)
    p.add_argument("--udp", metavar="HOST[:PORT]", help="talk to the ESP8266 over Wi-Fi/UDP instead of serial")
    p.add_argument("--host", help="WebSocket bind address (default 0.0.0.0)")
    p.add_argument("--port", type=int, help="WebSocket port (default 8765)")
    p.add_argument("--token", help="require ?token=... from WebSocket clients")
    p.add_argument("--db", help="SQLite path")
    p.add_argument("--voice", choices=["auto", "vosk", "mock", "keyboard"], help="voice backend")
    p.add_argument("--vision", choices=["auto", "opencv", "sim"], help="vision backend")
    p.add_argument("--speed", type=float, help="global speed multiplier (0.1-1.0); try 0.4 for first hardware runs")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    s = load_settings(args.config)
    s.sim = args.sim or s.sim
    if args.host:
        s.server.host = args.host
    if args.port is not None:
        s.server.port = args.port
    if args.token:
        s.server.token = args.token
    if args.voice:
        s.voice.backend = args.voice
    if args.vision:
        s.vision.backend = args.vision
    if args.speed is not None:
        s.motion.speed_scale = max(0.05, min(args.speed, 1.0))
    if args.log_level:
        s.log_level = args.log_level
    if args.baud:
        s.link.baud = args.baud
    if args.serial:
        s.link.kind, s.link.serial_port = "serial", args.serial
    if args.udp:
        host, _, port = args.udp.partition(":")
        s.link.kind, s.link.udp_host = "udp", host
        if port:
            s.link.udp_port = int(port)
    if s.sim:
        s.link.kind = "sim"
        s.db_path = args.db or "data/laptop_brain_sim.sqlite3"       # never mix sim edits into real calibration
        s.server.allow_inject = True
    elif args.db:
        s.db_path = args.db
    if not s.sim and s.link.kind == "serial" and not args.serial and s.link.serial_port == "auto":
        pass                                                         # 'auto' is fine: first CH340/CP210x/FTDI port
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)-7s %(name)-8s %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(run_forever(s))
    except KeyboardInterrupt:
        print("\nbye (ESP was told to freeze)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
