"""Tiny WebSocket test client / REPL for the laptop brain (no phone needed).

    python -m laptop_brain.cli_client                       # interactive
    python -m laptop_brain.cli_client -c "preset ready" -c "watch 3" -c estop
    python -m laptop_brain.cli_client --url ws://192.168.1.20:8765 --token SECRET

Commands:  state | angles a0..a5 | nudge CH DELTA | preset NAME | routine NAME | chain NAME_OR_TOKENS... [@OBJECT]
           voice WORD [conf] | listen | scan [OBJECT] | estop | clear | stop | presets | routines | chains
           (OBJECT = a label such as cup, bottle, cell_phone or an ArUco id such as 7)
           cal | events [N] | watch [SECONDS] | raw JSON | quit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from itertools import count

from websockets.asyncio.client import connect

_ids = count(1)


def parse(line: str):
    """Translate one REPL line into a message dict, or a ('local', ...) tuple."""
    parts = line.split()
    if not parts:
        return None
    c, a = parts[0].lower(), parts[1:]
    if c in ("quit", "exit"):
        return ("local", "quit")
    if c == "watch":
        return ("local", "watch", float(a[0]) if a else 3.0)
    if c == "raw":
        return json.loads(line.partition(" ")[2])
    simple = {"state": "GET_STATE", "estop": "ESTOP", "clear": "CLEAR_ESTOP", "stop": "STOP", "scan": "TRIGGER_VISION_SCAN",
              "listen": "TRIGGER_VOICE_LISTEN", "presets": "LIST_PRESETS", "routines": "LIST_ROUTINES",
              "chains": "LIST_CHAINS", "cal": "GET_CALIBRATION"}
    if c == "scan" and a:
        return {"type": "TRIGGER_VISION_SCAN", "target": a[0].lstrip("@")}
    if c in simple:
        return {"type": simple[c]}
    if c == "angles":
        vals = [None if v in ("_", "-") else float(v) for v in a]
        return {"type": "SET_ANGLES", "angles": vals, "transition_ms": 0}
    if c == "nudge":
        return {"type": "NUDGE_JOINT", "channel": int(a[0]), "delta_deg": float(a[1])}
    if c == "preset":
        return {"type": "RUN_PRESET", "name": a[0]}
    if c == "routine":
        return {"type": "RUN_ROUTINE", "name": a[0]}
    if c == "chain":
        target = next((t[1:] for t in a if t.startswith("@")), None)
        a = [t for t in a if not t.startswith("@")]
        msg = {"type": "RUN_CHAIN", "name": a[0]} if len(a) == 1 else {"type": "RUN_CHAIN", "tokens": a}
        if target:
            msg["target"] = target
        return msg
    if c == "voice":
        return {"type": "INJECT_VOICE", "word": a[0], "confidence": float(a[1]) if len(a) > 1 else 0.95}
    if c == "events":
        return {"type": "GET_EVENTS", "limit": int(a[0]) if a else 15}
    raise ValueError(f"unknown command '{c}' (try: state, preset ready, chain scan grab lift release, estop ...)")


def fmt_telemetry(m: dict) -> str:
    ang = " ".join(f"{x:6.1f}" for x in m["current_angles"])
    job = m.get("job")
    extra = f" job={job['kind']}:{job['name']}[{job['step']}]" if job else ""
    return f"{m['state']:<16} [{ang}] esp={'ok' if m['esp_connected'] else 'DOWN'} q={m['queue_len']}{extra}"


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.pending: dict = {}
        self.latest: dict = {}
        self.show_telemetry = False
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = json.loads(raw)
            t = m.get("type")
            if t in ("ack", "error") and m.get("id") in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif t == "telemetry":
                self.latest = m
                if self.show_telemetry:
                    print("  tele", fmt_telemetry(m))
            elif t == "event":
                print(f"  event {m['name']}: {json.dumps(m['data'])[:160]}")
            elif t == "hello":
                self.latest = m["snapshot"]
                print(f"connected (sim={m['sim']})  {fmt_telemetry(m['snapshot'])}")

    async def call(self, msg: dict, timeout: float = 5.0) -> dict:
        msg["id"] = f"c{next(_ids)}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[msg["id"]] = fut
        await self.ws.send(json.dumps(msg))
        return await asyncio.wait_for(fut, timeout)

    async def run_line(self, line: str) -> bool:
        try:
            msg = parse(line)
        except Exception as exc:
            print("  !", exc)
            return True
        if msg is None:
            return True
        if isinstance(msg, tuple):
            if msg[1] == "quit":
                return False
            self.show_telemetry = True
            await asyncio.sleep(msg[2])
            self.show_telemetry = False
            return True
        reply = await self.call(msg)
        if reply["type"] == "error":
            print(f"  ERROR {reply['code']}: {reply['message']}")
        else:
            print("  ok", json.dumps(reply["data"], default=str)[:600])
        return True


async def amain(args) -> int:
    url = args.url + (f"?token={args.token}" if args.token else "")
    async with connect(url) as ws:
        cli = Client(ws)
        await asyncio.sleep(0.3)
        if args.command:
            for line in args.command:
                print(f"> {line}")
                if not await cli.run_line(line):
                    break
            return 0
        print(__doc__)
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            if not await cli.run_line(line):
                break
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    p.add_argument("--token", default="")
    p.add_argument("-c", "--command", action="append", help="run a command non-interactively (repeatable)")
    try:
        return asyncio.run(amain(p.parse_args()))
    except (KeyboardInterrupt, ConnectionRefusedError, OSError) as exc:
        print("connection problem:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
