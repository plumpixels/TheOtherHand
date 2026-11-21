"""WebSocket API for the React Native / Expo app (and the CLI test client).

Every client message is a JSON object ``{"type": "<COMMAND>", "id": "<optional>", ...}``.
Replies echo the id:  ``{"type":"ack","id":..,"cmd":..,"ok":true,"data":{..}}`` or
``{"type":"error","id":..,"cmd":..,"code":"BUSY","message":".."}``.
The server also pushes ``telemetry`` (state, angles, esp_connected, last_vision, last_voice, ...)
and ``event`` messages (job_finished, estop, vision, voice, ...) to every connected client.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from http import HTTPStatus
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import broadcast, serve
from websockets.exceptions import ConnectionClosed

from .config import NUM_SERVOS, Settings
from .db import ValidationError
from .state_machine import Brain, BrainError

log = logging.getLogger("server")


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValidationError(f"{what} must be a finite number")
    return float(v)


def _mode(msg: dict, default: str = "queue") -> str:
    return str(msg.get("mode", default))


class WsServer:
    def __init__(self, brain: Brain, settings: Settings, sim_transport=None):
        self.brain, self.settings, self.s = brain, settings, settings.server
        self.sim_transport = sim_transport
        self.clients: set = set()
        self._server = None
        self._tele_task: Optional[asyncio.Task] = None
        self._last_sent: Optional[str] = None
        self._last_sent_t = 0.0
        brain.add_listener(self._on_brain_event)
        self._handlers = {
            "PING": lambda m: {"pong": time.time()},
            "GET_STATE": lambda m: self.brain.snapshot(),
            "SET_ANGLES": self._set_angles,
            "NUDGE_JOINT": self._nudge,
            "RUN_PRESET": lambda m: self.brain.run_preset(str(m["name"]), "app", _mode(m)),
            "RUN_ROUTINE": lambda m: self.brain.run_routine(str(m["name"]), "app", _mode(m)),
            "RUN_CHAIN": self._run_chain,
            "TRIGGER_VOICE_LISTEN": self._voice_listen,
            "TRIGGER_VISION_SCAN": lambda m: self.brain.run_scan("app", _mode(m), m.get("target"), m.get("detector")),
            "INJECT_VOICE": self._inject_voice,
            "ESTOP": lambda m: self.brain.estop("app", str(m.get("reason", "ESTOP from app"))),
            "CLEAR_ESTOP": lambda m: self.brain.clear_estop("app"),
            "STOP": lambda m: self.brain.stop("app", "STOP from app"),
            "LIST_PRESETS": lambda m: {"presets": [p.to_dict() for p in self.db.presets.list()]},
            "UPSERT_PRESET": self._upsert_preset,
            "DELETE_PRESET": lambda m: {"deleted": self.db.presets.delete(str(m["name"]))},
            "LIST_ROUTINES": lambda m: {"routines": [r.to_dict() for r in self.db.routines.list()]},
            "UPSERT_ROUTINE": self._upsert_routine,
            "DELETE_ROUTINE": lambda m: {"deleted": self.db.routines.delete(str(m["name"]))},
            "LIST_CHAINS": lambda m: {"chains": [c.to_dict() for c in self.db.chains.list()]},
            "UPSERT_CHAIN": self._upsert_chain,
            "DELETE_CHAIN": self._delete_chain,
            "GET_CALIBRATION": lambda m: {"servos": [c.to_dict() for c in self.db.calibration.all()]},
            "SET_CALIBRATION": self._set_calibration,
            "GET_EVENTS": lambda m: {"events": self.db.events.recent(int(m.get("limit", 50)), m.get("source"), m.get("before_id"))},
            "SIM_ESP_DISCONNECT": self._sim_disconnect,
        }

    @property
    def db(self):
        return self.brain.db

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if not self.s.token:
            log.warning("no --token set: anyone on your network can drive the arm. Use --token for real hardware.")
        self._server = await serve(self._handler, self.s.host, self.s.port, process_request=self._auth,
                                   max_size=2 ** 16, ping_interval=10, ping_timeout=10)
        self._tele_task = asyncio.create_task(self._telemetry_loop(), name="telemetry")
        log.info("WebSocket API listening on ws://%s:%d", self.s.host, self.port)

    async def stop(self) -> None:
        if self._tele_task:
            self._tele_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _auth(self, connection, request):
        if not self.s.token:
            return None
        query = parse_qs(urlparse(request.path).query)
        supplied = (query.get("token") or [""])[0]
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            supplied = supplied or bearer[7:]
        if supplied != self.s.token:
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    # ------------------------------------------------------------------ #
    # Connection handling                                                 #
    # ------------------------------------------------------------------ #
    async def _handler(self, ws) -> None:
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "hello", "version": 1, "sim": self.settings.sim,
                                      "snapshot": self.brain.snapshot()}))
            async for raw in ws:
                await self._on_message(ws, raw)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def _on_message(self, ws, raw) -> None:
        mid, cmd = None, ""
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError("message must be a JSON object")
            mid, cmd = msg.get("id"), str(msg.get("type", "")).upper()
            handler = self._handlers.get(cmd)
            if handler is None:
                raise BrainError("UNKNOWN_COMMAND", f"unknown command '{cmd}'")
            data = handler(msg)
            if asyncio.iscoroutine(data):
                data = await data
            reply = {"type": "ack", "id": mid, "cmd": cmd, "ok": True, "data": data}
        except BrainError as exc:
            reply = {"type": "error", "id": mid, "cmd": cmd, "code": exc.code, "message": exc.message}
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            reply = {"type": "error", "id": mid, "cmd": cmd, "code": "BAD_REQUEST", "message": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            log.exception("handler for %s failed", cmd)
            reply = {"type": "error", "id": mid, "cmd": cmd, "code": "INTERNAL", "message": str(exc)}
        try:
            await ws.send(json.dumps(reply, default=str))
        except ConnectionClosed:
            pass

    # ------------------------------------------------------------------ #
    # Push                                                                #
    # ------------------------------------------------------------------ #
    def _on_brain_event(self, name: str, data: dict) -> None:
        if name in ("job_finished", "estop", "estop_cleared", "vision", "voice", "esp"):
            if self.clients:
                broadcast(self.clients, json.dumps({"type": "event", "name": name, "data": data}, default=str))
        if name in ("estop", "estop_cleared", "esp", "state"):
            self._push_telemetry(force=True)

    def _push_telemetry(self, force: bool = False) -> None:
        if not self.clients:
            return
        snap = self.brain.snapshot()
        body = json.dumps({k: v for k, v in snap.items() if k != "ts"}, sort_keys=True, default=str)
        now = time.monotonic()
        if force or body != self._last_sent or now - self._last_sent_t >= 1.0:
            self._last_sent, self._last_sent_t = body, now
            broadcast(self.clients, json.dumps(snap, default=str))

    async def _telemetry_loop(self) -> None:
        period = 1.0 / max(self.s.telemetry_hz, 1.0)
        while True:
            await asyncio.sleep(period)
            self._push_telemetry()

    # ------------------------------------------------------------------ #
    # Handlers                                                            #
    # ------------------------------------------------------------------ #
    def _set_angles(self, m: dict) -> dict:
        raw = m["angles"]
        if isinstance(raw, dict):                           # {"0": 90, "3": 45}
            angles: list = [None] * NUM_SERVOS
            for k, v in raw.items():
                ch = int(k)
                if not 0 <= ch < NUM_SERVOS:
                    raise ValidationError("channel keys must be 0-5")
                angles[ch] = _num(v, f"angle[{k}]")
        elif isinstance(raw, list) and len(raw) == NUM_SERVOS:
            angles = [None if v is None else _num(v, "angle") for v in raw]
        else:
            raise ValidationError("angles must be a list of 6 (null = keep) or an object {channel: angle}")
        tr = int(_num(m.get("transition_ms", 0), "transition_ms"))
        return self.brain.cmd_set_angles(angles, "app", max(0, min(tr, 60_000)))

    def _nudge(self, m: dict) -> dict:
        return self.brain.cmd_nudge(int(_num(m["channel"], "channel")), _num(m["delta_deg"], "delta_deg"), "app")

    def _run_chain(self, m: dict) -> dict:
        extra = {"target": m.get("target"), "detector": m.get("detector")}
        if "tokens" in m:
            return self.brain.run_chain(name=m.get("name"), tokens=m["tokens"], source="app", policy=_mode(m), **extra)
        return self.brain.run_chain(name=str(m["name"]), source="app", policy=_mode(m), **extra)

    def _voice_listen(self, m: dict) -> dict:
        seconds = self.brain.voice.trigger_listen(float(m["seconds"]) if "seconds" in m else None)
        return {"listening_s": seconds, "backend": self.brain.voice.backend_name}

    def _inject_voice(self, m: dict) -> dict:
        if not (self.settings.sim or self.s.allow_inject):
            raise BrainError("FORBIDDEN", "INJECT_VOICE is only enabled in --sim or with server.allow_inject")
        self.brain.voice.inject(str(m["word"]), _num(m.get("confidence", 1.0), "confidence"))
        return dict(self.brain.last_voice or {})

    def _upsert_preset(self, m: dict) -> dict:
        p = self.db.presets.upsert(str(m["name"]), m["angles"], str(m.get("description", "")),
                                   int(m.get("default_transition_ms", 1000)))
        self.db.events.log("app", "preset_saved", p.name)
        warnings = []
        cals = self.db.calibration.all()
        for c, a in zip(cals, p.angles):
            if not c.min_angle <= a <= c.max_angle:
                warnings.append(f"ch{c.channel} {a:.0f} is outside {c.min_angle:.0f}..{c.max_angle:.0f} and will be clamped")
        return {"preset": p.to_dict(), "warnings": warnings}

    def _upsert_routine(self, m: dict) -> dict:
        r = self.db.routines.upsert(str(m["name"]), m["steps"], int(m.get("repeat", 1)), str(m.get("description", "")))
        self.db.events.log("app", "routine_saved", r.name)
        return {"routine": r.to_dict()}

    def _upsert_chain(self, m: dict) -> dict:
        c = self.db.chains.upsert(str(m["name"]), m["tokens"], m.get("aliases"), str(m.get("description", "")))
        self.brain.refresh_voice_grammar()
        self.db.events.log("app", "chain_saved", c.name)
        return {"chain": c.to_dict()}

    def _delete_chain(self, m: dict) -> dict:
        deleted = self.db.chains.delete(str(m["name"]))
        self.brain.refresh_voice_grammar()
        return {"deleted": deleted}

    def _set_calibration(self, m: dict) -> dict:
        """Update limits/pulses live (the calibration workflow).  Refused while a job runs or in ESTOP."""
        if self.brain._job or self.brain._queue:
            raise BrainError("BUSY", "stop the running job before changing calibration")
        ch = int(_num(m["channel"], "channel"))
        changes = {k: m[k] for k in ("name", "model", "min_angle", "max_angle", "home_angle", "min_pulse",
                                     "max_pulse", "max_step_deg") if k in m}
        cal = self.db.calibration.update(ch, **changes)
        self.brain.reload_calibration()
        self.db.events.log("app", "calibration", f"ch{ch} updated", cal.to_dict())
        return {"servo": cal.to_dict()}

    def _sim_disconnect(self, m: dict) -> dict:
        if not self.sim_transport:
            raise BrainError("FORBIDDEN", "only available with --sim")
        seconds = float(m.get("seconds", 3))
        self.sim_transport.set_offline(seconds)
        return {"offline_for_s": seconds}
