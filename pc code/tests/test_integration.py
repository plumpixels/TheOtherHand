"""End-to-end: real server + brain + motion + simulated ESP, driven over a real WebSocket."""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from itertools import count

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from laptop_brain.app import App
from laptop_brain.config import Settings

_ids = count(1)


def fast_settings(**server) -> Settings:
    s = Settings()
    s.sim = True
    s.link.kind = "sim"
    s.db_path = ":memory:"
    s.server.host, s.server.port, s.server.allow_inject = "127.0.0.1", 0, True
    for k, v in server.items():
        setattr(s.server, k, v)
    s.motion.tick_hz, s.motion.speed_scale = 100.0, 6.0         # fast sim so tests take seconds
    s.link.heartbeat_s, s.link.rx_timeout_s = 0.05, 0.4
    s.link.reconnect_min_s, s.link.reconnect_max_s = 0.1, 0.2
    s.tasks.hold_ms, s.tasks.grip_settle_ms = 50, 20
    s.voice.cooldown_s = 0.3
    return s


class Cli:
    def __init__(self, ws):
        self.ws, self.pending, self.events, self.tele = ws, {}, [], {}
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if m["type"] in ("ack", "error"):
                fut = self.pending.pop(m["id"], None)      # errors for unparseable input carry id=None
                if fut:
                    fut.set_result(m)
            elif m["type"] == "telemetry":
                self.tele = m
            elif m["type"] == "event":
                self.events.append(m)
            elif m["type"] == "hello":
                self.tele = m["snapshot"]

    async def call(self, type_, **kw):
        mid = f"t{next(_ids)}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[mid] = fut
        await self.ws.send(json.dumps({"type": type_, "id": mid, **kw}))
        return await asyncio.wait_for(fut, 5)

    async def ok(self, type_, **kw):
        r = await self.call(type_, **kw)
        assert r["type"] == "ack", r
        return r["data"]

    async def err(self, type_, **kw):
        r = await self.call(type_, **kw)
        assert r["type"] == "error", r
        return r["code"]

    async def until(self, pred, timeout=8.0, what="condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            st = await self.ok("GET_STATE")
            if pred(st):
                return st
            await asyncio.sleep(0.02)
        raise AssertionError(f"timeout waiting for {what}; last state {st}")

    def finished(self, name):
        return [e["data"] for e in self.events if e["name"] == "job_finished" and e["data"]["name"] == name]


@asynccontextmanager
async def running(**server):
    app = App(fast_settings(**server))
    await app.start()
    try:
        url = f"ws://127.0.0.1:{app.server.port}"
        yield app, url
    finally:
        await app.stop()


@asynccontextmanager
async def client(url):
    async with connect(url) as ws:
        c = Cli(ws)
        await asyncio.sleep(0.15)
        yield c
        c.task.cancel()


def run(coro):
    return asyncio.run(coro)


def test_preset_jog_clamp_and_busy():
    async def go():
        async with running() as (app, url), client(url) as c:
            assert c.tele["state"] == "IDLE" and c.tele["esp_connected"]
            await c.ok("RUN_PRESET", name="ready")
            await c.until(lambda s: s["state"] == "IDLE" and s["current_angles"][1] == 60.0, what="ready pose")
            assert c.finished("ready")[0]["ok"]
            d = await c.ok("SET_ANGLES", angles=[200, None, None, None, None, None])
            assert d["clamped"] and d["target"][0] == 180
            d = await c.ok("NUDGE_JOINT", channel=0, delta_deg=-1)
            assert d["target"][0] == 179
            await c.until(lambda s: s["state"] == "IDLE", what="jog settled")
            await c.ok("RUN_ROUTINE", name="demo_wave")
            assert await c.err("SET_ANGLES", angles=[90] * 6) == "BUSY"
            assert await c.err("RUN_PRESET", name="home", mode="reject") == "BUSY"
            assert await c.err("RUN_PRESET", name="nope") == "NOT_FOUND"
            await c.ok("STOP")
            await c.until(lambda s: s["state"] == "IDLE" and s["job"] is None, what="stopped")
            # sim ESP really received pulses within calibration
            assert all(600 <= p <= 2400 for p in app.sim_transport.pulses)
    run(go())


def test_estop_preempts_chain_and_latches():
    async def go():
        async with running() as (app, url), client(url) as c:
            await c.ok("RUN_CHAIN", tokens=["scan", "align", "grab", "lift", "release"])
            await c.until(lambda s: s["state"] == "EXECUTING_CHAIN" and s["job"] and s["job"]["step"].startswith("2/"), what="chain at align")
            t0 = time.monotonic()
            await c.ok("ESTOP")
            st = await c.ok("GET_STATE")
            assert st["state"] == "ESTOP" and st["estop"] and st["job"] is None
            assert time.monotonic() - t0 < 0.25
            assert app.sim_transport.frozen and app.sim_transport.estops >= 1       # firmware got E
            frozen_at = st["current_angles"]
            await asyncio.sleep(0.3)
            st2 = await c.ok("GET_STATE")
            assert st2["current_angles"] == frozen_at and app.sim_transport.frozen
            for cmd, kw in [("RUN_PRESET", {"name": "home"}), ("SET_ANGLES", {"angles": [90] * 6}),
                            ("RUN_CHAIN", {"name": "pickup"}), ("NUDGE_JOINT", {"channel": 0, "delta_deg": 1})]:
                assert await c.err(cmd, **kw) == "ESTOP_ACTIVE"
            await c.ok("GET_STATE")                                              # reads still work
            await c.ok("CLEAR_ESTOP")
            assert not app.sim_transport.frozen
            st3 = await c.ok("GET_STATE")
            assert st3["state"] == "IDLE" and st3["current_angles"] == frozen_at   # no jump on clear
            await c.ok("RUN_PRESET", name="home")
            await c.until(lambda s: s["state"] == "IDLE" and s["current_angles"] == [90, 90, 90, 90, 90, 40], what="home after estop")
            assert any(e["source"] == "estop" for e in app.db.events.recent(20))
    run(go())


def test_chain_with_vision_runs_to_completion():
    async def go():
        async with running() as (app, url), client(url) as c:
            await c.ok("RUN_CHAIN", name="fetch")                       # scan grab lift release
            await c.until(lambda s: s["state"] == "IDLE" and s["job"] is None and c.finished("fetch"), timeout=15, what="fetch done")
            fin = c.finished("fetch")[0]
            assert fin["ok"], fin
            st = await c.ok("GET_STATE")
            assert st["last_vision"]["found"] and st["last_vision"]["space"] == "table"
            assert 170 < st["last_vision"]["x"] < 190
            assert st["current_angles"][5] == 25.0                      # released: gripper open
    run(go())


def test_chain_without_target_aborts_cleanly():
    async def go():
        async with running() as (app, url), client(url) as c:
            await c.ok("RUN_CHAIN", tokens=["align"])
            await c.until(lambda s: c.finished("align"), what="abort")
            fin = c.finished("align")[0]
            assert not fin["ok"] and "needs a target" in fin["message"]
            st = await c.ok("GET_STATE")
            assert st["state"] == "IDLE" and "needs a target" in st["last_error"]
    run(go())


def test_voice_gating_and_rejection_when_busy():
    async def go():
        async with running() as (app, url), client(url) as c:
            d = await c.ok("INJECT_VOICE", word="ready")                 # no push-to-talk window yet
            assert not d["accepted"] and d["reason"] == "not_listening"
            await c.ok("TRIGGER_VOICE_LISTEN", seconds=3)
            d = await c.ok("INJECT_VOICE", word="ready", confidence=0.4)
            assert d["reason"] == "low_confidence"
            d = await c.ok("INJECT_VOICE", word="ready", confidence=0.95)
            assert d["accepted"] and d["action"] == "chain:ready"
            await c.until(lambda s: s["state"] == "IDLE" and c.finished("ready"), what="voice ready")
            # busy: start a slow routine, then a spoken command must be dropped, not queued
            await c.ok("RUN_ROUTINE", name="pen_demo")
            await c.ok("TRIGGER_VOICE_LISTEN")
            d = await c.ok("INJECT_VOICE", word="home")
            assert d["accepted"] and d["action"] == "rejected:BUSY"
            st = await c.ok("GET_STATE")
            assert st["queue_len"] == 0
            # spoken STOP always works and does not latch ESTOP
            d = await c.ok("INJECT_VOICE", word="stop")
            assert d["action"] == "stop"
            st = await c.until(lambda s: s["job"] is None, what="voice stop")
            assert not st["estop"]
            # chain alias works by voice
            await asyncio.sleep(0.35)
            await c.ok("TRIGGER_VOICE_LISTEN")
            d = await c.ok("INJECT_VOICE", word="tidy")
            assert d["action"] == "chain:reset"
    run(go())


def test_queue_and_preempt():
    async def go():
        async with running() as (app, url), client(url) as c:
            r1 = await c.ok("RUN_PRESET", name="ready")
            r2 = await c.ok("RUN_PRESET", name="pen_grip")
            assert r2["queued_behind"] >= 1
            await c.until(lambda s: s["state"] == "IDLE" and s["job"] is None and s["queue_len"] == 0, what="both done")
            assert [f["name"] for f in c.events and [e["data"] for e in c.events if e["name"] == "job_finished"]] == ["ready", "pen_grip"]
            await c.ok("RUN_ROUTINE", name="pen_demo")
            await asyncio.sleep(0.1)
            await c.ok("RUN_PRESET", name="home", mode="preempt")
            await c.until(lambda s: s["state"] == "IDLE" and s["current_angles"] == [90, 90, 90, 90, 90, 40], what="preempted to home")
            assert any(not f["ok"] and f["message"] == "cancelled" for f in c.finished("pen_demo"))
    run(go())


def test_esp_disconnect_is_a_safe_hold_and_recovers():
    async def go():
        async with running() as (app, url), client(url) as c:
            await c.ok("RUN_PRESET", name="ready")
            await asyncio.sleep(0.15)
            await c.ok("SIM_ESP_DISCONNECT", seconds=1.0)
            st = await c.until(lambda s: not s["esp_connected"], what="disconnect noticed")
            assert st["job"] is None and "ESP link lost" in (st["last_error"] or "")
            held = st["current_angles"]
            assert await c.err("RUN_PRESET", name="home") == "ESP_DISCONNECTED"
            assert await c.err("SET_ANGLES", angles=[90] * 6) == "ESP_DISCONNECTED"
            await asyncio.sleep(0.3)
            assert (await c.ok("GET_STATE"))["current_angles"] == held
            st = await c.until(lambda s: s["esp_connected"], timeout=6, what="reconnect")
            assert st["state"] == "IDLE" and st["current_angles"] == held
            await c.ok("RUN_PRESET", name="home")
            await c.until(lambda s: s["state"] == "IDLE" and s["current_angles"] == [90, 90, 90, 90, 90, 40], what="home after reconnect")
    run(go())


def test_crud_and_calibration_over_ws():
    async def go():
        async with running() as (app, url), client(url) as c:
            d = await c.ok("UPSERT_PRESET", name="wave_a", angles=[100, 70, 100, 100, 90, 40], description="t")
            assert d["preset"]["name"] == "wave_a" and d["warnings"] == []
            d = await c.ok("UPSERT_PRESET", name="low_sh", angles=[90, 5, 90, 90, 90, 40])   # shoulder min is 15
            assert d["warnings"]
            assert await c.err("UPSERT_PRESET", name="bad", angles=[90] * 5) == "BAD_REQUEST"
            names = [p["name"] for p in (await c.ok("LIST_PRESETS"))["presets"]]
            assert "wave_a" in names and "home" in names
            assert await c.err("DELETE_PRESET", name="home") == "BAD_REQUEST"
            assert (await c.ok("DELETE_PRESET", name="wave_a"))["deleted"]
            d = await c.ok("UPSERT_ROUTINE", name="r1", steps=[{"preset": "ready", "hold_ms": 10}, {"preset": "home"}], repeat=2)
            assert d["routine"]["repeat"] == 2
            assert await c.err("UPSERT_ROUTINE", name="r2", steps=[{"preset": "missing"}]) == "BAD_REQUEST"
            d = await c.ok("UPSERT_CHAIN", name="grabit", tokens=["scan", "grab"], aliases=["snag"])
            assert "snag" in app.voice.gate.allow                                   # voice grammar refreshed
            assert await c.err("UPSERT_CHAIN", name="bad", tokens=["scan", "fly"]) == "BAD_REQUEST"
            assert (await c.ok("DELETE_CHAIN", name="grabit"))["deleted"] and "snag" not in app.voice.gate.allow
            # live calibration: tighten base limit, clamping follows immediately
            await c.ok("SET_CALIBRATION", channel=0, max_angle=120)
            d = await c.ok("SET_ANGLES", angles=[170, None, None, None, None, None])
            assert d["clamped"] and d["target"][0] == 120
            assert await c.err("SET_CALIBRATION", channel=0, min_angle=150) == "BAD_REQUEST"
            ev = (await c.ok("GET_EVENTS", limit=5))["events"]
            assert ev and "kind" in ev[0]
            assert await c.err("NOT_A_COMMAND") == "UNKNOWN_COMMAND"
    run(go())


def test_vision_scan_job_and_telemetry_push():
    async def go():
        async with running() as (app, url), client(url) as c:
            await c.ok("TRIGGER_VISION_SCAN")
            await c.until(lambda s: s["last_vision"] and s["last_vision"].get("found"), what="scan result")
            await asyncio.sleep(0.3)
            assert c.tele["last_vision"]["found"]                                # pushed telemetry carries it too
            assert any(e["name"] == "vision" for e in c.events)
    run(go())


def test_token_auth_and_bad_json():
    async def go():
        async with running(token="s3cret") as (app, url):
            with pytest.raises(InvalidStatus):
                async with connect(url):
                    pass
            with pytest.raises(InvalidStatus):
                async with connect(url + "?token=wrong"):
                    pass
            async with client(url + "?token=s3cret") as c:
                assert (await c.ok("PING"))["pong"]
                await c.ws.send("not json")
                await asyncio.sleep(0.1)
                assert (await c.ok("PING"))["pong"]                              # server survived garbage
    run(go())
