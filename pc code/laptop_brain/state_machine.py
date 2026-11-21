"""Central command arbiter / state machine.

Everything that can move the arm goes through ``Brain``; it runs on one asyncio loop, so there are no
locks and no races between app, voice and vision.  Rules:

* ``ESTOP`` is *never queued*.  ``Brain.estop()`` is a plain synchronous call that (1) writes the E frame
  to the ESP, (2) freezes the motion loop, (3) cancels the running job and clears the queue, (4) latches
  state ESTOP.  Nothing but ``CLEAR_ESTOP`` leaves that state.
* Jobs (preset / routine / chain / scan) run one at a time from a FIFO queue.  The submitter picks a
  policy: ``queue`` (wait your turn, bounded), ``reject`` (fail with BUSY - used for voice so a stale word
  never moves the arm later) or ``preempt`` (brake the current job, drop the queue, run immediately).
* Live jogging (SET_ANGLES / NUDGE_JOINT) is accepted only while no job is running.
* Losing the ESP link freezes motion ("safe hold"), aborts jobs, and blocks motion commands until it is back.

States: IDLE, MOVING (preset or manual jog), VISION_SCAN, EXECUTING_CHAIN (chains and routines), ESTOP.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

from . import config as cfg
from .config import NUM_SERVOS, Settings
from .db import Database
from .esp_link import EspLink
from .kinematics import ArmKinematics, square_path
from .motion import Limits, MotionController
from .vision import VisionError, VisionResult, VisionService
from .voice import VoiceEvent, VoiceListener

log = logging.getLogger("brain")


class State(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    VISION_SCAN = "VISION_SCAN"
    EXECUTING_CHAIN = "EXECUTING_CHAIN"
    ESTOP = "ESTOP"


class BrainError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class JobAbort(Exception):
    """A job cannot continue safely (no target, unreachable pose, timeout...).  Not a crash."""


@dataclass
class Job:
    id: int
    kind: str                 # preset | routine | chain | scan
    name: str
    source: str
    tokens: list = field(default_factory=list)
    step: str = ""
    target: Optional[str] = None      # object label a scan should look for ("cup", "7", ...)
    detector: Optional[str] = None    # override vision.detector for this job

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "source": self.source, "step": self.step,
                "target": self.target}


@dataclass
class ChainCtx:
    target: Optional[VisionResult] = None
    aligned: bool = False
    wanted: Optional[str] = None          # object label the chain's scan looks for
    detector: Optional[str] = None


class Brain:
    def __init__(self, settings: Settings, db: Database, motion: MotionController, link: EspLink,
                 vision: VisionService, voice: VoiceListener, kin: ArmKinematics):
        self.settings, self.db, self.motion, self.link = settings, db, motion, link
        self.vision, self.voice, self.kin = vision, voice, kin
        self.state, self.detail = State.IDLE, ""
        self._estop = False
        self._job: Optional[Job] = None
        self._job_task: Optional[asyncio.Task] = None
        self._queue: deque = deque()
        self._wake = asyncio.Event()
        self._worker: Optional[asyncio.Task] = None
        self._next_id = 1
        self.last_vision: Optional[dict] = None
        self.last_voice: Optional[dict] = None
        self.last_error: Optional[str] = None
        self._listeners: list = []
        self._log_last: dict = {}
        self._tokens: dict = {
            "home": self._t_home, "ready": self._t_ready, "scan": self._t_scan, "align": self._t_align,
            "grab": self._t_grab, "lift": self._t_lift, "hold": self._t_hold, "release": self._t_release,
            "write": self._t_write,
        }
        link.on_change = self._on_link_change
        voice.on_event = self.on_voice_event

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self.reload_calibration()
        self.refresh_voice_grammar()
        self.motion.set_on_settled(self._refresh_state)
        self.motion.start()
        self.link.start()
        self.voice.start(sim=self.settings.sim)
        self._worker = asyncio.create_task(self._worker_loop(), name="job-worker")
        self.db.events.log("system", "start", "brain started",
                           {"sim": self.settings.sim, "vision": self.vision.backend_name, "voice": self.voice.backend_name})

    async def shutdown(self) -> None:
        self.link.send_estop()                      # leave the arm explicitly frozen when we exit
        self._cancel_current(brake=False)
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        await self.motion.stop()
        self.voice.stop()
        self.link.stop()
        self.db.events.log("system", "stop", "brain stopped")

    def reload_calibration(self) -> None:
        cals = self.db.calibration.all()
        self.motion.set_limits([Limits.from_cal(c) for c in cals])
        self.link.set_calibration(cals)
        self.kin.set_limits([(c.min_angle, c.max_angle) for c in cals[:4]])

    def refresh_voice_grammar(self) -> None:
        words = set(self.settings.voice.allowlist) | self.db.chains.voice_words()
        self.voice.set_allow(words)

    # ------------------------------------------------------------------ #
    # Observers / telemetry                                               #
    # ------------------------------------------------------------------ #
    def add_listener(self, fn: Callable[[str, dict], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, name: str, data: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(name, data)
            except Exception:
                log.exception("listener failed")

    def snapshot(self) -> dict:
        return {
            "type": "telemetry",
            "ts": round(time.time(), 3),
            "state": self.state.value,
            "detail": self.detail,
            "estop": self._estop,
            "current_angles": [round(a, 1) for a in self.motion.angles],
            "target_angles": [round(a, 1) for a in self.motion.targets],
            "esp_connected": self.link.connected,
            "last_vision": self.last_vision,
            "last_voice": self.last_voice,
            "job": self._job.to_dict() if self._job else None,
            "queue_len": len(self._queue),
            "sim": self.settings.sim,
            "last_error": self.last_error,
        }

    def _log_throttled(self, key: str, source: str, kind: str, message: str, payload: Optional[dict] = None,
                       interval: float = 1.0) -> None:
        now = time.monotonic()
        if now - self._log_last.get(key, -1e9) >= interval:
            self._log_last[key] = now
            self.db.events.log(source, kind, message, payload)

    # ------------------------------------------------------------------ #
    # State helpers                                                       #
    # ------------------------------------------------------------------ #
    def _set_state(self, state: State, detail: str = "", record: bool = True) -> None:
        if state == self.state and detail == self.detail:
            return
        old = self.state
        self.state, self.detail = state, detail
        if record and old != state:
            self.db.events.log("system", "state", f"{old.value} -> {state.value}", {"detail": detail})
        self._notify("state", {"state": state.value, "detail": detail})

    def _refresh_state(self) -> None:
        if self._estop:
            self._set_state(State.ESTOP, self.detail, record=False)
        elif self._job or self._queue:
            return
        elif not self.motion.is_settled:
            self._set_state(State.MOVING, "manual", record=False)
        else:
            self._set_state(State.IDLE, "", record=False)

    def _require_motion(self) -> None:
        if self._estop:
            raise BrainError("ESTOP_ACTIVE", "ESTOP is latched; send CLEAR_ESTOP first")
        if not self.link.connected:
            raise BrainError("ESP_DISCONNECTED", "ESP link is down; motion commands are blocked")

    # ------------------------------------------------------------------ #
    # Safety commands (synchronous, never queued)                         #
    # ------------------------------------------------------------------ #
    def estop(self, source: str = "app", reason: str = "ESTOP") -> dict:
        first = not self._estop
        self._estop = True
        self.link.send_estop()                      # 1. hardware first
        self.motion.freeze()                        # 2. stop generating motion
        self._queue.clear()                         # 3. drop pending work
        self._cancel_current(brake=False)           # 4. kill the running job + scan
        self.detail = reason
        self._set_state(State.ESTOP, reason, record=False)
        if first:
            self.db.events.log("estop", "estop", reason, {"from": source}, level="WARNING")
            log.warning("ESTOP (%s): %s", source, reason)
            self._notify("estop", {"source": source, "reason": reason})
        return {"estop": True}

    def clear_estop(self, source: str = "app") -> dict:
        if not self._estop:
            return {"cleared": False, "note": "not in ESTOP"}
        if not self.link.connected:
            raise BrainError("ESP_DISCONNECTED", "cannot clear ESTOP while the ESP is unreachable")
        self._estop = False
        self.last_error = None
        self.link.send_clear()
        self.motion.unfreeze()
        self.detail = ""
        self._set_state(State.IDLE, "", record=False)
        self.db.events.log("estop", "clear_estop", "ESTOP cleared", {"from": source})
        self._notify("estop_cleared", {"source": source})
        return {"cleared": True}

    def stop(self, source: str = "app", reason: str = "STOP") -> dict:
        """Soft stop: brake, cancel the job, drop the queue.  Not latched."""
        if self._estop:
            return {"stopped": False, "note": "ESTOP already active"}
        had_work = bool(self._job or self._queue)
        self._queue.clear()
        self._cancel_current(brake=True)
        self.db.events.log(source, "stop", reason, {"had_work": had_work})
        self._refresh_state()
        return {"stopped": True, "had_work": had_work}

    def _cancel_current(self, brake: bool) -> None:
        if self._job_task and not self._job_task.done():
            self._job_task.cancel()
        self.vision.cancel()
        if brake and not self.motion.frozen:
            self.motion.halt()

    def _on_link_change(self, connected: bool) -> None:
        self.db.events.log("esp", "link", "connected" if connected else "disconnected",
                           self.link.status(), level="INFO" if connected else "WARNING")
        if not connected:
            self._queue.clear()
            self._cancel_current(brake=False)
            self.motion.freeze()                    # safe hold
            self.last_error = "ESP link lost: arm holding position, motion blocked"
        else:
            if self._estop:
                self.link.send_estop()              # keep the firmware latched
            else:
                self.link.send_clear()
                self.motion.unfreeze()
                if self.last_error and self.last_error.startswith("ESP link lost"):
                    self.last_error = None
        self._refresh_state()
        self._notify("esp", {"connected": connected})

    # ------------------------------------------------------------------ #
    # Live jogging                                                        #
    # ------------------------------------------------------------------ #
    def _require_no_job(self) -> None:
        if self._job or self._queue:
            name = self._job.name if self._job else "queued work"
            raise BrainError("BUSY", f"'{name}' is running; send STOP first")

    def cmd_set_angles(self, angles: Sequence[Optional[float]], source: str = "app", transition_ms: int = 0) -> dict:
        self._require_motion()
        self._require_no_job()
        target, clamped = self.motion.move_to(angles, transition_ms)
        self._set_state(State.MOVING, "manual", record=False)
        self._log_throttled("set_angles", source, "set_angles", "manual angles", {"target": [round(t, 1) for t in target]})
        return {"target": [round(t, 1) for t in target], "clamped": clamped}

    def cmd_nudge(self, channel: int, delta_deg: float, source: str = "app") -> dict:
        self._require_motion()
        self._require_no_job()
        target, clamped = self.motion.nudge(channel, delta_deg)
        self._set_state(State.MOVING, "manual", record=False)
        self._log_throttled("nudge", source, "nudge", f"ch{channel} {delta_deg:+.1f}", {"target": [round(t, 1) for t in target]})
        return {"target": [round(t, 1) for t in target], "clamped": clamped}

    # ------------------------------------------------------------------ #
    # Job submission                                                      #
    # ------------------------------------------------------------------ #
    def submit(self, kind: str, name: str, source: str, tokens: Optional[list] = None, policy: str = "queue",
               target: Optional[str] = None, detector: Optional[str] = None) -> dict:
        if policy not in ("queue", "reject", "preempt"):
            raise BrainError("BAD_REQUEST", "mode must be queue, reject or preempt")
        if self._estop:
            raise BrainError("ESTOP_ACTIVE", "ESTOP is latched; send CLEAR_ESTOP first")
        if kind != "scan" and not self.link.connected:
            raise BrainError("ESP_DISCONNECTED", "ESP link is down; motion commands are blocked")
        if kind == "preset" and self.db.presets.get(name) is None:
            raise BrainError("NOT_FOUND", f"no preset '{name}'")
        if kind == "routine" and self.db.routines.get(name) is None:
            raise BrainError("NOT_FOUND", f"no routine '{name}'")
        busy = bool(self._job or self._queue)
        if policy == "reject" and busy:
            raise BrainError("BUSY", "another command is running")
        if policy == "queue" and len(self._queue) >= self.settings.tasks.queue_max:
            raise BrainError("QUEUE_FULL", "command queue is full")
        if policy == "preempt" and busy:
            self._queue.clear()
            self._cancel_current(brake=True)
        if detector is not None and detector not in ("auto", "aruco", "color", "contour", "object"):
            raise BrainError("BAD_REQUEST", "detector must be auto, aruco, color, contour or object")
        job = Job(self._next_id, kind, name, source, list(tokens or []),
                  target=(str(target).strip() or None) if target else None, detector=detector)
        self._next_id += 1
        queued_behind = len(self._queue) + (1 if self._job else 0)
        self._queue.append(job)
        self._wake.set()
        self._refresh_state()
        return {"job_id": job.id, "queued_behind": queued_behind}

    def run_preset(self, name: str, source: str = "app", policy: str = "queue") -> dict:
        return self.submit("preset", name, source, policy=policy)

    def run_routine(self, name: str, source: str = "app", policy: str = "queue") -> dict:
        return self.submit("routine", name, source, policy=policy)

    def run_chain(self, name: Optional[str] = None, tokens: Optional[list] = None, source: str = "app",
                  policy: str = "queue", target: Optional[str] = None, detector: Optional[str] = None) -> dict:
        if tokens is None:
            chain = self.db.chains.get(name or "")
            if chain is None:
                raise BrainError("NOT_FOUND", f"no chain '{name}'")
            tokens, name = chain.tokens, chain.name
        else:
            try:
                tokens = self.db.chains.validate_tokens(tokens)
            except ValueError as exc:
                raise BrainError("BAD_REQUEST", str(exc))
            name = name or "->".join(tokens)
        return self.submit("chain", name, source, tokens, policy, target=target, detector=detector)

    def run_scan(self, source: str = "app", policy: str = "queue", target: Optional[str] = None,
                 detector: Optional[str] = None) -> dict:
        return self.submit("scan", "scan", source, policy=policy, target=target, detector=detector)

    # ------------------------------------------------------------------ #
    # Voice                                                               #
    # ------------------------------------------------------------------ #
    def on_voice_event(self, ev: VoiceEvent) -> None:
        self.last_voice = {**ev.to_dict(), "action": ""}
        if ev.accepted or ev.reason in ("low_confidence", "cooldown", "wake_required"):
            self.db.events.log("voice", "heard", ev.word, ev.to_dict(), level="INFO" if ev.accepted else "DEBUG")
        action = ""
        if ev.accepted:
            action = self._dispatch_voice(ev.word)
        self.last_voice["action"] = action
        self._notify("voice", self.last_voice)

    def _dispatch_voice(self, word: str) -> str:
        if word in self.voice.gate.stop_words:
            if self.settings.voice.stop_latches_estop:
                self.estop("voice", "voice stop")
                return "estop"
            self.stop("voice", "voice stop")
            return "stop"
        chain = self.db.chains.resolve_word(word)
        if chain:
            name, tokens = chain.name, chain.tokens
        elif word in cfg.KNOWN_TOKENS:
            name, tokens = word, [word]
        else:
            return "ignored"
        try:
            self.submit("chain", name, "voice", tokens, policy="reject")      # never queue a spoken command
            return f"chain:{name}"
        except BrainError as exc:
            self.db.events.log("voice", "rejected", f"{word}: {exc.message}", {"code": exc.code}, level="WARNING")
            return f"rejected:{exc.code}"

    # ------------------------------------------------------------------ #
    # Worker + job execution                                              #
    # ------------------------------------------------------------------ #
    async def _worker_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue and not self._estop:
                job = self._queue.popleft()
                self._job = job
                self._job_task = asyncio.create_task(self._run_job(job), name=f"job-{job.id}")
                try:
                    await asyncio.wait({self._job_task})
                finally:
                    if not self._job_task.done():
                        self._job_task.cancel()
                    self._job, self._job_task = None, None
                self._refresh_state()
            self._refresh_state()

    async def _run_job(self, job: Job) -> None:
        job_state = {"preset": State.MOVING, "routine": State.EXECUTING_CHAIN,
                     "chain": State.EXECUTING_CHAIN, "scan": State.VISION_SCAN}[job.kind]
        self._set_state(job_state, f"{job.kind}:{job.name}")
        self.db.events.log(job.source, f"job_{job.kind}", f"start {job.name}", job.to_dict())
        ok, message = False, ""
        try:
            if job.kind == "preset":
                await self._goto_preset(job.name)
            elif job.kind == "routine":
                await self._run_routine(job)
            elif job.kind == "chain":
                await self._run_chain(job)
            else:
                res = await self._do_scan(job.target, job.detector)
                message = f"found {res.label}" if res else self._not_found_message(job.target)
            ok = True
        except asyncio.CancelledError:
            message = "cancelled"
            raise
        except JobAbort as exc:
            message = str(exc)
            self.last_error = message
            log.warning("job %s aborted: %s", job.name, message)
        except Exception as exc:
            message = f"internal error: {exc}"
            self.last_error = message
            log.exception("job %s crashed", job.name)
            if not self.motion.frozen:
                self.motion.halt()
        finally:
            self.db.events.log(job.source, "job_done", f"{job.name}: {'ok' if ok else message}",
                               {**job.to_dict(), "ok": ok}, level="INFO" if ok else "WARNING")
            self._notify("job_finished", {**job.to_dict(), "ok": ok, "message": message})

    # -- primitives ---------------------------------------------------------- #
    async def _move(self, angles: Sequence[Optional[float]], transition_ms: float = 0) -> None:
        if self.motion.frozen:
            raise JobAbort("motion is frozen")
        target, _ = self.motion.move_to(angles, transition_ms)
        timeout = max(self.motion.estimate_seconds(target), transition_ms / 1000.0) * 1.6 + 2.0
        if not await self.motion.wait_settled(timeout):
            raise JobAbort("motion timeout (servo did not reach target in time)")

    async def _goto_preset(self, name: str) -> None:
        preset = self.db.presets.get(name)
        if preset is None:
            raise JobAbort(f"preset '{name}' does not exist")
        await self._move(preset.angles, preset.default_transition_ms)

    async def _goto_xyz(self, x: float, y: float, z: float, transition_ms: float) -> None:
        sol = self.kin.solve(x, y, z)
        if not sol.ok:
            raise JobAbort(sol.reason)
        await self._move(sol.angles + [None, None], transition_ms)

    async def _grip(self, closed: bool) -> None:
        a = self.settings.arm
        await self._move([None] * 5 + [a.gripper_closed_deg if closed else a.gripper_open_deg], 500)
        await asyncio.sleep(self.settings.tasks.grip_settle_ms / 1000.0)

    async def _do_scan(self, target: Optional[str] = None, detector: Optional[str] = None) -> Optional[VisionResult]:
        try:
            res = await self.vision.scan(detector=detector, label=target)
        except VisionError as exc:
            raise JobAbort(f"vision: {exc}")
        wanted = target or self.settings.vision.target_label
        self.last_vision = res.to_dict() if res else {"found": False, "wanted": wanted,
                                                      "objects": self.vision.last_seen, "ts": time.time()}
        self.db.events.log("vision", "scan", f"found {res.label}" if res else self._not_found_message(target),
                           self.last_vision)
        self._notify("vision", self.last_vision)
        return res

    def _not_found_message(self, target: Optional[str]) -> str:
        wanted = target or self.settings.vision.target_label
        seen = sorted({o["label"] for o in self.vision.last_seen})
        msg = f"'{wanted}' not found" if wanted else "nothing found"
        return msg + (f" (saw: {', '.join(seen)})" if seen else "")

    def _need_target(self, ctx: ChainCtx, token: str) -> VisionResult:
        if ctx.target is None:
            raise JobAbort(f"'{token}' needs a target: put 'scan' before it in the chain")
        if ctx.target.space != "table":
            raise JobAbort("camera is not calibrated (vision.calib_points); refusing to move to a pixel-space target")
        return ctx.target

    # -- routines / chains -------------------------------------------------------- #
    async def _run_routine(self, job: Job) -> None:
        routine = self.db.routines.get(job.name)
        if routine is None:
            raise JobAbort(f"routine '{job.name}' does not exist")
        total = routine.repeat * len(routine.steps)
        n = 0
        for _ in range(routine.repeat):
            for step in routine.steps:
                n += 1
                job.step = f"{n}/{total}"
                if "preset" in step:
                    preset = self.db.presets.get(step["preset"])
                    if preset is None:
                        raise JobAbort(f"preset '{step['preset']}' no longer exists")
                    angles = preset.angles
                else:
                    angles = step["angles"]
                await self._move(angles, step.get("transition_ms", 800))
                if step.get("hold_ms"):
                    await asyncio.sleep(step["hold_ms"] / 1000.0)

    async def _run_chain(self, job: Job) -> None:
        ctx = ChainCtx(wanted=job.target, detector=job.detector)
        for i, token in enumerate(job.tokens):
            job.step = f"{i + 1}/{len(job.tokens)}:{token}"
            handler = self._tokens.get(token)
            if handler is None:
                raise JobAbort(f"unknown token '{token}'")
            await handler(ctx)
        job.step = "done"

    # -- chain tokens --------------------------------------------------------------- #
    async def _t_home(self, ctx: ChainCtx) -> None:
        ctx.aligned = False
        await self._goto_preset("home")

    async def _t_ready(self, ctx: ChainCtx) -> None:
        ctx.aligned = False
        await self._goto_preset("ready")

    async def _t_scan(self, ctx: ChainCtx) -> None:
        self._set_state(State.VISION_SCAN, "scan")
        try:
            res = await self._do_scan(ctx.wanted, ctx.detector)
        finally:
            if self._job:
                self._set_state(State.EXECUTING_CHAIN, f"{self._job.kind}:{self._job.name}")
        if res is None:
            raise JobAbort(f"scan: {self._not_found_message(ctx.wanted)}")
        if res.confidence < self.settings.vision.min_confidence:
            raise JobAbort(f"scan: target confidence {res.confidence:.2f} is below {self.settings.vision.min_confidence}")
        ctx.target, ctx.aligned = res, False

    async def _t_align(self, ctx: ChainCtx) -> None:
        t = self._need_target(ctx, "align")
        await self._goto_xyz(t.x, t.y, self.settings.arm.hover_z_mm, self.settings.tasks.align_transition_ms)
        ctx.aligned = True

    async def _t_grab(self, ctx: ChainCtx) -> None:
        a = self.settings.arm
        if ctx.target is not None:
            t = self._need_target(ctx, "grab")
            if not ctx.aligned:
                await self._t_align(ctx)
            await self._goto_xyz(t.x, t.y, a.grip_z_mm, 1200)
        else:
            self.db.events.log("system", "grab", "no target in chain: closing gripper in place")
        await self._grip(closed=True)

    async def _t_lift(self, ctx: ChainCtx) -> None:
        if ctx.target is not None:
            t = self._need_target(ctx, "lift")
            await self._goto_xyz(t.x, t.y, self.settings.arm.lift_z_mm, 1200)
        else:
            ready = self.db.presets.get("ready")
            if ready is None:
                raise JobAbort("preset 'ready' is missing")
            await self._move(ready.angles[:4] + [None, None], 1200)      # keep roll + gripper as they are

    async def _t_hold(self, ctx: ChainCtx) -> None:
        await asyncio.sleep(self.settings.tasks.hold_ms / 1000.0)

    async def _t_release(self, ctx: ChainCtx) -> None:
        await self._grip(closed=False)

    async def _t_write(self, ctx: ChainCtx) -> None:
        a = self.settings.arm
        pts = square_path(a.write_center_mm[0], a.write_center_mm[1], a.write_size_mm)
        try:
            hover = self.kin.solve(pts[0][0], pts[0][1], a.pen_z_mm + 20)
            if not hover.ok:
                raise ValueError(hover.reason)
            path = self.kin.plan_path(pts, a.pen_z_mm)
        except ValueError as exc:
            self.db.events.log("system", "write_fallback", f"path not reachable ({exc}); using preset pen_grip", level="WARNING")
            await self._goto_preset("pen_grip")
            return
        await self._move(hover.angles + [None, None], 1500)
        for ang in path:
            await self._move(ang + [None, None], 700)
        await self._move(hover.angles + [None, None], 800)
