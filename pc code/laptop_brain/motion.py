"""Motion smoother / interpolator.

Every tick (default 40 Hz) each joint moves toward its setpoint under three limits:

* hard ``min_angle`` / ``max_angle`` clamp,
* a per-servo speed cap (``max_step_deg`` per tick, scaled by ``speed_scale``),
* an acceleration cap with a braking profile, so MG996R joints never see a step
  change in commanded speed (this is what keeps the current spikes down).

Optionally a move has a ``transition_ms`` duration: the setpoint then follows a
smootherstep curve from the start pose to the target and the limiter above follows
the setpoint, so the move is both time-shaped and speed-safe.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .config import MotionSettings, NUM_SERVOS

log = logging.getLogger("motion")


@dataclass
class Limits:
    min_angle: float
    max_angle: float
    home_angle: float
    max_step_deg: float

    @classmethod
    def from_cal(cls, cal) -> "Limits":
        return cls(cal.min_angle, cal.max_angle, cal.home_angle, cal.max_step_deg)


def smootherstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)


class MotionController:
    def __init__(self, limits: Sequence[Limits], settings: MotionSettings,
                 sink: Optional[Callable[[list], object]] = None,
                 initial: Optional[Sequence[float]] = None,
                 clock: Callable[[], float] = time.monotonic):
        if len(limits) != NUM_SERVOS:
            raise ValueError("need limits for 6 servos")
        self.s = settings
        self.limits = list(limits)
        self.sink = sink
        self._clock = clock
        self.cur = [float(a) for a in (initial if initial is not None else [l.home_angle for l in limits])]
        self.tgt = list(self.cur)
        self.vel = [0.0] * NUM_SERVOS
        self._plan: Optional[tuple] = None          # (start_angles, t0, duration_s)
        self.frozen = False
        self._dirty = True
        self._settled = asyncio.Event()
        self._settled.set()
        self._was_settled = True
        self._on_settled: Optional[Callable[[], None]] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    # ------------------------------------------------------------------ #
    # Configuration                                                       #
    # ------------------------------------------------------------------ #
    def set_limits(self, limits: Sequence[Limits]) -> None:
        self.limits = list(limits)
        self.tgt = [self._clamp(i, a) for i, a in enumerate(self.tgt)]

    def set_on_settled(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_settled = cb

    def _clamp(self, i: int, a: float) -> float:
        l = self.limits[i]
        return min(l.max_angle, max(l.min_angle, a))

    # ------------------------------------------------------------------ #
    # Commands                                                            #
    # ------------------------------------------------------------------ #
    def move_to(self, angles: Sequence[Optional[float]], transition_ms: float = 0) -> tuple:
        """Set new targets (``None`` keeps a joint's current target).

        Returns ``(targets, clamped)`` where ``clamped`` says if any value had to be limited.
        """
        if self.frozen:
            raise RuntimeError("motion is frozen")
        new, clamped = list(self.tgt), False
        for i, a in enumerate(angles):
            if a is None:
                continue
            c = self._clamp(i, float(a))
            clamped = clamped or abs(c - a) > 1e-9
            new[i] = c
        self.tgt = new
        self._plan = (list(self.cur), self._clock(), transition_ms / 1000.0) if transition_ms and transition_ms > 0 else None
        self._settled.clear()
        self._was_settled = False
        return list(self.tgt), clamped

    def nudge(self, channel: int, delta_deg: float) -> tuple:
        """Relative jog from the current *target* (so quick repeated taps accumulate)."""
        if not 0 <= channel < NUM_SERVOS:
            raise ValueError("channel must be 0-5")
        angles: list = [None] * NUM_SERVOS
        angles[channel] = self.tgt[channel] + float(delta_deg)
        return self.move_to(angles)

    def halt(self) -> None:
        """Soft stop: brake with the normal deceleration profile and hold where we end up."""
        new = []
        for i in range(NUM_SERVOS):
            a = max(self.limits[i].max_step_deg * self.s.speed_scale * self.s.accel_fraction, 1e-6)
            v = self.vel[i]
            new.append(self._clamp(i, self.cur[i] + math.copysign(v * v / (2 * a), v)))
        self.tgt, self._plan = new, None
        self._settled.clear()
        self._was_settled = False

    def freeze(self) -> None:
        """Hard stop: no more commanded motion until ``unfreeze``.  Used by ESTOP and link loss."""
        self.frozen = True
        self.vel = [0.0] * NUM_SERVOS
        self.tgt = list(self.cur)
        self._plan = None

    def unfreeze(self) -> None:
        """Resume from exactly where we stopped (target = current, so nothing jumps)."""
        self.tgt = list(self.cur)
        self.vel = [0.0] * NUM_SERVOS
        self._plan = None
        self.frozen = False
        self._dirty = True
        self._settled.set()
        self._was_settled = True

    def resync(self) -> None:
        """Force the next tick to (re)send the current pose (e.g. after the ESP reconnects)."""
        self._dirty = True

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #
    @property
    def angles(self) -> list:
        return list(self.cur)

    @property
    def targets(self) -> list:
        return list(self.tgt)

    @property
    def is_settled(self) -> bool:
        return self._was_settled or self.frozen

    def estimate_seconds(self, target: Sequence[float]) -> float:
        worst = 0.0
        for i in range(NUM_SERVOS):
            vmax = self.limits[i].max_step_deg * self.s.speed_scale
            a = max(vmax * self.s.accel_fraction, 1e-6)
            ramp_ticks = vmax / a
            ticks = abs(target[i] - self.cur[i]) / vmax + ramp_ticks
            worst = max(worst, ticks)
        return worst / self.s.tick_hz

    async def wait_settled(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._settled.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------ #
    # Control loop                                                        #
    # ------------------------------------------------------------------ #
    def _setpoint(self) -> list:
        if self._plan is None:
            return self.tgt
        start, t0, dur = self._plan
        t = (self._clock() - t0) / dur if dur > 0 else 1.0
        if t >= 1.0:
            self._plan = None
            return self.tgt
        s = smootherstep(t)
        return [start[i] + (self.tgt[i] - start[i]) * s for i in range(NUM_SERVOS)]

    def tick(self) -> None:
        if self.frozen:
            return
        sp = self._setpoint()
        moved = False
        for i in range(NUM_SERVOS):
            lim = self.limits[i]
            vmax = lim.max_step_deg * self.s.speed_scale
            a = max(vmax * self.s.accel_fraction, 1e-6)
            err = sp[i] - self.cur[i]
            if abs(err) < 1e-9 and abs(self.vel[i]) < 1e-9:
                continue
            v_des = math.copysign(min(vmax, math.sqrt(2.0 * a * abs(err))), err)
            self.vel[i] += max(-a, min(a, v_des - self.vel[i]))
            step = self.vel[i]
            if (err >= 0 and step >= err) or (err < 0 and step <= err):
                new, self.vel[i] = sp[i], 0.0          # arrive exactly, no overshoot
            else:
                new = self.cur[i] + step
            new = self._clamp(i, new)
            if new != self.cur[i]:
                moved = True
            self.cur[i] = new
        if moved or self._dirty:
            self._dirty = False
            if self.sink:
                try:
                    self.sink(list(self.cur))
                except Exception:                      # a broken link must never kill the loop
                    log.exception("motion sink failed")
        eps = self.s.settle_eps_deg
        settled = (self._plan is None
                   and all(abs(self.tgt[i] - self.cur[i]) <= eps and abs(self.vel[i]) < 1e-6 for i in range(NUM_SERVOS)))
        if settled and not self._was_settled:
            for i in range(NUM_SERVOS):                # snap the last sub-epsilon sliver
                self.cur[i] = self.tgt[i]
            self._was_settled = True
            self._settled.set()
            if self._on_settled:
                self._on_settled()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        period = 1.0 / self.s.tick_hz
        next_t = loop.time()
        while not self._stop:
            next_t += period
            try:
                self.tick()
            except Exception:
                log.exception("tick failed")
            delay = next_t - loop.time()
            if delay < 0:                              # fell behind: don't burst to catch up
                next_t = loop.time()
                delay = 0
            await asyncio.sleep(delay)

    def start(self) -> None:
        self._stop = False
        self._task = asyncio.get_running_loop().create_task(self.run(), name="motion-loop")

    async def stop(self) -> None:
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
