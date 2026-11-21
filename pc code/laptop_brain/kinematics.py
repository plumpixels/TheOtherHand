"""Geometry helpers: arm forward/inverse kinematics, camera->desk mapping, path planning.

Conventions (geometric angles, degrees).  Servo angle = offset + sign * geometric angle,
with offsets/signs in ``ArmSettings`` so you can match your own assembly:

* base yaw  theta : 0 = straight ahead (+x), positive turns toward +y (counter-clockwise from above).
* shoulder  a1    : elevation of the upper arm above horizontal (90 = straight up).
* elbow     e     : forearm angle relative to the upper arm (0 = fully straight, negative = bent downward,
                    which is the "elbow up" solution used here).
* wrist     w     : tool angle relative to the forearm (0 = straight).
* tool pitch p = a1 + e + w : absolute tool direction above horizontal (-90 = pointing straight down).

Desk frame: origin on the desk under the base axis, +x forward, +y left, +z up, millimetres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .config import ArmSettings


@dataclass
class IKSolution:
    ok: bool
    angles: list = field(default_factory=list)   # 4 servo angles: base, shoulder, elbow, wrist_pitch
    reason: str = ""
    pitch_deg: float = 0.0


class ArmKinematics:
    def __init__(self, arm: ArmSettings, limits: Optional[Sequence[tuple]] = None):
        self.arm = arm
        self.limits = list(limits) if limits else [(0.0, 180.0)] * 4   # (min, max) per joint 0..3

    def set_limits(self, limits: Sequence[tuple]) -> None:
        self.limits = list(limits)

    # -- servo <-> geometric angle ----------------------------------------- #
    def servo_to_geo(self, servo: Sequence[float]) -> list:
        a = self.arm
        return [(servo[i] - a.joint_offsets_deg[i]) / a.joint_signs[i] for i in range(4)]

    def geo_to_servo(self, geo: Sequence[float]) -> list:
        a = self.arm
        return [a.joint_offsets_deg[i] + a.joint_signs[i] * geo[i] for i in range(4)]

    # -- forward ------------------------------------------------------------ #
    def forward(self, servo: Sequence[float]) -> dict:
        a = self.arm
        th, a1, e, w = [math.radians(v) for v in self.servo_to_geo(servo)]
        a2, a3 = a1 + e, a1 + e + w
        r_e, z_e = a.l1_mm * math.cos(a1), a.base_height_mm + a.l1_mm * math.sin(a1)
        r_w, z_w = r_e + a.l2_mm * math.cos(a2), z_e + a.l2_mm * math.sin(a2)
        r_t, z_t = r_w + a.l3_mm * math.cos(a3), z_w + a.l3_mm * math.sin(a3)
        c, s = math.cos(th), math.sin(th)
        return {"x": r_t * c, "y": r_t * s, "z": z_t, "pitch_deg": math.degrees(a3),
                "elbow_z": z_e, "wrist_z": z_w, "reach_mm": r_t}

    # -- inverse ------------------------------------------------------------ #
    def inverse(self, x: float, y: float, z: float, pitch_deg: float = -90.0, elbow_up: bool = True) -> IKSolution:
        a = self.arm
        if z < a.min_z_mm:
            return IKSolution(False, reason=f"z={z:.0f}mm is below the desk limit ({a.min_z_mm}mm)")
        r = math.hypot(x, y)
        theta = math.degrees(math.atan2(y, x)) if r > 1e-6 else 0.0
        p = math.radians(pitch_deg)
        rw = r - a.l3_mm * math.cos(p)
        zw = z - a.base_height_mm - a.l3_mm * math.sin(p)
        d2 = rw * rw + zw * zw
        c = (d2 - a.l1_mm ** 2 - a.l2_mm ** 2) / (2 * a.l1_mm * a.l2_mm)
        if c > 1.0 + 1e-9 or c < -1.0 - 1e-9:
            return IKSolution(False, reason=f"target ({x:.0f},{y:.0f},{z:.0f}) unreachable at pitch {pitch_deg:.0f} deg")
        c = max(-1.0, min(1.0, c))
        e = -math.acos(c) if elbow_up else math.acos(c)
        a1 = math.atan2(zw, rw) - math.atan2(a.l2_mm * math.sin(e), a.l1_mm + a.l2_mm * math.cos(e))
        w = p - (a1 + e)
        servo = self.geo_to_servo([theta, math.degrees(a1), math.degrees(e), math.degrees(w)])
        for i, (lo, hi) in enumerate(self.limits[:4]):
            if not (lo - 1e-6 <= servo[i] <= hi + 1e-6):
                return IKSolution(False, reason=f"joint {i} would need {servo[i]:.1f} deg (limit {lo:.0f}..{hi:.0f})")
        elbow_z = a.base_height_mm + a.l1_mm * math.sin(a1)
        wrist_z = a.base_height_mm + a.l1_mm * math.sin(a1) + a.l2_mm * math.sin(a1 + e)
        if min(elbow_z, wrist_z) < a.min_z_mm:
            return IKSolution(False, reason="solution would put the elbow/wrist below the desk")
        return IKSolution(True, servo, "", pitch_deg)

    def solve(self, x: float, y: float, z: float, pitch_candidates: Optional[Sequence[float]] = None) -> IKSolution:
        """Try the preferred tool pitch first, then progressively more relaxed ones."""
        last = IKSolution(False, reason="no pitch candidates")
        for p in (pitch_candidates or self.arm.pitch_candidates_deg):
            sol = self.inverse(x, y, z, p)
            if sol.ok:
                return sol
            last = sol
        return IKSolution(False, reason=f"no reachable pose: {last.reason}")

    def plan_path(self, points_xy: Sequence[tuple], z: float) -> list:
        """IK for a polyline on a horizontal plane.  Raises ValueError naming the first bad point."""
        out = []
        for i, (x, y) in enumerate(points_xy):
            sol = self.solve(x, y, z)
            if not sol.ok:
                raise ValueError(f"waypoint {i} ({x:.0f},{y:.0f},{z:.0f}): {sol.reason}")
            out.append(sol.angles)
        return out


def square_path(cx: float, cy: float, size: float) -> list:
    h = size / 2.0
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]


# --------------------------------------------------------------------------- #
# Camera pixel -> desk plane                                                   #
# --------------------------------------------------------------------------- #
def fit_homography(px: Sequence[tuple], mm: Sequence[tuple]) -> np.ndarray:
    """Direct linear transform from >= 4 pixel/desk point pairs (3x3 matrix)."""
    if len(px) != len(mm) or len(px) < 4:
        raise ValueError("need at least 4 matching point pairs")
    rows = []
    for (u, v), (x, y) in zip(px, mm):
        rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


class CameraMapping:
    """Maps webcam pixels to desk millimetres using the calibration points in the config."""

    def __init__(self, points: Optional[Sequence[dict]] = None):
        self.H: Optional[np.ndarray] = None
        if points:
            self.set_points(points)

    def set_points(self, points: Sequence[dict]) -> None:
        self.H = fit_homography([tuple(p["px"]) for p in points], [tuple(p["mm"]) for p in points])

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def fit_report(self, points: Sequence[dict]) -> dict:
        """How well the calibration points agree, in desk mm.

        ``rms_mm``/``max_mm``: residual of each point against the fit using all points.  With exactly 4 points
        the fit is exact (residual 0) and says nothing about accuracy, so add more.  ``loo_rms_mm`` (5+ points):
        leave-one-out error, i.e. each point predicted by a fit that did not see it -- an honest estimate of
        the accuracy you will get on a new object.
        """
        px = [tuple(p["px"]) for p in points]
        mm = [tuple(p["mm"]) for p in points]
        H = fit_homography(px, mm)

        def err(h, u, v, x, y):
            a, b, w = h @ np.array([u, v, 1.0])
            return float(np.hypot(a / w - x, b / w - y))

        res = [err(H, u, v, x, y) for (u, v), (x, y) in zip(px, mm)]
        out = {"points": len(points), "rms_mm": float(np.sqrt(np.mean(np.square(res)))), "max_mm": max(res),
               "per_point_mm": [round(r, 2) for r in res], "loo_rms_mm": None, "loo_per_point_mm": None}
        if len(points) >= 5:
            loo = []
            for i in range(len(points)):
                h = fit_homography(px[:i] + px[i + 1:], mm[:i] + mm[i + 1:])
                loo.append(err(h, *px[i], *mm[i]))
            out["loo_rms_mm"] = float(np.sqrt(np.mean(np.square(loo))))
            out["loo_per_point_mm"] = [round(r, 2) for r in loo]
        return out

    def px_to_mm(self, u: float, v: float) -> Optional[tuple]:
        if self.H is None:
            return None
        x, y, w = self.H @ np.array([u, v, 1.0])
        return (float(x / w), float(y / w))

    def mm_to_px(self, x: float, y: float) -> Optional[tuple]:
        if self.H is None:
            return None
        u, v, w = np.linalg.inv(self.H) @ np.array([x, y, 1.0])
        return (float(u / w), float(v / w))
