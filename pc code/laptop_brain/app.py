"""Wiring: build every component from ``Settings`` and run until interrupted."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .config import Settings
from .db import Database
from .esp_link import EspLink, SerialTransport, SimTransport, UdpTransport
from .kinematics import ArmKinematics, CameraMapping
from .motion import Limits, MotionController
from .server import WsServer
from .state_machine import Brain
from .vision import VisionService, make_backend
from .voice import VoiceListener

log = logging.getLogger("app")


def _report_camera_calibration(settings: Settings) -> None:
    pts = settings.vision.calib_points
    if not pts:
        if not settings.sim:
            log.warning("camera not calibrated (vision.calib_points): vision results stay in pixels and "
                        "align/grab/lift will refuse to move.  Run: python -m laptop_brain.calibrate_camera")
        return
    rep = CameraMapping().fit_report(pts)
    if rep["loo_rms_mm"] is None:
        log.info("camera calibration: %d points (add 2+ more to measure its accuracy)", rep["points"])
        return
    level = logging.WARNING if rep["loo_rms_mm"] > settings.vision.calib_warn_rms_mm else logging.INFO
    log.log(level, "camera calibration: %d points, expected position error ~%.1f mm (worst point %.1f mm)",
            rep["points"], rep["loo_rms_mm"], max(rep["loo_per_point_mm"]))


class App:
    def __init__(self, settings: Settings, db: Optional[Database] = None):
        self.settings = settings
        loop = asyncio.get_running_loop()
        self.db = db or Database(settings.db_path)
        cals = self.db.calibration.all()

        self.sim_transport: Optional[SimTransport] = None
        link_cfg = settings.link
        if settings.sim or link_cfg.kind == "sim":
            self.sim_transport = SimTransport()
            factory = lambda: self.sim_transport            # noqa: E731  (same object across reconnects)
        elif link_cfg.kind == "udp":
            factory = lambda: UdpTransport(link_cfg.udp_host, link_cfg.udp_port)   # noqa: E731
        else:
            factory = lambda: SerialTransport(link_cfg.serial_port, link_cfg.baud)  # noqa: E731

        self.link = EspLink(factory, link_cfg, cals, loop)
        self.motion = MotionController([Limits.from_cal(c) for c in cals], settings.motion, sink=self.link.send_angles)
        self.kin = ArmKinematics(settings.arm, [(c.min_angle, c.max_angle) for c in cals[:4]])
        _report_camera_calibration(settings)
        self.vision = VisionService(make_backend(settings.vision, settings.sim),
                                    CameraMapping(settings.vision.calib_points), settings.vision)
        self.voice = VoiceListener(settings.voice, loop)
        self.brain = Brain(settings, self.db, self.motion, self.link, self.vision, self.voice, self.kin)
        self.server = WsServer(self.brain, settings, self.sim_transport)

    async def start(self) -> None:
        await self.brain.start()
        await self.server.start()

    async def stop(self) -> None:
        await self.server.stop()
        await self.brain.shutdown()
        self.db.close()


async def run_forever(settings: Settings) -> None:
    app = App(settings)
    await app.start()
    try:
        await asyncio.Event().wait()
    finally:
        await app.stop()
