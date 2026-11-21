"""Write the RP2040-Zero firmware's calibration into the host database.

The firmware (RP2040-Zero + PCA9685) drives each servo with ticks = 150 + angle*450/180 at a 27 MHz oscillator
and sends "nominal microseconds" = ticks*20000/4096 over the wire.  That is angle 0 -> ~732 us, 180 -> ~2930 us,
so the host uses min_pulse=732 / max_pulse=2930: host angle == firmware angle (within 1 tick, ~0.2 deg).

The firmware REJECTS a whole move frame if any channel is outside its limits, so the host limits below sit 1 deg
inside the firmware's.  Wrist pitch (ch3) is locked at 70 deg by the firmware; the host keeps it there too.

    python tools/apply_rp2040_profile.py                    # real database (data/laptop_brain.sqlite3)
    python tools/apply_rp2040_profile.py --db other.sqlite3

Safe to re-run.  Presets other than home/open_grip/close_grip are placeholders built around home: re-pose them
on the real arm (cli: `angles ...`, then UPSERT_PRESET) before relying on them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laptop_brain.db import Database  # noqa: E402

PULSE = dict(min_pulse=732, max_pulse=2930)

# Firmware limits (deg):  base 0-140, shoulder 0-145, elbow 0-145, wrist pitch locked 70, roll 0-90, grip 0-50
SERVOS = [
    dict(channel=0, name="base",        model="MG996R", min_angle=1,  max_angle=139, home_angle=70,  max_step_deg=1.2),
    dict(channel=1, name="shoulder",    model="MG996R", min_angle=1,  max_angle=144, home_angle=140, max_step_deg=0.9),
    dict(channel=2, name="elbow",       model="MG996R", min_angle=1,  max_angle=144, home_angle=130, max_step_deg=1.0),
    dict(channel=3, name="wrist_pitch", model="MG90S",  min_angle=69, max_angle=71,  home_angle=70,  max_step_deg=2.0),
    dict(channel=4, name="wrist_roll",  model="MG90S",  min_angle=1,  max_angle=89,  home_angle=30,  max_step_deg=2.5),
    dict(channel=5, name="gripper",     model="MG90S",  min_angle=1,  max_angle=49,  home_angle=30,  max_step_deg=3.0),
]

HOME = [70, 140, 130, 70, 30, 30]          # == firmware DEFAULT_US: no jump when the host connects
GRIP_OPEN, GRIP_CLOSED = 48, 2             # firmware: 0 = closed, 50 = open


def with_grip(angles, g):
    return angles[:5] + [g]


PRESETS = [
    dict(name="home", angles=HOME, description="Rest pose = firmware boot pose (RP2040 profile).", default_transition_ms=1500),
    dict(name="ready", angles=HOME, description="PLACEHOLDER (= home): pose the arm and save your working pose.", default_transition_ms=1200),
    dict(name="open_grip", angles=with_grip(HOME, GRIP_OPEN), description="Home pose, gripper open.", default_transition_ms=600),
    dict(name="close_grip", angles=with_grip(HOME, GRIP_CLOSED), description="Home pose, gripper closed.", default_transition_ms=600),
    dict(name="pen_grip", angles=with_grip(HOME, GRIP_CLOSED), description="PLACEHOLDER: re-pose on the arm.", default_transition_ms=1200),
    dict(name="steady_hold", angles=with_grip(HOME, GRIP_CLOSED), description="PLACEHOLDER: re-pose on the arm.", default_transition_ms=1500),
]

DEMO_WAVE = dict(name="demo_wave", repeat=2, description="Sweep the base around home (RP2040 profile).", steps=[
    dict(preset="home", transition_ms=1500, hold_ms=200),
    dict(angles=[50] + HOME[1:], transition_ms=1200, hold_ms=100),
    dict(angles=[90] + HOME[1:], transition_ms=1800, hold_ms=100),
    dict(preset="home", transition_ms=1200, hold_ms=200),
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/laptop_brain.sqlite3")
    db = Database(ap.parse_args().db)
    try:
        for s in SERVOS:
            ch = s["channel"]
            fields = {k: v for k, v in s.items() if k != "channel"}
            # widen first so the update is valid whatever the old limits were, then set the real values
            db.calibration.update(ch, min_angle=0, max_angle=180, home_angle=90)
            cal = db.calibration.update(ch, **fields, **PULSE)
            print(f"ch{ch} {cal.name:<11} {cal.min_angle:>5.0f}..{cal.max_angle:<5.0f} home {cal.home_angle:>5.0f}")
        for p in PRESETS:
            db.presets.upsert(p["name"], p["angles"], p["description"], p["default_transition_ms"])
        db.routines.upsert(DEMO_WAVE["name"], DEMO_WAVE["steps"], DEMO_WAVE["repeat"], DEMO_WAVE["description"])
        db.events.log("system", "calibration", "applied RP2040-Zero profile")
        print(f"RP2040 profile written to {db.path}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
