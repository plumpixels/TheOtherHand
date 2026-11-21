"""Guided camera -> desk calibration, and a live position read-out to check it.

Collect calibration points (an ArUco marker from DICT_4X4_50, e.g. id 7, printed flat):

    python -m laptop_brain.calibrate_camera --config config.json --write

For each point you put the marker on the desk, measure where its centre is in desk millimetres
(origin under the base axis, +x forward, +y left) and type "x y".  The tool averages ~15 frames,
records the pixel, and at the end prints how accurate the calibration is.  Use 6-9 points spread
over the whole area the arm can reach (corners + middle).  --write saves them into the config file.

Check it live (prints the desk position of every marker / object it sees):

    python -m laptop_brain.calibrate_camera --config config.json --live
    python -m laptop_brain.calibrate_camera --config config.json --live --detector object

The arm server does not need to be stopped (it only opens the camera during a scan), but don't
trigger a scan while this tool holds the camera.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .config import load_settings
from .kinematics import CameraMapping
from .vision import OpenCVVision, VisionError, label_matches


def _measure_marker(vis: OpenCVVision, cap, marker: str, frames: int) -> tuple:
    us, vs, tries = [], [], 0
    while len(us) < frames and tries < frames * 4:
        tries += 1
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        dets = [d for d in vis._aruco_detect(frame) if label_matches(d[3], marker)]
        if dets:
            us.append(dets[0][0])
            vs.append(dets[0][1])
    if len(us) < max(3, frames // 3):
        raise VisionError(f"marker {marker} seen in only {len(us)}/{tries} frames; check lighting/focus")
    spread = max(statistics.pstdev(us), statistics.pstdev(vs))
    return statistics.median(us), statistics.median(vs), spread, len(us)


def _print_report(points: list) -> None:
    rep = CameraMapping().fit_report(points)
    print(f"\n{rep['points']} points.  Fit residual: rms {rep['rms_mm']:.2f} mm, worst {rep['max_mm']:.2f} mm")
    if rep["loo_rms_mm"] is None:
        print("Add at least 5 points to measure real accuracy (4 points always fit exactly).")
        return
    print(f"Expected accuracy on new positions (leave-one-out): ~{rep['loo_rms_mm']:.1f} mm rms")
    for p, e in zip(points, rep["loo_per_point_mm"]):
        flag = "   <-- re-measure this one?" if e > max(5.0, 2.5 * rep["loo_rms_mm"]) else ""
        print(f"  mm {p['mm']!s:>16}  px {p['px']!s:>20}  error {e:5.1f} mm{flag}")


def collect(args, settings) -> int:
    vis = OpenCVVision(settings.vision)
    cap = vis._open()
    points: list = []
    print(__doc__.split("Check it live")[0])
    print(f"Marker: {args.marker}.  Type 'x y' (mm), 'undo', 'report' or 'done'.")
    try:
        for _ in range(settings.vision.warmup_frames):
            cap.read()
        while True:
            try:
                line = input(f"point {len(points) + 1} > ").strip().lower()
            except EOFError:
                line = "done"
            if line in ("done", "q", "quit"):
                break
            if line == "undo":
                if points:
                    print("removed", points.pop())
                continue
            if line == "report":
                if len(points) >= 4:
                    _print_report(points)
                continue
            try:
                x, y = (float(v) for v in line.replace(",", " ").split())
            except ValueError:
                print("enter two numbers: x y   (desk mm)")
                continue
            for _ in range(3):        # flush buffered frames from before the marker was moved
                cap.read()
            try:
                u, v, spread, n = _measure_marker(vis, cap, args.marker, args.frames)
            except VisionError as exc:
                print("  !", exc)
                continue
            points.append({"px": [round(u, 2), round(v, 2)], "mm": [x, y]})
            note = "  (jittery: is the camera or marker moving?)" if spread > 1.5 else ""
            print(f"  marker at px ({u:.1f}, {v:.1f}) from {n} frames, jitter {spread:.2f} px{note}")
    finally:
        cap.release()
    if len(points) < 4:
        print("Need at least 4 points; nothing saved.")
        return 1
    _print_report(points)
    if args.write:
        path = Path(args.config)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault("vision", {})["calib_points"] = points
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {len(points)} points to {path}.  Restart the arm server to use them.")
    else:
        print('\nNot saved (no --write).  Paste into config.json under "vision":')
        print(json.dumps({"calib_points": points}))
    return 0


def live(args, settings) -> int:
    mapping = CameraMapping(settings.vision.calib_points)
    if not mapping.calibrated:
        print("No vision.calib_points in the config: positions will be shown in pixels only.")
    vis = OpenCVVision(settings.vision)
    cap = vis._open()
    print("Live read-out, Ctrl+C to stop.  Desk mm: +x forward, +y left of the base.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            _, seen = vis._detect(frame, args.detector, None)
            parts = []
            for u, v, conf, label, _bbox in seen:
                mm = mapping.px_to_mm(u, v)
                where = f"x={mm[0]:6.1f} y={mm[1]:6.1f} mm" if mm else f"px=({u:.0f},{v:.0f})"
                parts.append(f"{label} {conf:.2f} {where}")
            sys.stdout.write("\r" + (" | ".join(parts) or "nothing seen").ljust(110)[:160])
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="laptop_brain.calibrate_camera", description="camera -> desk calibration")
    p.add_argument("--config", default="config.json", help="config file to read (and update with --write)")
    p.add_argument("--camera", type=int, help="camera index (default: vision.camera_index)")
    p.add_argument("--marker", default="7", help="ArUco id used as the calibration target (default 7)")
    p.add_argument("--frames", type=int, default=15, help="frames averaged per point")
    p.add_argument("--write", action="store_true", help="save the points into --config")
    p.add_argument("--live", action="store_true", help="print live desk positions instead of calibrating")
    p.add_argument("--detector", default="aruco", help="detector for --live: aruco | object | color | contour | auto")
    args = p.parse_args(argv)
    settings = load_settings(args.config if Path(args.config).exists() else None)
    if args.camera is not None:
        settings.vision.camera_index = args.camera
    try:
        return live(args, settings) if args.live else collect(args, settings)
    except VisionError as exc:
        print("error:", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
