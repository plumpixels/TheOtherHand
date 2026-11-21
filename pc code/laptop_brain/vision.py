"""On-demand webcam vision.

The camera is opened only inside ``scan()`` and released immediately afterwards, so an idle
system costs no CPU/GPU and the webcam LED is off.  Detectors:

* ``aruco``   -- ArUco marker centre, sub-pixel corners (most precise; stick a printed marker on the object)
* ``object``  -- object recognition with a YOLO model (optional ``ultralytics`` package): "cup", "bottle",
                 "scissors", ... or the classes of your own trained model
* ``color``   -- largest blob inside an HSV range
* ``contour`` -- largest isolated blob on a plain desk (Otsu threshold)
* ``auto``    -- tries aruco, then color, then contour (no model needed).  When a scan names a target
                 label, ``auto`` tries aruco (label "7" / "aruco:7") then object.

A scan collects ``min_hits`` consistent detections of the chosen target over several frames, takes the
median pixel (rejects jitter and single-frame outliers) and maps it to desk millimetres with the
homography from ``vision.calib_points``.  Every object seen in the final frame is reported too, each
with its own desk coordinates.  Until you calibrate, results stay in pixel space and motion tokens
refuse to use them.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .config import VisionSettings
from .kinematics import CameraMapping

log = logging.getLogger("vision")

# A candidate detection inside one frame: (px, py, confidence, label, bbox or None)
LOCK_RADIUS_PX = 40.0          # later frames must stay this close to the first hit (same physical object)


class VisionError(RuntimeError):
    pass


def label_matches(label: str, wanted: Optional[str]) -> bool:
    """'cup' matches 'cup'; '7' and 'aruco:7' match 'aruco:7'; 'cell_phone' matches 'cell phone'.
    Case-insensitive.  None matches anything."""
    if not wanted:
        return True
    label, wanted = label.lower().strip(), wanted.lower().replace("_", " ").strip()
    return label == wanted or label.split(":", 1)[-1] == wanted


@dataclass
class RawDetection:
    px: float
    py: float
    confidence: float
    label: str
    table_xy: Optional[tuple] = None        # sim backend can supply desk coordinates directly
    frame_size: tuple = (0, 0)
    found: bool = True                      # False: the wanted target was not seen (objects may still be listed)
    objects: list = field(default_factory=list)   # everything seen: {"label","confidence","px","py","bbox"[,"mm"]}
    bbox: Optional[tuple] = None


@dataclass
class VisionResult:
    x: float                 # desk mm if space == "table", else pixels
    y: float
    confidence: float
    label: str
    space: str               # "table" | "pixel"
    px: float
    py: float
    ts: float
    frame_size: tuple = (0, 0)
    detector: str = ""
    objects: list = field(default_factory=list)
    wanted: Optional[str] = None
    bbox: Optional[tuple] = None

    def to_dict(self) -> dict:
        return {"found": True, "x": round(self.x, 1), "y": round(self.y, 1), "confidence": round(self.confidence, 3),
                "label": self.label, "space": self.space, "px": round(self.px, 1), "py": round(self.py, 1),
                "bbox": [round(b, 1) for b in self.bbox] if self.bbox else None,
                "frame_size": list(self.frame_size), "detector": self.detector, "wanted": self.wanted,
                "objects": self.objects, "ts": self.ts}


class VisionBackend(ABC):
    name = "backend"

    @abstractmethod
    def scan(self, detector: str, timeout: float, cancel: threading.Event,
             label: Optional[str] = None) -> Optional[RawDetection]: ...


class SimVision(VisionBackend):
    """Pretends to see the objects in ``sim_objects`` (default: one "sim_object" at ``sim_target_mm``)."""
    name = "sim"

    def __init__(self, settings: VisionSettings):
        self.s = settings

    def scan(self, detector, timeout, cancel, label=None):
        if cancel.wait(0.4):
            return None
        specs = self.s.sim_objects or [{"label": "sim_object", "mm": list(self.s.sim_target_mm)}]
        objects = []
        for spec in specs:
            x, y = spec["mm"][0] + random.uniform(-3, 3), spec["mm"][1] + random.uniform(-3, 3)
            objects.append({"label": spec["label"], "confidence": round(random.uniform(0.82, 0.97), 3),
                            "px": 320 + y, "py": 240 - x, "bbox": None, "mm": (x, y)})
        matches = [o for o in objects if label_matches(o["label"], label)]
        if not matches:
            return RawDetection(0, 0, 0, "", frame_size=(640, 480), found=False, objects=objects)
        best = max(matches, key=lambda o: o["confidence"])
        return RawDetection(best["px"], best["py"], best["confidence"], best["label"], table_xy=best["mm"],
                            frame_size=(640, 480), objects=objects)


class OpenCVVision(VisionBackend):
    name = "opencv"

    def __init__(self, settings: VisionSettings):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise VisionError("opencv-python is not installed (pip install opencv-python)") from exc
        import cv2
        self.cv2 = cv2
        self.s = settings
        self._aruco = None
        self._yolo_model = None

    # -- camera --------------------------------------------------------------- #
    def _open(self):
        cv2 = self.cv2
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.s.camera_index, api)
        if not cap.isOpened():
            cap.release()
            raise VisionError(f"cannot open camera index {self.s.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.height)
        return cap

    # -- detectors: each returns a list of (px, py, confidence, label, bbox) ---- #
    def _aruco_detect(self, frame) -> list:
        cv2 = self.cv2
        if not hasattr(cv2, "aruco"):
            return []
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.s.aruco_dict))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            if self._aruco is None:
                params = cv2.aruco.DetectorParameters()
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX     # sub-pixel corners
                self._aruco = cv2.aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = self._aruco.detectMarkers(gray)
        else:                                               # OpenCV < 4.7
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            return []
        ids = ids.reshape(-1)                                  # shape differs between OpenCV 4.x and 5.x
        out = []
        for i, c in enumerate(corners):
            pts = c.reshape(-1, 2).astype("float32")
            area = float(cv2.contourArea(pts))
            bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
            out.append((float(pts[:, 0].mean()), float(pts[:, 1].mean()), 1.0, f"aruco:{int(ids[i])}", bbox, area))
        out.sort(key=lambda d: -d[5])                         # biggest (closest) marker first
        return [d[:5] for d in out]

    def _largest_blob(self, mask, label: str) -> list:
        cv2 = self.cv2
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        best, best_area = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            if area >= self.s.min_area_px and area > best_area and area < 0.5 * w * h and not (label == "contour" and touches_border):
                best, best_area = c, area
        if best is None:
            return []
        m = cv2.moments(best)
        if m["m00"] == 0:
            return []
        x, y, bw, bh = cv2.boundingRect(best)
        return [(m["m10"] / m["m00"], m["m01"] / m["m00"], min(1.0, best_area / (4.0 * self.s.min_area_px)), label,
                 (float(x), float(y), float(x + bw), float(y + bh)))]

    def _color_detect(self, frame) -> list:
        cv2, np = self.cv2, __import__("numpy")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.s.color_hsv_low, dtype="uint8"), np.array(self.s.color_hsv_high, dtype="uint8"))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), "uint8"))
        return self._largest_blob(mask, "color")

    def _contour_detect(self, frame) -> list:
        cv2 = self.cv2
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return self._largest_blob(mask, "contour")

    def _yolo(self):
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise VisionError("object detection needs the 'ultralytics' package (pip install ultralytics)") from exc
            log.info("loading object model %s", self.s.object_model)
            self._yolo_model = YOLO(self.s.object_model)
        return self._yolo_model

    def _object_detect(self, frame) -> list:
        r = self._yolo().predict(frame, conf=self.s.object_min_confidence, imgsz=self.s.object_imgsz, verbose=False)[0]
        names, out = r.names, []
        for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            u = (x1 + x2) / 2.0
            v = y2 if self.s.object_anchor == "bottom" else (y1 + y2) / 2.0
            out.append((u, v, float(conf), str(names[int(cls)]), (x1, y1, x2, y2)))
        out.sort(key=lambda d: -d[2])
        return out

    def _detect(self, frame, detector: str, label: Optional[str] = None) -> tuple:
        """Return (matching candidates, every candidate seen) for the first detector that sees anything."""
        if detector == "auto":
            order = ("aruco", "object") if label else ("aruco", "color", "contour")
        else:
            order = (detector,)
        seen_all: list = []
        for name in order:
            fn = getattr(self, f"_{name}_detect", None)
            if fn is None:
                raise VisionError(f"unknown detector {name!r}")
            found = fn(frame)
            seen_all += found
            matches = [d for d in found if label_matches(d[3], label)]
            if matches:
                return matches, seen_all
        return [], seen_all

    def _save_debug(self, frame, chosen, seen):
        if not self.s.save_debug or frame is None:
            return
        try:
            cv2 = self.cv2
            img = frame.copy()
            for d in seen:
                if d[4]:
                    x1, y1, x2, y2 = (int(v) for v in d[4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 1)
                    cv2.putText(img, f"{d[3]} {d[2]:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            if chosen:
                cv2.drawMarker(img, (int(chosen[0]), int(chosen[1])), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                cv2.putText(img, f"TARGET {chosen[3]} {chosen[2]:.2f}", (int(chosen[0]) + 12, int(chosen[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            os.makedirs(self.s.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.s.debug_dir, "last_scan.jpg"), img)
        except Exception:
            log.debug("could not save debug frame", exc_info=True)

    # -- scan ------------------------------------------------------------------- #
    def scan(self, detector, timeout, cancel, label=None):
        cap = self._open()
        try:
            for _ in range(self.s.warmup_frames):
                cap.read()
            deadline, hits, frames, last, last_seen = time.monotonic() + timeout, [], 0, None, []
            while time.monotonic() < deadline and not cancel.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                frames, last = frames + 1, frame
                matches, seen = self._detect(frame, detector, label)
                last_seen = seen or last_seen
                if not matches:
                    continue
                if hits:   # stay on the same physical object as the first hit (two cups must not be averaged)
                    u0, v0 = hits[0][0], hits[0][1]
                    near = [d for d in matches if math.hypot(d[0] - u0, d[1] - v0) <= LOCK_RADIUS_PX]
                    if not near:
                        continue
                    det = min(near, key=lambda d: math.hypot(d[0] - u0, d[1] - v0))
                else:
                    det = matches[0]
                hits.append(det)
                if len(hits) >= self.s.min_hits:
                    break
            objects = [{"label": d[3], "confidence": round(d[2], 3), "px": d[0], "py": d[1], "bbox": d[4]}
                       for d in last_seen]
            h, w = last.shape[:2] if last is not None else (0, 0)
            if cancel.is_set():
                return None
            if len(hits) < self.s.min_hits:
                self._save_debug(last, None, last_seen)
                return RawDetection(0, 0, 0, "", frame_size=(w, h), found=False, objects=objects)
            xs, ys = sorted(d[0] for d in hits), sorted(d[1] for d in hits)
            u, v = xs[len(xs) // 2], ys[len(ys) // 2]                  # median of the hits
            conf = (sum(d[2] for d in hits) / len(hits)) * (len(hits) / max(frames, 1))
            label_out, bbox = hits[-1][3], hits[-1][4]
            self._save_debug(last, (u, v, conf, label_out), last_seen)
            return RawDetection(u, v, min(1.0, conf), label_out, frame_size=(w, h), objects=objects, bbox=bbox)
        finally:
            cap.release()                                              # camera is only held during a scan


class VisionService:
    def __init__(self, backend: VisionBackend, mapping: CameraMapping, settings: VisionSettings):
        self.backend, self.mapping, self.s = backend, mapping, settings
        self._cancel = threading.Event()
        self._busy = False
        self.last_seen: list = []          # objects from the most recent scan, found or not

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def cancel(self) -> None:
        self._cancel.set()

    def _map_object(self, o: dict) -> dict:
        mm = o.get("mm") or self.mapping.px_to_mm(o["px"], o["py"])
        out = {"label": o["label"], "confidence": o["confidence"], "px": round(o["px"], 1), "py": round(o["py"], 1),
               "space": "table" if mm else "pixel"}
        if mm:
            out["x"], out["y"] = round(mm[0], 1), round(mm[1], 1)
        if o.get("bbox"):
            out["bbox"] = [round(b, 1) for b in o["bbox"]]
        return out

    async def scan(self, detector: Optional[str] = None, timeout: Optional[float] = None,
                   label: Optional[str] = None) -> Optional[VisionResult]:
        if self._busy:
            raise VisionError("a scan is already running")
        self._busy = True
        self._cancel.clear()
        detector = detector or self.s.detector
        timeout = timeout or self.s.scan_timeout_s
        label = label or self.s.target_label
        try:
            raw = await asyncio.to_thread(self.backend.scan, detector, timeout, self._cancel, label)
        except asyncio.CancelledError:
            self._cancel.set()
            raise
        finally:
            self._busy = False
        self.last_seen = [self._map_object(o) for o in raw.objects] if raw else []
        if raw is None or not raw.found:
            return None
        table = raw.table_xy or self.mapping.px_to_mm(raw.px, raw.py)
        space = "table" if table else "pixel"
        x, y = table if table else (raw.px, raw.py)
        return VisionResult(x, y, raw.confidence, raw.label, space, raw.px, raw.py, time.time(), raw.frame_size,
                            detector=self.backend.name, objects=self.last_seen, wanted=label, bbox=raw.bbox)


def make_backend(settings: VisionSettings, sim: bool) -> VisionBackend:
    choice = "sim" if sim and settings.backend == "auto" else settings.backend
    if choice == "sim":
        return SimVision(settings)
    if choice in ("auto", "opencv"):
        try:
            return OpenCVVision(settings)
        except VisionError as exc:
            if choice == "opencv":
                raise
            log.warning("vision falling back to simulator: %s", exc)
            return SimVision(settings)
    raise VisionError(f"unknown vision backend {choice!r}")
