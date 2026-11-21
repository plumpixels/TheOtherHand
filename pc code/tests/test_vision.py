"""Vision: label targeting, object recognition (stubbed YOLO), ArUco precision, calibration report, API."""
import asyncio
import threading
from contextlib import asynccontextmanager

import numpy as np
import pytest

from laptop_brain import config as cfg
from laptop_brain.app import App
from laptop_brain.cli_client import parse
from laptop_brain.kinematics import CameraMapping
from laptop_brain.vision import OpenCVVision, VisionService, label_matches
from tests.test_integration import Cli, client, fast_settings, run

cv2 = pytest.importorskip("cv2")


# ------------------------------------------------------------------ helpers
class FakeCap:
    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def read(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return True, f.copy()

    def release(self):
        pass


class FakeBoxes:
    def __init__(self, dets):
        self.xyxy = np.array([d[0] for d in dets], dtype=float).reshape(-1, 4)
        self.conf = np.array([d[1] for d in dets], dtype=float)
        self.cls = np.array([d[2] for d in dets], dtype=float)


class FakeYolo:
    """Mimics ultralytics: predict(frame) -> [result] with .names and .boxes (xyxy/conf/cls)."""
    names = {0: "cup", 1: "bottle", 2: "scissors"}

    def __init__(self, per_frame):
        self.per_frame, self.calls = per_frame, 0

    def predict(self, frame, conf=0.25, imgsz=640, verbose=False):
        dets = [d for d in self.per_frame(self.calls) if d[1] >= conf]
        self.calls += 1
        return [type("R", (), {"names": self.names, "boxes": FakeBoxes(dets)})()]


def opencv_vision(frames, **vision):
    s = cfg.VisionSettings(warmup_frames=0, min_hits=5, save_debug=False, **vision)
    v = OpenCVVision(s)
    v._open = lambda: FakeCap(frames)
    return v, s


BLANK = np.full((480, 640, 3), 255, np.uint8)


# ------------------------------------------------------------------ pure logic
def test_label_matching():
    assert label_matches("cup", "CUP") and label_matches("aruco:7", "7") and label_matches("aruco:7", "aruco:7")
    assert label_matches("cell phone", "cell_phone") and label_matches("anything", None)
    assert not label_matches("aruco:17", "7") and not label_matches("cup", "bottle")


def test_calibration_report_measures_accuracy():
    H = np.array([[0.0, -0.5, 300.0], [-0.5, 0.0, 160.0], [0.0, 0.0002, 1.0]])   # a mildly tilted camera

    def to_mm(u, v):
        x, y, w = H @ np.array([u, v, 1.0])
        return [x / w, y / w]

    grid = [(u, v) for u in (80, 320, 560) for v in (60, 240, 420)]
    exact = [{"px": [u, v], "mm": to_mm(u, v)} for u, v in grid]
    rep = CameraMapping().fit_report(exact)
    assert rep["rms_mm"] < 0.01 and rep["loo_rms_mm"] < 0.01

    four = CameraMapping().fit_report(exact[:4])
    assert four["loo_rms_mm"] is None                              # 4 points can't measure accuracy

    noisy = [dict(p) for p in exact]
    noisy[4] = {"px": noisy[4]["px"], "mm": [noisy[4]["mm"][0] + 12.0, noisy[4]["mm"][1]]}   # one mis-measured point
    rep = CameraMapping().fit_report(noisy)
    assert rep["loo_rms_mm"] > 3.0
    assert max(range(9), key=lambda i: rep["loo_per_point_mm"][i]) == 4    # the bad point stands out


# ------------------------------------------------------------------ object recognition (stubbed model)
def test_object_detector_picks_named_object_and_reports_all():
    cups = [((100, 100, 140, 160), 0.9, 0), ((400, 300, 440, 360), 0.8, 0)]
    bottle = ((250, 200, 290, 300), 0.7, 1)
    v, _ = opencv_vision([BLANK], detector="object")
    v._yolo_model = FakeYolo(lambda i: cups + [bottle])
    raw = v.scan("object", 2.0, threading.Event(), label="bottle")
    assert raw.found and raw.label == "bottle"
    assert (raw.px, raw.py) == (270.0, 250.0)                      # box centre
    assert sorted(o["label"] for o in raw.objects) == ["bottle", "cup", "cup"]

    v._yolo_model = FakeYolo(lambda i: cups + [bottle])
    raw = v.scan("object", 0.5, threading.Event(), label="scissors")
    assert not raw.found and {o["label"] for o in raw.objects} == {"cup", "bottle"}


def test_object_detector_bottom_anchor_and_lock_on():
    # The best-scoring cup alternates between two cups; the scan must stay on the first one it saw.
    def frames(i):
        a, b = ((100, 100, 140, 160), 0.9 if i % 2 == 0 else 0.6, 0), ((400, 300, 440, 360), 0.6 if i % 2 == 0 else 0.9, 0)
        return [a, b]
    v, _ = opencv_vision([BLANK], detector="object", object_anchor="bottom")
    v._yolo_model = FakeYolo(frames)
    raw = v.scan("object", 2.0, threading.Event(), label="cup")
    assert (raw.px, raw.py) == (120.0, 160.0)                      # bottom-centre of the first cup


def test_object_detector_without_ultralytics_gives_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    v, _ = opencv_vision([BLANK])
    with pytest.raises(Exception, match="pip install ultralytics"):
        v.scan("object", 0.5, threading.Event(), label="cup")


# ------------------------------------------------------------------ ArUco precision on a synthetic frame
def test_aruco_subpixel_centre_and_mm_mapping():
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco not available")
    frame = BLANK.copy()
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (x0, y0) in ((7, (200, 150)), (3, (420, 300))):
        img = cv2.aruco.generateImageMarker(d, mid, 80)
        frame[y0:y0 + 80, x0:x0 + 80] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    v, s = opencv_vision([frame], detector="aruco")
    raw = v.scan("aruco", 2.0, threading.Event(), label="7")
    assert raw.label == "aruco:7"
    assert abs(raw.px - 239.5) < 0.6 and abs(raw.py - 189.5) < 0.6      # true centre of the 80 px marker

    # 1 px = 0.5 mm, origin at pixel (320, 400): every object gets desk coordinates
    pts = [{"px": [u, v_], "mm": [(400 - v_) * 0.5, (320 - u) * 0.5]} for u, v_ in ((0, 0), (640, 0), (640, 480), (0, 480))]
    svc = VisionService(v, CameraMapping(pts), s)
    res = asyncio.run(svc.scan(label="3"))
    assert res.space == "table" and res.label == "aruco:3"
    assert abs(res.x - (400 - 339.5) * 0.5) < 0.5 and abs(res.y - (320 - 459.5) * 0.5) < 0.5
    assert {o["label"] for o in res.objects} == {"aruco:7", "aruco:3"} and all("x" in o for o in res.objects)


# ------------------------------------------------------------------ API, end to end over the simulator
@asynccontextmanager
async def running_with_objects(objects):
    s = fast_settings()
    s.vision.sim_objects = objects
    app = App(s)
    await app.start()
    try:
        yield app, f"ws://127.0.0.1:{app.server.port}"
    finally:
        await app.stop()


def test_scan_and_chain_target_a_named_object():
    objects = [{"label": "cup", "mm": [180, 40]}, {"label": "bottle", "mm": [200, -50]}]

    async def go():
        async with running_with_objects(objects) as (app, url), client(url) as c:
            await c.ok("TRIGGER_VISION_SCAN", target="bottle")
            st = await c.until(lambda s: s["last_vision"] and s["state"] == "IDLE", what="scan")
            lv = st["last_vision"]
            assert lv["found"] and lv["label"] == "bottle" and lv["wanted"] == "bottle"
            assert abs(lv["x"] - 200) < 5 and abs(lv["y"] + 50) < 5
            assert {o["label"] for o in lv["objects"]} == {"cup", "bottle"}

            await c.ok("RUN_CHAIN", name="fetch", target="cup")
            await c.until(lambda s: c.finished("fetch"), timeout=15, what="fetch cup")
            assert c.finished("fetch")[0]["ok"]
            st = await c.ok("GET_STATE")
            assert st["last_vision"]["label"] == "cup" and abs(st["last_vision"]["y"] - 40) < 5

            await c.ok("RUN_CHAIN", tokens=["scan", "grab"], target="spoon")
            await c.until(lambda s: c.finished("scan->grab"), timeout=10, what="spoon chain")
            fin = c.finished("scan->grab")[0]
            assert not fin["ok"] and "'spoon' not found" in fin["message"] and "bottle" in fin["message"]

            assert await c.err("TRIGGER_VISION_SCAN", detector="laser") == "BAD_REQUEST"
    run(go())


def test_cli_parses_object_targets():
    assert parse("scan cup") == {"type": "TRIGGER_VISION_SCAN", "target": "cup"}
    assert parse("scan") == {"type": "TRIGGER_VISION_SCAN"}
    assert parse("chain fetch @bottle") == {"type": "RUN_CHAIN", "name": "fetch", "target": "bottle"}
    assert parse("chain scan grab lift @7") == {"type": "RUN_CHAIN", "tokens": ["scan", "grab", "lift"], "target": "7"}
