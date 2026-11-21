#!/usr/bin/env python3
"""
ROBOT ARM STUDIO
================
Simulation + serial control + webcam "AR" overlay for the 6-servo arm
(RP2040-Zero + PCA9685 firmware, protocol  M,<seq>,us0..us5 | H | E | C).

Install:   pip install pyserial opencv-python pillow numpy matplotlib
Run:       python robot_arm_studio.py

Everything you are expected to tune is in the USER CONFIG block right below.
If an  arm_settings.json  file exists next to this script (written by the
"Save" button in the Calibration tab) it OVERRIDES the calibration, geometry
and overlay values below.  Delete that file to go back to the values here.
"""

# =============================================================================
#                               USER CONFIG
# =============================================================================

SERIAL_BAUD = 115200          # ignored by USB-CDC, any value works
SEND_RATE_HZ = 25             # how often the GUI sends M frames (firmware watchdog = 500 ms)

CAMERA_INDEX = 0              # 0 = built-in laptop webcam, 1/2... = USB cameras
CAMERA_CAPTURE_SIZE = (1280, 720)
CAMERA_DISPLAY_WIDTH = 900    # width of the video in the window (pixels)

# ---- Arm geometry (millimetres).  Measure PIVOT-CENTRE to PIVOT-CENTRE. ------
GEOMETRY = {
    "base_height":     70.0,   # table surface -> shoulder servo axis (height)
    "shoulder_offset":  0.0,   # horizontal distance base rotation axis -> shoulder axis
    "upper_arm":      103.0,   # shoulder axis -> elbow axis        (your 103.0 mm caliper reading)
    "forearm":        131.9,   # elbow axis    -> wrist-pitch axis  (your 131.9 mm caliper reading)
    "gripper":         90.0,   # wrist-pitch axis -> fingertip
    "gripper_max_gap": 45.0,   # jaw opening when fully open (only for the drawing)
}

# ---- Servo limits, in SERVO degrees (must match the firmware!) ---------------
# These are the same numbers as the ANGLE_US(...) values in the firmware.
SERVO_LIMITS = {
    "base":        (0, 140),
    "shoulder":    (0, 145),
    "elbow":       (0, 145),
    "wrist_pitch": (20, 120),  # MUST equal the range you put in the firmware for ch3.
                               # (70, 70) = wrist still locked -> auto-level switches itself off.
    "wrist_roll":  (0, 90),
    "gripper":     (0, 50),
}
SERVO_HOME = {                 # firmware DEFAULT_US pose (servo degrees)
    "base": 70, "shoulder": 140, "elbow": 130,
    "wrist_pitch": 70, "wrist_roll": 30, "gripper": 30,
}
GRIPPER_CLOSED_SERVO = 0       # servo degree where the gripper is closed
GRIPPER_OPEN_SERVO = 50        # servo degree where the gripper is fully open

# ---- Gripper levelling ---------------------------------------------------------
AUTO_LEVEL_DEFAULT = True      # wrist cancels shoulder+elbow so the gripper stays level
LEVEL_ANGLE_DEG = 0.0          # gripper angle vs the ground: 0 = parallel, + = nose up, - = nose down

# ---- CALIBRATION: servo degree  <->  REAL joint angle -------------------------
#
#   real_angle = real_ref + direction * scale * (servo_deg - servo_ref)
#
#   servo_ref : a servo degree you commanded
#   real_ref  : the REAL joint angle you measured at that servo degree
#   direction : +1 or -1  (flip it if the real arm moves the opposite way to the sim)
#   scale     : real degrees per servo degree (1.0 for direct drive; e.g. 0.5 with a 2:1 gear)
#
# Real-angle conventions (look at the "Simulation" side view, arm facing RIGHT):
#   base        : 0 = arm points straight forward. + = counter-clockwise seen from above.
#   shoulder    : angle of upper arm ABOVE horizontal toward the front.
#                 0 = horizontal forward, 90 = straight up, 180 = horizontal backward.
#   elbow       : forearm angle relative to the upper-arm line. 0 = straight/in line,
#                 negative = forearm bent down/forward (the normal working pose).
#   wrist_pitch : gripper angle relative to forearm. 0 = in line, + = tip up.
#   wrist_roll  : gripper twist about its own axis. 0 = jaws open sideways.
#
# The values below are GUESSES. Calibrate them (see README / Calibration tab).
CALIBRATION = {
    "base":        {"servo_ref": 70, "real_ref": 0,   "direction": +1, "scale": 1.0},
    "shoulder":    {"servo_ref": 90, "real_ref": 90,  "direction": -1, "scale": 1.0},
    "elbow":       {"servo_ref": 90, "real_ref": -90, "direction": +1, "scale": 1.0},
    "wrist_pitch": {"servo_ref": 70, "real_ref": 0,   "direction": +1, "scale": 1.0},
    "wrist_roll":  {"servo_ref": 45, "real_ref": 0,   "direction": +1, "scale": 1.0},
}

# ---- Webcam overlay (side view) default placement -----------------------------
# Pixel position of the SHOULDER AXIS in the video, pixels-per-mm, and whether the
# arm faces LEFT in the video (mirror). Easier to set with "Align overlay" in the app.
OVERLAY = {"ox": 380.0, "oy": 300.0, "scale": 1.3, "mirror": False}

# ---- Behaviour ------------------------------------------------------------------
FIRMWARE_US_STEP_PER_20MS = 12   # copy of MAX_US_STEP_PER_20MS (used for the sim speed)
IK_TARGET = "tip"                # "tip" = gripper fingertip follows mouse, "wrist" = wrist axis
IK_GRID_STEP_DEG = 0.5           # IK search resolution (servo degrees)
PATH_SPEED_MM_S = 30.0           # default speed when playing a drawn path
SETTINGS_FILE = "arm_settings.json"

# =============================================================================
#                          END OF USER CONFIG
# =============================================================================

import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None

JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]  # = PCA9685 ch0..ch5
CAL_JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
NICE = {"base": "Base yaw", "shoulder": "Shoulder", "elbow": "Elbow",
        "wrist_pitch": "Wrist pitch", "wrist_roll": "Wrist roll", "gripper": "Gripper"}
CONVENTION = {
    "base": "0° = arm points straight forward. Positive = counter-clockwise seen from ABOVE.",
    "shoulder": "Angle of the upper arm (shoulder→elbow) above horizontal, toward the front.\n"
                "0° = horizontal forward, 90° = straight up, 180° = horizontal backward.",
    "elbow": "Angle of the forearm relative to the upper-arm line.\n0° = straight (in line). "
             "Negative = bent down/forward (normal pose), positive = bent up/back.",
    "wrist_pitch": "Gripper angle relative to the forearm. 0° = in line, positive = tip up.\n"
                   "(Locked in firmware: only real_ref matters, it sets how the sim draws it.)",
    "wrist_roll": "Twist of the gripper about its own axis. 0° = jaws open sideways.",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(SCRIPT_DIR, SETTINGS_FILE)


# =============================================================================
#  Firmware-exact pulse maths
# =============================================================================
def fw_angle_ticks(a):            # #define ANGLE_TICKS(a)
    return 150 + (int(a) * 450) // 180


def fw_ticks_to_us(t):            # #define TICKS_TO_US(t)
    return (int(t) * 20000 + 4095) // 4096


def fw_angle_us(a):
    return fw_ticks_to_us(fw_angle_ticks(a))


US_MIN = [fw_angle_us(SERVO_LIMITS[j][0]) for j in JOINTS]
US_MAX = [fw_angle_us(SERVO_LIMITS[j][1]) for j in JOINTS]


def servo_deg_to_us(ch, deg):
    """Servo degree -> microseconds that the firmware turns into the exact same tick
    count the calibrated web sketch used (2.5 ticks / degree)."""
    t = int(math.floor(150 + float(deg) * 2.5 + 1e-6))   # truncates like Arduino map()
    us = fw_ticks_to_us(t)
    return max(US_MIN[ch], min(US_MAX[ch], us))


def us_to_servo_deg(us):
    return (us * 4096.0 / 20000.0 - 150.0) / 2.5


def is_locked(j):
    lo, hi = SERVO_LIMITS[j]
    return lo == hi


def clamp_servo(j, s):
    lo, hi = SERVO_LIMITS[j]
    return max(lo, min(hi, s))


# =============================================================================
#  Calibration
# =============================================================================
def servo_to_real(j, s):
    c = CALIBRATION[j]
    return c["real_ref"] + c["direction"] * c["scale"] * (s - c["servo_ref"])


def real_to_servo(j, r):
    c = CALIBRATION[j]
    return c["servo_ref"] + (r - c["real_ref"]) / (c["direction"] * c["scale"])


def level_wrist_servo(sh_servo, el_servo, level_deg):
    """Wrist-pitch servo degree that makes the gripper point at `level_deg` above horizontal,
    clamped to the wrist's limits.  Works with floats and numpy arrays."""
    a2 = servo_to_real("shoulder", sh_servo) + servo_to_real("elbow", el_servo)
    lo, hi = SERVO_LIMITS["wrist_pitch"]
    return np.clip(real_to_servo("wrist_pitch", level_deg - a2), lo, hi)


def gripper_open_frac(s):
    span = GRIPPER_OPEN_SERVO - GRIPPER_CLOSED_SERVO
    return max(0.0, min(1.0, (s - GRIPPER_CLOSED_SERVO) / span)) if span else 0.0


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        return None
    try:
        with open(SETTINGS_PATH) as f:
            d = json.load(f)
        for j, c in d.get("calibration", {}).items():
            if j in CALIBRATION:
                CALIBRATION[j].update({k: float(v) for k, v in c.items()})
        GEOMETRY.update({k: float(v) for k, v in d.get("geometry", {}).items() if k in GEOMETRY})
        OVERLAY.update(d.get("overlay", {}))
        return f"Loaded overrides from {SETTINGS_FILE}"
    except Exception as e:
        return f"Could not read {SETTINGS_FILE}: {e}"


def save_settings():
    with open(SETTINGS_PATH, "w") as f:
        json.dump({"calibration": CALIBRATION, "geometry": GEOMETRY, "overlay": OVERLAY}, f, indent=2)


def python_snippet():
    lines = ["GEOMETRY = {"]
    for k, v in GEOMETRY.items():
        lines.append(f'    "{k}": {v:.1f},')
    lines.append("}")
    lines.append("CALIBRATION = {")
    for j, c in CALIBRATION.items():
        lines.append(f'    "{j}": {{"servo_ref": {c["servo_ref"]:g}, "real_ref": {c["real_ref"]:g}, '
                     f'"direction": {int(c["direction"]):+d}, "scale": {c["scale"]:.4g}}},')
    lines.append("}")
    lines.append(f"OVERLAY = {{\"ox\": {OVERLAY['ox']:.0f}, \"oy\": {OVERLAY['oy']:.0f}, "
                 f"\"scale\": {OVERLAY['scale']:.3f}, \"mirror\": {bool(OVERLAY['mirror'])}}}")
    return "\n".join(lines)


# =============================================================================
#  Kinematics
# =============================================================================
def planar_chain(sh_real, el_real, wp_real):
    """Side-view chain in the arm plane. Returns [(r,z)...] for
    base-floor, shoulder, elbow, wrist, tip, and the absolute gripper angle (rad)."""
    g = GEOMETRY
    r0, z0 = g["shoulder_offset"], g["base_height"]
    a1 = math.radians(sh_real)
    e = (r0 + g["upper_arm"] * math.cos(a1), z0 + g["upper_arm"] * math.sin(a1))
    a2 = a1 + math.radians(el_real)
    w = (e[0] + g["forearm"] * math.cos(a2), e[1] + g["forearm"] * math.sin(a2))
    a3 = a2 + math.radians(wp_real)
    t = (w[0] + g["gripper"] * math.cos(a3), w[1] + g["gripper"] * math.sin(a3))
    return [(0.0, 0.0), (r0, z0), e, w, t], a3


def arm_pose(servo):
    """Full pose from a dict of servo degrees."""
    yaw = math.radians(servo_to_real("base", servo["base"]))
    plane, a3 = planar_chain(servo_to_real("shoulder", servo["shoulder"]),
                             servo_to_real("elbow", servo["elbow"]),
                             servo_to_real("wrist_pitch", servo["wrist_pitch"]))
    cy, sy = math.cos(yaw), math.sin(yaw)
    pts3 = [(r * cy, r * sy, z) for r, z in plane]
    # gripper jaws (3D)
    roll = math.radians(servo_to_real("wrist_roll", servo["wrist_roll"]))
    gap = 4.0 + gripper_open_frac(servo["gripper"]) * GEOMETRY["gripper_max_gap"]
    d = np.array([math.cos(a3) * cy, math.cos(a3) * sy, math.sin(a3)])
    n = np.array([-sy, cy, 0.0])
    n_r = n * math.cos(roll) + np.cross(d, n) * math.sin(roll)
    w = np.array(pts3[3])
    t = np.array(pts3[4])
    mid = w + d * GEOMETRY["gripper"] * 0.45
    jaws = []
    for sgn in (1, -1):
        off = n_r * gap / 2 * sgn
        jaws.append([tuple(mid + off), tuple(t + off)])
    return {"yaw": yaw, "plane": plane, "a3": a3, "pts3": pts3, "jaws": jaws,
            "gap": gap, "roll": roll}


class IKSolver:
    """Brute-force IK over the (shoulder, elbow) servo grid. Always respects the joint
    limits and returns the closest reachable point if the target is unreachable."""

    def __init__(self):
        self._key = None

    def _build(self, wp_servo, level):
        key = (json.dumps(CALIBRATION, sort_keys=True), json.dumps(GEOMETRY, sort_keys=True),
               json.dumps(SERVO_LIMITS, sort_keys=True), wp_servo, IK_TARGET, level)
        if key == self._key:
            return
        self._key = key
        st = IK_GRID_STEP_DEG
        s1 = np.arange(SERVO_LIMITS["shoulder"][0], SERVO_LIMITS["shoulder"][1] + 1e-9, st)
        s2 = np.arange(SERVO_LIMITS["elbow"][0], SERVO_LIMITS["elbow"][1] + 1e-9, st)
        S1, S2 = np.meshgrid(s1, s2, indexing="ij")
        g = GEOMETRY
        a1 = np.radians(servo_to_real("shoulder", S1))
        a2 = a1 + np.radians(servo_to_real("elbow", S2))
        if level is not None:
            wp = level_wrist_servo(S1, S2, level)
        else:
            wp = wp_servo
        a3 = a2 + np.radians(servo_to_real("wrist_pitch", wp))
        R = g["shoulder_offset"] + g["upper_arm"] * np.cos(a1) + g["forearm"] * np.cos(a2)
        Z = g["base_height"] + g["upper_arm"] * np.sin(a1) + g["forearm"] * np.sin(a2)
        if IK_TARGET == "tip":
            R = R + g["gripper"] * np.cos(a3)
            Z = Z + g["gripper"] * np.sin(a3)
        self.S1, self.S2, self.R, self.Z = S1.ravel(), S2.ravel(), R.ravel(), Z.ravel()

    def solve(self, r, z, cur_sh, cur_el, wp_servo, level=None):
        self._build(wp_servo, level)
        d2 = (self.R - r) ** 2 + (self.Z - z) ** 2
        # small preference for staying near the current configuration (no elbow flipping)
        cost = d2 + 0.02 * ((self.S1 - cur_sh) ** 2 + (self.S2 - cur_el) ** 2)
        i = int(np.argmin(cost))
        s = np.array([self.S1[i], self.S2[i]])
        err = math.sqrt(float(d2[i]))
        # refine the grid answer with a few damped Gauss-Newton steps (sub-mm accuracy)
        goal = np.array([r, z])
        lo = np.array([SERVO_LIMITS["shoulder"][0], SERVO_LIMITS["elbow"][0]], float)
        hi = np.array([SERVO_LIMITS["shoulder"][1], SERVO_LIMITS["elbow"][1]], float)
        for _ in range(8):
            p = self._fk(s, wp_servo, level)
            e = goal - p
            if np.hypot(*e) < 0.05:
                break
            J = np.column_stack([(self._fk(s + d, wp_servo, level) - p) / 0.01
                                 for d in (np.array([0.01, 0]), np.array([0, 0.01]))])
            ds = J.T @ np.linalg.solve(J @ J.T + 1.0 * np.eye(2), e)
            s2 = np.clip(s + np.clip(ds, -3, 3), lo, hi)
            e2 = float(np.hypot(*(goal - self._fk(s2, wp_servo, level))))
            if e2 >= err:
                break
            s, err = s2, e2
        return float(s[0]), float(s[1]), err

    @staticmethod
    def _fk(s, wp_servo, level=None):
        if level is not None:
            wp_servo = float(level_wrist_servo(s[0], s[1], level))
        pose, _ = planar_chain(servo_to_real("shoulder", s[0]), servo_to_real("elbow", s[1]),
                               servo_to_real("wrist_pitch", wp_servo))
        return np.array(pose[4] if IK_TARGET == "tip" else pose[3])


def catmull_rom(points, samples_per_seg=24):
    if len(points) < 2:
        return list(points)
    P = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = (np.array(P[k], float) for k in (i - 1, i, i + 1, i + 2))
        for k in range(samples_per_seg):
            t = k / samples_per_seg
            t2, t3 = t * t, t * t * t
            p = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(p[0]), float(p[1])))
    out.append(tuple(points[-1]))
    return out


# =============================================================================
#  Arm model (target + simulated "real" position at firmware slew speed)
# =============================================================================
class ArmModel:
    def __init__(self):
        self.target = {j: float(SERVO_HOME[j]) for j in JOINTS}
        self.est_us = [float(servo_deg_to_us(i, self.target[j])) for i, j in enumerate(JOINTS)]
        self._acc = 0.0

    def set_servo(self, j, s):
        if is_locked(j):
            s = SERVO_LIMITS[j][0]
        self.target[j] = float(clamp_servo(j, s))

    def target_us(self):
        return [servo_deg_to_us(i, self.target[j]) for i, j in enumerate(JOINTS)]

    def est_servo(self):
        return {j: us_to_servo_deg(self.est_us[i]) for i, j in enumerate(JOINTS)}

    def moving(self):
        return any(abs(a - b) > 0.5 for a, b in zip(self.est_us, self.target_us()))

    def tick(self, dt, frozen):
        if frozen:
            self._acc = 0.0
            return
        self._acc += dt
        n = int(self._acc / 0.02)
        if n <= 0:
            return
        self._acc -= n * 0.02
        n = min(n, 50)
        lim = n * FIRMWARE_US_STEP_PER_20MS
        for i, t in enumerate(self.target_us()):
            diff = t - self.est_us[i]
            self.est_us[i] += max(-lim, min(lim, diff))

    def freeze_target_to_est(self):
        for i, j in enumerate(JOINTS):
            self.target[j] = clamp_servo(j, us_to_servo_deg(self.est_us[i]))


# =============================================================================
#  Serial link
# =============================================================================
class SerialLink:
    def __init__(self):
        self.ser = None
        self.rx = queue.Queue()
        self.seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open

    @staticmethod
    def ports():
        if serial is None:
            return []
        return [(p.device, f"{p.device}  {p.description}") for p in serial.tools.list_ports.comports()]

    def connect(self, port):
        self.disconnect()
        self.ser = serial.Serial(port, SERIAL_BAUD, timeout=0.1, write_timeout=0.2)
        self._stop.clear()
        threading.Thread(target=self._reader, daemon=True).start()

    def disconnect(self):
        self._stop.set()
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def _reader(self):
        buf = b""
        while not self._stop.is_set():
            try:
                data = self.ser.read(256)
            except Exception as e:
                self.rx.put(f"__ERR__ {e}")
                break
            if data:
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.rx.put(line.decode(errors="replace").strip())

    def send(self, cmd, args=None):
        if not self.connected:
            return
        self.seq = (self.seq + 1) % 100000
        line = f"{cmd},{self.seq}"
        if args:
            line += "," + ",".join(str(int(a)) for a in args)
        try:
            with self._lock:
                self.ser.write((line + "\n").encode())
        except Exception as e:
            self.rx.put(f"__ERR__ {e}")


# =============================================================================
#  Camera
# =============================================================================
class Camera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.fid = 0
        self.running = False
        self.lock = threading.Lock()

    def start(self, index):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        self.stop()
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAPTURE_SIZE[1])
        self.cap = cap
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and self.cap is not None:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.fid += 1
            else:
                time.sleep(0.02)

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.frame = None

    def get(self):
        with self.lock:
            return self.fid, self.frame


# =============================================================================
#  Side view (used both for the plain simulation and the webcam overlay)
# =============================================================================
COL_EST = "#3b8cff"
COL_TGT = "#ffb020"
COL_PATH = "#ff4fd8"
COL_BAD = "#ff3b3b"


class SideView:
    HANDLE_R = 9

    def __init__(self, app, canvas, camera_mode):
        self.app = app
        self.c = canvas
        self.camera_mode = camera_mode
        if camera_mode:
            self.ox, self.oy, self.scale = OVERLAY["ox"], OVERLAY["oy"], OVERLAY["scale"]
            self.mirror = bool(OVERLAY["mirror"])
        else:
            self.ox, self.oy, self.scale, self.mirror = 260.0, 260.0, 1.15, False
        self.align = False
        self.drag = None
        c = canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<ButtonPress-3>", self.on_right)
        c.bind("<MouseWheel>", lambda e: self.on_wheel(e.x, e.y, 1 if e.delta > 0 else -1))
        c.bind("<Button-4>", lambda e: self.on_wheel(e.x, e.y, 1))
        c.bind("<Button-5>", lambda e: self.on_wheel(e.x, e.y, -1))

    # --- coordinate transforms --------------------------------------------------
    def yaw(self):
        return math.radians(servo_to_real("base", self.app.model.target["base"]))

    def plane_to_view(self, r):
        return r * math.cos(self.yaw()) if self.camera_mode else r

    def view_to_plane(self, x):
        if not self.camera_mode:
            return x
        cy = math.cos(self.yaw())
        if abs(cy) < 0.3:
            return None
        return x / cy

    def w2p(self, x, z):
        sx = -1 if self.mirror else 1
        return (self.ox + sx * x * self.scale,
                self.oy - (z - GEOMETRY["base_height"]) * self.scale)

    def p2w(self, px, py):
        sx = -1 if self.mirror else 1
        return ((px - self.ox) / (sx * self.scale),
                GEOMETRY["base_height"] - (py - self.oy) / self.scale)

    def plane_px(self, r, z):
        return self.w2p(self.plane_to_view(r), z)

    def save_overlay(self):
        if self.camera_mode:
            OVERLAY.update({"ox": self.ox, "oy": self.oy, "scale": self.scale, "mirror": self.mirror})

    # --- drawing ------------------------------------------------------------------
    def _chain(self, pose, color, width, dash=None):
        c = self.c
        pts = [self.plane_px(r, z) for r, z in pose["plane"]]
        widths = [width + 4, width + 2, width, width - 1]
        for k in range(4):
            c.create_line(*pts[k], *pts[k + 1], fill=color, width=max(2, widths[k]),
                          capstyle=tk.ROUND, dash=dash, tags="ov")
        for p in pts[1:4]:
            r = width * 0.6 + 3
            c.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, outline=color,
                          width=2, fill="" if dash else "#10141c", tags="ov")
        # jaws (in-plane projection)
        a3 = pose["a3"]
        sx = -1 if self.mirror else 1
        ca = math.cos(a3) * (math.cos(pose["yaw"]) if self.camera_mode else 1.0)
        dvx, dvy = sx * ca, -math.sin(a3)
        nrm = math.hypot(dvx, dvy) or 1
        dvx, dvy = dvx / nrm, dvy / nrm
        nx, ny = -dvy, dvx
        half = pose["gap"] / 2 * abs(math.cos(pose["roll"])) * self.scale + 2
        tip = pts[4]
        base = (tip[0] - dvx * GEOMETRY["gripper"] * 0.5 * self.scale,
                tip[1] - dvy * GEOMETRY["gripper"] * 0.5 * self.scale)
        for s in (1, -1):
            c.create_line(base[0] + nx * half * s, base[1] + ny * half * s,
                          tip[0] + nx * half * s, tip[1] + ny * half * s,
                          fill=color, width=max(2, width - 2), dash=dash, tags="ov")
        return pts

    def draw(self):
        app, c = self.app, self.c
        c.delete("ov")
        W, H = int(c.winfo_width()), int(c.winfo_height())
        if not self.camera_mode:
            self.ox, self.oy = W * 0.36, H * 0.62
            step = 50 * self.scale
            x0, _ = self.w2p(0, 0)
            _, y0 = self.w2p(0, 0)
            k = -int(x0 / step) - 1
            while x0 + k * step < W:
                x = x0 + k * step
                c.create_line(x, 0, x, H, fill="#1f2633" if k else "#2f3a4d", tags="ov")
                k += 1
            k = -int(y0 / step) - 1
            while y0 + k * step < H:
                y = y0 + k * step
                c.create_line(0, y, W, y, fill="#1f2633", tags="ov")
                k += 1
            c.create_rectangle(0, y0, W, y0 + 6, fill="#3a4458", outline="", tags="ov")
            c.create_text(8, y0 + 14, anchor="w", fill="#7f8aa3", text="table", tags="ov")
            # base block
            bx, by = self.w2p(0, GEOMETRY["base_height"])
            hw = 45 * self.scale
            c.create_rectangle(bx - hw, by, bx + hw, y0, fill="#2a3242", outline="#46516a", tags="ov")
            c.create_text(8, 12, anchor="w", fill="#7f8aa3", tags="ov",
                          text="Drag anywhere = move gripper tip (IK) · drag orange elbow dot = shoulder only · "
                               "wheel = zoom", font=("TkDefaultFont", 9))

        # path
        if app.path:
            samples = catmull_rom(app.path)
            flat = []
            for r, z in samples:
                flat += self.plane_px(r, z)
            if len(flat) >= 4:
                c.create_line(*flat, fill=COL_PATH, width=2, dash=(6, 4), tags="ov", smooth=False)
            for i, (r, z) in enumerate(app.path):
                x, y = self.plane_px(r, z)
                c.create_oval(x - 6, y - 6, x + 6, y + 6, outline=COL_PATH, width=2, tags="ov")
                c.create_text(x + 10, y - 10, text=str(i + 1), fill=COL_PATH, tags="ov")

        est = arm_pose(app.model.est_servo())
        tgt = arm_pose(app.model.target)
        self._chain(est, COL_EST, 9)
        tp = self._chain(tgt, COL_TGT, 4, dash=(5, 4))

        # handles
        R = self.HANDLE_R
        ex, ey = tp[2]
        c.create_oval(ex - R, ey - R, ex + R, ey + R, fill=COL_TGT, outline="white", width=2, tags="ov")
        tx, ty = tp[4] if IK_TARGET == "tip" else tp[3]
        col = COL_BAD if app.ik_err > 3 else COL_TGT
        c.create_oval(tx - R - 3, ty - R - 3, tx + R + 3, ty + R + 3, outline=col, width=3, tags="ov")
        c.create_oval(tx - 4, ty - 4, tx + 4, ty + 4, fill=col, outline="", tags="ov")
        if app.ik_err > 3 and app.ik_goal_px.get(id(self)):
            gx, gy = app.ik_goal_px[id(self)]
            c.create_line(tx, ty, gx, gy, fill=COL_BAD, dash=(2, 3), tags="ov")
            c.create_text(gx + 8, gy, anchor="w", fill=COL_BAD, text="out of reach", tags="ov")
        sx, sy = tp[1]
        if self.camera_mode:
            col = "#00ff9c" if self.align else "#9aa4b8"
            c.create_rectangle(sx - 7, sy - 7, sx + 7, sy + 7, outline=col, width=2, tags="ov")
            c.create_line(sx - 14, sy, sx + 14, sy, fill=col, tags="ov")
            c.create_line(sx, sy - 14, sx, sy + 14, fill=col, tags="ov")
            if self.align:
                c.create_text(W / 2, 18, fill="#00ff9c", font=("TkDefaultFont", 12, "bold"), tags="ov",
                              text="ALIGN MODE: drag = move shoulder cross onto real shoulder axis · "
                                   "wheel = scale until links match")
            if abs(math.cos(self.yaw())) < 0.3:
                c.create_text(W / 2, H - 20, fill=COL_BAD, tags="ov",
                              text="Base is turned ~90° away from the camera plane - side dragging disabled")
        if app.path_edit.get():
            c.create_text(W - 10, H - 12, anchor="e", fill=COL_PATH, tags="ov",
                          text="PATH EDIT: click = add point · drag = move · right-click = delete")

    # --- mouse --------------------------------------------------------------------
    def _nearest(self, x, y, items):
        best, bd = None, 16
        for key, (px, py) in items:
            d = math.hypot(px - x, py - y)
            if d < bd:
                best, bd = key, d
        return best

    def on_press(self, e):
        app = self.app
        self.c.focus_set()
        if self.camera_mode and self.align:
            self.drag = ("anchor", e.x - self.ox, e.y - self.oy)
            return
        items = [(("path", i), self.plane_px(r, z)) for i, (r, z) in enumerate(app.path)]
        tp = [self.plane_px(r, z) for r, z in arm_pose(app.model.target)["plane"]]
        items.append((("elbow",), tp[2]))
        hit = self._nearest(e.x, e.y, items)
        if hit is None and app.path_edit.get():
            r_z = self._mouse_plane(e.x, e.y)
            if r_z is not None:
                app.path.append(list(r_z))
                hit = ("path", len(app.path) - 1)
        if hit is None:
            hit = ("tip",)
        self.drag = hit
        self.on_motion(e)

    def _mouse_plane(self, x, y):
        vx, z = self.p2w(x, y)
        r = self.view_to_plane(vx)
        return None if r is None else (r, z)

    def on_motion(self, e):
        if not self.drag:
            return
        app = self.app
        kind = self.drag[0]
        if kind == "anchor":
            self.ox, self.oy = e.x - self.drag[1], e.y - self.drag[2]
            self.save_overlay()
            return
        rz = self._mouse_plane(e.x, e.y)
        if rz is None:
            return
        if kind == "tip":
            app.stop_path()
            app.ik_goal_px[id(self)] = (e.x, e.y)
            app.move_tip_to(*rz)
        elif kind == "elbow":
            app.stop_path()
            app.point_shoulder_at(*rz)
        elif kind == "path":
            app.path[self.drag[1]] = list(rz)

    def on_release(self, e):
        self.drag = None

    def on_right(self, e):
        items = [(i, self.plane_px(r, z)) for i, (r, z) in enumerate(self.app.path)]
        hit = self._nearest(e.x, e.y, items)
        if hit is not None:
            self.app.path.pop(hit)

    def on_wheel(self, x, y, d):
        if self.camera_mode and not self.align:
            return
        f = 1.06 if d > 0 else 1 / 1.06
        # zoom around the shoulder anchor in camera mode, around the cursor otherwise
        cx, cy = (self.ox, self.oy)
        self.ox = cx + (self.ox - cx) * f
        self.oy = cy + (self.oy - cy) * f
        self.scale *= f
        self.save_overlay()


# =============================================================================
#  Top view (base yaw + reach)
# =============================================================================
class TopView:
    def __init__(self, app, canvas):
        self.app = app
        self.c = canvas
        self.scale = 0.42
        canvas.bind("<ButtonPress-1>", self.on_drag)
        canvas.bind("<B1-Motion>", self.on_drag)

    def center(self):
        return self.c.winfo_width() / 2, self.c.winfo_height() * 0.62

    def w2p(self, x, y):
        cx, cy = self.center()
        return cx - y * self.scale, cy - x * self.scale

    def draw(self):
        c = self.c
        c.delete("all")
        cx, cy = self.center()
        for rr in (100, 200, 300):
            r = rr * self.scale
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#253047")
            c.create_text(cx + 3, cy - r - 7, text=f"{rr}mm", fill="#55607a", anchor="w", font=("TkDefaultFont", 7))
        # yaw limits
        for s in SERVO_LIMITS["base"]:
            a = math.radians(servo_to_real("base", s))
            x, y = self.w2p(330 * math.cos(a), 330 * math.sin(a))
            c.create_line(cx, cy, x, y, fill="#5a3a3a", dash=(3, 3))
        c.create_oval(cx - 18, cy - 18, cx + 18, cy + 18, fill="#2a3242", outline="#46516a")
        for pose, col, w, dash in ((arm_pose(self.app.model.est_servo()), COL_EST, 7, None),
                                   (arm_pose(self.app.model.target), COL_TGT, 3, (4, 3))):
            pts = [self.w2p(p[0], p[1]) for p in pose["pts3"][1:]]
            flat = [v for p in pts for v in p]
            c.create_line(*flat, fill=col, width=w, dash=dash, capstyle=tk.ROUND)
        c.create_text(6, 10, anchor="w", fill="#7f8aa3", text="TOP VIEW · drag = turn base + reach",
                      font=("TkDefaultFont", 8))
        fx, fy = self.w2p(345, 0)
        c.create_text(fx - 40, max(30, fy + 6), text="front ↑", fill="#55607a", font=("TkDefaultFont", 8))

    def on_drag(self, e):
        cx, cy = self.center()
        x = (cy - e.y) / self.scale
        y = (cx - e.x) / self.scale
        if math.hypot(x, y) < 15:
            return
        self.app.stop_path()
        self.app.set_yaw_reach(math.degrees(math.atan2(y, x)), math.hypot(x, y))


# =============================================================================
#  Main application
# =============================================================================
class App:
    def __init__(self, root, settings_msg):
        self.root = root
        root.title("Robot Arm Studio")
        root.geometry("1480x900")
        root.configure(bg="#0e1219")
        self.model = ArmModel()
        self.ik = IKSolver()
        self.ik_err = 0.0
        self.ik_goal_px = {}
        self.link = SerialLink()
        self.cam = Camera()
        self.estop = False
        self.live = tk.BooleanVar(value=False)
        self.path = []
        self.path_edit = tk.BooleanVar(value=False)
        self.path_loop = tk.BooleanVar(value=True)
        self.path_speed = tk.DoubleVar(value=PATH_SPEED_MM_S)
        self.playing = False
        self.play_s = 0.0
        self.ack_count = 0
        self.last_watchdog_log = 0
        self.slider_vars = {}
        self.slider_info = {}
        self.photo = None
        self.cam_fid = -1
        self.last_3d = 0
        self._last_t = time.time()
        self.auto_level = tk.BooleanVar(value=AUTO_LEVEL_DEFAULT)
        self.level_deg = tk.DoubleVar(value=LEVEL_ANGLE_DEG)
        self.slider_widgets = {}
        self.level_lbl = None

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self._build_ui()
        self.log(settings_msg or f"Using values from the top of the script (no {SETTINGS_FILE}).")
        if serial is None:
            self.log("pyserial missing -> pip install pyserial  (simulation still works)")
        if cv2 is None or ImageTk is None:
            self.log("opencv-python / pillow missing -> camera tab disabled")
        root.bind_all("<Escape>", lambda e: self.do_estop())
        root.bind_all("<space>", self._space)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.ui_tick)
        self.root.after(200, self.send_tick)

    # ------------------------------------------------------------------ UI build
    def _build_ui(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pw, padding=6, width=360)
        pw.add(left, weight=0)
        self.nb = ttk.Notebook(pw)
        pw.add(self.nb, weight=1)

        # connection
        f = ttk.LabelFrame(left, text="Connection", padding=6)
        f.pack(fill=tk.X)
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(f, textvariable=self.port_var, width=26, state="readonly")
        self.port_box.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="↻", width=3, command=self.refresh_ports).grid(row=0, column=2)
        self.conn_btn = ttk.Button(f, text="Connect", command=self.toggle_connect)
        self.conn_btn.grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Checkbutton(f, text="LIVE (send moves to arm)", variable=self.live,
                        command=self._live_changed).grid(row=1, column=1, columnspan=2, sticky="w")
        self.conn_lbl = ttk.Label(f, text="Not connected", foreground="#888")
        self.conn_lbl.grid(row=2, column=0, columnspan=3, sticky="w")
        f.columnconfigure(1, weight=1)
        self.refresh_ports()

        # e-stop
        f = ttk.Frame(left, padding=(0, 6))
        f.pack(fill=tk.X)
        self.estop_btn = tk.Button(f, text="E-STOP  (Esc / Space)", bg="#d62828", fg="white",
                                   activebackground="#ff4040", font=("TkDefaultFont", 13, "bold"),
                                   height=2, command=self.do_estop)
        self.estop_btn.pack(fill=tk.X)
        g = ttk.Frame(f)
        g.pack(fill=tk.X, pady=3)
        ttk.Button(g, text="Clear E-stop", command=self.clear_estop).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(g, text="Home pose", command=self.go_home).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # joints
        self.joint_frame = ttk.LabelFrame(left, text="Joints:  real angle · servo° · µs", padding=6)
        self.joint_frame.pack(fill=tk.X)
        self.build_sliders()

        # path
        f = ttk.LabelFrame(left, text="Spline path (draw on Simulation or Camera view)", padding=6)
        f.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(f, text="Edit path (click to add points)", variable=self.path_edit).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(f, text="▶ Play", command=self.play_path).grid(row=1, column=0, sticky="ew")
        ttk.Button(f, text="■ Stop", command=self.stop_path).grid(row=1, column=1, sticky="ew")
        ttk.Button(f, text="Clear", command=lambda: (self.stop_path(), self.path.clear())).grid(
            row=1, column=2, sticky="ew")
        ttk.Checkbutton(f, text="Loop", variable=self.path_loop).grid(row=2, column=0, sticky="w")
        ttk.Label(f, text="mm/s").grid(row=2, column=1, sticky="e")
        tk.Scale(f, from_=5, to=80, orient=tk.HORIZONTAL, variable=self.path_speed, length=110,
                 showvalue=True).grid(row=2, column=2)
        for i in range(3):
            f.columnconfigure(i, weight=1)

        # log
        f = ttk.LabelFrame(left, text="Log", padding=4)
        f.pack(fill=tk.BOTH, expand=True)
        self.logbox = tk.Text(f, height=8, width=44, bg="#0e1219", fg="#c9d1e0", font=("Courier", 9), wrap="word")
        self.logbox.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(left, text="", font=("Courier", 9), justify=tk.LEFT, width=46)
        self.status.pack(fill=tk.X)

        self._build_sim_tab()
        self._build_cam_tab()
        self._build_cal_tab()

    def build_sliders(self):
        for w in self.joint_frame.winfo_children():
            w.destroy()
        self.slider_vars.clear()
        self.slider_info.clear()
        self.slider_widgets.clear()
        for row, j in enumerate(JOINTS):
            ttk.Label(self.joint_frame, text=NICE[j], width=11).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar()
            if j == "gripper":
                sc = tk.Scale(self.joint_frame, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL,
                              variable=var, length=170, showvalue=False, sliderlength=18, width=12,
                              command=lambda v, j=j: self._slider_moved(j, v))
            else:
                lo, hi = SERVO_LIMITS[j]
                sc = tk.Scale(self.joint_frame, from_=round(servo_to_real(j, lo), 1),
                              to=round(servo_to_real(j, hi), 1), resolution=0.1, orient=tk.HORIZONTAL,
                              variable=var, length=170, showvalue=False, sliderlength=18, width=12,
                              command=lambda v, j=j: self._slider_moved(j, v))
            sc.grid(row=row, column=1, sticky="ew", pady=1)
            info = ttk.Label(self.joint_frame, text="", font=("Courier", 8), width=21)
            info.grid(row=row, column=2, sticky="w")
            self.slider_info[j] = info
            self.slider_vars[j] = var
            self.slider_widgets[j] = sc
        n = len(JOINTS)
        ttk.Checkbutton(self.joint_frame, text="Auto-level gripper (wrist compensates)",
                        variable=self.auto_level, command=self._level_changed).grid(
            row=n, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(self.joint_frame, text="Gripper tilt", width=11).grid(row=n + 1, column=0, sticky="w")
        tk.Scale(self.joint_frame, from_=-90, to=90, resolution=1, orient=tk.HORIZONTAL,
                 variable=self.level_deg, length=170, showvalue=False, sliderlength=18, width=12,
                 command=lambda v: self._level_changed()).grid(row=n + 1, column=1, sticky="ew")
        self.level_lbl = ttk.Label(self.joint_frame, text="", font=("Courier", 8), width=21)
        self.level_lbl.grid(row=n + 1, column=2, sticky="w")
        self.joint_frame.columnconfigure(1, weight=1)
        self._update_wrist_slider_state()
        self.sync_sliders()

    def _build_sim_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Simulation  ")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=0)
        tab.rowconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        side = tk.Canvas(tab, bg="#121722", highlightthickness=0, width=620)
        side.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.side_view = SideView(self, side, camera_mode=False)
        top = tk.Canvas(tab, bg="#121722", highlightthickness=0, height=300, width=330)
        top.grid(row=0, column=1, sticky="nsew")
        self.top_view = TopView(self, top)
        self.ax = None
        if Figure is not None:
            fig = Figure(figsize=(4, 3.4), dpi=90, facecolor="#121722")
            ax = fig.add_subplot(projection="3d")
            ax.set_facecolor("#121722")
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.set_pane_color((0.1, 0.12, 0.17, 1))
                a.label.set_color("#8a94aa")
            ax.tick_params(colors="#55607a", labelsize=6)
            ax.set_xlim(-250, 330)
            ax.set_ylim(-290, 290)
            ax.set_zlim(0, 400)
            try:
                ax.set_box_aspect((580, 580, 400))
            except Exception:
                pass
            th = np.linspace(0, 2 * np.pi, 40)
            ax.plot(45 * np.cos(th), 45 * np.sin(th), 0 * th, color="#46516a")
            ax.plot(45 * np.cos(th), 45 * np.sin(th), 0 * th + GEOMETRY["base_height"] * 0.8, color="#46516a")
            self.l_est, = ax.plot([], [], [], "-o", color=COL_EST, lw=5, ms=6)
            self.l_tgt, = ax.plot([], [], [], "--", color=COL_TGT, lw=1.5)
            self.l_jaws = [ax.plot([], [], [], "-", color=COL_EST, lw=3)[0] for _ in range(2)]
            self.l_path, = ax.plot([], [], [], ":", color=COL_PATH, lw=1.5)
            ax.set_xlabel("x fwd")
            ax.set_ylabel("y")
            ax.set_title("3D (drag to rotate)", color="#8a94aa", fontsize=9)
            self.fig_canvas = FigureCanvasTkAgg(fig, master=tab)
            self.fig_canvas.get_tk_widget().grid(row=1, column=1, sticky="nsew")
            self.ax = ax

    def _build_cam_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Camera (AR)  ")
        bar = ttk.Frame(tab, padding=4)
        bar.pack(fill=tk.X)
        self.cam_btn = ttk.Button(bar, text="Start camera", command=self.toggle_camera)
        self.cam_btn.pack(side=tk.LEFT)
        ttk.Label(bar, text=" index").pack(side=tk.LEFT)
        self.cam_index = tk.IntVar(value=CAMERA_INDEX)
        ttk.Spinbox(bar, from_=0, to=9, width=3, textvariable=self.cam_index).pack(side=tk.LEFT)
        self.cam_mirror = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Mirror video", variable=self.cam_mirror).pack(side=tk.LEFT, padx=8)
        self.align_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Align overlay", variable=self.align_var,
                        command=lambda: setattr(self.cam_view, "align", self.align_var.get())).pack(side=tk.LEFT)
        self.ov_mirror = tk.BooleanVar(value=bool(OVERLAY["mirror"]))
        ttk.Checkbutton(bar, text="Arm faces LEFT in video", variable=self.ov_mirror,
                        command=self._ov_mirror_changed).pack(side=tk.LEFT, padx=8)
        self.show_ov = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show overlay", variable=self.show_ov).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save overlay", command=self.save_all).pack(side=tk.RIGHT)
        h = int(CAMERA_DISPLAY_WIDTH * 9 / 16)
        cv = tk.Canvas(tab, bg="#05070b", width=CAMERA_DISPLAY_WIDTH, height=h, highlightthickness=0)
        cv.pack(anchor="nw", padx=4, pady=4)
        self.cam_canvas = cv
        self.cam_img_item = cv.create_image(0, 0, anchor="nw")
        self.cam_msg = cv.create_text(CAMERA_DISPLAY_WIDTH / 2, h / 2, fill="#55607a",
                                      text="Camera off - press 'Start camera'.\n"
                                           "(The overlay still works without video.)", justify=tk.CENTER)
        self.cam_view = SideView(self, cv, camera_mode=True)
        ttk.Label(tab, padding=4, foreground="#666", text=(
            "Put the webcam at the SIDE of the arm (camera looking at the arm's side, base yaw ≈ 0).  "
            "Tick 'Align overlay', drag the green cross onto the real shoulder axis and scroll until the blue "
            "skeleton covers the real links. Untick it, then drag in the video: the gripper tip follows "
            "the mouse, the orange elbow dot moves only the shoulder.")).pack(fill=tk.X)

    def _build_cal_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="  Calibration  ")
        ttk.Label(tab, text="1. Pick a joint.  2. Move it with the RAW servo slider (LIVE must be on).  "
                            "3. Measure the real angle (protractor / phone inclinometer app).  "
                            "4. Record point A, move far away, record point B.  5. Apply → Save.",
                  wraplength=900, foreground="#444").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))
        ttk.Label(tab, text="Joint").grid(row=1, column=0, sticky="w")
        self.cal_joint = tk.StringVar(value="shoulder")
        cb = ttk.Combobox(tab, textvariable=self.cal_joint, values=CAL_JOINTS, state="readonly", width=14)
        cb.grid(row=1, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self._cal_load())
        self.cal_conv = ttk.Label(tab, text="", foreground="#1a5fb4", wraplength=700, justify=tk.LEFT)
        self.cal_conv.grid(row=2, column=0, columnspan=6, sticky="w", pady=6)

        ttk.Label(tab, text="RAW servo degree").grid(row=3, column=0, sticky="w")
        self.cal_raw = tk.DoubleVar()
        self.cal_raw_scale = tk.Scale(tab, from_=0, to=180, resolution=0.5, orient=tk.HORIZONTAL,
                                      length=420, variable=self.cal_raw, command=self._cal_raw_moved)
        self.cal_raw_scale.grid(row=3, column=1, columnspan=4, sticky="w")
        self.cal_now = ttk.Label(tab, text="", font=("Courier", 10))
        self.cal_now.grid(row=4, column=0, columnspan=6, sticky="w")

        lf = ttk.LabelFrame(tab, text="Two-point calibration", padding=8)
        lf.grid(row=5, column=0, columnspan=6, sticky="ew", pady=8)
        ttk.Label(lf, text="Measured REAL angle (°)").grid(row=0, column=0)
        self.cal_meas = tk.StringVar()
        ttk.Entry(lf, textvariable=self.cal_meas, width=8).grid(row=0, column=1)
        ttk.Button(lf, text="Record A", command=lambda: self._cal_record("A")).grid(row=0, column=2, padx=4)
        ttk.Button(lf, text="Record B", command=lambda: self._cal_record("B")).grid(row=0, column=3, padx=4)
        ttk.Button(lf, text="Compute from A & B", command=self._cal_compute).grid(row=0, column=4, padx=4)
        self.cal_pts = {}
        self.cal_pts_lbl = ttk.Label(lf, text="A: -    B: -", font=("Courier", 10))
        self.cal_pts_lbl.grid(row=1, column=0, columnspan=5, sticky="w", pady=4)

        lf = ttk.LabelFrame(tab, text="Calibration values for this joint (edit directly if you prefer)", padding=8)
        lf.grid(row=6, column=0, columnspan=6, sticky="ew")
        self.cal_fields = {}
        for i, k in enumerate(["servo_ref", "real_ref", "direction", "scale"]):
            ttk.Label(lf, text=k).grid(row=0, column=2 * i, padx=(8, 2))
            v = tk.StringVar()
            ttk.Entry(lf, textvariable=v, width=8).grid(row=0, column=2 * i + 1)
            self.cal_fields[k] = v
        ttk.Button(lf, text="Apply", command=self._cal_apply).grid(row=0, column=8, padx=8)
        ttk.Button(lf, text="Flip direction", command=self._cal_flip).grid(row=0, column=9)

        lf = ttk.LabelFrame(tab, text="Geometry (mm, pivot centre to pivot centre)", padding=8)
        lf.grid(row=7, column=0, columnspan=6, sticky="ew", pady=8)
        self.geo_fields = {}
        for i, k in enumerate(GEOMETRY):
            ttk.Label(lf, text=k).grid(row=i // 3, column=(i % 3) * 2, padx=(8, 2), sticky="e")
            v = tk.StringVar(value=f"{GEOMETRY[k]:g}")
            ttk.Entry(lf, textvariable=v, width=8).grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w")
            self.geo_fields[k] = v
        ttk.Button(lf, text="Apply geometry", command=self._geo_apply).grid(row=3, column=0, columnspan=2, pady=4)

        f = ttk.Frame(tab)
        f.grid(row=8, column=0, columnspan=6, sticky="w")
        ttk.Button(f, text=f"Save → {SETTINGS_FILE}", command=self.save_all).pack(side=tk.LEFT)
        ttk.Button(f, text="Print as Python (paste into top of script)", command=self._cal_print).pack(
            side=tk.LEFT, padx=6)
        self._cal_load()

    # ----------------------------------------------------------- calibration tab
    def _cal_load(self):
        j = self.cal_joint.get()
        c = CALIBRATION[j]
        for k, v in self.cal_fields.items():
            v.set(f"{c[k]:g}")
        lo, hi = SERVO_LIMITS[j]
        self.cal_raw_scale.configure(from_=lo, to=hi, state=tk.DISABLED if is_locked(j) else tk.NORMAL)
        self.cal_raw.set(self.model.target[j])
        self.cal_conv.configure(text=f"{NICE[j]}: {CONVENTION[j]}")
        self.cal_pts = {}
        self.cal_pts_lbl.configure(text="A: -    B: -")

    def _cal_raw_moved(self, v):
        j = self.cal_joint.get()
        if abs(float(v) - self.model.target[j]) < 0.01:
            return
        self.stop_path()
        self.model.set_servo(j, float(v))
        self.sync_sliders()

    def _cal_record(self, which):
        try:
            real = float(self.cal_meas.get())
        except ValueError:
            messagebox.showerror("Calibration", "Type the measured real angle first.")
            return
        self.cal_pts[which] = (self.model.target[self.cal_joint.get()], real)
        txt = "    ".join(f"{k}: servo {s:.1f}° → real {r:.1f}°" if (k in self.cal_pts) else f"{k}: -"
                         for k, (s, r) in [(k, self.cal_pts.get(k, (0, 0))) for k in "AB"])
        self.cal_pts_lbl.configure(text=txt)

    def _cal_compute(self):
        if "A" not in self.cal_pts or "B" not in self.cal_pts:
            messagebox.showerror("Calibration", "Record both A and B.")
            return
        (sa, ra), (sb, rb) = self.cal_pts["A"], self.cal_pts["B"]
        if abs(sb - sa) < 5:
            messagebox.showerror("Calibration", "Points A and B are too close. Move at least ~30 servo degrees.")
            return
        k = (rb - ra) / (sb - sa)
        self.cal_fields["servo_ref"].set(f"{sa:g}")
        self.cal_fields["real_ref"].set(f"{ra:g}")
        self.cal_fields["direction"].set("1" if k >= 0 else "-1")
        self.cal_fields["scale"].set(f"{abs(k):.4f}")
        self._cal_apply()

    def _cal_apply(self):
        j = self.cal_joint.get()
        try:
            vals = {k: float(v.get()) for k, v in self.cal_fields.items()}
        except ValueError:
            messagebox.showerror("Calibration", "All fields must be numbers.")
            return
        if vals["scale"] <= 0:
            messagebox.showerror("Calibration", "scale must be > 0 (use direction for the sign).")
            return
        vals["direction"] = 1 if vals["direction"] >= 0 else -1
        CALIBRATION[j].update(vals)
        self.build_sliders()
        self.log(f"Calibration applied for {j}: {CALIBRATION[j]}")

    def _cal_flip(self):
        d = self.cal_fields["direction"].get().strip()
        self.cal_fields["direction"].set("1" if d.startswith("-") else "-1")
        self._cal_apply()

    def _geo_apply(self):
        try:
            for k, v in self.geo_fields.items():
                GEOMETRY[k] = float(v.get())
        except ValueError:
            messagebox.showerror("Geometry", "All fields must be numbers.")
            return
        self.log("Geometry applied.")

    def _cal_print(self):
        self.log("---- paste into the USER CONFIG block ----\n" + python_snippet() + "\n------------------------------------------")
        print(python_snippet())

    def save_all(self):
        self.cam_view.save_overlay()
        save_settings()
        self.log(f"Saved calibration, geometry and overlay → {SETTINGS_PATH}")

    # -------------------------------------------------------------- arm control
    def _slider_moved(self, j, v):
        v = float(v)
        if j == "gripper":
            s = GRIPPER_CLOSED_SERVO + (GRIPPER_OPEN_SERVO - GRIPPER_CLOSED_SERVO) * v / 100.0
            if abs(gripper_open_frac(self.model.target[j]) * 100 - v) < 0.6:
                return
        else:
            if abs(servo_to_real(j, self.model.target[j]) - v) < 0.11:
                return
            s = real_to_servo(j, v)
        self.stop_path()
        self.model.set_servo(j, s)
        if j in ("shoulder", "elbow"):
            self.auto_level_wrist()
        self.ik_err = 0.0

    def sync_sliders(self):
        for j, var in self.slider_vars.items():
            if j == "gripper":
                var.set(round(gripper_open_frac(self.model.target[j]) * 100))
            else:
                var.set(round(servo_to_real(j, self.model.target[j]), 1))
        if hasattr(self, "cal_raw"):
            self.cal_raw.set(self.model.target[self.cal_joint.get()])

    def level_arg(self):
        """Gripper angle to hold, or None when auto-level is off / the wrist is locked."""
        if self.auto_level.get() and not is_locked("wrist_pitch"):
            return float(self.level_deg.get())
        return None

    def auto_level_wrist(self):
        lv = self.level_arg()
        if lv is None:
            return
        m = self.model.target
        self.model.set_servo("wrist_pitch", float(level_wrist_servo(m["shoulder"], m["elbow"], lv)))

    def _update_wrist_slider_state(self):
        sc = self.slider_widgets.get("wrist_pitch")
        if sc is not None:
            auto = self.level_arg() is not None
            sc.configure(state=tk.DISABLED if (auto or is_locked("wrist_pitch")) else tk.NORMAL)

    def _level_changed(self):
        self.auto_level_wrist()
        self._update_wrist_slider_state()
        self.sync_sliders()

    def move_tip_to(self, r, z):
        m = self.model.target
        s1, s2, err = self.ik.solve(r, z, m["shoulder"], m["elbow"], m["wrist_pitch"], self.level_arg())
        self.model.set_servo("shoulder", s1)
        self.model.set_servo("elbow", s2)
        self.auto_level_wrist()
        self.ik_err = err
        self.sync_sliders()

    def point_shoulder_at(self, r, z):
        r0, z0 = GEOMETRY["shoulder_offset"], GEOMETRY["base_height"]
        a = math.degrees(math.atan2(z - z0, r - r0))
        cur = servo_to_real("shoulder", self.model.target["shoulder"])
        a += 360 * round((cur - a) / 360)
        self.model.set_servo("shoulder", real_to_servo("shoulder", a))
        self.auto_level_wrist()
        self.ik_err = 0.0
        self.sync_sliders()

    def current_tip_rz(self):
        p = arm_pose(self.model.target)["plane"]
        return p[4] if IK_TARGET == "tip" else p[3]

    def set_yaw_reach(self, yaw_deg, reach):
        cur = servo_to_real("base", self.model.target["base"])
        yaw_deg += 360 * round((cur - yaw_deg) / 360)
        _, z = self.current_tip_rz()
        self.model.set_servo("base", real_to_servo("base", yaw_deg))
        self.move_tip_to(reach, z)

    def go_home(self):
        self.stop_path()
        for j in JOINTS:
            self.model.set_servo(j, SERVO_HOME[j])
        self.auto_level_wrist()
        self.ik_err = 0.0
        self.sync_sliders()

    # ------------------------------------------------------------------- path
    def play_path(self):
        if len(self.path) < 2:
            self.log("Path needs at least 2 points (tick 'Edit path' and click in a side view).")
            return
        self.path_edit.set(False)
        samples = np.array(catmull_rom(self.path))
        seg = np.hypot(*np.diff(samples, axis=0).T)
        self._ps = samples
        self._pcum = np.concatenate([[0], np.cumsum(seg)])
        self.play_s = 0.0
        self.playing = True

    def stop_path(self):
        self.playing = False

    def _path_step(self, dt):
        if not self.playing:
            return
        total = self._pcum[-1]
        # wait at the start until the (simulated) arm has caught up
        if self.play_s == 0.0 and self.model.moving():
            r, z = self._ps[0]
            self.move_tip_to(r, z)
            return
        self.play_s += self.path_speed.get() * dt
        if self.play_s >= total:
            if self.path_loop.get():
                self.play_s = 0.0
            else:
                self.play_s = total
                self.playing = False
        r = float(np.interp(self.play_s, self._pcum, self._ps[:, 0]))
        z = float(np.interp(self.play_s, self._pcum, self._ps[:, 1]))
        self.move_tip_to(r, z)

    # ---------------------------------------------------------------- serial
    def refresh_ports(self):
        ports = SerialLink.ports()
        self._ports = [p[0] for p in ports]
        self.port_box["values"] = [p[1] for p in ports]
        if ports and not self.port_var.get():
            self.port_box.current(0)

    def toggle_connect(self):
        if self.link.connected:
            self.link.disconnect()
            self.live.set(False)
            self.conn_btn.configure(text="Connect")
            self.conn_lbl.configure(text="Not connected", foreground="#888")
            self.log("Disconnected (firmware watchdog will hold the arm where it is).")
            return
        if serial is None:
            messagebox.showerror("Serial", "pip install pyserial")
            return
        idx = self.port_box.current()
        if idx < 0:
            messagebox.showerror("Serial", "No port selected. Plug the RP2040 in and press ↻.")
            return
        port = self._ports[idx]
        try:
            self.link.connect(port)
        except Exception as e:
            messagebox.showerror("Serial", f"Could not open {port}:\n{e}")
            return
        self.conn_btn.configure(text="Disconnect")
        self.conn_lbl.configure(text=f"Connected: {port}", foreground="#1a7f37")
        self.log(f"Connected to {port}. Tick LIVE to start moving the real arm.")

    def _live_changed(self):
        if self.live.get():
            if not self.link.connected:
                self.live.set(False)
                messagebox.showinfo("LIVE", "Connect to the arm first.")
                return
            self.log("LIVE on: the real arm now follows the orange (target) pose at firmware speed.")
        else:
            self.log("LIVE off: only heartbeats are sent, the real arm holds still.")

    def send_tick(self):
        try:
            if self.link.connected:
                if self.live.get() and not self.estop:
                    self.link.send("M", self.model.target_us())
                else:
                    self.link.send("H")
        finally:
            self.root.after(max(10, int(1000 / SEND_RATE_HZ)), self.send_tick)

    def do_estop(self):
        self.estop = True
        self.stop_path()
        self.link.send("E")
        self.model.freeze_target_to_est()
        self.sync_sliders()
        self.estop_btn.configure(text="E-STOP LATCHED – press Clear", bg="#7a0000")
        self.log("E-STOP sent. Arm frozen. Press 'Clear E-stop' to continue.")

    def clear_estop(self):
        self.model.freeze_target_to_est()
        self.sync_sliders()
        self.link.send("C")
        self.estop = False
        self.estop_btn.configure(text="E-STOP  (Esc / Space)", bg="#d62828")
        self.log("E-stop cleared.")

    def _space(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text)):
            return
        self.do_estop()

    # ---------------------------------------------------------------- camera
    def toggle_camera(self):
        if self.cam.running:
            self.cam.stop()
            self.cam_btn.configure(text="Start camera")
            self.cam_canvas.itemconfigure(self.cam_img_item, image="")
            self.cam_canvas.itemconfigure(self.cam_msg, state=tk.NORMAL)
            return
        if cv2 is None or ImageTk is None:
            messagebox.showerror("Camera", "pip install opencv-python pillow")
            return
        try:
            self.cam.start(int(self.cam_index.get()))
        except Exception as e:
            messagebox.showerror("Camera", str(e))
            return
        self.cam_btn.configure(text="Stop camera")
        self.cam_canvas.itemconfigure(self.cam_msg, state=tk.HIDDEN)

    def _ov_mirror_changed(self):
        self.cam_view.mirror = self.ov_mirror.get()
        self.cam_view.save_overlay()

    def _update_camera(self):
        fid, frame = self.cam.get()
        if frame is None or fid == self.cam_fid:
            return
        self.cam_fid = fid
        h, w = frame.shape[:2]
        dw = CAMERA_DISPLAY_WIDTH
        dh = int(dw * h / w)
        if int(self.cam_canvas.cget("height")) != dh:
            self.cam_canvas.configure(height=dh)
        img = cv2.resize(frame, (dw, dh))
        if self.cam_mirror.get():
            img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.cam_canvas.itemconfigure(self.cam_img_item, image=self.photo)
        self.cam_canvas.tag_lower(self.cam_img_item)

    # ---------------------------------------------------------------- 3D
    def _update_3d(self):
        if self.ax is None:
            return
        est = arm_pose(self.model.est_servo())
        tgt = arm_pose(self.model.target)
        p = np.array(est["pts3"])
        self.l_est.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        q = np.array(tgt["pts3"])
        self.l_tgt.set_data_3d(q[:, 0], q[:, 1], q[:, 2])
        for line, jaw in zip(self.l_jaws, est["jaws"]):
            j = np.array(jaw)
            line.set_data_3d(j[:, 0], j[:, 1], j[:, 2])
        if self.path:
            yaw = tgt["yaw"]
            s = np.array(catmull_rom(self.path))
            self.l_path.set_data_3d(s[:, 0] * math.cos(yaw), s[:, 0] * math.sin(yaw), s[:, 1])
        else:
            self.l_path.set_data_3d([], [], [])
        self.fig_canvas.draw_idle()

    # ---------------------------------------------------------------- loop
    def ui_tick(self):
        now = time.time()
        dt = min(0.2, now - self._last_t)
        self._last_t = now
        try:
            self._path_step(dt)
            self.model.tick(dt, frozen=self.estop)
            self._drain_rx()
            tab = self.nb.index("current")
            if tab == 0:
                self.side_view.draw()
                self.top_view.draw()
                if now - self.last_3d > 0.1:
                    self.last_3d = now
                    self._update_3d()
            elif tab == 1:
                self._update_camera()
                if self.show_ov.get():
                    self.cam_view.draw()
                else:
                    self.cam_canvas.delete("ov")
            elif tab == 2:
                j = self.cal_joint.get()
                s = self.model.target[j]
                self.cal_now.configure(text=f"{NICE[j]}: servo {s:6.1f}°  →  real {servo_to_real(j, s):7.1f}°   "
                                            f"({servo_deg_to_us(JOINTS.index(j), s)} µs)")
            self._update_status()
        except Exception as e:  # keep the GUI alive whatever happens
            self.log(f"UI error: {e!r}")
        self.root.after(33, self.ui_tick)

    def _update_status(self):
        us = self.model.target_us()
        for i, j in enumerate(JOINTS):
            if j in self.slider_info:
                v = (f"{gripper_open_frac(self.model.target[j]) * 100:5.0f}%" if j == "gripper"
                     else f"{servo_to_real(j, self.model.target[j]):6.1f}°")
                self.slider_info[j].configure(text=f"{v} s{self.model.target[j]:5.1f} {us[i]}")
        if self.level_lbl is not None:
            a3 = math.degrees(arm_pose(self.model.target)["a3"])
            self.level_lbl.configure(text=f"{self.level_deg.get():+.0f}° now {a3:+.0f}°")
        r, z = self.current_tip_rz()
        yaw = servo_to_real("base", self.model.target["base"])
        mode = "E-STOP" if self.estop else ("LIVE" if self.live.get() else "SIM ONLY")
        self.status.configure(text=f"[{mode}]  tip r={r:6.1f} z={z:6.1f} mm  yaw={yaw:6.1f}°\n"
                                   f"IK error {self.ik_err:5.1f} mm   acks {self.ack_count}"
                                   f"{'   ▶ path' if self.playing else ''}")

    def _drain_rx(self):
        for _ in range(200):
            try:
                line = self.link.rx.get_nowait()
            except queue.Empty:
                return
            if line.startswith("A,"):
                self.ack_count += 1
            elif line.startswith("__ERR__"):
                self.log("Serial error: " + line[7:])
                self.link.disconnect()
                self.live.set(False)
                self.conn_btn.configure(text="Connect")
                self.conn_lbl.configure(text="Connection lost", foreground="#c00")
            elif line == "W,watchdog":
                if time.time() - self.last_watchdog_log > 5:
                    self.last_watchdog_log = time.time()
                    self.log("Firmware: watchdog (no frames for 500 ms) – holding position.")
            elif line:
                self.log("FW: " + line)

    def log(self, msg):
        self.logbox.insert(tk.END, time.strftime("%H:%M:%S ") + msg + "\n")
        self.logbox.see(tk.END)
        if int(self.logbox.index("end-1c").split(".")[0]) > 500:
            self.logbox.delete("1.0", "100.0")

    def on_close(self):
        self.cam.stop()
        self.link.disconnect()
        self.root.destroy()


def main():
    msg = load_settings()
    root = tk.Tk()
    App(root, msg)
    root.mainloop()


if __name__ == "__main__":
    main()
