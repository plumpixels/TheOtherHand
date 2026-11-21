"""Configuration dataclasses + first-run seed data for the laptop brain.

Everything tunable lives here.  Override any value with a JSON file
(``--config config.json``); nested keys mirror the dataclasses below, e.g.::

    {"arm": {"l1_mm": 135.0}, "vision": {"camera_index": 1}}

Keys starting with ``_`` are ignored (use them for comments).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

NUM_SERVOS = 6
SERVO_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]

# Tokens usable in command chains (and as voice words).  See state_machine.py.
KNOWN_TOKENS = ("home", "ready", "scan", "align", "grab", "lift", "hold", "release", "write")


# --------------------------------------------------------------------------- #
# Seed data (inserted on first run only).  ALL ANGLES ARE PLACEHOLDERS:        #
# verify each one slowly on your own arm before trusting it.                   #
# --------------------------------------------------------------------------- #
# max_step_deg is "degrees per control tick".  At the default 40 Hz tick:
#   1.2 deg/tick = 48 deg/s (MG996R base), 0.9 = 36 deg/s (shoulder, heavy load).
DEFAULT_SERVO_CALIBRATION = [
    dict(channel=0, name="base",        model="MG996R", min_angle=0,  max_angle=180, home_angle=90, min_pulse=600, max_pulse=2400, max_step_deg=1.2),
    dict(channel=1, name="shoulder",    model="MG996R", min_angle=15, max_angle=165, home_angle=90, min_pulse=600, max_pulse=2400, max_step_deg=0.9),
    dict(channel=2, name="elbow",       model="MG996R", min_angle=15, max_angle=165, home_angle=90, min_pulse=600, max_pulse=2400, max_step_deg=1.0),
    dict(channel=3, name="wrist_pitch", model="MG90S",  min_angle=0,  max_angle=180, home_angle=90, min_pulse=600, max_pulse=2400, max_step_deg=2.0),
    dict(channel=4, name="wrist_roll",  model="MG90S",  min_angle=0,  max_angle=180, home_angle=90, min_pulse=600, max_pulse=2400, max_step_deg=2.5),
    dict(channel=5, name="gripper",     model="MG90S",  min_angle=20, max_angle=120, home_angle=40, min_pulse=600, max_pulse=2400, max_step_deg=3.0),
]

DEFAULT_PRESETS = [
    dict(name="home",        angles=[90, 90, 90, 90, 90, 40],  description="Rest pose; all joints centred, gripper open.", default_transition_ms=1500),
    dict(name="ready",       angles=[90, 60, 100, 100, 90, 40], description="Neutral working pose above the desk.",         default_transition_ms=1200),
    dict(name="open_grip",   angles=[90, 60, 100, 100, 90, 25], description="Ready pose with gripper fully open.",         default_transition_ms=600),
    dict(name="close_grip",  angles=[90, 60, 100, 100, 90, 100], description="Ready pose with gripper closed.",             default_transition_ms=600),
    dict(name="pen_grip",    angles=[90, 55, 110, 120, 90, 75], description="Tool/pen gripped, tip near the desk.",         default_transition_ms=1200),
    dict(name="steady_hold", angles=[90, 70, 95, 105, 90, 75],  description="Tool held steady in a raised hold pose.",     default_transition_ms=1500),
]

DEFAULT_ROUTINES = [
    dict(
        name="demo_wave", description="Ready, sweep the base left/right, return home.", repeat=2,
        steps=[
            dict(preset="ready", transition_ms=1500, hold_ms=200),
            dict(angles=[60, 60, 100, 100, 90, 40], transition_ms=1200, hold_ms=100),
            dict(angles=[120, 60, 100, 100, 90, 40], transition_ms=1800, hold_ms=100),
            dict(preset="ready", transition_ms=1200, hold_ms=200),
        ],
    ),
    dict(
        name="pen_demo", description="Pick the pen pose, hold steady, then home.", repeat=1,
        steps=[
            dict(preset="ready", transition_ms=1500, hold_ms=200),
            dict(preset="pen_grip", transition_ms=1500, hold_ms=800),
            dict(preset="steady_hold", transition_ms=1500, hold_ms=1500),
            dict(preset="home", transition_ms=2000, hold_ms=0),
        ],
    ),
]

DEFAULT_CHAINS = [
    dict(name="pickup", tokens=["scan", "align", "grab", "hold"], aliases=["pick"],
         description="Find the target, hover over it, grab it and hold."),
    dict(name="fetch", tokens=["scan", "grab", "lift", "release"], aliases=["bring"],
         description="Find, grab, lift and release again (demo of SCAN->GRAB->LIFT->RELEASE)."),
    dict(name="reset", tokens=["release", "home"], aliases=["tidy"],
         description="Open the gripper and go home."),
]


# --------------------------------------------------------------------------- #
# Settings                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class ServerSettings:
    host: str = "0.0.0.0"          # 0.0.0.0 so the phone on your LAN can connect
    port: int = 8765
    token: str = ""                # if set, clients must send ?token=... (or Authorization: Bearer ...)
    telemetry_hz: float = 10.0
    allow_inject: bool = False     # allow INJECT_VOICE from clients (always on in --sim)


@dataclass
class MotionSettings:
    tick_hz: float = 40.0
    accel_fraction: float = 0.15   # per-tick accel = max_step_deg * this (smooth start/stop)
    speed_scale: float = 1.0       # global multiplier on every max_step_deg (use <1 for cautious first runs)
    settle_eps_deg: float = 0.05


@dataclass
class LinkSettings:
    kind: str = "serial"           # serial | udp | sim
    serial_port: str = "auto"      # "auto" or e.g. COM5 or /dev/ttyUSB0
    baud: int = 115200
    udp_host: str = ""
    udp_port: int = 4210
    frame_format: str = "csv"      # csv | json
    heartbeat_s: float = 0.25
    rx_timeout_s: float = 1.5      # no reply from ESP for this long => disconnected
    connect_grace_s: float = 6.0   # ESP8266 resets when the serial port opens; give it time to boot (RP2040 doesn't)
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 5.0
    require_ack: bool = True       # False: treat "port opened" as connected (firmware never replies)


@dataclass
class VisionSettings:
    backend: str = "auto"          # auto | opencv | sim
    camera_index: int = 0
    width: int = 640
    height: int = 480
    detector: str = "auto"         # auto | aruco | color | contour | object
    scan_timeout_s: float = 4.0
    warmup_frames: int = 8
    min_hits: int = 3              # consistent detections required
    min_confidence: float = 0.5    # chain tokens ignore weaker targets
    aruco_dict: str = "DICT_4X4_50"
    color_hsv_low: list = field(default_factory=lambda: [5, 120, 120])
    color_hsv_high: list = field(default_factory=lambda: [25, 255, 255])
    min_area_px: int = 400
    save_debug: bool = True
    debug_dir: str = "debug"
    # Pixel -> desk-plane homography.  >= 4 points: [{"px": [u, v], "mm": [x, y]}, ...]
    calib_points: list = field(default_factory=list)
    calib_warn_rms_mm: float = 5.0  # warn at start-up if the calibration points fit worse than this
    # Object recognition (detector "object"): a YOLO model via the optional `ultralytics` package.
    # "yolov8n.pt" knows the 80 COCO classes (cup, bottle, scissors, cell phone, book, ...); point this at
    # your own trained .pt to recognise custom objects.  The weights download on first use.
    object_model: str = "yolov8n.pt"
    object_min_confidence: float = 0.35   # per-frame YOLO score below this is ignored
    object_imgsz: int = 640
    # Which pixel of a detection is mapped to the desk: "center" (best with a top-down camera) or
    # "bottom" (bottom-centre of the box, i.e. where an object touches the desk, for an angled camera).
    object_anchor: str = "center"
    target_label: Optional[str] = None    # default object for scans that don't name one (e.g. voice "fetch")
    sim_target_mm: list = field(default_factory=lambda: [180.0, 0.0])
    # Simulator only: [{"label": "cup", "mm": [x, y]}, ...]; empty = one "sim_object" at sim_target_mm.
    sim_objects: list = field(default_factory=list)


@dataclass
class VoiceSettings:
    backend: str = "auto"          # auto | vosk | mock | keyboard
    model_path: str = "models/vosk-model-small-en-us-0.15"
    sample_rate: int = 16000
    device: Optional[Any] = None   # sounddevice device index/name
    allowlist: list = field(default_factory=lambda: [
        "home", "ready", "scan", "align", "grab", "lift", "hold", "release", "write", "stop"])
    min_confidence: float = 0.80
    cooldown_s: float = 1.5
    mode: str = "ptt"              # ptt (push-to-talk window) | always
    listen_window_s: float = 5.0
    wake_word: str = ""            # optional: say this first, then the command
    wake_window_s: float = 4.0
    stop_words: list = field(default_factory=lambda: ["stop"])
    stop_latches_estop: bool = False   # True: voice "stop" latches ESTOP instead of a soft stop


@dataclass
class ArmSettings:
    """Geometry.  MEASURE THESE on your arm (centre of joint to centre of joint)."""
    base_height_mm: float = 60.0   # table to shoulder axis
    l1_mm: float = 120.0           # shoulder -> elbow
    l2_mm: float = 120.0           # elbow -> wrist pitch axis
    l3_mm: float = 90.0            # wrist pitch axis -> tool tip (gripper/pen tip)
    # servo_angle = offset + sign * geometric_angle  for [base, shoulder, elbow, wrist_pitch].
    # Geometric conventions are in kinematics.py.  Flip a sign if a joint moves the wrong way.
    joint_offsets_deg: list = field(default_factory=lambda: [90.0, 0.0, 0.0, 90.0])
    joint_signs: list = field(default_factory=lambda: [1, 1, -1, 1])
    min_z_mm: float = -2.0         # never plan below the desk plane (elbow/wrist/tip)
    pitch_candidates_deg: list = field(default_factory=lambda: [-90, -75, -60, -45, -30, 0])
    hover_z_mm: float = 80.0
    grip_z_mm: float = 15.0
    lift_z_mm: float = 100.0
    pen_z_mm: float = 10.0
    gripper_open_deg: float = 25.0
    gripper_closed_deg: float = 100.0
    write_center_mm: list = field(default_factory=lambda: [170.0, 0.0])
    write_size_mm: float = 40.0


@dataclass
class TaskSettings:
    hold_ms: int = 1500            # duration of the `hold` token
    grip_settle_ms: int = 450
    align_transition_ms: int = 1500
    queue_max: int = 8


@dataclass
class Settings:
    sim: bool = False
    db_path: str = "data/laptop_brain.sqlite3"
    log_level: str = "INFO"
    server: ServerSettings = field(default_factory=ServerSettings)
    motion: MotionSettings = field(default_factory=MotionSettings)
    link: LinkSettings = field(default_factory=LinkSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    arm: ArmSettings = field(default_factory=ArmSettings)
    tasks: TaskSettings = field(default_factory=TaskSettings)


def _merge(obj: Any, data: dict, path: str = "") -> None:
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_settings(path: Optional[str] = None) -> Settings:
    settings = Settings()
    if path:
        _merge(settings, json.loads(Path(path).read_text(encoding="utf-8")))
    return settings
