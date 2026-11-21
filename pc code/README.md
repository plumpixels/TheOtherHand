# theOtherHand — laptop host ("laptop brain")

Python backend for a 6-servo assistive / precision-tool-holding arm (PCA9685 + ESP8266, MG996R ×3 + MG90S ×3).
It owns the database, motion smoothing, safety rules, webcam vision, voice commands and the WebSocket API for
your Expo app. The ESP8266 stays dumb: it receives pulse widths and a watchdog heartbeat.

> **Scope.** This is a desk-assistance / rehab-demo / tool-holding prototype on hobby servos. It has no
> force sensing, no encoders and no position feedback, so it is **not** suitable for clinical or surgical use.
> The software `ESTOP` is a convenience layer, not a safety-rated stop: put a physical switch in the servo
> power line.

---

## 1. Quick start (no hardware)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                       # core + pytest; enough for --sim
python -m laptop_brain --sim                              # WebSocket API on ws://0.0.0.0:8765
```

In a second terminal, use the bundled test client:

```bash
python -m laptop_brain.cli_client                         # interactive
python -m laptop_brain.cli_client -c "preset ready" -c "watch 3" -c "chain scan align grab lift release" -c "watch 8"
```

Handy commands inside the client:

| command | does |
|---|---|
| `state` / `watch 3` | one snapshot / print live telemetry for 3 s |
| `preset ready` · `routine demo_wave` · `chain fetch` · `chain scan grab lift release` | run a preset / routine / named chain / ad-hoc token chain |
| `angles 90 60 _ _ _ _` · `nudge 0 -1` | set joint angles (`_` = keep) · nudge joint 0 by −1° |
| `scan` · `scan cup` · `chain fetch @bottle` | scan for anything / for a named object · run a chain on a named object |
| `listen` · `voice grab 0.95` | open the push-to-talk window · inject a spoken word |
| `estop` · `clear` · `stop` | latching ESTOP · clear it · soft stop (no latch) |
| `presets` · `routines` · `chains` · `cal` · `events 20` | read the database |
| `raw {"type":"SIM_ESP_DISCONNECT","seconds":3}` | simulate the ESP cable being pulled |

`--sim` uses a **separate** database (`data/laptop_brain_sim.sqlite3`), a simulated ESP8266 that speaks the real
wire protocol, simulated vision (finds an object near `vision.sim_target_mm`) and mock voice (words arrive
through `INJECT_VOICE`). Run the tests with `python -m pytest -q` (36 tests, ~30 s).

## 2. Layout

```
laptop_brain/
  config.py         all settings + seed data (servo limits, presets, routines, chains)
  db.py             SQLite schema + repositories: servo_calibration, presets, routines, command_chains, event_logs
  motion.py         30-50 Hz smoother: min/max clamp, per-servo speed cap, acceleration cap, timed transitions
  kinematics.py     FK/IK for base+shoulder+elbow+wrist-pitch, camera pixel -> desk homography, path helper
  esp_link.py       transports (Serial primary, UDP optional, Sim), frame protocol, heartbeat, health, reconnect
  vision.py         on-demand webcam scan (ArUco / YOLO objects / colour / contour) + simulator
  calibrate_camera.py  guided camera->desk calibration with accuracy report + live mm read-out
  voice.py          Vosk single-word recogniser, VoiceGate (allowlist/confidence/cooldown/PTT/wake), mock + keyboard
  state_machine.py  the arbiter: IDLE, MOVING, VISION_SCAN, EXECUTING_CHAIN, ESTOP + job queue + chain tokens
  server.py         WebSocket API + telemetry push
  app.py / __main__.py   wiring and CLI flags
  cli_client.py     test client / REPL
tests/              unit + end-to-end tests over a real WebSocket
```

## 3. Going to real hardware

1. `pip install -r requirements.txt -r requirements-optional.txt`
2. Copy `config.example.json` → `config.json` and edit it (see §7 for what to measure).
3. **Before powering the servos**, move the arm by hand close to its `home` pose (§7.1). On connect, the host
   sends the home angles; a servo that is somewhere else will snap there at full speed.
4. Power: servos from their own 5–6 V supply (MG996R can pull ~2.5 A stalled each), common ground with the
   PCA9685/ESP, a bulk capacitor on the servo rail, and a physical cut-off switch.
5. First run slow: `python -m laptop_brain --config config.json --serial auto --token SECRET --speed 0.4`
   (`--serial COM5` or `/dev/ttyUSB0` to pick a port; `--udp 192.168.1.50:4210` for Wi-Fi).
6. Point the app at `ws://<laptop-ip>:8765/?token=SECRET`. Without `--token` anyone on your network can drive
   the arm; the server warns about that on start-up.

Opening the serial port resets most ESP8266 boards, so the link allows `connect_grace_s` (6 s) for the firmware
to boot and answer. If your firmware never replies, set `link.require_ack` to `false` (you then lose disconnect
detection, so don't do that for anything but bring-up).

## 4. ESP8266 firmware contract

The laptop converts angles to **pulse widths in µs** using each servo's `min_pulse` (= 0°) and `max_pulse` (= 180°),
so the firmware needs no calibration. One frame per line, `\n` terminated, 115200 baud:

```
laptop -> ESP   M,<seq>,<us0>,<us1>,<us2>,<us3>,<us4>,<us5>     move: channels 0-5, e.g. M,42,1500,1500,1500,1500,1500,1000
                H,<seq>                                           heartbeat every 250 ms (also feeds your watchdog)
                E,<seq>                                           ESTOP: freeze outputs at the last pulses, ignore M until C
                C,<seq>                                           clear ESTOP
ESP -> laptop   R,<fw>          after boot            A,<seq>[,info]   ack of H (required for health tracking)
                W[,reason]      watchdog tripped / bad frame (informational)
```

`frame_format: "json"` sends the same frames as `{"c":"M","s":42,"p":[...]}`. What the firmware should do:

* PCA9685 at 50 Hz, 12-bit: `ticks = us * 4096 / 20000` (≈ `us * 0.2048`; trim with `setOscillatorFrequency`).
* **Watchdog**: if no valid frame (M or H) arrives for ~500 ms, hold the last pulses and answer `W`; resume on the next valid `M`.
* **`E` latches**: ignore `M` until `C`. "Freeze" keeps holding torque, which is the safest default for a
  tool-holding arm; add a separate release-torque command only if you want one.
* Reply `A,<seq>` to every `H` (and `E`/`C`). The host declares the link dead after `rx_timeout_s` (1.5 s) of silence.

The firmware itself is not part of this package; the protocol above is what the sim ESP in `esp_link.py` implements
and what the tests exercise. Ask if you want a matching Arduino sketch.

## 5. WebSocket API

Client → server: `{"type": "<COMMAND>", "id": "<any>", ...}`. Replies carry the same `id`:
`{"type":"ack","ok":true,"data":{...}}` or `{"type":"error","code":"BUSY","message":"..."}`.
Error codes: `ESTOP_ACTIVE`, `ESP_DISCONNECTED`, `BUSY`, `QUEUE_FULL`, `NOT_FOUND`, `BAD_REQUEST`, `FORBIDDEN`, `UNKNOWN_COMMAND`, `INTERNAL`.

| command | fields | notes |
|---|---|---|
| `SET_ANGLES` | `angles`: list of 6 (`null` = keep) or `{"0": 90}`; `transition_ms` | clamped to min/max; reply says `clamped` |
| `NUDGE_JOINT` | `channel`, `delta_deg` (e.g. `-1` / `1`) | relative to the current *target*, so taps accumulate |
| `RUN_PRESET` / `RUN_ROUTINE` | `name`, optional `mode` | `mode`: `queue` (default), `reject`, `preempt` |
| `RUN_CHAIN` | `name` **or** `tokens: ["scan","grab",...]`; optional `target`, `detector` | tokens: home ready scan align grab lift hold release write; `target` = object the `scan` token looks for |
| `TRIGGER_VISION_SCAN` | optional `target` (`"cup"`, `"7"`), `detector` | one on-demand scan; result lands in `last_vision` |
| `TRIGGER_VOICE_LISTEN` | `seconds?` | opens the push-to-talk window (default 5 s) |
| `ESTOP` / `CLEAR_ESTOP` / `STOP` | – | ESTOP latches; STOP is a braked soft stop |
| `GET_STATE`, `PING` | – | snapshot / liveness |
| `LIST_/UPSERT_/DELETE_` `PRESET(S)`, `ROUTINE(S)`, `CHAIN(S)` | see below | CRUD; chains refresh the voice grammar |
| `GET_CALIBRATION` / `SET_CALIBRATION` | `channel` + any of `min_angle max_angle home_angle min_pulse max_pulse max_step_deg name model` | applies live; refused while a job runs |
| `GET_EVENTS` | `limit`, `source?`, `before_id?` | audit log |
| `INJECT_VOICE`, `SIM_ESP_DISCONNECT` | `word`, `confidence` / `seconds` | test aids: `--sim` (or `server.allow_inject`) only |

CRUD payloads: preset `{name, angles[6], description, default_transition_ms}` · routine
`{name, repeat, steps:[{preset|angles, hold_ms, transition_ms}]}` · chain `{name, tokens[], aliases[], description}`.
Chain names/aliases that are single lowercase words become voice commands automatically.

Server → client: `hello` on connect, then `telemetry` (≈10 Hz when something changed, at least 1 Hz):

```json
{"type":"telemetry","state":"EXECUTING_CHAIN","detail":"chain:fetch","estop":false,
 "current_angles":[90,60,100,100,90,40],"target_angles":[...],"esp_connected":true,
 "last_vision":{"found":true,"x":182.8,"y":3.0,"confidence":0.91,"label":"cup","wanted":"cup","space":"table","px":323,"py":57,
                "bbox":[300,20,346,94],"objects":[{"label":"cup","confidence":0.91,"x":182.8,"y":3.0,...},{"label":"bottle",...}]},
 "last_voice":{"word":"grab","confidence":0.96,"accepted":true,"reason":"ok","action":"chain:grab"},
 "job":{"id":4,"kind":"chain","name":"fetch","step":"2/4:grab"},"queue_len":0,"last_error":null,"sim":false}
```

plus `event` messages (`job_finished`, `estop`, `estop_cleared`, `vision`, `voice`, `esp`).
`last_vision.space` is `"table"` (desk mm) when the camera is calibrated, otherwise `"pixel"`. `objects` lists
everything seen in the last frame, each with its own desk `x`/`y`. When the wanted object isn't found,
`last_vision` is `{"found":false,"wanted":"cup","objects":[...]}` and the job message says what *was* seen.

## 6. How commands are arbitrated

* **ESTOP is never queued.** It writes `E` to the ESP first, then freezes the motion loop, cancels the running
  job (and any scan), clears the queue and latches `ESTOP`. Only `CLEAR_ESTOP` leaves it, and only while the
  ESP is reachable. Clearing resumes from where the arm stopped; nothing jumps.
* **Jobs run one at a time.** App commands default to `queue` (max 8), voice commands use `reject` (a spoken word
  never moves the arm later), and `preempt` brakes the current job and runs immediately.
* **Manual jogging** (`SET_ANGLES`, `NUDGE_JOINT`) is accepted only while no job is running (`BUSY` otherwise).
* **ESP link lost** → safe hold: motion frozen, jobs and queue dropped, motion commands return `ESP_DISCONNECTED`.
  On reconnect the host resumes from the held pose.
* **Voice `stop`** is a soft stop and bypasses cooldown/window. Set `voice.stop_latches_estop` to make it latch.
* Every clamp is enforced in `MotionController` (the single place that produces angles), so no path can bypass limits.

## 7. What you must calibrate on *your* arm

### 7.1 Servo limits, home and pulses (`servo_calibration`)
The seed values are conservative placeholders. Do this with the arm unloaded and `--speed 0.3`:

1. For each joint, jog with `NUDGE_JOINT` (±1°) toward each mechanical stop and write down the last angle that
   does **not** strain or buzz. Save them: `SET_CALIBRATION {channel, min_angle, max_angle}` (applies live and persists).
2. Set `home_angle` to the neutral pose you want at start-up (arm folded safely, gripper open).
3. `min_pulse` / `max_pulse` map 0° / 180°. 600/2400 µs is a safe start; many MG996R/MG90S go to 500/2500, and
   cheap units differ. Widen only if the servo reaches the full 180° without hitting its stop.
4. `max_step_deg` is **degrees per control tick**: speed = `max_step_deg × tick_hz × speed_scale` (1.2 → 48°/s at 40 Hz).
   Lower it for heavy loads or jittery joints (shoulder/elbow carry the most).
5. Edit the presets (`home`, `ready`, `open_grip`, `close_grip`, `pen_grip`, `steady_hold`): the seeded angles are
   placeholders. Use `SET_ANGLES` to pose the arm, then `UPSERT_PRESET`. Saving warns if an angle will be clamped.

### 7.2 Arm geometry and joint directions (`arm.*`)
`base_height_mm`, `l1_mm` (shoulder→elbow), `l2_mm` (elbow→wrist axis), `l3_mm` (wrist axis→tool tip, **including**
the gripper or pen) are measured axis-to-axis in mm. `joint_offsets_deg` / `joint_signs` define servo angle =
offset + sign × geometric angle for `[base, shoulder, elbow, wrist_pitch]` (conventions are at the top of
`kinematics.py`: base 0° = straight ahead, shoulder 90° = upper arm vertical, elbow 0° = straight with negative =
bent down, pitch −90° = tool pointing down). The defaults assume servo 90° ≈ base forward, shoulder angle =
elevation, elbow bent by `servo`, wrist neutral. Verify:

```bash
python - <<'EOF'
from laptop_brain.config import ArmSettings
from laptop_brain.kinematics import ArmKinematics
k = ArmKinematics(ArmSettings())
print(k.forward([90, 60, 100, 100, 90, 40]))          # predicted tool tip x, y, z in mm
print(k.solve(180, 0, 15))                             # IK for a desk point 180 mm ahead
EOF
```

Put the arm in that pose (`angles 90 60 100 100 _ _`) and measure the real tool tip from the base axis.
If a joint moves the wrong way, flip its sign; if the pose is off by a constant, adjust the offset; if the
distance is off, fix the link lengths. Do this before using `align`/`grab`/`write`: unreachable or below-desk
targets are refused (`min_z_mm`), but wrong signs/offsets will happily aim somewhere unintended.
Also set `hover_z_mm`, `grip_z_mm`, `lift_z_mm`, `pen_z_mm`, `gripper_open_deg`, `gripper_closed_deg`, and
`write_center_mm` (where `write` draws a 40 mm square; if it's unreachable the token falls back to the `pen_grip` preset).

### 7.3 Camera → desk mapping (`vision.calib_points`)
The desk frame is: origin under the base axis, +x forward, +y left, millimetres. The camera is mapped to the desk
plane with a homography, so a pixel becomes a desk (x, y) in mm.

1. Fix the webcam so it sees the whole workspace and does not move afterwards (top-down is most accurate).
2. Print an ArUco marker (dictionary `DICT_4X4_50`, id 7) at a known size and keep it flat.
3. Run the guided tool and place the marker at 6–9 **measured** points spread over the reachable area:
   ```bash
   python -m laptop_brain.calibrate_camera --config config.json --write
   ```
   It averages ~15 frames per point and prints the **expected accuracy in mm** (leave-one-out error), flagging
   any point that looks mis-measured. 4 points always fit perfectly and tell you nothing, so use more.
4. Check it live: `python -m laptop_brain.calibrate_camera --live` prints the desk position of every marker it sees.
   Put the marker on a ruler mark and compare. The server also logs the expected error at start-up.

Until calibrated, results stay `space: "pixel"` and `align`/`grab`/`lift` refuse to move to them.

### 7.3b Recognising objects (`detector: "object"`)
`pip install ultralytics`, then scan with a target: `scan cup`, `RUN_CHAIN {"name":"fetch","target":"bottle"}`.
The default `vision.object_model` `yolov8n.pt` downloads (~6 MB) on first use and knows the 80 COCO classes
(cup, bottle, scissors, book, cell phone, remote, ...). For your own objects (a specific tool, pill box, pen), train a
YOLO model on ~100 photos of each and set `object_model` to its `.pt` file. Set `vision.target_label` to give voice
commands like `fetch` a default object. Note: `ultralytics` and its models are AGPL-3.0; check the licence before
shipping it in a product.

**Where precision comes from / how to get more**
* ArUco markers: sub-pixel corners, ~1 mm with a good calibration. Use them on objects you must place precisely.
* YOLO boxes: the box centre wobbles by a few % of the object size, so expect ~3–10 mm. A scan takes the median of
  `min_hits` frames (raise it to 5–7 for steadier results) and stays locked on the first object it saw.
* The map is for the **desk surface**. A tall object's centre seen from an angle lands too far away; mount the
  camera top-down, or set `object_anchor: "bottom"` (bottom of the box = where it touches the desk).
* Calibrate with the same resolution (`vision.width/height`) you scan at, and re-calibrate if the camera moves.
* `debug/last_scan.jpg` shows every box seen and a green cross on the chosen target.

### 7.4 Voice
```bash
pip install -r requirements-optional.txt
# download a small model from https://alphacephei.com/vosk/models and unzip so this exists:
#   models/vosk-model-small-en-us-0.15/
python -m laptop_brain --serial auto --voice vosk
```
The recogniser is constrained to `voice.allowlist` plus your chain names/aliases (so random speech maps to
`[unk]`). A word only acts if it passes `VoiceGate`: in the allowlist, confidence ≥ `min_confidence` (0.80), outside the
`cooldown_s` (1.5 s) of the last accepted word, and inside a push-to-talk window (`TRIGGER_VOICE_LISTEN`, default
mode) or after the optional `wake_word`. Set `voice.mode` to `"always"` for an always-listening mic. If Vosk, the
mic or the model is missing, the host logs a warning and keeps running with voice via `INJECT_VOICE`/`--voice keyboard` only.

## 8. What has and hasn't been verified

Verified in this repo: the 36 tests (unit + end-to-end over a real WebSocket against the simulated ESP) pass on Python
3.11, 3.12 and 3.13 (and 3.14 here): ESTOP preemption and latch, ESP-disconnect safe hold and recovery, speed/accel/limit
enforcement, DB validation, voice gating, queueing/preempt, CRUD, token auth. The OpenCV detectors (ArUco, colour,
contour) and homography mapping were exercised on synthetic frames with a stubbed camera; ArUco centres come out
within 0.6 px on a synthetic frame. Object recognition was tested with a stubbed YOLO model (label targeting,
lock-on, bottom anchor, mm mapping), **not** with real `ultralytics` weights or a real webcam.

**Not** verified (no hardware in the build environment): a real ESP8266/PCA9685 over serial or UDP, a real webcam,
Vosk with a microphone, and Python 3.10 (the code uses nothing newer than 3.10, but it wasn't run there). Expect to
tune pulse ranges, joint signs/offsets, speeds and presets on the real arm.
