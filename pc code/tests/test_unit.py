"""Pure-logic tests: voice gate, kinematics, motion limits, database, frame encoding."""
import asyncio
import math

import pytest

from laptop_brain import config as cfg
from laptop_brain.db import Database, ValidationError
from laptop_brain.esp_link import SimTransport, angles_to_pulses, encode_cmd, encode_move
from laptop_brain.kinematics import ArmKinematics, CameraMapping, fit_homography
from laptop_brain.motion import Limits, MotionController
from laptop_brain.voice import VoiceGate


# ----------------------------------------------------------------- voice gate
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_gate(**kw):
    s = cfg.VoiceSettings(**kw)
    c = Clock()
    return VoiceGate(s, c), c


def test_gate_ptt_requires_window_and_is_one_shot():
    g, c = make_gate(mode="ptt")
    assert g.evaluate("grab", 0.99) == (False, "not_listening")
    g.open_window(5)
    assert g.evaluate("grab", 0.99) == (True, "ok")
    c.t += 10
    g.open_window(5)
    assert g.evaluate("grab", 0.99)[0] is True
    assert g.evaluate("home", 0.99) == (False, "not_listening")   # window closed after one command


def test_gate_rejects_unknown_low_conf_and_cooldown():
    g, c = make_gate(mode="always", cooldown_s=1.5)
    assert g.evaluate("banana", 1.0) == (False, "not_in_allowlist")
    assert g.evaluate("[unk]", 1.0) == (False, "unknown")
    assert g.evaluate("grab", 0.5) == (False, "low_confidence")
    assert g.evaluate("grab", 0.9) == (True, "ok")
    assert g.evaluate("home", 0.9) == (False, "cooldown")
    c.t += 2
    assert g.evaluate("home", 0.9) == (True, "ok")


def test_gate_stop_bypasses_cooldown_and_window():
    g, c = make_gate(mode="ptt")
    assert g.evaluate("stop", 0.9) == (True, "ok")
    assert g.evaluate("stop", 0.9) == (True, "ok")


def test_gate_wake_word():
    g, c = make_gate(mode="always", wake_word="arm", wake_window_s=3)
    assert g.evaluate("grab", 0.99) == (False, "wake_required")
    assert g.evaluate("arm", 0.99) == (False, "wake_armed")
    assert g.evaluate("grab", 0.99) == (True, "ok")
    c.t += 5
    assert g.evaluate("home", 0.99) == (False, "wake_required")


# ----------------------------------------------------------------- kinematics
@pytest.fixture
def kin():
    return ArmKinematics(cfg.ArmSettings(), [(c["min_angle"], c["max_angle"]) for c in cfg.DEFAULT_SERVO_CALIBRATION[:4]])


@pytest.mark.parametrize("x,y,z", [(180, 0, 15), (160, 40, 60), (150, -50, 100), (200, 20, 30)])
def test_ik_fk_roundtrip(kin, x, y, z):
    sol = kin.solve(x, y, z)
    assert sol.ok, sol.reason
    fk = kin.forward(sol.angles + [90, 90])
    assert fk["x"] == pytest.approx(x, abs=0.01)
    assert fk["y"] == pytest.approx(y, abs=0.01)
    assert fk["z"] == pytest.approx(z, abs=0.01)


def test_ik_rejects_unreachable_and_below_desk(kin):
    assert not kin.solve(600, 0, 50).ok
    assert not kin.solve(180, 0, -30).ok
    assert not kin.solve(-150, 0, 50).ok        # behind the base needs |base| > limits


def test_homography_recovers_scale_and_offset():
    px = [(0, 0), (640, 0), (640, 480), (0, 480)]
    mm = [(300, 160), (300, -160), (100, -160), (100, 160)]
    m = CameraMapping([{"px": p, "mm": q} for p, q in zip(px, mm)])
    assert m.calibrated
    x, y = m.px_to_mm(320, 240)
    assert (x, y) == pytest.approx((200, 0), abs=1e-6)
    u, v = m.mm_to_px(x, y)
    assert (u, v) == pytest.approx((320, 240), abs=1e-6)
    assert CameraMapping().px_to_mm(1, 1) is None


# ----------------------------------------------------------------- motion
def make_motion(**kw):
    limits = [Limits(c["min_angle"], c["max_angle"], c["home_angle"], c["max_step_deg"]) for c in cfg.DEFAULT_SERVO_CALIBRATION]
    t = [0.0]
    clock = lambda: t[0]
    sent = []
    m = MotionController(limits, cfg.MotionSettings(**kw), sink=lambda a: sent.append(a), clock=clock)
    return m, t, sent, limits


def run_ticks(m, t, n, dt=1 / 40):
    for _ in range(n):
        t[0] += dt
        m.tick()


def test_motion_respects_speed_accel_and_bounds():
    async def go():
        m, t, sent, limits = make_motion()
        m.move_to([0, 180, 15, 0, 0, 20])              # base 90->0, shoulder 90->165 (clamped), etc.
        prev, prev_v = m.angles, [0.0] * 6
        for _ in range(600):
            run_ticks(m, t, 1)
            now = m.angles
            for i in range(6):
                step = now[i] - prev[i]
                assert abs(step) <= limits[i].max_step_deg + 1e-9
                assert limits[i].min_angle - 1e-9 <= now[i] <= limits[i].max_angle + 1e-9
                a = limits[i].max_step_deg * m.s.accel_fraction
                assert abs(step - prev_v[i]) <= a + 1e-6 or abs(now[i] - m.tgt[i]) < 1e-6   # smooth speed changes
                prev_v[i] = step
            prev = now
        assert m.angles == pytest.approx(m.targets, abs=1e-6)
        assert m.is_settled
        assert m.targets[1] == 165                      # clamp applied
    asyncio.run(go())


def test_motion_clamp_flag_and_halt_brakes():
    async def go():
        m, t, sent, _ = make_motion()
        _, clamped = m.move_to([200, None, None, None, None, None])
        assert clamped
        run_ticks(m, t, 30)
        before = m.angles[0]
        m.halt()
        run_ticks(m, t, 60)
        stop = m.angles[0]
        assert before < stop < 180                       # braked, did not run to the old target
        v = [m.angles[0]]
        run_ticks(m, t, 10)
        assert m.angles[0] == pytest.approx(stop, abs=1e-6)
    asyncio.run(go())


def test_motion_freeze_stops_output():
    async def go():
        m, t, sent, _ = make_motion()
        m.move_to([0, None, None, None, None, None])
        run_ticks(m, t, 10)
        m.freeze()
        n, pos = len(sent), m.angles
        run_ticks(m, t, 50)
        assert len(sent) == n and m.angles == pos
        assert m.is_settled
        m.unfreeze()
        assert m.targets == m.angles
    asyncio.run(go())


def test_timed_transition_follows_duration():
    async def go():
        m, t, _, _ = make_motion(speed_scale=1.0)
        m.move_to([None, None, None, 150, None, None], transition_ms=1000)   # 60 deg at 2 deg/tick needs >= 30 ticks
        run_ticks(m, t, 20)
        assert m.angles[3] < 150 - 1
        run_ticks(m, t, 60)
        assert m.angles[3] == pytest.approx(150, abs=0.1)
    asyncio.run(go())


# ----------------------------------------------------------------- database
def test_db_seed_and_validation():
    db = Database(":memory:")
    assert [c.channel for c in db.calibration.all()] == [0, 1, 2, 3, 4, 5]
    assert {"home", "ready", "open_grip", "pen_grip", "steady_hold"} <= {p.name for p in db.presets.list()}
    assert db.routines.get("demo_wave") and db.chains.get("pickup")
    assert db.chains.resolve_word("fetch").name == "fetch"
    assert db.chains.resolve_word("bring").name == "fetch"
    with pytest.raises(ValidationError):
        db.calibration.update(1, min_angle=100, max_angle=90)
    with pytest.raises(ValidationError):
        db.calibration.update(0, max_step_deg=50)
    with pytest.raises(ValidationError):
        db.calibration.update(0, min_pulse=100)
    with pytest.raises(ValidationError):
        db.presets.upsert("bad name!", [90] * 6)
    with pytest.raises(ValidationError):
        db.presets.upsert("p", [90] * 5)
    with pytest.raises(ValidationError):
        db.presets.upsert("p", [90, 90, 90, 90, 90, 181])
    with pytest.raises(ValidationError):
        db.presets.delete("home")
    p = db.presets.upsert("mine", [1, 2, 3, 4, 5, 6], "x", 500)
    assert p.angles == [1, 2, 3, 4, 5, 6] and db.presets.delete("mine")


def test_db_chains_and_routines_validation():
    db = Database(":memory:")
    with pytest.raises(ValidationError):
        db.chains.upsert("c1", ["scan", "explode"])
    with pytest.raises(ValidationError):
        db.chains.upsert("grab", ["scan"])                      # name collides with a token
    with pytest.raises(ValidationError):
        db.chains.upsert("c1", ["scan"], aliases=["fetch"])     # alias used by another chain
    c = db.chains.upsert("c1", ["SCAN", "grab"], aliases=["Go"])
    assert c.tokens == ["scan", "grab"] and c.aliases == ["go"]
    with pytest.raises(ValidationError):
        db.routines.upsert("r", [{"preset": "nope"}])
    with pytest.raises(ValidationError):
        db.routines.upsert("r", [{"preset": "home", "angles": [90] * 6}])
    r = db.routines.upsert("r", [{"preset": "home"}, {"angles": [90, None, 90, 90, 90, 40], "hold_ms": 100}], repeat=3)
    assert r.repeat == 3 and r.steps[0]["transition_ms"] == 800


def test_db_event_log():
    db = Database(":memory:")
    db.events.log("voice", "heard", "grab", {"conf": 0.9})
    db.events.log("estop", "estop", "x", level="WARNING")
    rows = db.events.recent(10)
    assert rows[0]["source"] == "estop" and rows[1]["payload"] == {"conf": 0.9}
    assert len(db.events.recent(10, source="voice")) == 1


# ----------------------------------------------------------------- link frames
def test_pulses_and_frames_and_sim_esp():
    db = Database(":memory:")
    cals = db.calibration.all()
    p = angles_to_pulses([0, 90, 180, 90, 90, 0], cals)
    assert p[0] == 600 and p[1] == 1500                           # 0 deg / 90 deg
    assert p[2] == round(600 + 1800 * 165 / 180)
    assert p[5] == round(600 + 1800 * 20 / 180)                   # gripper min angle is 20
    assert encode_move(7, p, "csv").startswith(b"M,7,")
    assert b'"c":"M"' in encode_move(7, p, "json")
    sim = SimTransport()
    sim.open()
    sim.write(encode_move(1, [1000] * 6))
    assert sim.pulses == [1000] * 6
    sim.write(encode_cmd("E", 2))
    sim.write(encode_move(3, [2000] * 6))
    assert sim.pulses == [1000] * 6 and sim.frozen               # frozen after E
    sim.write(encode_cmd("C", 4, "json"))
    sim.write(encode_move(5, [2000] * 6, "json"))
    assert sim.pulses == [2000] * 6
