"""SQLite persistence: schema, seed data and a small repository layer.

One ``Database`` object owns the connection; the repositories
(``calibration``, ``presets``, ``routines``, ``chains``, ``events``) hold all SQL.
Everything is validated on write so bad data never reaches the motion code.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config as cfg

NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")
WORD_RE = re.compile(r"^[a-z]{2,24}$")
PROTECTED_PRESETS = {"home", "ready"}      # the state machine depends on these two
MODELS = ("MG996R", "MG90S")
MAX_STEP_CAP = {"MG996R": 3.0, "MG90S": 6.0}   # deg/tick hard caps per servo model


class ValidationError(ValueError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS servo_calibration (
    channel      INTEGER PRIMARY KEY CHECK (channel BETWEEN 0 AND 5),
    name         TEXT NOT NULL,
    model        TEXT NOT NULL CHECK (model IN ('MG996R','MG90S')),
    min_angle    REAL NOT NULL,
    max_angle    REAL NOT NULL,
    home_angle   REAL NOT NULL,
    min_pulse    INTEGER NOT NULL,
    max_pulse    INTEGER NOT NULL,
    max_step_deg REAL NOT NULL,
    CHECK (min_angle < max_angle),
    CHECK (home_angle BETWEEN min_angle AND max_angle),
    CHECK (min_pulse < max_pulse)
);

CREATE TABLE IF NOT EXISTS presets (
    name                  TEXT PRIMARY KEY,
    angles_json           TEXT NOT NULL,
    description           TEXT NOT NULL DEFAULT '',
    default_transition_ms INTEGER NOT NULL DEFAULT 1000,
    builtin               INTEGER NOT NULL DEFAULT 0,
    updated_at            REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS routines (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    repeat      INTEGER NOT NULL DEFAULT 1,
    steps_json  TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS command_chains (
    name         TEXT PRIMARY KEY,
    tokens_json  TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    description  TEXT NOT NULL DEFAULT '',
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS event_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    source       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    level        TEXT NOT NULL DEFAULT 'INFO',
    message      TEXT NOT NULL DEFAULT '',
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_event_logs_ts ON event_logs (ts);
"""


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ServoCal:
    channel: int
    name: str
    model: str
    min_angle: float
    max_angle: float
    home_angle: float
    min_pulse: int
    max_pulse: int
    max_step_deg: float

    def validate(self) -> "ServoCal":
        if not (isinstance(self.channel, int) and 0 <= self.channel < cfg.NUM_SERVOS):
            raise ValidationError("channel must be 0-5")
        if self.model not in MODELS:
            raise ValidationError(f"model must be one of {MODELS}")
        for f in ("min_angle", "max_angle", "home_angle", "max_step_deg"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValidationError(f"{f} must be a finite number")
        if not (0 <= self.min_angle < self.max_angle <= 180):
            raise ValidationError("need 0 <= min_angle < max_angle <= 180")
        if not (self.min_angle <= self.home_angle <= self.max_angle):
            raise ValidationError("home_angle must lie within [min_angle, max_angle]")
        # 0 deg / 180 deg pulse.  Up to 3000: firmware calibrated as "ticks 150..600 at a 27 MHz oscillator"
        # maps 180 deg to ~2930 nominal us.  min/max_angle keep the pulses actually sent far inside this.
        if not (400 <= int(self.min_pulse) < int(self.max_pulse) <= 3000):
            raise ValidationError("pulses must satisfy 400 <= min_pulse < max_pulse <= 3000 (us)")
        cap = MAX_STEP_CAP[self.model]
        if not (0.05 <= self.max_step_deg <= cap):
            raise ValidationError(f"max_step_deg for {self.model} must be within 0.05..{cap}")
        self.min_pulse, self.max_pulse = int(self.min_pulse), int(self.max_pulse)
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Preset:
    name: str
    angles: list
    description: str = ""
    default_transition_ms: int = 1000
    builtin: bool = False
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Routine:
    name: str
    steps: list
    repeat: int = 1
    description: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chain:
    name: str
    tokens: list
    aliases: list = field(default_factory=list)
    description: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _check_angles(angles: Any, allow_none: bool = False) -> list:
    if not isinstance(angles, (list, tuple)) or len(angles) != cfg.NUM_SERVOS:
        raise ValidationError(f"angles must be a list of {cfg.NUM_SERVOS} numbers")
    out = []
    for a in angles:
        if a is None and allow_none:
            out.append(None)
            continue
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or not (0 <= a <= 180):
            raise ValidationError("every angle must be a finite number within 0..180")
        out.append(float(a))
    return out


def _check_name(name: Any, what: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValidationError(f"{what} name must match [a-z0-9_]{{1,32}}")
    return name


# --------------------------------------------------------------------------- #
# Database + repositories                                                      #
# --------------------------------------------------------------------------- #
class Database:
    def __init__(self, path: str = "data/laptop_brain.sqlite3", max_events: int = 50_000):
        self.path = path
        self.max_events = max_events
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
        self.calibration = CalibrationRepo(self)
        self.presets = PresetRepo(self)
        self.routines = RoutineRepo(self)
        self.chains = ChainRepo(self)
        self.events = EventRepo(self)
        self._seed()

    # -- plumbing ----------------------------------------------------------- #
    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    @contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- seeding ------------------------------------------------------------ #
    def _seed(self) -> None:
        if self.query("SELECT 1 FROM meta WHERE key='seeded'"):
            return
        with self.tx():
            for row in cfg.DEFAULT_SERVO_CALIBRATION:
                self.calibration._save(ServoCal(**row).validate())
            for p in cfg.DEFAULT_PRESETS:
                self.presets._save(Preset(builtin=True, **p))
            for r in cfg.DEFAULT_ROUTINES:
                self.routines._save(Routine(**r))
            for c in cfg.DEFAULT_CHAINS:
                self.chains._save(Chain(**c))
            self.execute("INSERT INTO meta(key,value) VALUES('seeded', ?)", (str(time.time()),))
        self.events.log("system", "seed", "database seeded with defaults")


class CalibrationRepo:
    FIELDS = ("name", "model", "min_angle", "max_angle", "home_angle", "min_pulse", "max_pulse", "max_step_deg")

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r) -> ServoCal:
        return ServoCal(**{k: r[k] for k in r.keys()})

    def all(self) -> list:
        return [self._row(r) for r in self.db.query("SELECT * FROM servo_calibration ORDER BY channel")]

    def get(self, channel: int) -> ServoCal:
        rows = self.db.query("SELECT * FROM servo_calibration WHERE channel=?", (channel,))
        if not rows:
            raise ValidationError(f"no calibration for channel {channel}")
        return self._row(rows[0])

    def _save(self, cal: ServoCal) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO servo_calibration"
            "(channel,name,model,min_angle,max_angle,home_angle,min_pulse,max_pulse,max_step_deg)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (cal.channel, cal.name, cal.model, cal.min_angle, cal.max_angle, cal.home_angle,
             cal.min_pulse, cal.max_pulse, cal.max_step_deg))

    def update(self, channel: int, **changes: Any) -> ServoCal:
        unknown = set(changes) - set(self.FIELDS)
        if unknown:
            raise ValidationError(f"unknown calibration fields: {sorted(unknown)}")
        cal = self.get(channel)
        for k, v in changes.items():
            setattr(cal, k, v)
        cal.validate()
        self._save(cal)
        return cal


class PresetRepo:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r) -> Preset:
        return Preset(name=r["name"], angles=json.loads(r["angles_json"]), description=r["description"],
                      default_transition_ms=r["default_transition_ms"], builtin=bool(r["builtin"]),
                      updated_at=r["updated_at"])

    def list(self) -> list:
        return [self._row(r) for r in self.db.query("SELECT * FROM presets ORDER BY name")]

    def get(self, name: str) -> Optional[Preset]:
        rows = self.db.query("SELECT * FROM presets WHERE name=?", (name,))
        return self._row(rows[0]) if rows else None

    def _save(self, p: Preset) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO presets(name,angles_json,description,default_transition_ms,builtin,updated_at)"
            " VALUES(?,?,?,?,?,?)",
            (p.name, json.dumps(p.angles), p.description, p.default_transition_ms, int(p.builtin), time.time()))

    def upsert(self, name: str, angles: Any, description: str = "", default_transition_ms: int = 1000) -> Preset:
        _check_name(name, "preset")
        angles = _check_angles(angles)
        if not isinstance(default_transition_ms, int) or isinstance(default_transition_ms, bool) \
                or not (0 <= default_transition_ms <= 60_000):
            raise ValidationError("default_transition_ms must be an int within 0..60000")
        existing = self.get(name)
        p = Preset(name, angles, str(description)[:200], default_transition_ms,
                   builtin=existing.builtin if existing else False)
        self._save(p)
        return self.get(name)  # type: ignore[return-value]

    def delete(self, name: str) -> bool:
        if name in PROTECTED_PRESETS:
            raise ValidationError(f"preset '{name}' is protected (the state machine uses it)")
        if self.db.query("SELECT 1 FROM presets WHERE name=?", (name,)):
            self.db.execute("DELETE FROM presets WHERE name=?", (name,))
            return True
        return False


class RoutineRepo:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r) -> Routine:
        return Routine(name=r["name"], steps=json.loads(r["steps_json"]), repeat=r["repeat"],
                       description=r["description"], updated_at=r["updated_at"])

    def list(self) -> list:
        return [self._row(r) for r in self.db.query("SELECT * FROM routines ORDER BY name")]

    def get(self, name: str) -> Optional[Routine]:
        rows = self.db.query("SELECT * FROM routines WHERE name=?", (name,))
        return self._row(rows[0]) if rows else None

    def _save(self, r: Routine) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO routines(name,description,repeat,steps_json,updated_at) VALUES(?,?,?,?,?)",
            (r.name, r.description, r.repeat, json.dumps(r.steps), time.time()))

    def _validate_steps(self, steps: Any) -> list:
        if not isinstance(steps, list) or not (1 <= len(steps) <= 200):
            raise ValidationError("steps must be a list of 1..200 entries")
        clean = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                raise ValidationError(f"step {i} must be an object")
            has_p, has_a = "preset" in s, "angles" in s
            if has_p == has_a:
                raise ValidationError(f"step {i} needs exactly one of 'preset' or 'angles'")
            step: dict = {}
            if has_p:
                if not isinstance(s["preset"], str) or self.db.presets.get(s["preset"]) is None:
                    raise ValidationError(f"step {i}: unknown preset {s['preset']!r}")
                step["preset"] = s["preset"]
            else:
                step["angles"] = _check_angles(s["angles"], allow_none=True)
            for key, default in (("transition_ms", 800), ("hold_ms", 0)):
                v = s.get(key, default)
                if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= 60_000):
                    raise ValidationError(f"step {i}: {key} must be an int within 0..60000")
                step[key] = v
            clean.append(step)
        return clean

    def upsert(self, name: str, steps: Any, repeat: int = 1, description: str = "") -> Routine:
        _check_name(name, "routine")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not (1 <= repeat <= 100):
            raise ValidationError("repeat must be an int within 1..100")
        r = Routine(name, self._validate_steps(steps), repeat, str(description)[:200])
        self._save(r)
        return self.get(name)  # type: ignore[return-value]

    def delete(self, name: str) -> bool:
        if self.db.query("SELECT 1 FROM routines WHERE name=?", (name,)):
            self.db.execute("DELETE FROM routines WHERE name=?", (name,))
            return True
        return False


class ChainRepo:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r) -> Chain:
        return Chain(name=r["name"], tokens=json.loads(r["tokens_json"]), aliases=json.loads(r["aliases_json"]),
                     description=r["description"], updated_at=r["updated_at"])

    def list(self) -> list:
        return [self._row(r) for r in self.db.query("SELECT * FROM command_chains ORDER BY name")]

    def get(self, name: str) -> Optional[Chain]:
        rows = self.db.query("SELECT * FROM command_chains WHERE name=?", (name,))
        return self._row(rows[0]) if rows else None

    def _save(self, c: Chain) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO command_chains(name,tokens_json,aliases_json,description,updated_at)"
            " VALUES(?,?,?,?,?)",
            (c.name, json.dumps(c.tokens), json.dumps(c.aliases), c.description, time.time()))

    @staticmethod
    def validate_tokens(tokens: Any) -> list:
        if not isinstance(tokens, list) or not (1 <= len(tokens) <= 32):
            raise ValidationError("tokens must be a list of 1..32 words")
        out = [str(t).strip().lower() for t in tokens]
        bad = [t for t in out if t not in cfg.KNOWN_TOKENS]
        if bad:
            raise ValidationError(f"unknown tokens {bad}; allowed: {list(cfg.KNOWN_TOKENS)}")
        return out

    def upsert(self, name: str, tokens: Any, aliases: Any = None, description: str = "") -> Chain:
        _check_name(name, "chain")
        if name in cfg.KNOWN_TOKENS:
            raise ValidationError(f"chain name '{name}' collides with a built-in token")
        tokens = self.validate_tokens(tokens)
        aliases = [str(a).strip().lower() for a in (aliases or [])]
        for a in aliases:
            if not WORD_RE.match(a):
                raise ValidationError(f"alias {a!r} must be a single lowercase word (a-z, 2-24 chars)")
            if a in cfg.KNOWN_TOKENS or a in [w for w in ("stop",)]:
                raise ValidationError(f"alias '{a}' collides with a built-in word")
        taken = {}
        for c in self.list():
            if c.name == name:
                continue
            for w in [c.name, *c.aliases]:
                taken[w] = c.name
        for w in [name, *aliases]:
            if w in taken:
                raise ValidationError(f"'{w}' is already used by chain '{taken[w]}'")
        chain = Chain(name, tokens, sorted(set(aliases)), str(description)[:200])
        self._save(chain)
        return self.get(name)  # type: ignore[return-value]

    def delete(self, name: str) -> bool:
        if self.db.query("SELECT 1 FROM command_chains WHERE name=?", (name,)):
            self.db.execute("DELETE FROM command_chains WHERE name=?", (name,))
            return True
        return False

    def resolve_word(self, word: str) -> Optional[Chain]:
        """Voice / accessibility lookup: chain name or alias -> chain."""
        word = word.strip().lower()
        for c in self.list():
            if word == c.name or word in c.aliases:
                return c
        return None

    def voice_words(self) -> set:
        words: set = set()
        for c in self.list():
            for w in [c.name, *c.aliases]:
                if WORD_RE.match(w):
                    words.add(w)
        return words


class EventRepo:
    def __init__(self, db: Database):
        self.db = db
        self._n = 0

    def log(self, source: str, kind: str, message: str = "", payload: Optional[dict] = None,
            level: str = "INFO") -> int:
        cur = self.db.execute(
            "INSERT INTO event_logs(ts,source,kind,level,message,payload_json) VALUES(?,?,?,?,?,?)",
            (time.time(), source, kind, level, message[:500],
             json.dumps(payload, default=str) if payload is not None else None))
        self._n += 1
        if self._n % 500 == 0:
            self.db.execute("DELETE FROM event_logs WHERE id <= (SELECT MAX(id) FROM event_logs) - ?",
                            (self.db.max_events,))
        return cur.lastrowid

    def recent(self, limit: int = 50, source: Optional[str] = None, before_id: Optional[int] = None) -> list:
        limit = max(1, min(int(limit), 500))
        sql, args = "SELECT * FROM event_logs WHERE 1=1", []
        if source:
            sql += " AND source=?"
            args.append(source)
        if before_id:
            sql += " AND id<?"
            args.append(int(before_id))
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.db.query(sql, args):
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json")) if d.get("payload_json") else None
            out.append(d)
        return out

    def count(self) -> int:
        return self.db.query("SELECT COUNT(*) AS n FROM event_logs")[0]["n"]
