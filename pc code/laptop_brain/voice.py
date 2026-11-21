"""Single-word voice commands.

Layers, from audio to action:

1. A backend produces ``(word, confidence)`` pairs:
   * ``vosk``     -- offline recognizer constrained to a strict grammar (allowlist + ``[unk]``)
   * ``keyboard`` -- type a word + Enter in the server console (testing without a mic)
   * ``mock``     -- no audio at all; words arrive only via ``inject()`` (WebSocket INJECT_VOICE)
2. ``VoiceGate`` decides whether a word is allowed to trigger anything: allowlist, confidence
   threshold, debounce cooldown, and push-to-talk window or wake word.  It is pure logic with an
   injectable clock, so it is unit-tested without audio.
3. ``VoiceListener`` ties it together and hands ``VoiceEvent`` objects to the state machine.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .config import VoiceSettings

log = logging.getLogger("voice")


@dataclass
class VoiceEvent:
    word: str
    confidence: float
    accepted: bool
    reason: str
    ts: float

    def to_dict(self) -> dict:
        return {"word": self.word, "confidence": round(self.confidence, 3), "accepted": self.accepted,
                "reason": self.reason, "ts": self.ts}


class VoiceGate:
    def __init__(self, settings: VoiceSettings, clock: Callable[[], float] = time.monotonic):
        self.s, self._clock = settings, clock
        self.allow: set = set(w.lower() for w in settings.allowlist)
        self.stop_words: set = set(w.lower() for w in settings.stop_words)
        self._last_accept = -1e9
        self._window_until = 0.0
        self._wake_until = 0.0

    def set_allow(self, words: Iterable[str]) -> None:
        self.allow = {w.lower() for w in words}

    def open_window(self, seconds: Optional[float] = None) -> float:
        seconds = seconds or self.s.listen_window_s
        self._window_until = self._clock() + seconds
        return seconds

    def is_listening(self) -> bool:
        """Should audio be processed right now?  (Wake-word mode must always listen for the wake word.)"""
        return bool(self.s.wake_word) or self.s.mode == "always" or self._clock() < self._window_until

    def evaluate(self, word: str, conf: float) -> tuple:
        now = self._clock()
        w = (word or "").strip().lower()
        if not w or w == "[unk]":
            return False, "unknown"
        if w in self.stop_words:                       # stopping must always be easy: skips cooldown + window
            if conf < self.s.min_confidence * 0.75:
                return False, "low_confidence"
            self._last_accept = now
            return True, "ok"
        wake = self.s.wake_word.lower()
        if wake and w == wake:
            self._wake_until = now + self.s.wake_window_s
            return False, "wake_armed"
        if w not in self.allow:
            return False, "not_in_allowlist"
        if conf < self.s.min_confidence:
            return False, "low_confidence"
        if wake:
            if now > self._wake_until:
                return False, "wake_required"
        elif self.s.mode == "ptt" and now > self._window_until:
            return False, "not_listening"
        if now - self._last_accept < self.s.cooldown_s:
            return False, "cooldown"
        self._last_accept = now
        self._window_until = 0.0                       # push-to-talk: one command per press
        self._wake_until = 0.0
        return True, "ok"


class VoiceListener:
    def __init__(self, settings: VoiceSettings, loop, on_event: Optional[Callable[[VoiceEvent], None]] = None):
        self.s, self._loop, self.on_event = settings, loop, on_event
        self.gate = VoiceGate(settings)
        self.backend_name = "mock"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._grammar_dirty = threading.Event()

    # -- control --------------------------------------------------------------- #
    def start(self, sim: bool = False) -> None:
        choice = self.s.backend
        if choice == "auto":
            choice = "mock" if sim else "vosk"
        if choice == "vosk":
            problem = self._vosk_problem()
            if problem:
                log.warning("voice recognition disabled (%s); use INJECT_VOICE / keyboard instead", problem)
                choice = "mock"
        self.backend_name = choice
        target = {"vosk": self._run_vosk, "keyboard": self._run_keyboard}.get(choice)
        if target:
            self._stop.clear()
            self._thread = threading.Thread(target=target, name=f"voice-{choice}", daemon=True)
            self._thread.start()
        log.info("voice backend: %s (mode=%s)", self.backend_name, self.s.mode)

    def stop(self) -> None:
        self._stop.set()

    def set_allow(self, words: Iterable[str]) -> None:
        words = set(words)
        if self.s.wake_word:
            words.add(self.s.wake_word.lower())
        self.gate.set_allow(words)
        self._grammar_dirty.set()

    def trigger_listen(self, seconds: Optional[float] = None) -> float:
        return self.gate.open_window(seconds)

    def inject(self, word: str, confidence: float = 1.0) -> VoiceEvent:
        """Feed a word through the real gate (used by tests, the keyboard backend and the app)."""
        accepted, reason = self.gate.evaluate(word, confidence)
        ev = VoiceEvent(word.strip().lower(), float(confidence), accepted, reason, time.time())
        if self.on_event:
            self.on_event(ev)
        return ev

    def _emit_threadsafe(self, word: str, conf: float) -> None:
        self._loop.call_soon_threadsafe(self.inject, word, conf)

    # -- vosk ------------------------------------------------------------------- #
    def _vosk_problem(self) -> Optional[str]:
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except ImportError as exc:
            return f"missing package: {exc.name}"
        if not os.path.isdir(self.s.model_path):
            return f"model not found at {self.s.model_path}"
        return None

    def _grammar(self) -> str:
        return json.dumps(sorted(self.gate.allow) + ["[unk]"])

    def _run_vosk(self) -> None:
        import sounddevice as sd
        import vosk
        vosk.SetLogLevel(-1)
        model = vosk.Model(self.s.model_path)

        def new_rec():
            rec = vosk.KaldiRecognizer(model, self.s.sample_rate, self._grammar())
            rec.SetWords(True)
            return rec

        rec = new_rec()
        audio: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        def callback(indata, frames, t, status):
            try:
                audio.put_nowait(bytes(indata))
            except queue.Full:
                pass

        was_listening = True
        try:
            with sd.RawInputStream(samplerate=self.s.sample_rate, blocksize=4000, device=self.s.device,
                                   dtype="int16", channels=1, callback=callback):
                while not self._stop.is_set():
                    if self._grammar_dirty.is_set():
                        self._grammar_dirty.clear()
                        rec = new_rec()
                    try:
                        data = audio.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    listening = self.gate.is_listening()
                    if not listening:
                        was_listening = False
                        continue
                    if not was_listening:                 # window just opened: start from clean state
                        rec.Reset()
                        was_listening = True
                    if rec.AcceptWaveform(data):
                        for item in json.loads(rec.Result()).get("result", []):
                            self._emit_threadsafe(item["word"], float(item.get("conf", 1.0)))
        except Exception:
            log.exception("vosk/audio thread crashed; voice commands now only via inject")

    # -- keyboard --------------------------------------------------------------- #
    def _run_keyboard(self) -> None:
        log.info("keyboard voice: type a word + Enter in this console")
        for line in sys.stdin:
            if self._stop.is_set():
                break
            word = line.strip()
            if word:
                self._emit_threadsafe(word, 1.0)
