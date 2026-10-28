"""Inspection outcomes and what each engine's history teaches the alerts.

When an alert sends a technician to an engine, the outcome is recorded:
"fault confirmed" or "no fault found". Most systems throw that answer away.
Here it becomes per-engine memory:

* no fault found at risk r: this engine's alert level rises to r + 0.10
  (at most 0.95) for the next MEMORY_CYCLES cycles, so the same borderline
  reading does not send the technician straight back
* fault confirmed: recorded against the engine; the alert level is unchanged

The raised level only filters borderline risk scores. It never silences an
engine the LSTM puts inside the failure window (30 cycles or fewer): the
alert rule's remaining-life clause is independent of the level.

Every outcome is also a labelled example for retraining, which is what the
"inspect next" suggestions in src.active_inspection are designed to collect.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime

from src import config
from src.artifact_io import atomic_write_text

LOG_FILE = config.MODELS_DIR / "inspections.json"
MEMORY_CYCLES = 30
RAISE_BY = 0.10
MAX_LEVEL = 0.95
OUTCOMES = ("fault_confirmed", "no_fault")


class InspectionLog:
    def __init__(self, path=LOG_FILE):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.entries = []

    def _save(self):
        try:
            atomic_write_text(self.path, json.dumps(self.entries, indent=1))
        except OSError:
            pass

    def record(self, engine, cycle: int, outcome: str, risk: float | None,
               predicted_rul: float | None, note: str = "") -> dict:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        entry = {"engine": str(engine), "cycle": int(cycle), "outcome": outcome,
                 "risk": None if risk is None else round(float(risk), 4),
                 "predicted_rul": None if predicted_rul is None else round(float(predicted_rul), 1),
                 "note": note.strip()[:300], "time": datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            self.entries.append(entry)
            self._save()
        return entry

    def history(self, engine) -> list[dict]:
        with self.lock:
            return [e for e in self.entries if e["engine"] == str(engine)]

    def clear(self, engine) -> None:
        with self.lock:
            self.entries = [e for e in self.entries if e["engine"] != str(engine)]
            self._save()

    def memory(self, engine) -> dict | None:
        """The raised alert level from the most recent no-fault inspection."""
        last = next((e for e in reversed(self.history(engine)) if e["outcome"] == "no_fault"), None)
        if last is None or last["risk"] is None:
            return None
        return {"level": round(min(MAX_LEVEL, last["risk"] + RAISE_BY), 4),
                "from_cycle": last["cycle"], "until_cycle": last["cycle"] + MEMORY_CYCLES,
                "inspected_at_risk": last["risk"]}

    def level_at(self, engine, cycle: int, base: float) -> float:
        m = self.memory(engine)
        if m and m["from_cycle"] <= cycle <= m["until_cycle"]:
            return max(base, m["level"])
        return base
