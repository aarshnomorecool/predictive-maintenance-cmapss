"""The findings Bob works from, and the deterministic plan derived from them.

A Findings object is a snapshot of what the system already concluded about one
unit at one moment: its status (from the alert rule, not from Bob), failure
risk, remaining life with its likely range, the sensors pushing the risk up
(SHAP), each sensor's drift, the sensor guard's flags and the unfamiliar-wear
flag. The Plan is what a work order may contain: its priority, the suspected
components (looked up in the profile from the driving sensors), and the
checklist steps allowed for each. Neither depends on a language model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.profiles import INSTRUMENT_PREFIX, Profile

DRIFTING = 0.45            # same band as the dashboard's "drifting" colour
URGENT_LIFE = 15           # predicted life at which the workshop planner calls an engine due
MAX_COMPONENTS = 3
SHARE_OF_TOP = 0.10        # a component must carry this share of the top component's push


@dataclass
class Findings:
    machine: str
    unit: str
    cycle: int | None
    status: str                          # "alert" | "watch" | "normal", from the system's alert rule
    probability: float
    threshold: float
    model: str
    rul: float | None = None
    rul_range: tuple[float, float] | None = None
    shap: dict[str, float] = field(default_factory=dict)       # sensor -> SHAP contribution
    drift: dict[str, float] = field(default_factory=dict)      # sensor -> drift (0 healthy, 1 failure level)
    flagged: list[str] = field(default_factory=list)           # sensors the guard (or a technician) flagged
    unfamiliar: bool = False
    fault_mode: str | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["rul_range"] = list(self.rul_range) if self.rul_range else None
        return d


def status_from_rule(probability: float, threshold: float, rul: float | None, failure_window: float | None,
                     drift: dict[str, float] | None = None) -> str:
    """The system's alert rule, as the dashboard applies it to an engine's latest cycle."""
    if probability >= threshold or (rul is not None and failure_window is not None and rul <= failure_window):
        return "alert"
    if probability >= threshold / 2 or any(v >= DRIFTING for v in (drift or {}).values()):
        return "watch"
    return "normal"


@dataclass
class Plan:
    priority: str
    components: list[str]                 # suspected components, most likely first (instrument:* first)
    steps: dict[str, list[dict]]          # component id -> allowed checklist items
    drivers: list[str]                    # sensors behind the suspicion, strongest first
    notes: list[str]


def priority_of(f: Findings) -> str:
    if f.status == "alert" and f.rul is not None and f.rul <= URGENT_LIFE:
        return "Urgent"
    if f.status == "alert" or f.unfamiliar:
        return "High"
    if f.status == "watch" or f.flagged:
        return "Medium"
    return "Routine"


def make_plan(f: Findings, profile: Profile) -> Plan:
    flagged = [s for s in f.flagged if profile.sensor(s)]
    # sensors pushing the risk up, excluding the ones the guard says are broken
    push = {s: v for s, v in f.shap.items() if v > 0 and s not in flagged and profile.sensor(s)}
    by_component: dict[str, float] = {}
    for s, v in push.items():
        if f.drift and f.drift.get(s, 1.0) < DRIFTING and f.status != "alert":
            continue
        c = profile.component_of(s)
        if c:
            by_component[c] = by_component.get(c, 0.0) + v
    ranked = sorted(by_component, key=by_component.get, reverse=True)
    if ranked:
        top = by_component[ranked[0]]
        ranked = [c for c in ranked if by_component[c] >= SHARE_OF_TOP * top][:MAX_COMPONENTS]
    if f.status == "normal" and not f.unfamiliar:
        ranked = []
    instruments = [f"{INSTRUMENT_PREFIX}{s}" for s in flagged]
    components = instruments + ranked
    steps = {}
    for k, c in enumerate(ranked):
        items = profile.checklist(c)
        steps[c] = items if k == 0 else items[:2]
    for c in instruments:
        steps[c] = profile.checklist(c)
    drivers = sorted(push, key=push.get, reverse=True)
    notes = []
    if f.unfamiliar:
        notes.append("The wear pattern is unfamiliar to the models: do not plan on the remaining-life estimate, "
                     "and inspect the whole unit.")
    if flagged:
        notes.append("A suspected faulty sensor is listed first: check the sensor before replacing any part, "
                     "because a second sensor disagreeing too would point to unfamiliar wear instead.")
    if not components:
        notes.append("No component action is needed now; continue routine monitoring.")
    return Plan(priority=priority_of(f), components=components, steps=steps, drivers=drivers, notes=notes)
