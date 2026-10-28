"""Machine profiles: one file describes one machine type.

Everything that is specific to a machine (its columns, sensors, the component
each sensor measures, inspection checklists, how remaining life is labelled,
which tasks its data supports and where its trained models live) is kept in
``machines/<id>/profile.json``. Shared code reads the profile instead of
naming turbofan parts, so a pump, a milling machine or a robotic arm with its
own sensors and components goes through the same code.

The turbofan profile is written by hand from the C-MAPSS documentation;
profiles for new machines are written by the onboarding pipeline
(src/onboarding). Baselines are always computed from data:

    python -m src.profiles                       # validate every profile
    python -m src.profiles --refresh turbofan    # recompute turbofan baselines from training data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACHINES_DIR = PROJECT_ROOT / "machines"
DEFAULT_MACHINE = "turbofan"
INSTRUMENT_PREFIX = "instrument:"      # component id Bob uses for a suspected faulty sensor itself


class ProfileError(ValueError):
    pass


@dataclass
class Profile:
    data: dict
    path: Path | None = None
    _sensors: dict = field(default_factory=dict, init=False, repr=False)
    _components: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._sensors = {s["id"]: s for s in self.data.get("sensors", [])}
        self._components = {c["id"]: c for c in self.data.get("components", [])}

    # identity ---------------------------------------------------------
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def unit_label(self) -> str:
        return self.data.get("unit_label", "unit")

    @property
    def time_unit(self) -> str:
        return self.data.get("time_unit", "record")

    @property
    def columns(self) -> dict:
        return self.data["columns"]

    @property
    def tasks(self) -> dict:
        return self.data.get("tasks_enabled", {})

    @property
    def labelling(self) -> dict:
        return self.data.get("labelling", {})

    @property
    def folder(self) -> Path:
        return self.path.parent if self.path else MACHINES_DIR / self.id

    @property
    def models_dir(self) -> Path:
        rel = self.data.get("models", {}).get("dir", "models")
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / rel if self.data.get("models", {}).get("relative_to") == "project"
                                          else self.folder / rel)

    # sensors and components ---------------------------------------------
    @property
    def sensors(self) -> list[dict]:
        return list(self._sensors.values())

    @property
    def components(self) -> list[dict]:
        return list(self._components.values())

    def sensor(self, sensor_id: str) -> dict | None:
        return self._sensors.get(sensor_id)

    def component(self, component_id: str) -> dict | None:
        if component_id.startswith(INSTRUMENT_PREFIX):
            s = self.sensor(component_id[len(INSTRUMENT_PREFIX):])
            return None if s is None else {"id": component_id, "name": f"{s['label']} sensor",
                                           "checklist": self.instrument_checklist}
        return self._components.get(component_id)

    def component_of(self, sensor_id: str) -> str | None:
        s = self.sensor(sensor_id)
        return s.get("component_id") if s else None

    def sensors_of(self, component_id: str) -> list[str]:
        return [s["id"] for s in self.sensors if s.get("component_id") == component_id]

    def checklist(self, component_id: str) -> list[dict]:
        c = self.component(component_id)
        return list(c.get("checklist", [])) if c else []

    @property
    def instrument_checklist(self) -> list[dict]:
        return list(self.data.get("instrument_checklist", DEFAULT_INSTRUMENT_CHECKLIST))

    def label_of(self, sensor_id: str) -> str:
        s = self.sensor(sensor_id)
        return s["label"] if s else sensor_id

    # checks -------------------------------------------------------------
    def problems(self) -> list[str]:
        out = []
        for key in ("id", "name", "columns", "sensors", "components"):
            if key not in self.data:
                out.append(f"missing '{key}'")
        if out:
            return out
        if not self.sensors:
            out.append("no sensors")
        for s in self.sensors:
            if s.get("component_id") not in self._components:
                out.append(f"sensor {s['id']} points at unknown component {s.get('component_id')!r}")
        for c in self.components:
            ids = [i["id"] for i in c.get("checklist", [])]
            if len(ids) != len(set(ids)):
                out.append(f"component {c['id']} has duplicate checklist ids")
        cols = self.columns
        if not cols.get("sensors"):
            out.append("columns.sensors is empty")
        for col in cols.get("sensors", []):
            if col not in self._sensors:
                out.append(f"sensor column {col} has no sensor entry")
        if self.tasks.get("rul") and not (cols.get("unit") and cols.get("time")):
            out.append("RUL is enabled but the unit or time column is missing")
        if self.tasks.get("classification") and not (cols.get("label") or self.tasks.get("rul")):
            out.append("classification is enabled but there is no label column and no RUL to derive one")
        return out

    def validate(self) -> "Profile":
        p = self.problems()
        if p:
            raise ProfileError(f"profile {self.data.get('id', '?')}: " + "; ".join(p))
        return self

    def public(self) -> dict:
        """What the dashboard needs: no file paths."""
        d = {k: v for k, v in self.data.items() if k != "models"}
        d["has_models"] = (self.models_dir / "classifier.joblib").exists() or self.id == DEFAULT_MACHINE
        return d

    def save(self, path: Path | None = None) -> Path:
        from src.artifact_io import atomic_write_text
        path = path or self.path or (MACHINES_DIR / self.id / "profile.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self.data, indent=1, ensure_ascii=False))
        self.path = path
        return path


DEFAULT_INSTRUMENT_CHECKLIST = [
    {"id": "instrument.1", "text": "Check the sensor's connector and wiring for damage, corrosion or moisture"},
    {"id": "instrument.2", "text": "Compare the reading with a portable reference instrument or a redundant channel"},
    {"id": "instrument.3", "text": "Recalibrate or replace the sensor, then confirm that its reading agrees with the other sensors"},
]


def profile_path(machine_id: str) -> Path:
    return MACHINES_DIR / machine_id / "profile.json"


def load_profile(machine_id: str = DEFAULT_MACHINE) -> Profile:
    path = profile_path(machine_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ProfileError(f"no profile for machine {machine_id!r} at {path}") from e
    return Profile(data, path).validate()


def list_profiles() -> list[Profile]:
    out = []
    for p in sorted(MACHINES_DIR.glob("*/profile.json")):
        try:
            out.append(Profile(json.loads(p.read_text(encoding="utf-8")), p))
        except ValueError:
            continue
    # the default machine first
    out.sort(key=lambda pr: (pr.data.get("id") != DEFAULT_MACHINE, pr.data.get("name", "")))
    return out


def refresh_turbofan_baselines() -> None:
    """Healthy level, spread and drift direction of each turbofan sensor, from training data."""
    from src import config
    from src.data_pipeline import add_train_rul, load_train
    prof = load_profile(DEFAULT_MACHINE)
    train = add_train_rul(load_train(config.SUBSET), clip=config.RUL_CLIP)
    healthy = train[train[config.RUL_COL] >= config.RUL_CLIP]
    failing = train[train[config.RUL_COL] < 10]
    for s in prof.data["sensors"]:
        col = s["id"]
        s["baseline_mean"] = round(float(healthy[col].mean()), 6)
        s["baseline_std"] = round(float(healthy[col].std()), 6)
        gap = float(failing[col].mean() - healthy[col].mean())
        s["drift_direction"] = "up" if gap > 0 else "down" if gap < 0 else "flat"
    prof.save()
    print(f"refreshed baselines for {len(prof.sensors)} sensors in {prof.path}")


def main() -> None:
    import sys
    if "--refresh" in sys.argv:
        refresh_turbofan_baselines()
    for p in list_profiles():
        problems = p.problems()
        status = "ok" if not problems else "PROBLEMS: " + "; ".join(problems)
        print(f"{p.data.get('id'):<12} {p.data.get('name')}: {len(p.sensors)} sensors, "
              f"{len(p.components)} components, tasks {[k for k, v in p.tasks.items() if v]} - {status}")


if __name__ == "__main__":
    main()
