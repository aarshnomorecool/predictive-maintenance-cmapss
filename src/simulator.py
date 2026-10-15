"""A virtual turbofan that streams new cycles live, for demonstrations.

The trained models only understand readings that behave like C-MAPSS data,
so the simulator is built from statistics measured on the real training
engines rather than invented numbers:

* healthy level of every channel (mean while RUL sits at the 125 ceiling)
* failure level (mean over the last 10 cycles before failure)
* end-of-life level (the final recorded cycle)
* noise within one engine, and the spread between engines

One hidden number drives the engine: ``damage``, on the same scale as the
dashboard's sensor drift, 0 for a healthy engine and 1 for one that reads
like it is in its last 10 cycles. Every sensor reads

    healthy level + engine offset + damage x (failure level - healthy level) + noise

Damage grows exponentially, slowly at first and then faster, which is the
shape degradation takes in C-MAPSS. FD001 has one fault mode, high-pressure
compressor (HPC) degradation, so "inject fault" makes that same degradation
progress faster; it does not invent a fault type the models never saw.

Because the wear rate is known, the simulator also knows the engine's true
remaining life at the current rate. The dashboard shows it as "projected life
left" so the LSTM's prediction can be compared against a real answer.

    python -m src.simulator     # checks the trained models read simulated
                                # engines the way they read real ones
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.data_pipeline import add_train_rul, load_train

# Growth law: damage += rate * (OFFSET + damage) per cycle. OFFSET keeps a
# brand-new engine (damage 0) degrading at all.
GROWTH_OFFSET = 0.05

# A fresh engine at the normal rate reaches end of life after about this many
# cycles. Mean FD001 training life is 206.
NORMAL_LIFE = 200

WEAR_PRESETS = {"normal": 1.0, "fast": 2.0, "severe": 4.0}

# "Inject HPC fault": degradation progresses this many times faster.
FAULT_MULTIPLIER = 3.0

SPEEDS = (1, 2, 5, 10)     # cycles per second

# Initial wear of a new engine, drawn uniformly. C-MAPSS engines also start
# with unknown, differing amounts of wear.
INITIAL_DAMAGE = (0.0, 0.08)


# --------------------------------------------------------------- profile ----

@dataclass
class Profile:
    """Per-channel statistics measured on the training engines."""

    columns: list[str]
    healthy: dict[str, float]
    failure: dict[str, float]
    within_std: dict[str, float]      # noise inside one engine, healthy phase
    between_std: dict[str, float]     # spread of healthy level across engines
    low: dict[str, float]             # observed range, for manual overrides
    high: dict[str, float]
    end_damage: float                 # damage at the final recorded cycle

    def damage_coeff(self, column: str) -> float:
        return self.failure[column] - self.healthy[column]


def measure_profile(feature_cols: list[str], subset: str = config.SUBSET) -> Profile:
    train = add_train_rul(load_train(subset), clip=10_000)        # unclipped RUL
    columns = config.SETTING_COLS + config.SENSOR_COLS
    healthy = train[train[config.RUL_COL] >= config.RUL_CLIP]
    failing = train[train[config.RUL_COL] < 10]
    end = train[train[config.RUL_COL] == 0]

    within = healthy.groupby(config.UNIT_COL)[columns].std().mean()
    between = healthy.groupby(config.UNIT_COL)[columns].mean().std()

    profile = Profile(
        columns=columns,
        healthy=healthy[columns].mean().to_dict(),
        failure=failing[columns].mean().to_dict(),
        within_std=within.fillna(0).to_dict(),
        between_std=between.fillna(0).to_dict(),
        low=train[columns].min().to_dict(),
        high=train[columns].max().to_dict(),
        end_damage=1.0,
    )
    # Only channels that move with wear carry damage; the rest stay at their
    # healthy level. Settings never degrade.
    moving = [c for c in feature_cols if c in config.SENSOR_COLS
              and abs(profile.damage_coeff(c)) > 1e-9]
    for column in columns:
        if column not in moving:
            profile.failure[column] = profile.healthy[column]
    # End of life on the damage scale, averaged over the channels that move
    ratios = [(end[c].mean() - profile.healthy[c]) / profile.damage_coeff(c) for c in moving]
    profile.end_damage = float(np.mean(ratios))
    return profile


def normal_rate(profile: Profile) -> float:
    """Rate at which a fresh engine reaches end of life in NORMAL_LIFE cycles."""
    return math.log((profile.end_damage + GROWTH_OFFSET) / GROWTH_OFFSET) / NORMAL_LIFE


def cycles_to_end(damage: float, rate: float, profile: Profile) -> float:
    """Closed form of the growth law: cycles until damage reaches end of life."""
    if damage >= profile.end_damage:
        return 0.0
    return math.log((profile.end_damage + GROWTH_OFFSET) / (damage + GROWTH_OFFSET)) / rate


# ----------------------------------------------------------- generation ----

@dataclass
class EngineState:
    """One simulated engine. Pure data generation, no models involved."""

    profile: Profile
    seed: int
    wear: str = "normal"
    fault: bool = False
    overrides: dict[str, float] = field(default_factory=dict)
    damage: float = 0.0
    cycle: int = 0
    rows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.damage = float(self.rng.uniform(*INITIAL_DAMAGE))
        # This engine's own manufacturing offset, fixed for its whole life
        self.offset = {c: float(self.rng.normal(0, self.profile.between_std[c]))
                       for c in self.profile.columns}
        self.base_rate = normal_rate(self.profile)

    @property
    def rate(self) -> float:
        return self.base_rate * WEAR_PRESETS[self.wear] * (FAULT_MULTIPLIER if self.fault else 1.0)

    @property
    def failed(self) -> bool:
        return self.damage >= self.profile.end_damage

    def projected_life(self) -> float:
        return cycles_to_end(self.damage, self.rate, self.profile)

    def step(self) -> dict:
        """Advance one cycle and return the new raw row."""
        if self.cycle > 0:
            self.damage = min(self.damage + self.rate * (GROWTH_OFFSET + self.damage),
                              self.profile.end_damage)
        self.cycle += 1
        p = self.profile
        row = {config.UNIT_COL: 0, config.CYCLE_COL: self.cycle}
        for c in p.columns:
            noise = float(self.rng.normal(0, p.within_std[c]))
            if c in self.overrides:
                row[c] = self.overrides[c] + noise
            else:
                row[c] = (p.healthy[c] + self.offset[c]
                          + self.damage * p.damage_coeff(c) + noise)
        row["damage"] = self.damage
        row["projected_life"] = self.projected_life()
        self.rows.append(row)
        return row

    def note(self, text: str) -> None:
        self.events.append({"cycle": self.cycle, "text": text})


def simulate_run(profile: Profile, seed: int, wear: str = "normal",
                 fault_at: int | None = None) -> pd.DataFrame:
    """A whole engine life at once, for offline validation."""
    engine = EngineState(profile, seed=seed, wear=wear)
    while not engine.failed:
        if fault_at is not None and engine.cycle == fault_at:
            engine.fault = True
        engine.step()
    return pd.DataFrame(engine.rows)


# -------------------------------------------------------------- scoring ----

class Scorer:
    """Scores the newest cycle with every trained model, one cycle at a time."""

    def __init__(self, artifacts, rul):
        from src.inference import PROBABILITY_COLUMNS
        self.artifacts = artifacts
        self.rul = rul
        self.prob_cols = PROBABILITY_COLUMNS
        # Single-row prediction is faster without a process pool; copy so the
        # dashboard's batch scoring keeps all cores.
        self.models = {}
        for name, model in artifacts.models.items():
            model = copy.copy(model)
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            self.models[name] = model
        self.lstm = None
        if rul is not None:
            # Calling a Keras model directly runs eagerly, about 100 ms for one
            # window. Compiling it to a graph with a fixed input shape makes a
            # single prediction a few milliseconds.
            import tensorflow as tf
            spec = tf.TensorSpec((1, rul.window_size, len(rul.feature_cols)), tf.float32)
            self.lstm = tf.function(lambda x: rul.model(x, training=False),
                                    input_signature=[spec], reduce_retracing=True)

    def score(self, history: pd.DataFrame) -> dict:
        from src.preprocessing import apply_scaler, make_windows_every_cycle
        last = history.tail(1)
        X = apply_scaler(last, self.artifacts.scaler, self.artifacts.feature_cols)[
            self.artifacts.feature_cols].to_numpy(dtype=np.float32)
        out = {self.prob_cols[name]: float(model.predict_proba(X)[0, 1])
               for name, model in self.models.items()}
        out["predicted_rul"] = None
        if self.rul is not None:
            window_rows = history.tail(self.rul.window_size)
            scaled = apply_scaler(window_rows, self.rul.scaler, self.rul.feature_cols)
            W, _ = make_windows_every_cycle(scaled, self.rul.feature_cols, self.rul.window_size)
            raw = float(np.asarray(self.lstm(W[-1:]))[0, 0])
            ceiling = float(self.rul.metadata["rul_clip"])
            out["predicted_rul"] = float(np.clip(raw * ceiling, 0, ceiling))
        return out


# --------------------------------------------------------------- runner ----

class SimulationRunner:
    """Owns the live engine and a background thread that advances it.

    Every cycle is scored the moment it is generated and handed to
    ``on_cycle`` (the alert manager), so alerts fire in real time.
    """

    def __init__(self, on_cycle=None):
        self.on_cycle = on_cycle or (lambda runner, scores: None)
        self.lock = threading.RLock()
        self.engine: EngineState | None = None
        self.scorer: Scorer | None = None
        self.profile: Profile | None = None
        self.scores: list[dict] = []
        self.running = False
        self.speed = 2
        self.demo: dict | None = None
        self.version = 0                  # bumps on every change, for clients
        self.thread = threading.Thread(target=self._loop, daemon=True, name="simulation")
        self.thread.start()

    def bind(self, artifacts, rul, profile: Profile) -> None:
        """Attach freshly loaded models. Called on start-up and after retraining."""
        with self.lock:
            had_engine = self.engine is not None
            self.scorer = Scorer(artifacts, rul)
            self.profile = profile
            self.reset()
            if had_engine:
                self.engine.note("Models were retrained, so the simulation restarted")

    # ----------------------------------------------------------- controls --

    def reset(self, seed: int | None = None) -> None:
        with self.lock:
            self.running = False
            self.demo = None
            seed = int(time.time() * 1000) % 2**31 if seed is None else seed
            self.engine = EngineState(self.profile, seed=seed)
            self.scores = []
            self.engine.note("New engine fitted, simulation reset")
            self._advance()                 # one cycle so there is always data
            self.version += 1

    def start(self) -> None:
        with self.lock:
            if self.engine.failed:
                self.reset()
            self.running = True
            self.version += 1

    def pause(self) -> None:
        with self.lock:
            self.running = False
            self.version += 1

    def set_speed(self, speed: int) -> None:
        with self.lock:
            if speed not in SPEEDS:
                raise ValueError(f"speed must be one of {SPEEDS}")
            self.speed = speed
            self.version += 1

    def set_wear(self, wear: str) -> None:
        with self.lock:
            if wear not in WEAR_PRESETS:
                raise ValueError(f"wear must be one of {list(WEAR_PRESETS)}")
            if wear != self.engine.wear:
                self.engine.wear = wear
                self.engine.note(f"Wear rate set to {wear}")
            self.version += 1

    def set_fault(self, active: bool) -> None:
        with self.lock:
            if active != self.engine.fault:
                self.engine.fault = active
                self.engine.note("HPC fault injected: degradation now 3x faster" if active
                                 else "HPC fault cleared")
            self.version += 1

    def set_override(self, column: str, value: float | None) -> None:
        with self.lock:
            if column not in self.profile.columns:
                raise ValueError(f"unknown channel {column}")
            if value is None:
                if self.engine.overrides.pop(column, None) is not None:
                    self.engine.note(f"{column} back to simulated values")
            else:
                self.engine.overrides[column] = float(value)
                self.engine.note(f"{column} set by hand to {value:g}")
            self.version += 1

    def run_demo(self) -> None:
        """Scripted scenario for presentations: healthy running, then an HPC
        fault, then the alert, then end of life. Fixed seed, so it plays out
        the same way every time."""
        with self.lock:
            # About a minute in all: a long healthy stretch to talk over, the
            # fault, the alert roughly 45 seconds in, then end of life.
            self.reset(seed=2024)
            self.speed = 2
            self.demo = {"fault_at": 75}
            self.engine.note("Demo: a healthy engine starts its service life")
            self.running = True
            self.version += 1

    # ---------------------------------------------------------- internals --

    def _advance(self) -> None:
        engine = self.engine
        if self.demo and engine.cycle == self.demo["fault_at"] and not engine.fault:
            engine.fault = True
            engine.note("Demo: HPC fault begins, degradation now 3x faster")
        engine.step()
        history = pd.DataFrame(engine.rows)
        scores = self.scorer.score(history)
        self.scores.append(scores)
        if engine.failed:
            self.running = False
            engine.note(f"Engine reached end of life at cycle {engine.cycle}")
        self.version += 1
        try:
            self.on_cycle(self, scores)
        except Exception as exc:             # an alert problem must not stop the engine
            engine.note(f"Alert check failed: {exc}")

    def _loop(self) -> None:
        """Advance on a fixed clock. Sleeping a fixed time *after* each step
        would make the real speed depend on how long scoring took."""
        next_tick = time.perf_counter()
        while True:
            with self.lock:
                running, speed = self.running and self.engine is not None, self.speed
            if not running:
                time.sleep(0.1)
                next_tick = time.perf_counter()
                continue
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.1))
                continue
            with self.lock:
                if self.running:
                    self._advance()
            # If a step overran, start counting again from now rather than
            # racing through a backlog of missed cycles
            next_tick = max(next_tick + 1.0 / speed, time.perf_counter() - 1.0 / speed)

    # ------------------------------------------------------------ reading --

    def snapshot(self) -> tuple[pd.DataFrame, list[dict], dict]:
        """History with scores attached, the event log, and status, copied
        under the lock so the thread can keep running."""
        with self.lock:
            e = self.engine
            history = pd.DataFrame(e.rows)
            for key in ("prob_rf", "prob_xgb", "predicted_rul"):
                history[key] = [s.get(key) for s in self.scores]
            status = {
                "running": self.running,
                "speed": self.speed,
                "speeds": list(SPEEDS),
                "wear": e.wear,
                "wear_presets": list(WEAR_PRESETS),
                "fault": e.fault,
                "failed": e.failed,
                "demo": self.demo is not None,
                "cycle": e.cycle,
                "damage": round(e.damage, 4),
                "end_damage": round(self.profile.end_damage, 4),
                "projected_life": round(e.projected_life(), 1),
                "overrides": dict(e.overrides),
                "version": self.version,
            }
            return history, list(e.events), status


# ------------------------------------------------------------------- cli ----

def main() -> None:
    """Do the trained models read simulated engines like real ones?

    Simulates engines at every wear setting, scores every cycle in batch, and
    compares against the simulator's known remaining life.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.metrics import roc_auc_score
    from src.inference import load_artifacts, load_rul_artifacts, predict_rul_every_cycle, score_frame

    artifacts = load_artifacts()
    rul = load_rul_artifacts()
    profile = measure_profile(artifacts.feature_cols, artifacts.subset)
    print(f"\nEnd of life on the damage scale: {profile.end_damage:.2f} "
          f"(1.0 = reads like the last 10 cycles)")

    runs = []
    plan = [("normal", None)] * 12 + [("fast", None)] * 6 + [("normal", 60)] * 6
    for k, (wear, fault_at) in enumerate(plan):
        run = simulate_run(profile, seed=1000 + k, wear=wear, fault_at=fault_at)
        run[config.UNIT_COL] = k + 1
        runs.append(run)
    sim = pd.concat(runs, ignore_index=True)

    scored = score_frame(artifacts, sim)
    pred = predict_rul_every_cycle(rul, sim)
    truth = np.minimum(sim["projected_life"].to_numpy(), config.RUL_CLIP)
    lstm = pred["predicted_rul"].to_numpy()
    failing = truth <= config.FAILURE_THRESHOLD

    lives = sim.groupby(config.UNIT_COL)[config.CYCLE_COL].max()
    print(f"Simulated {len(plan)} engines, {len(sim):,} cycles, "
          f"lives {lives.min()} to {lives.max()} cycles")
    print(f"\n  LSTM remaining life   RMSE {np.sqrt(np.mean((lstm - truth) ** 2)):6.2f} cycles "
          f"(real test engines: {rul.metadata['metrics']['LSTM']['test']['rmse']:.2f})")
    for name, col in (("XGBoost", "prob_xgb"), ("Random Forest", "prob_rf")):
        auc = roc_auc_score(failing, scored[col])
        print(f"  {name:<14} failure ROC-AUC {auc:.4f}")

    # How much warning does an alert at 50% XGBoost risk give?
    lead = []
    for unit, g in scored.assign(truth=truth).groupby(config.UNIT_COL):
        hit = g[g["prob_xgb"] >= 0.5]
        lead.append(float(hit["truth"].iloc[0]) if len(hit) else 0.0)
    print(f"\n  Alert at 50% XGBoost risk fires with a median of {np.median(lead):.0f} "
          f"projected cycles left (range {min(lead):.0f} to {max(lead):.0f})")


if __name__ == "__main__":
    main()
