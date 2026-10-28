"""Sensor integrity guard: is it the engine that is failing, or the sensor?

Predictive maintenance models trust their sensors. A sensor that sticks,
freezes or drifts looks to the model exactly like engine wear, so the model
raises a false alarm, or a frozen sensor hides wear that is really happening.

In C-MAPSS FD001 every informative sensor is driven by the same degradation
(one fault mode, HPC wear). Each sensor's drift, its position between its
healthy and failure levels (src.sensor_health), is therefore an independent
estimate of the same quantity. A healthy sensor agrees with the others; a
broken one does not. For every sensor and cycle:

    residual = drift(sensor) - median drift of the other sensors

scaled by how much that sensor naturally disagrees on the 100 training
engines (a robust standard deviation, so outliers in training do not inflate
it). A sensor is flagged when its score stays above SUSPECT_Z for
PERSIST_CYCLES cycles in a row, and released when it falls back below
RELEASE_Z. The median of the *other* sensors is used so that one broken
sensor cannot drag the reference along with it.

A flagged sensor's readings are rebuilt from the consensus of the healthy
sensors (healthy level + consensus drift x its healthy-to-failure range) and
the models are scored again on the repaired data, so a broken instrument
does not turn into a false failure alarm.

This is analytical redundancy, a long-standing idea in aerospace fault
detection, applied to an ML prognostics pipeline and measured:

    python -m src.sensor_integrity     # injects sensor faults into real test
                                       # engines and reports detection rates
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.sensor_health import Calibration, score_history

# Thresholds chosen on the training engines (see README): at 8 robust standard
# deviations no clean training or test engine is ever flagged.
SUSPECT_Z = 8.0
RELEASE_Z = 4.0
PERSIST_CYCLES = 3
# When this many sensors disagree at the same cycle, the engine is not showing
# broken sensors but a wear pattern the reference does not know (see
# src.unfamiliar_wear). Individual sensors are then not flagged or rebuilt.
# Two, not three: injected single faults never pull a second sensor over the
# line (0 wrong flags), while FD003 fan wear moves P30 and phi together. At
# three, 42 of 44 fan-wear engines first had those two blamed as broken.
PATTERN_SENSORS = 2
# Only sensors whose drift tracks the consensus this closely (robust std,
# drift units) can be cross-checked. In FD001 that is 10 of the 14 informative
# sensors; the four shaft-speed sensors (Nf, Nc, NRf, NRc) follow wear too
# loosely, so a fault on them would hide inside their natural scatter.
GUARD_SCALE = 0.12
# The smoothing window inside sensor_health needs this many cycles to fill;
# before that one noisy reading weighs too much.
WARMUP_CYCLES = 10


@dataclass
class IntegrityModel:
    sensors: list[str]                  # guarded sensors, cross-checked against each other
    unguarded: list[str]                # informative but too loose to cross-check
    center: pd.Series                   # typical residual per sensor on training engines
    scale: pd.Series                    # robust std of the residual per sensor
    calibrations: dict[str, Calibration]

    def public(self) -> dict:
        return {"sensors": self.sensors, "unguarded": self.unguarded,
                "suspect_z": SUSPECT_Z, "release_z": RELEASE_Z,
                "persist_cycles": PERSIST_CYCLES,
                "scale": {c: round(float(self.scale[c]), 4) for c in self.sensors}}


def _residuals(drift: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    D = drift[sensors].to_numpy(dtype=float)
    res = np.empty_like(D)
    for i in range(len(sensors)):
        res[:, i] = D[:, i] - np.median(np.delete(D, i, axis=1), axis=1)
    return pd.DataFrame(res, columns=sensors, index=drift.index)


def fit_integrity(calibrations: dict[str, Calibration], subset: str = config.SUBSET) -> IntegrityModel:
    """Measure how much each sensor naturally disagrees on the training engines,
    keep the ones that agree closely enough to cross-check, and measure again
    among those alone."""
    from src.data_pipeline import load_train
    informative = [c for c, cal in calibrations.items() if cal.informative]
    train = load_train(subset)
    drift = score_history(train, calibrations).sort_values([config.UNIT_COL, config.CYCLE_COL])
    keep = drift.groupby(config.UNIT_COL).cumcount().to_numpy() >= WARMUP_CYCLES

    def spread(sensors):
        res = _residuals(drift, sensors)[keep]
        center = res.median()
        return center, (res - center).abs().median() * 1.4826

    _, scale_all = spread(informative)
    guarded = [c for c in informative if scale_all[c] < GUARD_SCALE]
    center, scale = spread(guarded)
    return IntegrityModel(sensors=guarded, unguarded=[c for c in informative if c not in guarded],
                          center=center, scale=scale, calibrations=calibrations)


@dataclass
class IntegrityResult:
    drift: pd.DataFrame                 # per-cycle drift, sorted by unit and cycle
    z: pd.DataFrame                     # disagreement score per sensor and cycle
    flags: pd.DataFrame                 # bool: sensor considered faulty at that cycle
    consensus: pd.Series                # median drift of the sensors that are not flagged
    corrected: pd.DataFrame             # raw readings with faulty sensors rebuilt

    def faulty_at_last(self, unit) -> list[str]:
        rows = self.flags[self.drift[config.UNIT_COL].to_numpy() == unit]
        if rows.empty:
            return []
        last = rows.iloc[-1]
        return [c for c in rows.columns if last[c]]


def assess(raw: pd.DataFrame, model: IntegrityModel, forced: list[str] | tuple = ()) -> IntegrityResult:
    """Score, flag and repair every sensor of every engine in ``raw``.

    ``forced``: sensors a technician has marked faulty. They are treated as
    flagged at every cycle, whatever their score, and rebuilt like any other
    faulty sensor. Only guarded sensors can be rebuilt from the others."""
    raw = raw.sort_values([config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)
    drift = score_history(raw, model.calibrations).sort_values(
        [config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)
    sensors = model.sensors
    z = (_residuals(drift, sensors) - model.center) / model.scale
    absz = z.abs().to_numpy()
    units = drift[config.UNIT_COL].to_numpy()
    rank = drift.groupby(config.UNIT_COL).cumcount().to_numpy()

    flags = np.zeros_like(absz, dtype=bool)
    active = np.zeros_like(absz, dtype=bool)        # fault present, including its build-up
    for j in range(len(sensors)):
        run = 0
        flagged = False
        for t in range(len(absz)):
            if t and units[t] != units[t - 1]:
                run, flagged = 0, False
            if rank[t] < WARMUP_CYCLES:
                continue
            run = run + 1 if absz[t, j] > SUSPECT_Z else 0
            if not flagged and run >= PERSIST_CYCLES:
                flagged = True
                # the fault began before the score crossed the threshold:
                # walk back over the build-up while the score was elevated
                k = t
                while k > 0 and units[k - 1] == units[t] and absz[k - 1, j] > RELEASE_Z:
                    k -= 1
                active[k:t, j] = True
            elif flagged and absz[t, j] < RELEASE_Z:
                flagged = False
            flags[t, j] = flagged
            active[t, j] |= flagged
    # Many sensors disagreeing together is an unfamiliar wear pattern, not a set
    # of broken sensors: rebuilding them from a minority would be wrong
    pattern = (absz > SUSPECT_Z).sum(axis=1) >= PATTERN_SENSORS
    flags[pattern] = False
    active[pattern] = False
    for c in forced:
        if c in sensors:
            j = sensors.index(c)
            flags[:, j] = True
            active[:, j] = True
    flags_df = pd.DataFrame(flags, columns=sensors)

    # Consensus from the sensors that are not flagged at each cycle
    D = drift[sensors].to_numpy(dtype=float)
    masked = np.where(flags, np.nan, D)
    all_flagged = flags.all(axis=1)
    masked[all_flagged] = D[all_flagged]           # nothing left to trust: plain median
    consensus = pd.Series(np.nanmedian(masked, axis=1), index=drift.index)

    corrected = raw.copy()
    # some channels (htBleed) are recorded as whole numbers; rebuilt values are not
    corrected[sensors] = corrected[sensors].astype(float)
    for j, c in enumerate(sensors):
        if active[:, j].any():
            cal = model.calibrations[c]
            rebuilt = cal.healthy_mean + consensus.to_numpy() * (cal.failure_mean - cal.healthy_mean)
            corrected.loc[active[:, j], c] = rebuilt[active[:, j]]
    return IntegrityResult(drift=drift, z=z, flags=flags_df, consensus=consensus, corrected=corrected)


# ------------------------------------------------------------ evaluation ----

FAULTS = ("stuck at failure level", "frozen", "bias +4 sigma", "slow drift")


def inject(raw: pd.DataFrame, unit: int, sensor: str, kind: str, onset: int,
           cal: Calibration) -> pd.DataFrame:
    """Standard sensor-fault shapes from the fault-detection literature."""
    out = raw.copy()
    out[sensor] = out[sensor].astype(float)
    m = (out[config.UNIT_COL] == unit) & (out[config.CYCLE_COL] >= onset)
    gap = cal.failure_mean - cal.healthy_mean
    if kind == "stuck at failure level":
        out.loc[m, sensor] = cal.failure_mean
    elif kind == "frozen":
        frozen = out.loc[(out[config.UNIT_COL] == unit) & (out[config.CYCLE_COL] == onset), sensor].iloc[0]
        out.loc[m, sensor] = frozen
    elif kind == "bias +4 sigma":
        out.loc[m, sensor] += 4 * cal.healthy_std * np.sign(gap)
    elif kind == "slow drift":
        steps = (out.loc[m, config.CYCLE_COL] - onset).to_numpy()
        out.loc[m, sensor] += np.minimum(steps / 30.0, 1.0) * gap
    return out


def main() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src.data_pipeline import build_dataset
    from src.inference import load_artifacts, load_rul_artifacts, predict_rul_every_cycle, score_frame
    from src.sensor_health import calibrate

    artifacts = load_artifacts()
    rul = load_rul_artifacts()
    cals = calibrate(artifacts.feature_cols, artifacts.subset)
    model = fit_integrity(cals, artifacts.subset)
    from src.sensor_health import SPEC_BY_COLUMN
    sym = lambda cols: ", ".join(SPEC_BY_COLUMN[c].symbol for c in cols)
    print(f"\nGuarded ({len(model.sensors)}): {sym(model.sensors)}")
    print(f"Not cross-checkable ({len(model.unguarded)}): {sym(model.unguarded)}")
    print(f"Natural disagreement among guarded sensors (robust std, drift units): "
          f"{model.scale.min():.3f} to {model.scale.max():.3f}")

    test = build_dataset(artifacts.subset).test
    raw_cols = [config.UNIT_COL, config.CYCLE_COL, config.RUL_COL] + config.SETTING_COLS + config.SENSOR_COLS
    test = test[raw_cols]

    clean = assess(test, model)
    false_units = clean.drift[clean.flags.any(axis=1).to_numpy()][config.UNIT_COL].nunique()
    print(f"Clean test engines with any sensor flagged at any cycle: {false_units} of 100")

    lengths = test.groupby(config.UNIT_COL)[config.CYCLE_COL].max()
    truth_end = test.groupby(config.UNIT_COL)[config.RUL_COL].last()
    units = [u for u in lengths.index if lengths[u] >= 50]
    print(f"Injecting each fault type into {len(units)} test engines (those with 50+ cycles), "
          f"rotating through the sensors, fault starting 30 cycles before the end\n")

    print(f"  {'fault':<24}{'detected':>10}{'right sensor':>14}{'wrong flags':>13}{'median delay':>14}")
    impact = []
    saved = {"guarded": [SPEC_BY_COLUMN[c].symbol for c in model.sensors],
             "unguarded": [SPEC_BY_COLUMN[c].symbol for c in model.unguarded],
             "clean_test_engines_flagged": int(false_units), "test": [], "run_to_failure": []}
    for kind in FAULTS:
        detected = right = wrong = 0
        delays = []
        for k, unit in enumerate(units):
            sensor = model.sensors[k % len(model.sensors)]
            onset = int(lengths[unit]) - 30
            faulty = inject(test, unit, sensor, kind, onset, cals[sensor])
            one = faulty[faulty[config.UNIT_COL] == unit]
            res = assess(one, model)
            after = res.drift[config.CYCLE_COL].to_numpy() >= onset
            f = res.flags[after]
            if f[sensor].any():
                detected += 1
                delays.append(int(np.argmax(f[sensor].to_numpy())))
            if f.drop(columns=[sensor]).any().any():
                wrong += 1
            if f[sensor].any() and not f.drop(columns=[sensor]).any().any():
                right += 1
            if kind == "stuck at failure level" and truth_end[unit] > 60:
                impact.append((unit, one, res.corrected))
        n = len(units)
        print(f"  {kind:<24}{detected / n:>10.0%}{right / n:>14.0%}{wrong / n:>13.0%}"
              f"{(np.median(delays) if delays else float('nan')):>11.0f} cyc")
        saved["test"].append({"fault": kind, "detected": round(detected / n, 3), "wrong": round(wrong / n, 3),
                              "median_delay": float(np.median(delays)) if delays else None, "engines": n})

    # Engines that really wear out: the 20 validation engines, run to failure,
    # fault starting 60 cycles before the end. A frozen sensor here hides wear.
    import json
    from src.data_pipeline import load_train
    val_units = json.loads((config.MODELS_DIR / "classifier_metadata.json").read_text())["val_units"]
    val = load_train(artifacts.subset)
    val = val[val[config.UNIT_COL].isin(val_units)]
    val_last = val.groupby(config.UNIT_COL)[config.CYCLE_COL].max()
    print(f"\nThe 20 validation engines, run to failure, fault starting 60 cycles before failure\n")
    print(f"  {'fault':<24}{'detected':>10}{'right sensor':>14}{'wrong flags':>13}{'median delay':>14}")
    for kind in FAULTS:
        detected = right = wrong = 0
        delays = []
        for k, unit in enumerate(val_units):
            sensor = model.sensors[k % len(model.sensors)]
            onset = int(val_last[unit]) - 60
            one = inject(val[val[config.UNIT_COL] == unit], unit, sensor, kind, onset, cals[sensor])
            res = assess(one, model)
            f = res.flags[res.drift[config.CYCLE_COL].to_numpy() >= onset]
            hit, other = bool(f[sensor].any()), bool(f.drop(columns=[sensor]).any().any())
            detected += hit
            wrong += other
            right += hit and not other
            if hit:
                delays.append(int(np.argmax(f[sensor].to_numpy())))
        n = len(val_units)
        print(f"  {kind:<24}{detected / n:>10.0%}{right / n:>14.0%}{wrong / n:>13.0%}"
              f"{(np.median(delays) if delays else float('nan')):>11.0f} cyc")
        saved["run_to_failure"].append({"fault": kind, "detected": round(detected / n, 3), "wrong": round(wrong / n, 3),
                                        "median_delay": float(np.median(delays)) if delays else None, "engines": n})

    # What the guard saves: a healthy engine with a sensor stuck at its failure level
    raw_alarm = fixed_alarm = 0
    raw_err, fixed_err = [], []
    for unit, broken, repaired in impact:
        for frame, store in ((broken, "raw"), (repaired, "fixed")):
            p = score_frame(artifacts, frame)["prob_xgb"].iloc[-1]
            r = predict_rul_every_cycle(rul, frame)["predicted_rul"].iloc[-1]
            truth = min(float(truth_end[unit]), config.RUL_CLIP)
            if store == "raw":
                raw_alarm += p >= 0.5
                raw_err.append(abs(r - truth))
            else:
                fixed_alarm += p >= 0.5
                fixed_err.append(abs(r - truth))
    n = len(impact)
    print(f"\nHealthy engines (more than 60 cycles left) with one sensor stuck at its failure level, {n} engines:")
    print(f"  false failure alarms (XGBoost >= 0.5)   without guard {raw_alarm}/{n}   with guard {fixed_alarm}/{n}")
    print(f"  LSTM remaining-life error (MAE)        without guard {np.mean(raw_err):.1f}   "
          f"with guard {np.mean(fixed_err):.1f} cycles")
    saved["stuck_sensor_on_healthy_engines"] = {
        "engines": n, "false_alarms_without": int(raw_alarm), "false_alarms_with": int(fixed_alarm),
        "lstm_mae_without": round(float(np.mean(raw_err)), 1), "lstm_mae_with": round(float(np.mean(fixed_err)), 1)}
    import json
    from src.artifact_io import atomic_write_text
    out = config.MODELS_DIR / "insights" / "integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(saved, indent=1))


if __name__ == "__main__":
    main()
