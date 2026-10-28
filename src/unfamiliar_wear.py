""""I've never seen this kind of failure": flagging wear the models were not trained on.

The models learned one failure mode, high-pressure compressor (HPC) wear,
from FD001. In HPC wear every sensor moves towards its failure level
together, which is exactly what the sensor integrity guard checks. FD003
contains a second mode, fan degradation, with a very different fingerprint:
P30 and phi move the opposite way, five to six times further, and BPR, W31
and W32 reverse too.

The guard's disagreement scores separate three situations:

* everything agrees ............ familiar wear (or healthy)
* one sensor disagrees ......... a faulty sensor (src.sensor_integrity)
* two or more disagree,
  persistently .................. an unfamiliar wear pattern: the
                                  remaining-life prediction should not be trusted

A single disagreeing sensor can also be the first sign of unfamiliar wear,
before a second sensor joins it; the dashboard says so with every sensor fault.

Nothing from FD003 is used to build the detector. It is calibrated on FD001
alone and evaluated on the 100 FD003 training engines (which ran to failure),
labelled by their end-of-life fingerprint: engines whose pattern matches
FD001's are HPC wear, the rest fan wear.

    python -m src.unfamiliar_wear     # writes models/insights/unfamiliar.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.sensor_integrity import PERSIST_CYCLES, SUSPECT_Z, WARMUP_CYCLES, IntegrityModel, assess

INSIGHTS_DIR = config.MODELS_DIR / "insights"
RESULT_FILE = INSIGHTS_DIR / "unfamiliar.json"

from src.sensor_integrity import PATTERN_SENSORS  # this many disagreeing at once: a pattern, not a fault

FINGERPRINT_GAP = 6.0


def unfamiliar_flags(raw: pd.DataFrame, model: IntegrityModel) -> tuple[pd.Series, pd.Series]:
    """Per cycle: how many guarded sensors disagree, and whether the wear
    pattern is unfamiliar (PATTERN_SENSORS or more, for PERSIST_CYCLES cycles)."""
    return pattern_flags(assess(raw, model))


def pattern_flags(res) -> tuple[pd.Series, pd.Series]:
    """The same, from an existing integrity assessment."""
    count = (res.z.abs() > SUSPECT_Z).sum(axis=1)
    units = res.drift[config.UNIT_COL].to_numpy()
    rank = res.drift.groupby(config.UNIT_COL).cumcount().to_numpy()
    flag = np.zeros(len(count), dtype=bool)
    run = 0
    on = False
    for t in range(len(count)):
        if t and units[t] != units[t - 1]:
            run, on = 0, False
        if rank[t] < WARMUP_CYCLES:
            continue
        run = run + 1 if count.iloc[t] >= PATTERN_SENSORS else 0
        on = on or run >= PERSIST_CYCLES          # once unfamiliar, stays unfamiliar for that engine
        flag[t] = on
    return count.reset_index(drop=True), pd.Series(flag)


def fingerprint_labels(subset: str, calibrations, reference: pd.Series) -> pd.Series:
    """Label each engine by its end-of-life pattern: near the FD001 pattern is
    HPC wear, far from it is another mode."""
    from src.data_pipeline import add_train_rul, load_train
    from src.sensor_health import score_history
    informative = [c for c, cal in calibrations.items() if cal.informative]
    df = add_train_rul(load_train(subset), clip=10_000)
    drift = score_history(df, calibrations).merge(df[[config.UNIT_COL, config.CYCLE_COL, config.RUL_COL]],
                                                  on=[config.UNIT_COL, config.CYCLE_COL])
    end = drift[drift[config.RUL_COL] < 10].groupby(config.UNIT_COL)[informative].mean()
    distance = np.linalg.norm(end.to_numpy() - reference[informative].to_numpy(), axis=1)
    # FD003's distances fall into two groups with nothing between 4.4 and 8.5;
    # FD001's own engines reach at most 4.6. 6 sits in that gap.
    return pd.Series(np.where(distance < FINGERPRINT_GAP, "HPC wear", "fan wear"), index=end.index)


def main() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src.artifact_io import atomic_write_text
    from src.data_pipeline import add_train_rul, build_dataset, load_train
    from src.inference import load_artifacts, load_rul_artifacts, predict_rul_every_cycle
    from src.sensor_health import calibrate, score_history
    from src.sensor_integrity import fit_integrity

    artifacts = load_artifacts()
    rul = load_rul_artifacts()
    cals = calibrate(artifacts.feature_cols)
    model = fit_integrity(cals)
    informative = [c for c, cal in cals.items() if cal.informative]

    # FD001 reference pattern at end of life
    fd1 = add_train_rul(load_train("FD001"), clip=10_000)
    d1 = score_history(fd1, cals).merge(fd1[[config.UNIT_COL, config.CYCLE_COL, config.RUL_COL]],
                                        on=[config.UNIT_COL, config.CYCLE_COL])
    reference = d1[d1[config.RUL_COL] < 10][informative].mean()

    # False alarms on familiar wear: FD001 test and training engines
    raw_cols = [config.UNIT_COL, config.CYCLE_COL] + config.SETTING_COLS + config.SENSOR_COLS
    fd1_test = build_dataset("FD001").test[raw_cols]
    _, f_test = unfamiliar_flags(fd1_test, model)
    fd1_train = load_train("FD001")[raw_cols]
    _, f_train = unfamiliar_flags(fd1_train, model)
    false_test = fd1_test.sort_values([config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)[f_test.to_numpy()][config.UNIT_COL].nunique()
    false_train = fd1_train.sort_values([config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)[f_train.to_numpy()][config.UNIT_COL].nunique()

    # FD003: unseen engines with two failure modes
    labels = fingerprint_labels("FD003", cals, reference)
    fd3 = add_train_rul(load_train("FD003"), clip=10_000)
    fd3_sorted = fd3.sort_values([config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)
    res3 = assess(fd3_sorted[raw_cols], model)
    count, flags = pattern_flags(res3)
    fd3_sorted["unfamiliar"] = flags.to_numpy()
    fd3_sorted["sensor_flag"] = res3.flags.reset_index(drop=True).any(axis=1).to_numpy()

    per_engine = []
    for unit, g in fd3_sorted.groupby(config.UNIT_COL):
        hit = g[g["unfamiliar"]]
        # A lone disagreeing sensor first looks like a sensor fault, until a second joins it
        blamed = g[g["sensor_flag"] & ~g["unfamiliar"]]
        per_engine.append({"engine": int(unit), "mode": labels[unit], "flagged": bool(len(hit)),
                           "life_left_when_flagged": int(hit[config.RUL_COL].iloc[0]) if len(hit) else None,
                           "sensor_blamed_cycles": int(len(blamed))})
    pe = pd.DataFrame(per_engine)
    fan, hpc = pe[pe["mode"] == "fan wear"], pe[pe["mode"] == "HPC wear"]

    # Why it matters: the LSTM on each mode, last 60 cycles before failure
    pred = predict_rul_every_cycle(rul, fd3_sorted[raw_cols]).merge(
        fd3_sorted[[config.UNIT_COL, config.CYCLE_COL, config.RUL_COL]], on=[config.UNIT_COL, config.CYCLE_COL])
    pred["mode"] = pred[config.UNIT_COL].map(labels)
    near = pred[pred[config.RUL_COL] <= 60]
    rmse = {m: round(float(np.sqrt(np.mean((g["predicted_rul"] - np.minimum(g[config.RUL_COL], config.RUL_CLIP)) ** 2))), 2)
            for m, g in near.groupby("mode")}

    result = {
        "pattern_sensors": PATTERN_SENSORS, "suspect_z": SUSPECT_Z,
        "fd001_false_flags": {"test_engines": int(false_test), "training_engines": int(false_train)},
        "fd003": {"engines": int(len(pe)), "fan_wear": int(len(fan)), "hpc_wear": int(len(hpc)),
                  "fan_flagged": int(fan["flagged"].sum()), "hpc_flagged": int(hpc["flagged"].sum()),
                  "median_life_left_when_flagged": float(fan["life_left_when_flagged"].dropna().median())
                  if fan["flagged"].any() else None,
                  "fan_first_blamed_on_a_sensor": int((fan["sensor_blamed_cycles"] > 0).sum()),
                  "fan_blamed_cycles_median": float(fan.loc[fan["sensor_blamed_cycles"] > 0, "sensor_blamed_cycles"].median())
                  if (fan["sensor_blamed_cycles"] > 0).any() else None,
                  "hpc_blamed_on_a_sensor": int((hpc["sensor_blamed_cycles"] > 0).sum())},
        "lstm_rmse_last_60_cycles": rmse,
        "fingerprint": {"sensors": [c for c in informative],
                        "hpc": reference.round(2).tolist()},
    }
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULT_FILE, json.dumps(result, indent=1))

    r = result
    print(f"\nFamiliar wear (FD001) wrongly flagged as unfamiliar: {r['fd001_false_flags']['test_engines']}/100 test "
          f"engines, {r['fd001_false_flags']['training_engines']}/100 training engines")
    f = r["fd003"]
    print(f"FD003 (never seen): {f['fan_wear']} fan-wear engines, {f['hpc_wear']} HPC-wear engines")
    print(f"  fan wear flagged as unfamiliar: {f['fan_flagged']}/{f['fan_wear']}, "
          f"median {f['median_life_left_when_flagged']} cycles before failure")
    print(f"  HPC wear flagged as unfamiliar: {f['hpc_flagged']}/{f['hpc_wear']}")
    print(f"  fan-wear engines where one sensor was first blamed as faulty: {f['fan_first_blamed_on_a_sensor']}"
          f"/{f['fan_wear']} (median {f['fan_blamed_cycles_median']} cycles); HPC: {f['hpc_blamed_on_a_sensor']}")
    print(f"LSTM error in the last 60 cycles: {rmse}")


if __name__ == "__main__":
    main()
