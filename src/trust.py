"""How far can a remaining-life prediction be trusted, and from when?

Two measurements on the 20 validation engines, which ran to failure and
were never used to fit the LSTM's weights:

1. Likely range. The LSTM's real errors, grouped by how far from failure it
   says the engine is. Near failure it is precise; in mid-life it is not.
   The 10th and 90th percentiles of those errors give an 80% range that is
   attached to every prediction ("93 cycles, likely 75 to 110").

2. Time to trust. A retrofit: sensors are fitted to an engine that is already
   part-worn, so there is no history. Each validation engine is "first
   monitored" 120, 100, 80 or 60 cycles before failure, and the error is
   tracked as cycles of data accumulate. While fewer than 30 cycles are
   available, the LSTM's window is padded with the first reading, so early
   predictions are wider; the ranges for that phase are measured separately.

    python -m src.trust      # writes models/insights/trust.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config

INSIGHTS_DIR = config.MODELS_DIR / "insights"
TRUST_FILE = INSIGHTS_DIR / "trust.json"

# Predicted-life bands the ranges are measured in (cycles)
BANDS = [(0, 30), (30, 60), (60, 90), (90, 126)]
LATE_STARTS = (120, 100, 80, 60)                 # true RUL when monitoring begins
OBS_BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 45), (46, 60)]
SETTLED_TOLERANCE = 1.15                          # within 15% of the settled error


def band_of(pred: float) -> int:
    for k, (lo, hi) in enumerate(BANDS):
        if lo <= pred < hi:
            return k
    return len(BANDS) - 1


def likely_range(pred: float, cycles_observed: int, table: dict) -> tuple[float, float]:
    """80% range around a prediction, from the measured error table."""
    phase = "settled" if cycles_observed >= table["window"] else "building"
    q = table["ranges"][phase][band_of(pred)]
    # error = predicted - true, so true = predicted - error
    lo = max(0.0, pred - q["q90"])
    hi = max(lo, pred - q["q10"])
    return round(lo, 1), round(hi, 1)


def measure() -> dict:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src.data_pipeline import add_train_rul, load_train
    from src.inference import load_rul_artifacts, predict_rul_every_cycle

    rul = load_rul_artifacts()
    window = rul.window_size
    val_units = json.loads((config.MODELS_DIR / "classifier_metadata.json").read_text())["val_units"]
    train = add_train_rul(load_train(rul.metadata["subset"]), clip=config.RUL_CLIP)
    val = train[train[config.UNIT_COL].isin(val_units)]

    # Full histories: errors once the window is full
    full = predict_rul_every_cycle(rul, val).merge(
        val[[config.UNIT_COL, config.CYCLE_COL, config.RUL_COL]], on=[config.UNIT_COL, config.CYCLE_COL])
    full["obs"] = full[config.CYCLE_COL]
    full["err"] = full["predicted_rul"] - full[config.RUL_COL]

    # Late starts: monitoring begins on an already part-worn engine
    late = []
    last = val.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    life_left = last - val[config.CYCLE_COL]
    for start_rul in LATE_STARTS:
        part = val[life_left <= start_rul]
        p = predict_rul_every_cycle(rul, part).merge(
            part[[config.UNIT_COL, config.CYCLE_COL, config.RUL_COL]], on=[config.UNIT_COL, config.CYCLE_COL])
        p["obs"] = p.groupby(config.UNIT_COL).cumcount() + 1
        p["err"] = p["predicted_rul"] - p[config.RUL_COL]
        p["start"] = start_rul
        late.append(p)
    late = pd.concat(late, ignore_index=True)

    def stats(e: pd.Series) -> dict:
        return {"n": int(len(e)), "mae": round(float(e.abs().mean()), 2),
                "q10": round(float(e.quantile(0.10)), 2), "q90": round(float(e.quantile(0.90)), 2)}

    curve = [{"from": lo, "to": hi, **stats(late.loc[late["obs"].between(lo, hi), "err"])}
             for lo, hi in OBS_BUCKETS]
    settled_mae = float(late.loc[late["obs"] > window, "err"].abs().mean())
    trusted_after = next((c["from"] for c in curve if c["mae"] <= settled_mae * SETTLED_TOLERANCE), None)

    def band_stats(df):
        out = []
        for k, (lo, hi) in enumerate(BANDS):
            e = df.loc[df["predicted_rul"].between(lo, hi - 1e-9), "err"]
            out.append({"band": [lo, hi], **(stats(e) if len(e) >= 20 else stats(df["err"]))})
        return out

    settled = pd.concat([full[full["obs"] > window], late[late["obs"] > window]])
    building = late[late["obs"] <= window]
    ranges = {"settled": band_stats(settled), "building": band_stats(building)}

    # How often does the true value land inside the 80% range? (should be about 80%)
    table = {"window": window, "ranges": ranges}
    inside = []
    for df in (full, late):
        for pred, obs, err in df[["predicted_rul", "obs", "err"]].itertuples(index=False):
            lo, hi = likely_range(pred, obs, table)
            truth = pred - err
            inside.append(lo <= truth <= hi)

    return {
        "window": window,
        "late_starts": list(LATE_STARTS),
        "curve": curve,
        "settled_mae": round(settled_mae, 2),
        "trusted_after": trusted_after,
        "ranges": ranges,
        "coverage": round(float(np.mean(inside)), 3),
        "engines": len(val_units),
    }


def load_trust() -> dict | None:
    try:
        return json.loads(TRUST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    from src.artifact_io import atomic_write_text
    result = measure()
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(TRUST_FILE, json.dumps(result, indent=1))
    print(f"\nRetrofit test: monitoring starts {', '.join(map(str, LATE_STARTS))} cycles before failure, "
          f"{result['engines']} validation engines\n")
    print(f"  {'cycles of data':<16}{'MAE':>7}{'80% range of error':>22}")
    for c in result["curve"]:
        print(f"  {c['from']:>3} to {c['to']:<9}{c['mae']:>7.1f}{c['q10']:>12.1f} to {c['q90']:<6.1f}")
    print(f"\n  Settled error (window full): {result['settled_mae']:.1f} cycles. "
          f"Within 15% of that after {result['trusted_after']} cycles of data.")
    print("\nLikely range (80%) of the truth around a prediction, by predicted life:")
    for phase in ("building", "settled"):
        cells = ", ".join(f"{r['band'][0]}-{r['band'][1]}: -{r['q90']:.0f}/+{-r['q10']:.0f}"
                          for r in result["ranges"][phase])
        print(f"  {phase:<9} {cells}")
    print(f"\nTrue remaining life fell inside the stated 80% range {result['coverage']:.0%} of the time.")
    print(f"Saved {TRUST_FILE}")


if __name__ == "__main__":
    main()
