"""Per-sensor health: how far each reading has drifted towards failure.

The classifiers give one probability per engine. An operator looking at an
engine wants to know *where* the trouble is, so this module scores every
physical sensor on its own.

For each sensor, two reference levels are measured on the training split
only (test engines never inform the calibration):

* healthy level: mean reading while RUL sits at the 125-cycle ceiling,
  the early-life phase where no degradation is observable yet
* failure level: mean reading in the last 10 cycles before failure

A reading's drift is then where it sits between the two:

    drift = (reading - healthy) / (failure - healthy)

0 means "reads like a new engine", 1 means "reads like an engine about to
fail". The formula is direction-aware for free: sensors that fall with wear
(sensor_7, sensor_12, sensor_20, sensor_21) have failure < healthy, so the
ratio is still positive as they degrade.

Single readings are noisy, so drift is computed on a trailing rolling mean.
Trailing, not centred: a centred window would peek at future cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.data_pipeline import add_train_rul, load_train

# Trailing window for smoothing raw readings before scoring them.
SMOOTHING_WINDOW = 10

# Cycles before failure that define the failure reference level.
FAILURE_REFERENCE_CYCLES = 10

# Severity bands on the drift scale. Chosen by measuring how well they
# separate test engines inside the failure window from those outside it;
# python -m src.sensor_health prints that check.
WARNING_LEVEL = 0.45
CRITICAL_LEVEL = 0.75

# A sensor whose healthy-to-failure gap is smaller than this many healthy
# standard deviations barely moves with wear. It is still scored, but flagged
# so the UI does not present noise as a finding.
MIN_SIGNAL_TO_NOISE = 1.0


@dataclass(frozen=True)
class SensorSpec:
    """Physical meaning of one C-MAPSS channel (Saxena et al., Table 2)."""

    column: str
    symbol: str
    description: str
    units: str
    module: str


# Table 2 of Saxena et al., PHM 2008. `module` is where the measurement is
# taken on the engine, which is what places the sensor on the 3D model.
SENSOR_SPECS: tuple[SensorSpec, ...] = (
    SensorSpec("sensor_1", "T2", "Total temperature at fan inlet", "°R", "inlet"),
    SensorSpec("sensor_2", "T24", "Total temperature at LPC outlet", "°R", "lpc"),
    SensorSpec("sensor_3", "T30", "Total temperature at HPC outlet", "°R", "hpc"),
    SensorSpec("sensor_4", "T50", "Total temperature at LPT outlet", "°R", "lpt"),
    SensorSpec("sensor_5", "P2", "Pressure at fan inlet", "psia", "inlet"),
    SensorSpec("sensor_6", "P15", "Total pressure in bypass duct", "psia", "bypass"),
    SensorSpec("sensor_7", "P30", "Total pressure at HPC outlet", "psia", "hpc"),
    SensorSpec("sensor_8", "Nf", "Physical fan speed", "rpm", "fan"),
    SensorSpec("sensor_9", "Nc", "Physical core speed", "rpm", "core_shaft"),
    SensorSpec("sensor_10", "epr", "Engine pressure ratio (P50/P2)", "–", "nozzle"),
    SensorSpec("sensor_11", "Ps30", "Static pressure at HPC outlet", "psia", "hpc"),
    SensorSpec("sensor_12", "phi", "Ratio of fuel flow to Ps30", "pps/psi", "combustor"),
    SensorSpec("sensor_13", "NRf", "Corrected fan speed", "rpm", "fan"),
    SensorSpec("sensor_14", "NRc", "Corrected core speed", "rpm", "core_shaft"),
    SensorSpec("sensor_15", "BPR", "Bypass ratio", "–", "bypass"),
    SensorSpec("sensor_16", "farB", "Burner fuel-air ratio", "–", "combustor"),
    SensorSpec("sensor_17", "htBleed", "Bleed enthalpy", "–", "hpc"),
    SensorSpec("sensor_18", "Nf_dmd", "Demanded fan speed", "rpm", "fan"),
    SensorSpec("sensor_19", "PCNfR_dmd", "Demanded corrected fan speed", "rpm", "fan"),
    SensorSpec("sensor_20", "W31", "HPT coolant bleed", "lbm/s", "hpt"),
    SensorSpec("sensor_21", "W32", "LPT coolant bleed", "lbm/s", "lpt"),
)

SPEC_BY_COLUMN = {spec.column: spec for spec in SENSOR_SPECS}


@dataclass(frozen=True)
class Calibration:
    """Reference levels for one sensor, measured on training engines."""

    column: str
    healthy_mean: float
    healthy_std: float
    failure_mean: float

    @property
    def signal_to_noise(self) -> float:
        if self.healthy_std == 0:
            return 0.0
        return abs(self.failure_mean - self.healthy_mean) / self.healthy_std

    @property
    def informative(self) -> bool:
        return self.signal_to_noise >= MIN_SIGNAL_TO_NOISE

    @property
    def rises_with_wear(self) -> bool:
        return self.failure_mean > self.healthy_mean


def smooth(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Trailing rolling mean per engine. Never mixes two engines' readings."""
    ordered = df.sort_values([config.UNIT_COL, config.CYCLE_COL])
    smoothed = (
        ordered.groupby(config.UNIT_COL)[columns]
        .rolling(SMOOTHING_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out = ordered.copy()
    out[columns] = smoothed[columns]
    return out


def calibrate(feature_cols: list[str], subset: str = config.SUBSET) -> dict[str, Calibration]:
    """Measure healthy and failure levels for every scored sensor.

    Uses the full training file. That is the same pool the classifiers'
    training and validation units come from; test engines stay unseen.
    """
    train = add_train_rul(load_train(subset), clip=config.RUL_CLIP)
    healthy = train[train[config.RUL_COL] >= config.RUL_CLIP]
    failing = train[train[config.RUL_COL] < FAILURE_REFERENCE_CYCLES]

    calibrations = {}
    for column in scored_sensors(feature_cols):
        calibrations[column] = Calibration(
            column=column,
            healthy_mean=float(healthy[column].mean()),
            healthy_std=float(healthy[column].std()),
            failure_mean=float(failing[column].mean()),
        )
    return calibrations


def scored_sensors(feature_cols: list[str]) -> list[str]:
    """Physical sensors the classifiers use. Operating settings are not sensors."""
    return [column for column in feature_cols if column in SPEC_BY_COLUMN]


def drift(values: pd.Series | np.ndarray | float, calibration: Calibration):
    gap = calibration.failure_mean - calibration.healthy_mean
    if gap == 0:
        return np.zeros_like(np.asarray(values, dtype=float))
    return (np.asarray(values, dtype=float) - calibration.healthy_mean) / gap


def severity(drift_value: float, informative: bool = True) -> str:
    if not informative:
        return "low_signal"
    if drift_value >= CRITICAL_LEVEL:
        return "critical"
    if drift_value >= WARNING_LEVEL:
        return "warning"
    return "normal"


def score_history(
    raw_history: pd.DataFrame, calibrations: dict[str, Calibration]
) -> pd.DataFrame:
    """Drift per cycle for every scored sensor of one or more engines."""
    columns = list(calibrations)
    smoothed = smooth(raw_history, columns)
    scored = smoothed[[config.UNIT_COL, config.CYCLE_COL]].copy()
    for column, calibration in calibrations.items():
        scored[column] = drift(smoothed[column], calibration)
    return scored.reset_index(drop=True)


# -------------------------------------------------------------------- cli ----

def main() -> None:
    """Check that the severity bands actually separate failing engines."""
    from src.inference import latest_per_engine, load_artifacts, load_scored_test

    artifacts = load_artifacts()
    calibrations = calibrate(artifacts.feature_cols, artifacts.subset)
    scored = load_scored_test(artifacts)

    print(f"\n{'sensor':<10}{'symbol':<10}{'healthy':>12}{'failure':>12}{'SNR':>7}  trend")
    for column, cal in calibrations.items():
        spec = SPEC_BY_COLUMN[column]
        trend = "rises" if cal.rises_with_wear else "falls"
        flag = "" if cal.informative else "  (low signal)"
        print(
            f"{column:<10}{spec.symbol:<10}{cal.healthy_mean:>12.3f}"
            f"{cal.failure_mean:>12.3f}{cal.signal_to_noise:>7.2f}  {trend}{flag}"
        )

    drifts = score_history(scored, calibrations)
    latest = latest_per_engine(scored)[[config.UNIT_COL, config.LABEL_COL]]
    last_drift = latest_per_engine(drifts)
    merged = last_drift.merge(latest, on=config.UNIT_COL)
    informative = [c for c, cal in calibrations.items() if cal.informative]
    merged["critical_sensors"] = (merged[informative] >= CRITICAL_LEVEL).sum(axis=1)
    merged["warning_or_worse"] = (merged[informative] >= WARNING_LEVEL).sum(axis=1)

    print("\nLatest cycle, informative sensors per engine (mean):")
    for label, group in merged.groupby(config.LABEL_COL):
        name = "inside failure window" if label else "outside failure window"
        print(
            f"  {name:<24} n={len(group):>3}  critical {group['critical_sensors'].mean():5.2f}"
            f"  warning+ {group['warning_or_worse'].mean():5.2f}"
        )


if __name__ == "__main__":
    main()
