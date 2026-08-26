"""Scaling, class balancing and sequence windowing.

Two branches diverge here and must not be mixed up:

* the tabular branch feeds Random Forest / XGBoost and may use SMOTE,
* the sequence branch feeds the LSTM and must never see SMOTE (rule 2),
  because interpolating between two synthetic points on a continuous RUL
  curve invents an engine history that never happened.

Run directly for a shape report on FD001:

    python -m src.preprocessing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config
from src.data_pipeline import Dataset, build_dataset


# --------------------------------------------------------------- scaling ----

def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> MinMaxScaler:
    """Fit the scaler on training rows only.

    Fitting on the full frame lets test-set minima and maxima bleed into the
    transform, which is a leak even though no label is involved.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float64))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return a copy with the feature columns scaled in place."""
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(
        df[feature_cols].to_numpy(dtype=np.float64)
    )
    return scaled


def scale_splits(
    dataset: Dataset,
) -> tuple[Dataset, MinMaxScaler]:
    """Scale train, val and test with a scaler fitted on train alone."""
    scaler = fit_scaler(dataset.train, dataset.feature_cols)
    scaled = Dataset(
        subset=dataset.subset,
        train=apply_scaler(dataset.train, scaler, dataset.feature_cols),
        val=apply_scaler(dataset.val, scaler, dataset.feature_cols),
        test=apply_scaler(dataset.test, scaler, dataset.feature_cols),
        feature_cols=list(dataset.feature_cols),
        dropped_cols=list(dataset.dropped_cols),
    )
    return scaled, scaler


# ------------------------------------------------- tabular branch (RF/XGB) --

def to_tabular(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Flat feature matrix and binary failure labels, one row per cycle."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[config.LABEL_COL].to_numpy(dtype=np.int8)
    return X, y


def balance_with_smote(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class for the classification branch only.

    Rule 2. Apply this to the training split alone. Balancing validation or
    test data changes the class prior the metrics are measured against and
    makes precision and recall meaningless.
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# ------------------------------------------------- sequence branch (LSTM) ---

def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
    target_col: str = config.RUL_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows of consecutive cycles, one engine unit at a time.

    Rule 4. Windows are cut inside a single unit and never span the boundary
    between two engines, which would splice one engine's end onto another's
    beginning. The target is the value at the window's last cycle.

    Returns ``(X, y, units)`` where ``X`` has shape
    ``(n_windows, window_size, n_features)`` and ``units`` records which engine
    each window came from, so downstream code can group predictions per engine.

    Units shorter than ``window_size`` produce no windows and are skipped.
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    origin_units: list[int] = []

    for unit, unit_df in df.groupby(config.UNIT_COL, sort=True):
        unit_df = unit_df.sort_values(config.CYCLE_COL)
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        target_values = unit_df[target_col].to_numpy(dtype=np.float32)

        if len(unit_df) < window_size:
            continue

        for end in range(window_size, len(unit_df) + 1):
            sequences.append(values[end - window_size:end])
            targets.append(target_values[end - 1])
            origin_units.append(int(unit))

    if not sequences:
        n_features = len(feature_cols)
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
        np.asarray(origin_units, dtype=np.int32),
    )


def make_last_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
    target_col: str = config.RUL_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per engine, taken from its final cycles.

    This is the C-MAPSS test protocol: predict the RUL of each test engine at
    the moment its recording stops, one prediction per engine.

    Engines with fewer than ``window_size`` cycles are front-padded by
    repeating their first recorded cycle, so no test engine is silently
    dropped from the evaluation.
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    origin_units: list[int] = []

    for unit, unit_df in df.groupby(config.UNIT_COL, sort=True):
        unit_df = unit_df.sort_values(config.CYCLE_COL)
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)

        if len(values) >= window_size:
            window = values[-window_size:]
        else:
            pad = np.repeat(values[:1], window_size - len(values), axis=0)
            window = np.vstack([pad, values])

        sequences.append(window)
        targets.append(float(unit_df[target_col].iloc[-1]))
        origin_units.append(int(unit))

    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
        np.asarray(origin_units, dtype=np.int32),
    )


# -------------------------------------------------------------- bundles -----

@dataclass
class TabularData:
    """Ready-to-fit arrays for the RF / XGBoost classification branch."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: list[str]
    smote_applied: bool

    def summary(self) -> str:
        def rate(y: np.ndarray) -> str:
            return f"{y.mean():.2%}" if len(y) else "n/a"

        return "\n".join([
            "Tabular branch (RandomForest / XGBoost)",
            f"  SMOTE applied to train : {self.smote_applied}",
            f"  X_train {self.X_train.shape}  positives {rate(self.y_train)}",
            f"  X_val   {self.X_val.shape}  positives {rate(self.y_val)}",
            f"  X_test  {self.X_test.shape}  positives {rate(self.y_test)}",
        ])


@dataclass
class SequenceData:
    """Ready-to-fit arrays for the LSTM RUL regression branch."""

    X_train: np.ndarray
    y_train: np.ndarray
    units_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    units_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    units_test: np.ndarray
    feature_cols: list[str]
    window_size: int

    def summary(self) -> str:
        return "\n".join([
            f"Sequence branch (LSTM), window {self.window_size}",
            f"  X_train {self.X_train.shape}  y_train {self.y_train.shape}"
            f"  units {len(np.unique(self.units_train))}",
            f"  X_val   {self.X_val.shape}  y_val   {self.y_val.shape}"
            f"  units {len(np.unique(self.units_val))}",
            f"  X_test  {self.X_test.shape}   y_test  {self.y_test.shape}"
            f"  units {len(np.unique(self.units_test))} (last window per engine)",
        ])


def prepare_tabular(
    dataset: Dataset,
    use_smote: bool = True,
    seed: int = config.RANDOM_SEED,
) -> TabularData:
    """Build the classification arrays. SMOTE touches the training split only."""
    X_train, y_train = to_tabular(dataset.train, dataset.feature_cols)
    X_val, y_val = to_tabular(dataset.val, dataset.feature_cols)
    X_test, y_test = to_tabular(dataset.test, dataset.feature_cols)

    if use_smote:
        X_train, y_train = balance_with_smote(X_train, y_train, seed)

    return TabularData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_cols=list(dataset.feature_cols),
        smote_applied=use_smote,
    )


def prepare_sequences(
    dataset: Dataset,
    window_size: int = config.WINDOW_SIZE,
) -> SequenceData:
    """Build the LSTM arrays. No SMOTE anywhere in this branch."""
    X_train, y_train, units_train = make_windows(
        dataset.train, dataset.feature_cols, window_size
    )
    X_val, y_val, units_val = make_windows(
        dataset.val, dataset.feature_cols, window_size
    )
    X_test, y_test, units_test = make_last_windows(
        dataset.test, dataset.feature_cols, window_size
    )

    return SequenceData(
        X_train=X_train,
        y_train=y_train,
        units_train=units_train,
        X_val=X_val,
        y_val=y_val,
        units_val=units_val,
        X_test=X_test,
        y_test=y_test,
        units_test=units_test,
        feature_cols=list(dataset.feature_cols),
        window_size=window_size,
    )


def prepare_all(
    subset: str = config.SUBSET,
    use_smote: bool = True,
    window_size: int = config.WINDOW_SIZE,
) -> tuple[Dataset, MinMaxScaler, TabularData, SequenceData]:
    """One call that yields both branches from the same scaled split."""
    dataset = build_dataset(subset)
    scaled, scaler = scale_splits(dataset)
    return (
        scaled,
        scaler,
        prepare_tabular(scaled, use_smote=use_smote),
        prepare_sequences(scaled, window_size=window_size),
    )


if __name__ == "__main__":
    scaled, _, tabular, sequences = prepare_all()
    print(scaled.summary())
    print()
    print(tabular.summary())
    print()
    print(sequences.summary())
