"""Loading and labelling of the NASA C-MAPSS turbofan dataset.

Everything downstream (classification, LSTM regression, dashboard) reads its
data through this module, so the labelling rules live in exactly one place.

Run directly for a summary of the built FD001 dataset:

    python -m src.data_pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ------------------------------------------------------------- loading ------

def _read_space_separated(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a C-MAPSS text file.

    The files are space separated with no header and two trailing spaces on
    every line, which produces phantom all-NaN columns if not dropped.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: expected {len(columns)} columns, got {df.shape[1]}"
        )
    df.columns = columns
    return df


def load_train(subset: str = config.SUBSET) -> pd.DataFrame:
    """Training trajectories, each run to failure."""
    return _read_space_separated(
        config.DATA_DIR / f"train_{subset}.txt", config.RAW_COLUMNS
    )


def load_test(subset: str = config.SUBSET) -> pd.DataFrame:
    """Test trajectories, each truncated some time before failure."""
    return _read_space_separated(
        config.DATA_DIR / f"test_{subset}.txt", config.RAW_COLUMNS
    )


def load_test_rul(subset: str = config.SUBSET) -> pd.DataFrame:
    """True remaining cycles for each test unit at its final recorded cycle.

    The file is one value per line, in unit-number order, with no unit column.
    """
    df = _read_space_separated(
        config.DATA_DIR / f"RUL_{subset}.txt", ["true_rul_at_end"]
    )
    df.insert(0, config.UNIT_COL, np.arange(1, len(df) + 1))
    return df


# ------------------------------------------------------------ labelling -----

def add_train_rul(df: pd.DataFrame, clip: int = config.RUL_CLIP) -> pd.DataFrame:
    """RUL for run-to-failure trajectories: cycles left until the last one.

    Clipped at ``clip`` per rule 1 (piecewise linear RUL). Before the knee the
    engine is healthy and degradation is not yet observable in the sensors, so
    asking the model to distinguish 300 cycles left from 200 teaches it noise.
    """
    df = df.copy()
    max_cycle = df.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    df[config.RUL_COL] = (max_cycle - df[config.CYCLE_COL]).clip(upper=clip)
    return df


def add_test_rul(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    clip: int = config.RUL_CLIP,
) -> pd.DataFrame:
    """RUL for truncated test trajectories.

    A test unit stops before failure, so the true RUL at any cycle is the
    cycles remaining in the recording plus the ground-truth RUL that the
    provided vector gives for its final cycle.
    """
    df = df.copy()
    max_cycle = df.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    true_rul = df[config.UNIT_COL].map(
        rul_df.set_index(config.UNIT_COL)["true_rul_at_end"]
    )
    if true_rul.isna().any():
        missing = sorted(df.loc[true_rul.isna(), config.UNIT_COL].unique())
        raise ValueError(f"No ground-truth RUL for test units: {missing}")
    df[config.RUL_COL] = (
        (max_cycle - df[config.CYCLE_COL]) + true_rul
    ).clip(upper=clip)
    return df


def add_failure_label(
    df: pd.DataFrame,
    threshold: int = config.FAILURE_THRESHOLD,
) -> pd.DataFrame:
    """Binary target for the classification branch: is failure imminent."""
    df = df.copy()
    df[config.LABEL_COL] = (df[config.RUL_COL] <= threshold).astype(int)
    return df


# ------------------------------------------------- feature column pruning ---

def find_near_constant_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = config.NEAR_CONSTANT_STD_THRESHOLD,
) -> list[str]:
    """Columns whose standard deviation is effectively zero.

    Rule 5: measure the variance rather than hardcoding the usual suspects,
    because which sensors are flat changes between FD001 and FD002/FD004.
    """
    columns = columns or (config.SETTING_COLS + config.SENSOR_COLS)
    stds = df[columns].std(numeric_only=True)
    return [column for column in columns if stds.get(column, 0.0) <= threshold]


def select_feature_columns(
    df: pd.DataFrame,
    threshold: float = config.NEAR_CONSTANT_STD_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Split candidate features into kept and dropped, based on train variance."""
    candidates = config.SETTING_COLS + config.SENSOR_COLS
    dropped = find_near_constant_columns(df, candidates, threshold)
    kept = [column for column in candidates if column not in dropped]
    return kept, dropped


# ------------------------------------------------------------ splitting -----

def split_by_unit(
    df: pd.DataFrame,
    val_fraction: float = config.VAL_UNIT_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole engine units, never individual rows.

    Rule 3. Splitting rows at random puts later cycles of an engine into the
    training set and earlier cycles of the same engine into validation, which
    leaks the future and inflates every metric.
    """
    units = np.sort(df[config.UNIT_COL].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_units = set(shuffled[:n_val].tolist())

    is_val = df[config.UNIT_COL].isin(val_units)
    return df.loc[~is_val].copy(), df.loc[is_val].copy()


# -------------------------------------------------------------- bundle ------

@dataclass
class Dataset:
    """Everything the model-training modules need, labelled and split."""

    subset: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    dropped_cols: list[str] = field(default_factory=list)

    @property
    def train_units(self) -> list[int]:
        return sorted(self.train[config.UNIT_COL].unique().tolist())

    @property
    def val_units(self) -> list[int]:
        return sorted(self.val[config.UNIT_COL].unique().tolist())

    def summary(self) -> str:
        lines = [
            f"Dataset {self.subset}",
            f"  features kept    : {len(self.feature_cols)} "
            f"({', '.join(self.feature_cols)})",
            f"  features dropped : {len(self.dropped_cols)} "
            f"({', '.join(self.dropped_cols) or 'none'})",
            f"  train : {len(self.train):>6} rows / "
            f"{len(self.train_units):>3} units",
            f"  val   : {len(self.val):>6} rows / "
            f"{len(self.val_units):>3} units",
            f"  test  : {len(self.test):>6} rows / "
            f"{self.test[config.UNIT_COL].nunique():>3} units",
        ]
        for name, frame in (
            ("train", self.train), ("val", self.val), ("test", self.test)
        ):
            positives = int(frame[config.LABEL_COL].sum())
            rate = positives / len(frame) if len(frame) else 0.0
            lines.append(
                f"  {name:<5} failure_soon rate : {rate:6.2%} "
                f"({positives}/{len(frame)})"
            )
        return "\n".join(lines)


def build_dataset(
    subset: str = config.SUBSET,
    val_fraction: float = config.VAL_UNIT_FRACTION,
    seed: int = config.RANDOM_SEED,
    rul_clip: int = config.RUL_CLIP,
    failure_threshold: int = config.FAILURE_THRESHOLD,
) -> Dataset:
    """Load, label, prune and split one C-MAPSS subset."""
    train_raw = add_failure_label(
        add_train_rul(load_train(subset), clip=rul_clip),
        threshold=failure_threshold,
    )
    test_raw = add_failure_label(
        add_test_rul(load_test(subset), load_test_rul(subset), clip=rul_clip),
        threshold=failure_threshold,
    )

    train_df, val_df = split_by_unit(train_raw, val_fraction, seed)

    # Variance is measured on the training split only. Looking at validation or
    # test data to decide which columns to keep is a leak, quiet but real.
    feature_cols, dropped_cols = select_feature_columns(train_df)

    return Dataset(
        subset=subset,
        train=train_df,
        val=val_df,
        test=test_raw,
        feature_cols=feature_cols,
        dropped_cols=dropped_cols,
    )


if __name__ == "__main__":
    print(build_dataset().summary())
