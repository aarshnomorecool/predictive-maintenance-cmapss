"""Checks that the data foundation obeys the project design rules.

Every one of these guards a mistake that silently inflates metrics rather than
raising an error, which is exactly the kind an examiner looks for.

    python verify_foundation.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import config
from src.data_pipeline import (
    add_test_rul,
    add_train_rul,
    build_dataset,
    load_test,
    load_test_rul,
    load_train,
)
from src.preprocessing import (
    make_last_windows,
    make_windows,
    prepare_tabular,
    scale_splits,
)

CHECKS: list[tuple[str, callable]] = []
FAILURES: list[str] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


# ------------------------------------------------------------- raw data -----

@check("raw FD001 row and unit counts match the official readme")
def _raw_counts(ctx):
    train, test = ctx["train_raw"], ctx["test_raw"]
    assert train[config.UNIT_COL].nunique() == 100, (
        f"expected 100 train units, got {train[config.UNIT_COL].nunique()}"
    )
    assert test[config.UNIT_COL].nunique() == 100, (
        f"expected 100 test units, got {test[config.UNIT_COL].nunique()}"
    )
    assert len(train) == 20631, f"expected 20631 train rows, got {len(train)}"
    assert len(test) == 13096, f"expected 13096 test rows, got {len(test)}"
    return f"{len(train)} train rows, {len(test)} test rows, 100 units each"


@check("every unit's cycle index is contiguous and starts at 1")
def _cycles_contiguous(ctx):
    for name, frame in (("train", ctx["train_raw"]), ("test", ctx["test_raw"])):
        grouped = frame.groupby(config.UNIT_COL)[config.CYCLE_COL]
        assert (grouped.min() == 1).all(), f"{name}: some unit does not start at cycle 1"
        spans = grouped.max() - grouped.min() + 1
        assert (spans == grouped.count()).all(), f"{name}: gap in cycle numbering"
    return "cycles run 1..n with no gaps in either split"


@check("no missing values in any raw column")
def _no_nans(ctx):
    for name, frame in (("train", ctx["train_raw"]), ("test", ctx["test_raw"])):
        assert not frame.isna().any().any(), f"{name} contains NaN"
    return "train and test are complete"


# ------------------------------------------------------------ labelling -----

@check("rule 1: train RUL counts down to 0 and is clipped at 125")
def _train_rul(ctx):
    labelled = add_train_rul(ctx["train_raw"])
    per_unit_min = labelled.groupby(config.UNIT_COL)[config.RUL_COL].min()
    assert (per_unit_min == 0).all(), "some train unit does not reach RUL 0"
    assert labelled[config.RUL_COL].max() == config.RUL_CLIP, (
        f"max RUL {labelled[config.RUL_COL].max()}, expected {config.RUL_CLIP}"
    )
    assert (labelled[config.RUL_COL] >= 0).all(), "negative RUL produced"
    unit1 = labelled[labelled[config.UNIT_COL] == 1]
    diffs = unit1[config.RUL_COL].diff().dropna()
    assert (diffs <= 0).all(), "RUL is not monotonically decreasing within a unit"
    return f"min 0, max {config.RUL_CLIP}, monotone decreasing per unit"


@check("test RUL equals the provided ground truth at each unit's last cycle")
def _test_rul(ctx):
    truth = load_test_rul()
    labelled = add_test_rul(ctx["test_raw"], truth)
    last = labelled.sort_values(config.CYCLE_COL).groupby(config.UNIT_COL).tail(1)
    merged = last.merge(truth, on=config.UNIT_COL)
    expected = merged["true_rul_at_end"].clip(upper=config.RUL_CLIP)
    mismatches = int((merged[config.RUL_COL] != expected).sum())
    assert mismatches == 0, f"{mismatches} test units have a wrong terminal RUL"
    return f"all {len(merged)} test units match RUL_FD001.txt (after clipping)"


@check("failure label is exactly RUL <= threshold")
def _label_consistency(ctx):
    ds = ctx["dataset"]
    for name, frame in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
        expected = (frame[config.RUL_COL] <= config.FAILURE_THRESHOLD).astype(int)
        assert frame[config.LABEL_COL].equals(expected), f"{name} label mismatch"
        assert frame[config.LABEL_COL].isin([0, 1]).all(), f"{name} non-binary label"
    return f"binary label consistent at threshold {config.FAILURE_THRESHOLD} cycles"


# -------------------------------------------------------- feature pruning ---

@check("rule 5: dropped columns really are constant in the training split")
def _dropped_are_flat(ctx):
    ds = ctx["dataset"]
    stds = ds.train[ds.dropped_cols].std()
    assert (stds <= config.NEAR_CONSTANT_STD_THRESHOLD).all(), (
        f"a dropped column has real variance: {stds.to_dict()}"
    )
    kept_stds = ds.train[ds.feature_cols].std()
    assert (kept_stds > config.NEAR_CONSTANT_STD_THRESHOLD).all(), (
        "a kept column is flat and should have been dropped"
    )
    return f"{len(ds.dropped_cols)} flat dropped, {len(ds.feature_cols)} informative kept"


# ------------------------------------------------------------ splitting -----

@check("rule 3: train and validation share no engine unit")
def _no_unit_leak(ctx):
    ds = ctx["dataset"]
    overlap = set(ds.train_units) & set(ds.val_units)
    assert not overlap, f"units in both splits: {sorted(overlap)}"
    assert len(ds.train_units) + len(ds.val_units) == 100, "units lost in the split"
    return f"{len(ds.train_units)} train units, {len(ds.val_units)} val units, disjoint"


@check("the split is reproducible from the seed")
def _split_reproducible(ctx):
    again = build_dataset()
    assert again.val_units == ctx["dataset"].val_units, "split changed between runs"
    return f"same {len(again.val_units)} validation units on a repeat build"


# -------------------------------------------------------------- scaling -----

@check("scaler is fitted on train only, so val and test can exceed [0, 1]")
def _scaler_fit_on_train(ctx):
    scaled = ctx["scaled"]
    train_values = scaled.train[scaled.feature_cols].to_numpy()
    assert train_values.min() >= -1e-9 and train_values.max() <= 1 + 1e-9, (
        "training features are not in [0, 1], scaler was not fitted on train"
    )
    test_values = scaled.test[scaled.feature_cols].to_numpy()
    out_of_range = float(
        ((test_values < 0) | (test_values > 1)).mean()
    )
    return (
        f"train in [0, 1] exactly; {out_of_range:.2%} of test values fall outside, "
        "as they should when the scaler never saw them"
    )


# ------------------------------------------------------- class balancing ----

@check("rule 2: SMOTE balances train only, val and test keep their real prior")
def _smote_scope(ctx):
    ds = ctx["scaled"]
    balanced = prepare_tabular(ds, use_smote=True)
    raw = prepare_tabular(ds, use_smote=False)

    assert abs(balanced.y_train.mean() - 0.5) < 1e-6, "train not balanced"
    assert len(balanced.y_train) > len(raw.y_train), "SMOTE added no rows"
    assert np.array_equal(balanced.y_val, raw.y_val), "validation was resampled"
    assert np.array_equal(balanced.y_test, raw.y_test), "test was resampled"
    return (
        f"train {raw.y_train.mean():.2%} -> {balanced.y_train.mean():.0%} positive; "
        f"val {balanced.y_val.mean():.2%} and test {balanced.y_test.mean():.2%} untouched"
    )


# ------------------------------------------------------------ windowing -----

@check("rule 4: no LSTM window spans two engine units")
def _windows_within_unit(ctx):
    ds = ctx["scaled"]
    X, y, units = make_windows(ds.train, ds.feature_cols, config.WINDOW_SIZE)

    # Reconstruct expected window count: each unit contributes
    # (n_cycles - window_size + 1) windows, and nothing more.
    counts = ds.train.groupby(config.UNIT_COL).size()
    expected = int((counts - config.WINDOW_SIZE + 1).clip(lower=0).sum())
    assert len(X) == expected, f"expected {expected} windows, got {len(X)}"
    assert X.shape[1:] == (config.WINDOW_SIZE, len(ds.feature_cols)), (
        f"unexpected window shape {X.shape}"
    )
    assert len(np.unique(units)) == len(ds.train_units), "a unit produced no windows"
    return f"{len(X)} windows, all accounted for unit by unit"


@check("window contents match the source rows for a spot-checked engine")
def _window_contents(ctx):
    ds = ctx["scaled"]
    unit = ds.train_units[0]
    unit_df = ds.train[ds.train[config.UNIT_COL] == unit].sort_values(config.CYCLE_COL)
    X, y, units = make_windows(ds.train, ds.feature_cols, config.WINDOW_SIZE)

    first = np.flatnonzero(units == unit)[0]
    expected_window = unit_df[ds.feature_cols].to_numpy(dtype=np.float32)[
        : config.WINDOW_SIZE
    ]
    assert np.allclose(X[first], expected_window), "window rows do not match source"
    expected_target = float(unit_df[config.RUL_COL].iloc[config.WINDOW_SIZE - 1])
    assert y[first] == expected_target, (
        f"target {y[first]} should be the RUL at the window's last cycle "
        f"({expected_target})"
    )
    return f"unit {unit}: first window and its target reproduce the raw rows"


@check("test windowing yields exactly one window per engine, none dropped")
def _test_windows(ctx):
    ds = ctx["scaled"]
    X, y, units = make_last_windows(ds.test, ds.feature_cols, config.WINDOW_SIZE)
    n_units = ds.test[config.UNIT_COL].nunique()
    assert len(X) == n_units, f"expected {n_units} windows, got {len(X)}"
    assert len(np.unique(units)) == n_units, "duplicate or missing engine"

    short = ds.test.groupby(config.UNIT_COL).size()
    n_short = int((short < config.WINDOW_SIZE).sum())
    truth = load_test_rul()
    expected = truth.set_index(config.UNIT_COL)["true_rul_at_end"].clip(
        upper=config.RUL_CLIP
    )
    assert np.allclose(y, expected.loc[units].to_numpy()), (
        "test targets do not match the ground-truth RUL vector"
    )
    return (
        f"{len(X)} windows for {n_units} engines "
        f"({n_short} front-padded for being shorter than {config.WINDOW_SIZE} cycles)"
    )


# ------------------------------------------------------------------ main ----

def main() -> int:
    print("Building FD001 foundation...\n")
    train_raw = load_train()
    test_raw = load_test()
    dataset = build_dataset()
    scaled, _ = scale_splits(dataset)
    ctx = {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "dataset": dataset,
        "scaled": scaled,
    }

    width = max(len(name) for name, _ in CHECKS)
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except AssertionError as exc:
            FAILURES.append(name)
            print(f"FAIL  {name.ljust(width)}  {exc}")
        else:
            print(f"pass  {name.ljust(width)}  {detail}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} of {len(CHECKS)} checks FAILED: {FAILURES}")
        return 1
    print(f"All {len(CHECKS)} checks passed. Foundation is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
