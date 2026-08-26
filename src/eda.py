"""Exploratory analysis of FD001, for the report and for sanity.

Writes figures to reports/figures/ and a text summary to reports/eda_summary.md.

    python -m src.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write files, never try to open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data_pipeline import add_train_rul, build_dataset, load_train

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(config.PROJECT_ROOT).as_posix()


# --------------------------------------------------------------- figures ----

def plot_engine_lifetimes(train: pd.DataFrame) -> tuple[str, pd.Series]:
    """How long engines survive. Wide spread is the reason RUL is hard."""
    lifetimes = train.groupby(config.UNIT_COL)[config.CYCLE_COL].max()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(lifetimes, bins=25, kde=True, ax=ax, color="#2a6f97")
    ax.axvline(lifetimes.mean(), color="#c1121f", linestyle="--",
               label=f"mean {lifetimes.mean():.0f}")
    ax.set_xlabel("Cycles to failure")
    ax.set_ylabel("Number of engines")
    ax.set_title("FD001 engine lifetimes (run to failure)")
    ax.legend()
    return _save(fig, "engine_lifetimes.png"), lifetimes


def plot_rul_clipping(train: pd.DataFrame) -> str:
    """Raw countdown against the clipped target the models actually learn."""
    max_cycle = train.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    raw_rul = max_cycle - train[config.CYCLE_COL]
    clipped = raw_rul.clip(upper=config.RUL_CLIP)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(raw_rul, bins=40, ax=axes[0], color="#7d8597")
    axes[0].set_title("Raw RUL")
    axes[0].set_xlabel("Cycles remaining")
    sns.histplot(clipped, bins=40, ax=axes[1], color="#2a6f97")
    axes[1].axvline(config.RUL_CLIP, color="#c1121f", linestyle="--",
                    label=f"clip at {config.RUL_CLIP}")
    axes[1].set_title(f"Piecewise-linear RUL (clipped at {config.RUL_CLIP})")
    axes[1].set_xlabel("Cycles remaining")
    axes[1].legend()
    fig.suptitle("Clipping collapses the healthy-phase tail into one flat class")
    return _save(fig, "rul_clipping.png")


def plot_sensor_degradation(train: pd.DataFrame, sensors: list[str],
                            n_units: int = 6) -> str:
    """Sensor traces against remaining life. Trending sensors carry the signal."""
    units = sorted(train[config.UNIT_COL].unique())[:n_units]
    subset = train[train[config.UNIT_COL].isin(units)].copy()

    # Plot against the unclipped countdown. The clipped target would stack every
    # healthy-phase cycle onto a single x value and draw a vertical wall at 125.
    max_cycle = subset.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    subset["raw_rul"] = max_cycle - subset[config.CYCLE_COL]

    n_cols = 3
    n_rows = int(np.ceil(len(sensors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, sensor in zip(axes, sensors):
        for unit in units:
            unit_df = subset[subset[config.UNIT_COL] == unit]
            ax.plot(unit_df["raw_rul"], unit_df[sensor],
                    alpha=0.7, linewidth=0.9)
        ax.invert_xaxis()  # time flows left to right as RUL counts down
        ax.axvline(config.RUL_CLIP, color="#c1121f", linestyle=":", linewidth=1.2)
        ax.set_title(sensor)
        ax.set_xlabel("RUL (cycles remaining)")

    for ax in axes[len(sensors):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Sensor drift over remaining life, {n_units} engines "
        f"(time runs left to right; dotted line = RUL clip at {config.RUL_CLIP})",
        y=1.001,
    )
    fig.tight_layout()
    return _save(fig, "sensor_degradation.png")


def plot_rul_correlation(train: pd.DataFrame,
                         feature_cols: list[str]) -> tuple[str, pd.Series]:
    """Which sensors actually track remaining life, and in which direction."""
    corr = (
        train[feature_cols + [config.RUL_COL]]
        .corr(numeric_only=True)[config.RUL_COL]
        .drop(config.RUL_COL)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c1121f" if v < 0 else "#2a6f97" for v in corr]
    ax.barh(corr.index, corr.to_numpy(), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with RUL")
    ax.set_title("Sensor correlation with remaining useful life")
    return _save(fig, "rul_correlation.png"), corr


def plot_feature_correlation(train: pd.DataFrame,
                             feature_cols: list[str]) -> str:
    """Redundancy between sensors, which matters for tree feature importance."""
    corr = train[feature_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.3)
    ax.set_title("Sensor-to-sensor correlation (training split)")
    return _save(fig, "feature_correlation.png")


def plot_class_balance(dataset) -> str:
    """The imbalance SMOTE exists to address, and how the splits differ."""
    rows = [
        {"split": name, "rate": frame[config.LABEL_COL].mean() * 100}
        for name, frame in (
            ("train", dataset.train), ("val", dataset.val), ("test", dataset.test)
        )
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(df, x="split", y="rate", hue="split", palette=PALETTE,
                legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%")
    ax.set_ylabel(f"Rows with RUL <= {config.FAILURE_THRESHOLD} (%)")
    ax.set_xlabel("")
    ax.set_title("Class imbalance in the failure-soon target")
    return _save(fig, "class_balance.png")


# --------------------------------------------------------------- summary ----

def write_summary(lines: list[str]) -> str:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "eda_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.relative_to(config.PROJECT_ROOT).as_posix()


def main() -> None:
    dataset = build_dataset()
    train_labelled = add_train_rul(load_train())

    lifetimes_fig, lifetimes = plot_engine_lifetimes(train_labelled)
    clipping_fig = plot_rul_clipping(load_train())
    corr_fig, corr = plot_rul_correlation(dataset.train, dataset.feature_cols)

    top_sensors = corr.abs().sort_values(ascending=False).head(6).index.tolist()
    degradation_fig = plot_sensor_degradation(dataset.train, top_sensors)
    heatmap_fig = plot_feature_correlation(dataset.train, dataset.feature_cols)
    balance_fig = plot_class_balance(dataset)

    strong = corr[corr.abs() >= 0.5].sort_values()
    lines = [
        f"# EDA summary: C-MAPSS {dataset.subset}",
        "",
        "Generated by `python -m src.eda`.",
        "",
        "## Shape",
        "",
        f"- Train: {len(dataset.train)} rows across {len(dataset.train_units)} engines",
        f"- Validation: {len(dataset.val)} rows across {len(dataset.val_units)} engines "
        "(held out by engine, never by row)",
        f"- Test: {len(dataset.test)} rows across "
        f"{dataset.test[config.UNIT_COL].nunique()} engines, each truncated before failure",
        "",
        "## Engine lifetimes",
        "",
        f"- Shortest {int(lifetimes.min())} cycles, longest {int(lifetimes.max())}, "
        f"mean {lifetimes.mean():.1f}, median {lifetimes.median():.0f}",
        f"- Standard deviation {lifetimes.std():.1f} cycles. Engines start with "
        "different unknown initial wear, so cycle count alone is a weak predictor "
        "and the sensor trajectory is what carries the signal.",
        "",
        "## Features",
        "",
        f"- Kept {len(dataset.feature_cols)}: {', '.join(dataset.feature_cols)}",
        f"- Dropped {len(dataset.dropped_cols)} as near-constant: "
        f"{', '.join(dataset.dropped_cols)}",
        "- Dropping is decided by measured variance on the training split, not by "
        "a hardcoded list, so the same code works on FD002 and FD004 where "
        "different columns go flat.",
        "",
        "## Correlation with RUL",
        "",
        f"- {len(strong)} features correlate with RUL at |r| >= 0.5:",
    ]
    for name, value in strong.items():
        direction = "falls" if value > 0 else "rises"
        lines.append(f"  - `{name}`: r = {value:+.3f} ({direction} as the engine ages)")
    lines += [
        "",
        "## Class balance",
        "",
        f"- Failure-soon threshold: RUL <= {config.FAILURE_THRESHOLD} cycles",
        f"- Train {dataset.train[config.LABEL_COL].mean():.2%} positive, "
        f"validation {dataset.val[config.LABEL_COL].mean():.2%}, "
        f"test {dataset.test[config.LABEL_COL].mean():.2%}",
        "- The test rate is far lower because test trajectories stop well before "
        "failure. Report precision, recall and ROC-AUC rather than accuracy.",
        "",
        "## Figures",
        "",
        f"- `{lifetimes_fig}`",
        f"- `{clipping_fig}`",
        f"- `{degradation_fig}`",
        f"- `{corr_fig}`",
        f"- `{heatmap_fig}`",
        f"- `{balance_fig}`",
        "",
    ]

    summary_path = write_summary(lines)
    print("Wrote:")
    for artefact in (lifetimes_fig, clipping_fig, degradation_fig, corr_fig,
                     heatmap_fig, balance_fig, summary_path):
        print(f"  {artefact}")


if __name__ == "__main__":
    main()
