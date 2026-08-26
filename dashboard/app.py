"""Streamlit dashboard for the failure-classification models.

Run from the project root so `src` is importable:

    streamlit run dashboard/app.py

Everything shown here is bound to the artifacts in models/ via
src.inference. Retraining while the app is open is picked up automatically:
the caches are keyed on the artifact file timestamps, and a background
fragment watches those timestamps and reruns the app when they change.

The RUL and SHAP panels are placeholders. Those models are not built yet and
this file does not pretend otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Allow `streamlit run dashboard/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.inference import (
    PROBABILITY_COLUMNS,
    Artifacts,
    artifact_signature,
    artifacts_exist,
    engine_history,
    fleet_summary,
    load_artifacts,
    load_scored_test,
)

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

ALERT_COLOR = "#c1121f"
OK_COLOR = "#2a6f97"


# ------------------------------------------------------ cached data access --

@st.cache_resource(show_spinner="Loading trained models...")
def cached_artifacts(signature: tuple) -> Artifacts:
    """Reloads only when the files on disk change."""
    return load_artifacts()


@st.cache_data(show_spinner="Scoring test engines...")
def cached_scored(signature: tuple) -> pd.DataFrame:
    return load_scored_test(cached_artifacts(signature))


@st.fragment(run_every=5)
def watch_for_retrain(known_signature: tuple) -> None:
    """Rerun the whole app when src.train_classifier writes new artifacts."""
    if artifact_signature() != known_signature:
        st.rerun(scope="app")


# ------------------------------------------------------------------ views --

def render_status(artifacts: Artifacts, live: bool) -> None:
    left, right = st.columns([3, 1])
    with left:
        st.caption(
            f"Subset **{artifacts.subset}**  ·  "
            f"trained **{artifacts.trained_at:%d %b %Y, %H:%M}**  ·  "
            f"{len(artifacts.feature_cols)} features  ·  "
            f"failure defined as RUL ≤ {artifacts.failure_threshold} cycles"
        )
    with right:
        st.caption("🟢 watching models/" if live else "⚪ auto-refresh off")

    if artifacts.is_quick_run:
        st.warning(
            "These artifacts came from a `--quick` smoke run and are not "
            "meaningful. Run `python -m src.train_classifier` for real models.",
            icon="⚠️",
        )


def render_fleet(summary: pd.DataFrame, threshold: float) -> None:
    alerting = int(summary["alert"].sum())
    truly_failing = int(summary["actually_failing"].sum())
    caught = int((summary["alert"] & summary["actually_failing"]).sum())

    a, b, c, d = st.columns(4)
    a.metric("Engines monitored", len(summary))
    b.metric("Above alert threshold", alerting)
    c.metric("Actually within failure window", truly_failing)
    d.metric(
        "Correctly flagged",
        f"{caught}/{truly_failing}" if truly_failing else "n/a",
        help="Alerting engines that really are inside the failure window.",
    )

    st.markdown("##### Fleet risk, highest first")
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "engine": st.column_config.NumberColumn("Engine", width="small"),
            "cycles_observed": st.column_config.NumberColumn("Cycles"),
            "failure_probability": st.column_config.ProgressColumn(
                "Failure probability", min_value=0.0, max_value=1.0,
                format="%.3f",
            ),
            "true_rul": st.column_config.NumberColumn(
                "True RUL", help="Ground truth, for evaluation only."
            ),
            "actually_failing": st.column_config.CheckboxColumn("Really failing"),
            "alert": st.column_config.CheckboxColumn(
                f"Alert (≥ {threshold:.2f})"
            ),
        },
    )


def render_engine(history: pd.DataFrame, probability_col: str,
                  threshold: float, failure_threshold: int) -> None:
    latest = history.iloc[-1]
    probability = float(latest[probability_col])

    a, b, c, d = st.columns(4)
    a.metric("Failure probability", f"{probability:.1%}")
    a.progress(min(probability, 1.0))
    b.metric("Status", "ALERT" if probability >= threshold else "Normal")
    c.metric("Cycles observed", int(latest[config.CYCLE_COL]))
    d.metric(
        "True RUL", f"{int(latest[config.RUL_COL])} cycles",
        help="Ground truth from RUL_FD001.txt, shown for comparison only. "
             "The model never sees it.",
    )

    if probability >= threshold:
        st.error(
            f"Failure probability {probability:.1%} is at or above the "
            f"{threshold:.0%} threshold. Schedule inspection.",
            icon="🚨",
        )
    else:
        st.success(
            f"Failure probability {probability:.1%} is below the "
            f"{threshold:.0%} threshold.",
            icon="✅",
        )

    st.markdown("##### Failure probability across the engine's run")
    chart_df = history[[config.CYCLE_COL, probability_col, config.RUL_COL]].rename(
        columns={probability_col: "probability"}
    )
    line = (
        alt.Chart(chart_df)
        .mark_line(color=OK_COLOR, strokeWidth=2)
        .encode(
            x=alt.X(f"{config.CYCLE_COL}:Q", title="Cycle"),
            y=alt.Y("probability:Q", title="Failure probability",
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=[config.CYCLE_COL, alt.Tooltip("probability:Q", format=".3f"),
                     config.RUL_COL],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color=ALERT_COLOR, strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    entered = chart_df[chart_df[config.RUL_COL] <= failure_threshold]
    layers = [line, rule]
    if not entered.empty:
        onset = float(entered[config.CYCLE_COL].iloc[0])
        layers.append(
            alt.Chart(pd.DataFrame({"x": [onset]}))
            .mark_rule(color="#6c757d", strokeDash=[2, 3])
            .encode(x="x:Q")
        )
    st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")
    st.caption(
        f"Red dashed line is the alert threshold. Grey dashed line is where "
        f"true RUL first drops to {failure_threshold} cycles, the point the "
        "model is supposed to catch."
    )


def render_sensors(history: pd.DataFrame, feature_cols: list[str]) -> None:
    st.markdown("##### Sensor readings")
    default = [c for c in ("sensor_11", "sensor_4", "sensor_12") if c in feature_cols]
    chosen = st.multiselect(
        "Sensors to plot", feature_cols, default=default or feature_cols[:3],
        help="Raw units, not scaled.",
    )
    if not chosen:
        st.info("Pick at least one sensor.")
        return

    long = history.melt(
        id_vars=[config.CYCLE_COL], value_vars=chosen,
        var_name="sensor", value_name="reading",
    )
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X(f"{config.CYCLE_COL}:Q", title="Cycle"),
            y=alt.Y("reading:Q", title="Reading",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("sensor:N", title=None),
            facet=alt.Facet("sensor:N", columns=2, title=None),
            tooltip=[config.CYCLE_COL, "sensor", "reading"],
        )
        .resolve_scale(y="independent")
        .properties(height=160, width=340)
    )
    st.altair_chart(chart)


def render_performance(artifacts: Artifacts) -> None:
    st.markdown("##### Held-out performance")
    metrics = artifacts.metrics_frame()
    st.dataframe(
        metrics, width="stretch", hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="%.4f")
            for column in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
    )
    st.caption(
        "Test positives are rare because test engines stop well before "
        "failure, so accuracy runs high regardless. Recall and ROC-AUC are "
        "the meaningful columns."
    )

    st.markdown("##### Feature importance (Random Forest)")
    importance = artifacts.feature_importance().reset_index()
    importance.columns = ["feature", "importance"]
    chart = (
        alt.Chart(importance)
        .mark_bar(color=OK_COLOR)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Impurity-based importance. It is a rough guide, not an explanation. "
        "SHAP replaces this once src/explainability.py exists."
    )


def render_roadmap() -> None:
    st.markdown("##### Not built yet")
    st.info(
        "This dashboard covers the classification branch only. Three panels "
        "are still missing, each blocked on work that has not been done.",
        icon="🧭",
    )
    left, middle, right = st.columns(3)
    with left:
        st.markdown("**RUL estimate**")
        st.caption(
            "A predicted remaining-life figure per engine, from the LSTM. "
            "Blocked on `src/train_lstm.py`. The windowing that feeds it is "
            "already built and tested."
        )
    with middle:
        st.markdown("**SHAP explanations**")
        st.caption(
            "Per-prediction attribution showing which sensors drove a given "
            "alert. Blocked on `src/explainability.py`, and `shap` is not "
            "installed yet."
        )
    with right:
        st.markdown("**Email alerts**")
        st.caption(
            "Automatic notification when probability crosses the threshold. "
            "Blocked on `src/alerts.py`. The threshold control in the sidebar "
            "already drives the alert logic shown here."
        )


# ------------------------------------------------------------------- main --

def main() -> None:
    st.title("⚙️ Predictive Maintenance — Engine Failure Risk")

    if not artifacts_exist():
        st.error("No trained models found.", icon="🚫")
        st.markdown(
            "Train the classifiers first, then this page will populate:\n\n"
            "```powershell\npython -m src.train_classifier\n```"
        )
        st.stop()

    signature = artifact_signature()
    artifacts = cached_artifacts(signature)
    scored = cached_scored(signature)

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Model", list(PROBABILITY_COLUMNS))
        threshold = st.slider(
            "Alert threshold", 0.05, 0.95, 0.80, 0.05,
            help="Failure probability at or above which an engine is flagged.",
        )
        engines = sorted(scored[config.UNIT_COL].unique().tolist())
        engine = st.selectbox("Engine", engines)
        live = st.toggle(
            "Auto-refresh on retrain", value=True,
            help="Watches models/ and reloads when training writes new files.",
        )
        st.divider()
        st.caption(
            f"{len(engines)} test engines · {len(scored):,} scored cycles\n\n"
            "Retrain any time with `python -m src.train_classifier`."
        )

    if live:
        watch_for_retrain(signature)

    render_status(artifacts, live)

    probability_col = PROBABILITY_COLUMNS[model_name]
    summary = fleet_summary(scored, model_name, threshold)
    history = engine_history(scored, engine)

    fleet_tab, engine_tab, model_tab, next_tab = st.tabs(
        ["Fleet", f"Engine {engine}", "Model performance", "Roadmap"]
    )
    with fleet_tab:
        render_fleet(summary, threshold)
    with engine_tab:
        render_engine(
            history, probability_col, threshold, artifacts.failure_threshold
        )
        render_sensors(history, artifacts.feature_cols)
    with model_tab:
        render_performance(artifacts)
    with next_tab:
        render_roadmap()


main()
