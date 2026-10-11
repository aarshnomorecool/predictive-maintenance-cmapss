"""Flask backend for the web dashboard.

    python -m api.server            # then open http://127.0.0.1:8050

Serves the static frontend in web/ and a small JSON API over the same
src.inference / src.explainability / src.sensor_health code the CLI tools use.
Nothing is precomputed or mocked: every number comes from the trained models
in models/ and the C-MAPSS test split.

Live sync: the loaded state is keyed on the artifact file timestamps. When
src.train_classifier writes new models, the next request rebuilds the state,
and the frontend, which polls /api/signature, refetches everything.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from src import config
from src.data_pipeline import load_test_rul
from src.explainability import MODEL_UNITS, explain
from src.inference import (
    PROBABILITY_COLUMNS,
    Artifacts,
    RulArtifacts,
    artifact_signature,
    artifacts_exist,
    engine_history,
    latest_per_engine,
    load_artifacts,
    load_rul_artifacts,
    load_scored_test,
    lstm_artifacts_exist,
    lstm_signature,
    predict_rul_every_cycle,
)
from src.latent_space import (
    N_NEIGHBORS,
    LatentSpace,
    build_latent_space,
    connection_strengths,
    encode_frame,
    layer_model,
)
from src.preprocessing import apply_scaler, make_windows_every_cycle
from src.sensor_health import (
    CRITICAL_LEVEL,
    SENSOR_SPECS,
    SPEC_BY_COLUMN,
    WARNING_LEVEL,
    Calibration,
    calibrate,
    score_history,
    severity,
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = Flask(__name__, static_folder=None)


# ------------------------------------------------------------------ state ----

@dataclass
class State:
    """Everything derived from one set of trained artifacts."""

    signature: tuple
    artifacts: Artifacts
    scored: pd.DataFrame          # test rows with both models' probabilities
    drift: pd.DataFrame           # per-cycle drift for every scored sensor
    calibrations: dict[str, Calibration]
    true_rul_at_end: dict[int, int]
    rul: RulArtifacts | None = None          # LSTM, optional until trained
    predicted_rul: pd.DataFrame | None = None  # unit, cycle, predicted_rul
    space: LatentSpace | None = None          # training engines in LSTM space
    test_vectors: np.ndarray | None = None    # every test cycle in LSTM space
    test_index: pd.DataFrame | None = None    # unit, cycle for test_vectors rows
    display_center: np.ndarray | None = None  # PCA coords -> [-1, 1] for drawing
    display_scale: np.ndarray | None = None
    layers: object = None                     # (keras model, layer list) for activations


def combined_signature() -> tuple:
    """Classifier and LSTM files together, so retraining either reloads."""
    return artifact_signature() + lstm_signature()


_state: State | None = None
_lock = threading.Lock()


def state() -> State:
    """Current state, rebuilt only when the files in models/ change."""
    global _state
    if not artifacts_exist():
        abort(503, description="No trained models. Run: python -m src.train_classifier")
    signature = combined_signature()
    with _lock:
        if _state is None or _state.signature != signature:
            artifacts = load_artifacts()
            scored = load_scored_test(artifacts)
            calibrations = calibrate(artifacts.feature_cols, artifacts.subset)
            truth = load_test_rul(artifacts.subset)
            rul_artifacts, predicted, space = None, None, None
            test_vectors = test_index = center = scale = layers = None
            if lstm_artifacts_exist():
                rul_artifacts = load_rul_artifacts()
                predicted = predict_rul_every_cycle(rul_artifacts, scored)
                space = build_latent_space(rul_artifacts, artifacts.subset)
                test_vectors, test_index = encode_frame(rul_artifacts, scored)
                # Stretch each PCA axis to the same drawn size. PC1 alone holds
                # 90% of the variance, so unscaled the cloud would be a thin line.
                lo, hi = space.coords.min(axis=0), space.coords.max(axis=0)
                center, scale = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
                layers = layer_model(rul_artifacts.model)
            _state = State(
                signature=signature,
                artifacts=artifacts,
                scored=scored,
                drift=score_history(scored, calibrations),
                calibrations=calibrations,
                true_rul_at_end=dict(zip(truth[config.UNIT_COL], truth["true_rul_at_end"])),
                rul=rul_artifacts,
                predicted_rul=predicted,
                space=space,
                test_vectors=test_vectors,
                test_index=test_index,
                display_center=center,
                display_scale=scale,
                layers=layers,
            )
        return _state


def model_param() -> str:
    model = request.args.get("model", "XGBoost")
    if model not in PROBABILITY_COLUMNS:
        abort(400, description=f"model must be one of {list(PROBABILITY_COLUMNS)}")
    return model


def signature_token(signature: tuple) -> str:
    return "-".join(str(mtime) for _, mtime in signature)


def unclipped_rul(s: State, engine: int, cycles: pd.Series, last_cycle: int) -> np.ndarray:
    """True remaining life without the 125 ceiling, for display.

    The models train on the clipped target; showing the clipped value to a
    person would report 125 for an engine that really has 145 cycles left.
    """
    return (last_cycle - cycles.to_numpy()) + s.true_rul_at_end[engine]


def round_list(values, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


# -------------------------------------------------------------------- api ----

@app.get("/api/signature")
def api_signature():
    if not artifacts_exist():
        return jsonify(ready=False, token=None)
    return jsonify(ready=True, token=signature_token(combined_signature()))


@app.get("/api/meta")
def api_meta():
    s = state()
    a = s.artifacts
    scored_columns = set(s.calibrations)
    sensors = []
    for spec in SENSOR_SPECS:
        cal = s.calibrations.get(spec.column)
        sensors.append({
            "column": spec.column,
            "symbol": spec.symbol,
            "description": spec.description,
            "units": spec.units,
            "module": spec.module,
            "monitored": spec.column in scored_columns,
            "informative": bool(cal.informative) if cal else False,
            "rises_with_wear": bool(cal.rises_with_wear) if cal else None,
            "signal_to_noise": round(cal.signal_to_noise, 2) if cal else None,
        })
    metrics = a.metrics_frame().round(4).to_dict(orient="records")
    rul = None
    if s.rul is not None:
        rul = {
            "trained_at": s.rul.trained_at.isoformat(),
            "quick_run": s.rul.is_quick_run,
            "window_size": s.rul.window_size,
            "best_epoch": s.rul.metadata["best_epoch"],
            "epochs_run": s.rul.metadata["epochs_run"],
            "metrics": s.rul.metadata["metrics"],
            "history": s.rul.metadata["history"],
        }
    return jsonify(
        rul=rul,
        rul_ceiling=config.RUL_CLIP,
        token=signature_token(s.signature),
        subset=a.subset,
        trained_at=a.trained_at.isoformat(),
        quick_run=a.is_quick_run,
        failure_threshold=a.failure_threshold,
        feature_cols=a.feature_cols,
        models=list(PROBABILITY_COLUMNS),
        model_units=MODEL_UNITS,
        severity_levels={"warning": WARNING_LEVEL, "critical": CRITICAL_LEVEL},
        sensors=sensors,
        metrics=metrics,
        dropped_cols=[c for c in config.SETTING_COLS + config.SENSOR_COLS
                      if c not in a.feature_cols],
    )


@app.get("/api/fleet")
def api_fleet():
    s = state()
    model = model_param()
    probability_col = PROBABILITY_COLUMNS[model]
    latest = latest_per_engine(s.scored)
    latest_drift = latest_per_engine(s.drift).set_index(config.UNIT_COL)
    predicted_at_end = (
        latest_per_engine(s.predicted_rul).set_index(config.UNIT_COL)["predicted_rul"]
        if s.predicted_rul is not None else None
    )
    informative = [c for c, cal in s.calibrations.items() if cal.informative]

    engines = []
    for _, row in latest.iterrows():
        engine = int(row[config.UNIT_COL])
        drifts = latest_drift.loc[engine, informative]
        worst = drifts.idxmax()
        engines.append({
            "engine": engine,
            "cycles": int(row[config.CYCLE_COL]),
            "probability": round(float(row[probability_col]), 4),
            "true_rul": int(s.true_rul_at_end[engine]),
            "predicted_rul": (round(float(predicted_at_end[engine]), 1)
                              if predicted_at_end is not None else None),
            "in_failure_window": bool(row[config.LABEL_COL]),
            "critical_sensors": int((drifts >= CRITICAL_LEVEL).sum()),
            "warning_sensors": int(((drifts >= WARNING_LEVEL) & (drifts < CRITICAL_LEVEL)).sum()),
            "worst_sensor": SPEC_BY_COLUMN[worst].symbol,
        })
    engines.sort(key=lambda e: e["probability"], reverse=True)
    return jsonify(model=model, token=signature_token(s.signature), engines=engines)


@app.get("/api/engine/<int:engine>")
def api_engine(engine: int):
    s = state()
    model = model_param()
    try:
        history = engine_history(s.scored, engine)
    except ValueError:
        abort(404, description=f"Engine {engine} is not in the test split")
    drift = s.drift[s.drift[config.UNIT_COL] == engine].sort_values(config.CYCLE_COL)

    cycles = history[config.CYCLE_COL]
    last_cycle = int(cycles.iloc[-1])
    sensors = {}
    for column, cal in s.calibrations.items():
        drift_values = drift[column].to_numpy()
        sensors[column] = {
            "readings": round_list(history[column], 4),
            "drift": round_list(drift_values, 4),
            "healthy_level": round(cal.healthy_mean, 4),
            "failure_level": round(cal.failure_mean, 4),
            "severity": severity(float(drift_values[-1]), cal.informative),
        }
    predicted = None
    if s.predicted_rul is not None:
        own = s.predicted_rul[s.predicted_rul[config.UNIT_COL] == engine]
        predicted = round_list(
            own.set_index(config.CYCLE_COL)["predicted_rul"].reindex(cycles), 1)
    flat = {
        column: round(float(history[column].iloc[-1]), 4)
        for column in config.SENSOR_COLS if column not in s.calibrations
    }
    return jsonify(
        engine=engine,
        model=model,
        cycles=[int(c) for c in cycles],
        probability=round_list(history[PROBABILITY_COLUMNS[model]], 4),
        true_rul=[int(v) for v in unclipped_rul(s, engine, cycles, last_cycle)],
        predicted_rul=predicted,
        in_failure_window=[bool(v) for v in history[config.LABEL_COL]],
        sensors=sensors,
        flat_sensors=flat,
        settings={c: round(float(history[c].iloc[-1]), 4)
                  for c in config.SETTING_COLS},
    )


@app.get("/api/engine/<int:engine>/explain")
def api_explain(engine: int):
    """SHAP drivers at one cycle. Defaults to the engine's latest cycle."""
    s = state()
    model = model_param()
    try:
        history = engine_history(s.scored, engine)
    except ValueError:
        abort(404, description=f"Engine {engine} is not in the test split")

    cycle = request.args.get("cycle", type=int)
    row = history.tail(1) if cycle is None else history[history[config.CYCLE_COL] == cycle]
    if row.empty:
        abort(404, description=f"Engine {engine} has no cycle {cycle}")

    explanation = explain(s.artifacts, row, model)
    frame = explanation.contributions()
    return jsonify(
        engine=engine,
        model=model,
        cycle=int(row[config.CYCLE_COL].iloc[0]),
        units=explanation.units,
        base_value=round(explanation.base_value, 5),
        output=round(explanation.prediction(), 5),
        probability=round(float(row[PROBABILITY_COLUMNS[model]].iloc[0]), 4),
        contributions=[
            {"feature": r.feature, "shap": round(float(r.shap_value), 5),
             "reading": round(float(r.reading), 4)}
            for r in frame.itertuples()
        ],
    )


# ------------------------------------------------- vector space / network --

def require_lstm(s: State) -> None:
    if s.rul is None:
        abort(503, description="No LSTM trained. Run: python -m src.train_lstm")


def display_coords(s: State, coords: np.ndarray) -> np.ndarray:
    return (coords - s.display_center) / s.display_scale


@app.get("/api/space")
def api_space():
    """Every moment of every training engine, as a point in 3D."""
    s = state()
    require_lstm(s)
    sp = s.space
    xyz = display_coords(s, sp.coords)
    # How well "average the neighbours' remaining life" predicts each test
    # engine at its last cycle: evidence that nearness here means something.
    last = (s.test_index.groupby(config.UNIT_COL).cumcount(ascending=False) == 0).to_numpy()
    truth = latest_per_engine(s.scored)[config.RUL_COL].to_numpy()
    knn = sp.knn_life_estimate(s.test_vectors[last])
    return jsonify(
        knn_test_rmse=round(float(np.sqrt(np.mean((knn - truth) ** 2))), 2),
        lstm_test_rmse=round(float(s.rul.metadata["metrics"]["LSTM"]["test"]["rmse"]), 2),
        n=len(sp.reference),
        engines=int(sp.reference[config.UNIT_COL].nunique()),
        dims=int(sp.vectors.shape[1]),
        explained_variance=[round(v, 4) for v in sp.explained_variance],
        neighbors=N_NEIGHBORS,
        rul_ceiling=config.RUL_CLIP,
        xyz=round_list(xyz.reshape(-1), 3),
        life_left=[int(v) for v in sp.reference["life_left"]],
        unit=[int(v) for v in sp.reference[config.UNIT_COL]],
        cycle=[int(v) for v in sp.reference[config.CYCLE_COL]],
    )


@app.get("/api/engine/<int:engine>/space")
def api_engine_space(engine: int):
    """One test engine's path through the space, and its nearest training
    neighbours at every cycle, so replay needs no further requests."""
    s = state()
    require_lstm(s)
    mask = (s.test_index[config.UNIT_COL] == engine).to_numpy()
    if not mask.any():
        abort(404, description=f"Engine {engine} is not in the test split")
    vectors = s.test_vectors[mask]
    xyz = display_coords(s, s.space.project(vectors))
    indices, distances = s.space.neighbors(vectors)
    return jsonify(
        engine=engine,
        cycles=[int(c) for c in s.test_index.loc[mask, config.CYCLE_COL]],
        xyz=round_list(xyz.reshape(-1), 3),
        neighbors=indices.tolist(),
        distances=[round_list(row, 3) for row in distances],
        knn_life=round_list(s.space.knn_life_estimate(vectors), 1),
    )


@app.get("/api/network")
def api_network():
    """Layer sizes and trained connection strengths, scaled 0-1 per layer."""
    s = state()
    require_lstm(s)
    layers = []
    for layer in connection_strengths(s.rul.model):
        w = layer["weights"]
        layers.append({
            "name": layer["name"], "kind": layer["kind"], "units": int(layer["units"]),
            "weights": round_list((w / w.max()).reshape(-1), 3),
            "shape": list(w.shape),
        })
    return jsonify(
        parameters=int(s.rul.model.count_params()),
        features=[SPEC_BY_COLUMN[c].symbol if c in SPEC_BY_COLUMN
                  else c.replace("setting_", "Setting ") for c in s.rul.feature_cols],
        window_size=s.rul.window_size,
        layers=layers,
        rul_ceiling=config.RUL_CLIP,
    )


@app.get("/api/engine/<int:engine>/network")
def api_engine_network(engine: int):
    """What every layer computed for one engine at one cycle."""
    s = state()
    require_lstm(s)
    try:
        history = engine_history(s.scored, engine)
    except ValueError:
        abort(404, description=f"Engine {engine} is not in the test split")
    cycle = request.args.get("cycle", type=int, default=int(history[config.CYCLE_COL].iloc[-1]))
    upto = history[history[config.CYCLE_COL] <= cycle]
    if upto.empty:
        abort(404, description=f"Engine {engine} has no cycle {cycle}")

    scaled = apply_scaler(upto, s.rul.scaler, s.rul.feature_cols)
    X, _ = make_windows_every_cycle(scaled, s.rul.feature_cols, s.rul.window_size)
    model, layer_list = s.layers
    outputs = model.predict(X[-1:], verbose=0)
    values = []
    for layer, out in zip(layer_list, outputs):
        out = np.asarray(out)[0]
        entry = {"name": layer.name, "kind": type(layer).__name__}
        if out.ndim == 2:              # LSTM returning a sequence: (time, units)
            entry["sequence"] = round_list(out.reshape(-1), 3)
            entry["shape"] = list(out.shape)
            out = out[-1]
        entry["values"] = round_list(out, 4)
        values.append(entry)
    return jsonify(
        engine=engine,
        cycle=int(upto[config.CYCLE_COL].iloc[-1]),
        input=round_list(X[-1].reshape(-1), 3),
        input_shape=list(X[-1].shape),
        layers=values,
        predicted_rul=round(float(np.clip(values[-1]["values"][0] * config.RUL_CLIP, 0, config.RUL_CLIP)), 1),
    )


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(503)
def json_error(error):
    return jsonify(error=error.description), error.code


# ---------------------------------------------------------------- frontend ----

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(WEB_DIR, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    print(f"\n  Dashboard: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
