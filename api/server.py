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
from src.alerts import AlertManager, EmailSettings
from src.simulator import SimulationRunner, measure_profile
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

SIM_ID = "sim"
SIM_LABEL = "Simulated engine"

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

alerts = AlertManager()


def _sim_cycle(runner: SimulationRunner, scores: dict) -> None:
    """Called by the simulation thread after every new cycle."""
    probability = scores[PROBABILITY_COLUMNS[alerts.model]]

    def drivers() -> list[dict]:
        # Top sensors pushing the risk up at this cycle, by SHAP
        history = pd.DataFrame(runner.engine.rows)
        explanation = explain(runner.scorer.artifacts, history.tail(1), alerts.model)
        frame = explanation.contributions()
        frame = frame[frame["shap_value"] > 0]
        return [{"column": r.feature,
                 "symbol": SPEC_BY_COLUMN[r.feature].symbol if r.feature in SPEC_BY_COLUMN else r.feature,
                 "description": SPEC_BY_COLUMN[r.feature].description if r.feature in SPEC_BY_COLUMN else r.feature,
                 "reading": round(float(r.reading), 3)}
                for r in frame.head(3).itertuples()]

    alerts.evaluate(SIM_ID, SIM_LABEL, runner.engine.cycle, probability,
                    scores.get("predicted_rul"), drivers,
                    EmailSettings.from_env().dashboard_url.rstrip("/") + f"/#engine={SIM_ID}")


runner = SimulationRunner(on_cycle=_sim_cycle)


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
            runner.bind(artifacts, rul_artifacts,
                        measure_profile(artifacts.feature_cols, artifacts.subset))
            alerts.reset_engine(SIM_ID)
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


@dataclass
class EngineData:
    """One engine's recorded or simulated run, in the shape every view uses."""

    key: object                    # int test unit, or SIM_ID
    history: pd.DataFrame          # raw readings plus prob_rf / prob_xgb
    drift: pd.DataFrame
    predicted: list | None         # LSTM prediction per cycle
    truth: list[int]               # true (test) or projected (sim) life left
    in_window: list[bool]
    truth_label: str
    sim: dict | None = None        # simulator status and events, for SIM_ID


def engine_data(s: State, engine_id: str) -> EngineData:
    if engine_id == SIM_ID:
        history, events, status = runner.snapshot()
        drift = score_history(history, s.calibrations).sort_values(config.CYCLE_COL)
        truth = [int(round(v)) for v in history["projected_life"]]
        predicted = (None if history["predicted_rul"].isna().all()
                     else round_list(history["predicted_rul"], 1))
        return EngineData(
            key=SIM_ID, history=history, drift=drift, predicted=predicted, truth=truth,
            in_window=[t <= s.artifacts.failure_threshold for t in truth],
            truth_label="Projected life left",
            sim={**status, "events": events[-12:], "alert_state": alerts.status.get(SIM_ID, "normal"),
                 "last_alert_id": alerts.last_id()},
        )
    try:
        engine = int(engine_id)
        history = engine_history(s.scored, engine)
    except ValueError:
        abort(404, description=f"Engine {engine_id} is not in the test split")
    drift = s.drift[s.drift[config.UNIT_COL] == engine].sort_values(config.CYCLE_COL)
    cycles = history[config.CYCLE_COL]
    predicted = None
    if s.predicted_rul is not None:
        own = s.predicted_rul[s.predicted_rul[config.UNIT_COL] == engine]
        predicted = round_list(own.set_index(config.CYCLE_COL)["predicted_rul"].reindex(cycles), 1)
    return EngineData(
        key=engine, history=history, drift=drift, predicted=predicted,
        truth=[int(v) for v in unclipped_rul(s, engine, cycles, int(cycles.iloc[-1]))],
        in_window=[bool(v) for v in history[config.LABEL_COL]],
        truth_label="True life left",
    )


@app.get("/api/engine/<engine_id>")
def api_engine(engine_id: str):
    s = state()
    model = model_param()
    d = engine_data(s, engine_id)
    history = d.history
    sensors = {}
    for column, cal in s.calibrations.items():
        drift_values = d.drift[column].to_numpy()
        sensors[column] = {
            "readings": round_list(history[column], 4),
            "drift": round_list(drift_values, 4),
            "healthy_level": round(cal.healthy_mean, 4),
            "failure_level": round(cal.failure_mean, 4),
            "severity": severity(float(drift_values[-1]), cal.informative),
        }
    flat = {
        column: round(float(history[column].iloc[-1]), 4)
        for column in config.SENSOR_COLS if column not in s.calibrations
    }
    return jsonify(
        engine=d.key,
        model=model,
        cycles=[int(c) for c in history[config.CYCLE_COL]],
        probability=round_list(history[PROBABILITY_COLUMNS[model]], 4),
        true_rul=d.truth,
        truth_label=d.truth_label,
        predicted_rul=d.predicted,
        in_failure_window=d.in_window,
        sensors=sensors,
        flat_sensors=flat,
        settings={c: round(float(history[c].iloc[-1]), 4) for c in config.SETTING_COLS},
        sim=d.sim,
    )


@app.get("/api/engine/<engine_id>/explain")
def api_explain(engine_id: str):
    """SHAP drivers at one cycle. Defaults to the engine's latest cycle."""
    s = state()
    model = model_param()
    history = engine_data(s, engine_id).history
    cycle = request.args.get("cycle", type=int)
    row = history.tail(1) if cycle is None else history[history[config.CYCLE_COL] == cycle]
    if row.empty:
        abort(404, description=f"Engine {engine_id} has no cycle {cycle}")

    explanation = explain(s.artifacts, row, model)
    frame = explanation.contributions()
    return jsonify(
        engine=engine_data_key(engine_id),
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


def engine_data_key(engine_id: str):
    return SIM_ID if engine_id == SIM_ID else int(engine_id)


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


@app.get("/api/engine/<engine_id>/space")
def api_engine_space(engine_id: str):
    """One engine's path through the space, and its nearest training
    neighbours at every cycle, so replay needs no further requests."""
    s = state()
    require_lstm(s)
    if engine_id == SIM_ID:
        history = engine_data(s, SIM_ID).history
        vectors, index = encode_frame(s.rul, history)
        cycles = index[config.CYCLE_COL]
    else:
        engine = engine_data_key(engine_id)
        mask = (s.test_index[config.UNIT_COL] == engine).to_numpy()
        if not mask.any():
            abort(404, description=f"Engine {engine} is not in the test split")
        vectors = s.test_vectors[mask]
        cycles = s.test_index.loc[mask, config.CYCLE_COL]
    xyz = display_coords(s, s.space.project(vectors))
    indices, distances = s.space.neighbors(vectors)
    return jsonify(
        engine=engine_data_key(engine_id),
        cycles=[int(c) for c in cycles],
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


@app.get("/api/engine/<engine_id>/network")
def api_engine_network(engine_id: str):
    """What every layer computed for one engine at one cycle."""
    s = state()
    require_lstm(s)
    history = engine_data(s, engine_id).history
    engine = engine_data_key(engine_id)
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


# ------------------------------------------------------------ simulation --

@app.get("/api/sim/status")
def api_sim_status():
    """Small, cheap summary for the fleet list's live row."""
    s = state()
    model = model_param()
    with runner.lock:
        last = runner.scores[-1] if runner.scores else {}
        e = runner.engine
        return jsonify(
            cycle=e.cycle, running=runner.running, failed=e.failed, demo=runner.demo is not None,
            probability=last.get(PROBABILITY_COLUMNS[model]),
            predicted_rul=last.get("predicted_rul"),
            projected_life=round(e.projected_life(), 1),
            alert_state=alerts.status.get(SIM_ID, "normal"),
            last_alert_id=alerts.last_id(),
            version=runner.version,
        )


@app.get("/api/sim/ranges")
def api_sim_ranges():
    """Range each channel took in the training data, for manual overrides."""
    state()
    p = runner.profile
    return jsonify({c: {"low": round(p.low[c], 4), "high": round(p.high[c], 4),
                        "healthy": round(p.healthy[c], 4), "failure": round(p.failure[c], 4)}
                    for c in p.columns})


@app.post("/api/sim/control")
def api_sim_control():
    state()
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    try:
        if action == "start":
            runner.start()
        elif action == "pause":
            runner.pause()
        elif action == "reset":
            runner.reset()
            alerts.reset_engine(SIM_ID)
        elif action == "demo":
            runner.run_demo()
            alerts.reset_engine(SIM_ID)
        elif action == "speed":
            runner.set_speed(int(body["value"]))
        elif action == "wear":
            runner.set_wear(str(body["value"]))
        elif action == "fault":
            runner.set_fault(bool(body["value"]))
        elif action == "override":
            value = body.get("value")
            runner.set_override(str(body["column"]), None if value is None else float(value))
        else:
            abort(400, description=f"Unknown action {action!r}")
    except (KeyError, TypeError, ValueError) as exc:
        abort(400, description=str(exc))
    return jsonify(ok=True, version=runner.version)


# ---------------------------------------------------------------- alerts --

@app.get("/api/alerts")
def api_alerts():
    return jsonify(entries=alerts.entries(), email=EmailSettings.from_env().public(),
                   rule=alerts.rule(), last_id=alerts.last_id())


@app.post("/api/alerts/test")
def api_alerts_test():
    entry = alerts.send_test(EmailSettings.from_env().dashboard_url)
    return jsonify(entry)


@app.post("/api/alerts/rule")
def api_alerts_rule():
    body = request.get_json(silent=True) or {}
    model = body.get("model")
    if model is not None and model not in PROBABILITY_COLUMNS:
        abort(400, description=f"model must be one of {list(PROBABILITY_COLUMNS)}")
    alerts.set_rule(threshold=body.get("threshold"), model=model)
    return jsonify(alerts.rule())


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
