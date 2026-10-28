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
from src.sensor_integrity import IntegrityModel, assess, fit_integrity
from src.trust import likely_range, load_trust
from src.inspections import InspectionLog, MEMORY_CYCLES
from src.active_inspection import suggest as suggest_inspections
from src.maintenance_plan import DEFAULT_HORIZON, SAFETY, plan as repair_plan
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
from src.profiles import DEFAULT_MACHINE, MACHINES_DIR, load_profile
from src.bob.findings import Findings, status_from_rule
from src.bob.providers import get_provider, list_providers
from src.bob.workorder import make_work_order, render_html
from src.unfamiliar_wear import pattern_flags


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
    integrity: IntegrityModel | None = None   # sensor integrity guard


def combined_signature() -> tuple:
    """Classifier and LSTM files together, so retraining either reloads."""
    return artifact_signature() + lstm_signature()


_state: State | None = None
_lock = threading.Lock()

alerts = AlertManager()


class TechnicianNotes:
    """Sensors a technician has marked faulty, per engine. Kept on disk so the
    judgement survives a restart; models/ is gitignored."""

    path = config.MODELS_DIR / "technician_notes.json"

    def __init__(self):
        import json
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.lock = threading.Lock()

    def excluded(self, key) -> list[str]:
        with self.lock:
            return list(self.data.get(str(key), {}).get("excluded", []))

    def set_excluded(self, key, sensor: str, faulty: bool) -> list[str]:
        import json
        from src.artifact_io import atomic_write_text
        with self.lock:
            entry = self.data.setdefault(str(key), {})
            current = set(entry.get("excluded", []))
            (current.add if faulty else current.discard)(sensor)
            entry["excluded"] = sorted(current)
            try:
                atomic_write_text(self.path, json.dumps(self.data, indent=1))
            except OSError:
                pass
            return entry["excluded"]


notes = TechnicianNotes()
inspections = InspectionLog()


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

    link = EmailSettings.from_env().dashboard_url.rstrip("/") + f"/#engine={SIM_ID}"
    if _state is not None and _state.integrity is not None:
        # sensors the technician already marked faulty need no alert
        detected = [c for c in scores.get("faulty", []) if c not in runner.scorer.forced]
        alerts.check_sensors(SIM_ID, SIM_LABEL, runner.engine.cycle,
                             _fault_details(_state, pd.DataFrame(runner.engine.rows), detected),
                             probability, scores.get("predicted_rul"), link)
    if scores.get("unfamiliar"):
        res = assess(pd.DataFrame(runner.engine.rows), _state.integrity) if _state and _state.integrity else None
        n = int((res.z.abs().iloc[-1] > 8).sum()) if res is not None else 0
        alerts.check_unfamiliar(SIM_ID, SIM_LABEL, runner.engine.cycle, True, n,
                                probability, scores.get("predicted_rul"), link)
    level = inspections.level_at(SIM_ID, runner.engine.cycle, alerts.threshold)
    alerts.evaluate(SIM_ID, SIM_LABEL, runner.engine.cycle, probability,
                    scores.get("predicted_rul"), drivers, link, threshold=level)


def _fault_details(s, raw: pd.DataFrame, faulty: list[str]) -> list[dict]:
    """What each flagged sensor says against what the healthy ones agree on."""
    if not faulty:
        return []
    res = assess(raw, s.integrity)
    out = []
    for c in faulty:
        spec = SPEC_BY_COLUMN[c]
        out.append({"column": c, "symbol": spec.symbol, "description": spec.description,
                    "reading": round(float(raw[c].iloc[-1]), 3),
                    "sensor_drift": float(res.drift[c].iloc[-1]),
                    "consensus": float(res.consensus.iloc[-1]),
                    "z": round(float(res.z[c].iloc[-1]), 1),
                    "peers": len(s.integrity.sensors) - len(faulty)})
    return out


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
            integrity = fit_integrity(calibrations, artifacts.subset)
            runner.bind(artifacts, rul_artifacts,
                        measure_profile(artifacts.feature_cols, artifacts.subset), integrity)
            runner.scorer.forced = notes.excluded(SIM_ID)
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
                integrity=integrity,
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

    # Engines where a technician excluded a sensor are scored on repaired readings,
    # so the list agrees with the engine view
    repaired = {}
    for key in list(notes.data):
        if key.isdigit() and notes.excluded(key) and s.integrity is not None:
            d = engine_data(s, key)
            integrity_payload(s, d, model)
            repaired[int(key)] = (float(d.history[probability_col].iloc[-1]),
                                  d.predicted[-1] if d.predicted else None)

    engines = []
    for _, row in latest.iterrows():
        engine = int(row[config.UNIT_COL])
        drifts = latest_drift.loc[engine, informative]
        worst = drifts.idxmax()
        prob = repaired[engine][0] if engine in repaired else float(row[probability_col])
        engines.append({
            "engine": engine,
            "cycles": int(row[config.CYCLE_COL]),
            "probability": round(prob, 4),
            "sensor_excluded": engine in repaired,
            "true_rul": int(s.true_rul_at_end[engine]),
            "predicted_rul": (round(float(repaired[engine][1]), 1) if engine in repaired and repaired[engine][1] is not None
                              else round(float(predicted_at_end[engine]), 1) if predicted_at_end is not None else None),
            "in_failure_window": bool(row[config.LABEL_COL]),
            "critical_sensors": int((drifts >= CRITICAL_LEVEL).sum()),
            "warning_sensors": int(((drifts >= WARNING_LEVEL) & (drifts < CRITICAL_LEVEL)).sum()),
            "worst_sensor": SPEC_BY_COLUMN[worst].symbol,
        })
    engines.sort(key=lambda e: e["probability"], reverse=True)

    predicted_now = {e["engine"]: e["predicted_rul"] for e in engines if e["predicted_rul"] is not None}
    latest_for_suggest = latest.copy()
    for engine, (prob, _) in repaired.items():
        latest_for_suggest.loc[latest_for_suggest[config.UNIT_COL] == engine, probability_col] = prob
    inspect_next = suggest_inspections(latest_for_suggest, alerts.threshold, predicted_now)
    reasons = {d["engine"]: d["reasons"] for d in inspect_next}
    for e in engines:
        e["inspect_next"] = reasons.get(e["engine"])
        e["inspected"] = bool(inspections.history(e["engine"]))
        m = inspections.memory(e["engine"])
        e["memory_level"] = m["level"] if m and m["from_cycle"] <= e["cycles"] <= m["until_cycle"] else None
    return jsonify(model=model, token=signature_token(s.signature), engines=engines,
                   inspect_next=inspect_next,
                   repair_plan=repair_plan(predicted_now) if predicted_now else None)


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
    integrity, probability_raw, predicted_raw = integrity_payload(s, d, model)
    trust = load_trust()
    predicted_range = None
    if trust and d.predicted:
        # every engine here is monitored from new, so cycles observed = position in its record
        bounds = [likely_range(p, k + 1, trust) for k, p in enumerate(d.predicted)]
        predicted_range = {"lo": [b[0] for b in bounds], "hi": [b[1] for b in bounds],
                           "window": trust["window"], "trusted_after": trust["trusted_after"],
                           "coverage": trust["coverage"]}
    key = d.key
    companions = None
    if d.predicted and key != SIM_ID:
        latest_pred = d.predicted[-1]
        if latest_pred <= SAFETY + DEFAULT_HORIZON:
            fleet_pred = {}
            if s.predicted_rul is not None:
                fleet_pred = latest_per_engine(s.predicted_rul).set_index(config.UNIT_COL)["predicted_rul"].to_dict()
            fleet_pred[key] = latest_pred
            companions = repair_plan(fleet_pred, anchor=key)
            companions["join"] = [{"engine": int(e), "predicted_rul": round(float(fleet_pred[e]), 1)}
                                  for e in companions["join"]]
    return jsonify(
        inspections=inspections.history(key),
        alert_memory=inspections.memory(key),
        repair_group=companions,
        predicted_range=predicted_range,
        engine=d.key,
        model=model,
        cycles=[int(c) for c in history[config.CYCLE_COL]],
        probability=round_list(history[PROBABILITY_COLUMNS[model]], 4),
        probability_raw=probability_raw,
        predicted_rul_raw=predicted_raw,
        integrity=integrity,
        true_rul=d.truth,
        truth_label=d.truth_label,
        predicted_rul=d.predicted,
        in_failure_window=d.in_window,
        sensors=sensors,
        flat_sensors=flat,
        settings={c: round(float(history[c].iloc[-1]), 4) for c in config.SETTING_COLS},
        sim=d.sim,
    )


def integrity_payload(s: State, d: EngineData, model: str):
    """Cross-check of every guarded sensor at every cycle. When a sensor is
    flagged, also return what the models said before the repair."""
    if s.integrity is None:
        return None, None, None
    raw_cols = [config.UNIT_COL, config.CYCLE_COL] + config.SETTING_COLS + config.SENSOR_COLS
    forced = [c for c in notes.excluded(d.key) if c in s.integrity.sensors]
    res = assess(d.history[raw_cols], s.integrity, forced)
    payload = {
        **s.integrity.public(),
        "excluded_by_technician": forced,
        "z": {c: round_list(res.z[c], 2) for c in s.integrity.sensors},
        "flags": {c: [bool(v) for v in res.flags[c]] for c in s.integrity.sensors},
        "consensus": round_list(res.consensus, 4),
        "any_flagged": bool(res.flags.to_numpy().any()),
    }
    from src.unfamiliar_wear import pattern_flags
    count, unfamiliar = pattern_flags(res)
    payload["unfamiliar"] = [bool(v) for v in unfamiliar]
    payload["disagreeing"] = [int(v) for v in count]
    if not payload["any_flagged"]:
        return payload, None, None
    col = PROBABILITY_COLUMNS[model]
    if d.key == SIM_ID:
        # the simulator already scored repaired readings; it kept the raw scores too
        prob_raw = round_list(d.history["raw_" + col], 4)
        pred_raw = (None if d.history["raw_predicted_rul"].isna().all()
                    else round_list(d.history["raw_predicted_rul"], 1))
        return payload, prob_raw, pred_raw
    # A recorded engine with a flagged sensor: score the repaired readings now
    from src.inference import score_frame
    prob_raw = round_list(d.history[col], 4)
    repaired = score_frame(s.artifacts, res.corrected)
    d.history[col] = repaired[col].to_numpy()
    pred_raw = d.predicted
    if s.rul is not None:
        d.predicted = round_list(predict_rul_every_cycle(s.rul, res.corrected)["predicted_rul"], 1)
    return payload, prob_raw, pred_raw


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

    repaired = False
    if s.integrity is not None:
        raw_cols = [config.UNIT_COL, config.CYCLE_COL] + config.SETTING_COLS + config.SENSOR_COLS
        upto = history[history[config.CYCLE_COL] <= int(row[config.CYCLE_COL].iloc[0])][raw_cols]
        res = assess(upto, s.integrity, notes.excluded(engine_data_key(engine_id)))
        if res.flags.iloc[-1].any():
            # explain what the models actually scored: the repaired reading
            row = row.copy()
            for c in res.flags.columns[res.flags.iloc[-1].to_numpy()]:
                row[c] = res.corrected[c].iloc[-1]
            row[PROBABILITY_COLUMNS[model]] = score_frame_one(s, row, model)
            repaired = True
    explanation = explain(s.artifacts, row, model)
    frame = explanation.contributions()
    return jsonify(
        repaired=repaired,
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


def score_frame_one(s: State, row: pd.DataFrame, model: str) -> float:
    from src.inference import score_frame
    return float(score_frame(s.artifacts, row)[PROBABILITY_COLUMNS[model]].iloc[0])


@app.post("/api/engine/<engine_id>/inspection")
def api_inspection(engine_id: str):
    """Record what a technician found."""
    s = state()
    body = request.get_json(silent=True) or {}
    d = engine_data(s, engine_id)
    cycle = int(body.get("cycle") or d.history[config.CYCLE_COL].iloc[-1])
    row = d.history[d.history[config.CYCLE_COL] == cycle]
    if row.empty:
        abort(400, description=f"No cycle {cycle} for this engine")
    k = row.index[0] - d.history.index[0]
    risk = float(row[PROBABILITY_COLUMNS[alerts.model]].iloc[0])
    predicted = d.predicted[k] if d.predicted else None
    try:
        entry = inspections.record(d.key, cycle, str(body.get("outcome")), risk, predicted, str(body.get("note", "")))
    except ValueError as exc:
        abort(400, description=str(exc))
    if d.key == SIM_ID:
        alerts.acknowledge(SIM_ID)
    return jsonify(entry=entry, memory=inspections.memory(d.key), history=inspections.history(d.key))


@app.post("/api/engine/<engine_id>/sensor-note")
def api_sensor_note(engine_id: str):
    """A technician marks a sensor faulty, or trusts it again."""
    s = state()
    body = request.get_json(silent=True) or {}
    sensor, faulty = body.get("sensor"), bool(body.get("faulty"))
    if s.integrity is None or sensor not in s.integrity.sensors:
        abort(400, description="Only cross-checked sensors can be excluded; their value is rebuilt "
                               "from the other cross-checked sensors.")
    key = engine_data_key(engine_id)
    excluded = notes.set_excluded(key, sensor, faulty)
    if key == SIM_ID:
        with runner.lock:
            runner.scorer.forced = excluded
            runner.version += 1
    return jsonify(engine=key, excluded=excluded)


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
            inspections.clear(SIM_ID)
        elif action == "demo":
            runner.run_demo()
            alerts.reset_engine(SIM_ID)
            inspections.clear(SIM_ID)
        elif action == "speed":
            runner.set_speed(int(body["value"]))
        elif action == "wear":
            runner.set_wear(str(body["value"]))
        elif action == "fault":
            runner.set_fault(bool(body["value"]))
        elif action == "fan_fault":
            runner.set_fan_fault(bool(body["value"]))
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


# --------------------------------------------------------------- insights --

INSIGHT_COMMANDS = {
    "trust": "python -m src.trust",
    "inspection": "python -m src.active_inspection",
    "repairs": "python -m src.maintenance_plan",
    "sensor_kit": "python -m src.sensor_kit",
    "unfamiliar": "python -m src.unfamiliar_wear",
    "adaptation": "python -m src.adaptation",
    "integrity": "python -m src.sensor_integrity",
}


@app.get("/api/insights/<name>")
def api_insight(name: str):
    """Results of the fleet-level analyses, computed offline by their CLI."""
    if name not in INSIGHT_COMMANDS:
        abort(404, description=f"No analysis called {name}")
    import json
    path = config.MODELS_DIR / "insights" / f"{name}.json"
    try:
        return jsonify({"available": True, **json.loads(path.read_text(encoding="utf-8"))})
    except (OSError, ValueError):
        # A normal answer, not an error: the analysis simply has not been run
        return jsonify(available=False, command=INSIGHT_COMMANDS[name])


# ---------------------------------------------------------------------- bob ----

def engine_findings(s: State, engine_id: str, cycle: int | None = None,
                    model: str | None = None, threshold: float | None = None):
    model = model or model_param()
    profile = load_profile(DEFAULT_MACHINE)
    d = engine_data(s, engine_id)
    history = d.history
    if history.empty:
        abort(404, description=f"No data for engine {engine_id}")

    if cycle is not None and cycle in history[config.CYCLE_COL].values:
        idx = history.index[history[config.CYCLE_COL] == cycle][0]
        c_val = int(cycle)
    else:
        idx = history.index[-1]
        c_val = int(history[config.CYCLE_COL].iloc[-1])

    pos = history.index.get_loc(idx)
    prob_col = PROBABILITY_COLUMNS[model]
    prob = float(history[prob_col].iloc[pos])

    if threshold is None:
        try:
            threshold = float(request.args.get("threshold", 0.5))
        except (ValueError, TypeError):
            threshold = 0.5

    r_now = float(d.predicted[pos]) if (d.predicted and pos < len(d.predicted)) else None
    trust = load_trust()
    rng = likely_range(r_now, pos + 1, trust) if (trust and r_now is not None) else None

    drift = {c: float(d.drift[c].iloc[pos]) for c in s.calibrations if c in d.drift.columns}

    raw_cols = [config.UNIT_COL, config.CYCLE_COL] + config.SETTING_COLS + config.SENSOR_COLS
    forced = [c for c in notes.excluded(d.key) if c in s.integrity.sensors] if s.integrity else []
    res = assess(d.history.iloc[:pos + 1][raw_cols], s.integrity, forced) if s.integrity else None
    flagged = [c for c in res.flags.columns if bool(res.flags[c].iloc[-1])] if res is not None else []

    unfamiliar = False
    if res is not None:
        try:
            _, unf_series = pattern_flags(res)
            unfamiliar = bool(unf_series.iloc[-1]) if not unf_series.empty else False
        except Exception:
            unfamiliar = False

    row = d.history.iloc[[pos]][config.RAW_COLUMNS]
    if flagged and res is not None:
        row = res.corrected.iloc[[-1]]
    exp = explain(s.artifacts, row, model).contributions()
    shap = {r.feature: float(r.shap_value) for r in exp.itertuples() if profile.sensor(r.feature)}

    status = status_from_rule(prob, threshold, r_now, profile.labelling.get("failure_window", 30), drift)

    f = Findings(machine=profile.id, unit=str(d.key), cycle=c_val, status=status,
                 probability=prob, threshold=threshold, model=model, rul=r_now, rul_range=rng,
                 shap=shap, drift=drift, flagged=flagged, unfamiliar=unfamiliar)
    return profile, f


@app.get("/api/bob/providers")
def api_bob_providers():
    active = get_provider()
    return jsonify(providers=list_providers(), active=active.name, active_label=active.label)


@app.get("/api/bob/workorder/<engine_id>")
def api_bob_workorder(engine_id: str):
    s = state()
    cycle = request.args.get("cycle", type=int)
    model = request.args.get("model")
    prov_name = request.args.get("provider")
    threshold = request.args.get("threshold", type=float)

    profile, f = engine_findings(s, engine_id, cycle=cycle, model=model, threshold=threshold)
    provider = get_provider(prov_name) if prov_name else None
    order = make_work_order(f, profile, provider=provider)
    return jsonify(order)


@app.get("/api/bob/workorder/<engine_id>/html")
def api_bob_workorder_html(engine_id: str):
    s = state()
    cycle = request.args.get("cycle", type=int)
    model = request.args.get("model")
    prov_name = request.args.get("provider")
    threshold = request.args.get("threshold", type=float)

    profile, f = engine_findings(s, engine_id, cycle=cycle, model=model, threshold=threshold)
    provider = get_provider(prov_name) if prov_name else None
    order = make_work_order(f, profile, provider=provider)
    return render_html(order), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.post("/api/bob/chat")
def api_bob_chat():
    s = state()
    data = request.get_json(silent=True) or {}
    engine_id = str(data.get("engine_id") or "sim")
    cycle = data.get("cycle")
    if cycle is not None:
        try:
            cycle = int(cycle)
        except (ValueError, TypeError):
            cycle = None
    model = data.get("model")
    prov_name = data.get("provider")
    threshold = data.get("threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            threshold = 0.5
    message = str(data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return jsonify(error="Empty message"), 400

    profile, f = engine_findings(s, engine_id, cycle=cycle, model=model, threshold=threshold)
    from src.bob.chat import chat_with_bob
    res = chat_with_bob(f, profile, message=message, history=history, provider_name=prov_name)
    return jsonify(res)


@app.get("/api/bob/grounding")
def api_bob_grounding():
    import json
    path = config.MODELS_DIR / "insights" / "bob_grounding.json"
    if path.exists():
        try:
            return jsonify(available=True, **json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            return jsonify(available=False, error=str(exc))
    return jsonify(available=False, command="python -m src.bob")


@app.get("/api/machines")
def api_machines():
    import json
    machines = []
    if MACHINES_DIR.exists():
        for p in sorted(MACHINES_DIR.iterdir()):
            prof_file = p / "profile.json"
            if prof_file.exists():
                try:
                    data = json.loads(prof_file.read_text(encoding="utf-8"))
                    machines.append({
                        "id": data.get("id", p.name),
                        "name": data.get("name", p.name),
                        "machine_type": data.get("machine_type", "industrial equipment"),
                        "unit_label": data.get("unit_label", "unit"),
                        "time_unit": data.get("time_unit", "cycle"),
                        "tasks": data.get("tasks_enabled", {}),
                    })
                except Exception:
                    pass
    return jsonify(machines=machines, active=DEFAULT_MACHINE)


@app.get("/api/machines/<machine_id>")
def api_machine_profile(machine_id: str):
    try:
        prof = load_profile(machine_id)
        return jsonify(prof.data)
    except Exception as exc:
        abort(404, description=f"Machine profile {machine_id} not found: {exc}")


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
