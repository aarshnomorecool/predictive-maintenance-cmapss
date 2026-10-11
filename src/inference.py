"""Loading trained artifacts and scoring engines with them.

Kept separate from the dashboard so the Streamlit UI stays presentational and
the planned Flask API can call exactly the same code paths.

The scaler saved at training time is reused here rather than refitted. A
refitted scaler would usually match, but "usually" is how a transform quietly
drifts away from the one the model was trained against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_pipeline import build_dataset
from src.preprocessing import apply_scaler, make_windows_every_cycle

RF_FILE = "rf_classifier.joblib"
XGB_FILE = "xgb_classifier.joblib"
SCALER_FILE = "scaler.joblib"
METADATA_FILE = "classifier_metadata.json"

MODEL_FILES = (RF_FILE, XGB_FILE, SCALER_FILE, METADATA_FILE)

PROBABILITY_COLUMNS = {"RandomForest": "prob_rf", "XGBoost": "prob_xgb"}


def artifact_paths() -> list[Path]:
    return [config.MODELS_DIR / name for name in MODEL_FILES]


def artifacts_exist() -> bool:
    return all(path.exists() for path in artifact_paths())


def artifact_signature() -> tuple:
    """Fingerprint of the files on disk, used to detect a retrain.

    Caching on this means the dashboard picks up a newly trained model without
    a manual restart, and without reloading on every interaction.
    """
    return tuple(
        (path.name, path.stat().st_mtime_ns if path.exists() else 0)
        for path in artifact_paths()
    )


@dataclass
class Artifacts:
    """Everything produced by src.train_classifier, loaded back."""

    models: dict[str, object]
    scaler: object
    metadata: dict
    trained_at: pd.Timestamp

    @property
    def feature_cols(self) -> list[str]:
        return list(self.metadata["feature_cols"])

    @property
    def subset(self) -> str:
        return self.metadata["subset"]

    @property
    def is_quick_run(self) -> bool:
        return bool(self.metadata.get("quick_run", False))

    @property
    def failure_threshold(self) -> int:
        return int(self.metadata["failure_threshold"])

    def feature_importance(self) -> pd.Series:
        importance = self.metadata.get("rf_feature_importance", {})
        return pd.Series(importance, dtype=float).sort_values(ascending=False)

    def metrics_frame(self) -> pd.DataFrame:
        rows = [
            {"model": model, "split": split, **scores}
            for model, splits in self.metadata["metrics"].items()
            for split, scores in splits.items()
        ]
        return pd.DataFrame(rows)


def load_artifacts() -> Artifacts:
    """Read the trained models back off disk.

    Raises FileNotFoundError with an actionable message if training has not
    been run, which is the state the dashboard starts in.
    """
    missing = [path.name for path in artifact_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {config.MODELS_DIR}. "
            "Run: python -m src.train_classifier"
        )

    metadata = json.loads(
        (config.MODELS_DIR / METADATA_FILE).read_text(encoding="utf-8")
    )
    trained_at = pd.Timestamp(
        (config.MODELS_DIR / METADATA_FILE).stat().st_mtime, unit="s"
    )
    return Artifacts(
        models={
            "RandomForest": joblib.load(config.MODELS_DIR / RF_FILE),
            "XGBoost": joblib.load(config.MODELS_DIR / XGB_FILE),
        },
        scaler=joblib.load(config.MODELS_DIR / SCALER_FILE),
        metadata=metadata,
        trained_at=trained_at,
    )


def score_frame(
    artifacts: Artifacts,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one failure-probability column per model to raw engine rows.

    Takes unscaled rows so the caller keeps readable sensor units for plotting,
    and scales internally using the training-time scaler.
    """
    feature_cols = artifacts.feature_cols
    missing = [column for column in feature_cols if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Input is missing trained features: {missing}")

    scaled = apply_scaler(raw_df, artifacts.scaler, feature_cols)
    X = scaled[feature_cols].to_numpy(dtype=np.float32)

    scored = raw_df.copy()
    for name, model in artifacts.models.items():
        scored[PROBABILITY_COLUMNS[name]] = model.predict_proba(X)[:, 1]
    return scored


def load_scored_test(artifacts: Artifacts) -> pd.DataFrame:
    """The full labelled test split with both models' probabilities attached."""
    dataset = build_dataset(artifacts.subset)
    return score_frame(artifacts, dataset.test)


def latest_per_engine(scored: pd.DataFrame) -> pd.DataFrame:
    """Each engine's most recent cycle, which is what an operator acts on."""
    return (
        scored.sort_values(config.CYCLE_COL)
        .groupby(config.UNIT_COL, as_index=False)
        .tail(1)
        .sort_values(config.UNIT_COL)
        .reset_index(drop=True)
    )


def fleet_summary(
    scored: pd.DataFrame,
    model_name: str,
    alert_threshold: float,
) -> pd.DataFrame:
    """Per-engine risk table, highest risk first."""
    probability_col = PROBABILITY_COLUMNS[model_name]
    latest = latest_per_engine(scored)

    summary = pd.DataFrame({
        "engine": latest[config.UNIT_COL].to_numpy(),
        "cycles_observed": latest[config.CYCLE_COL].to_numpy(),
        "failure_probability": latest[probability_col].to_numpy(),
        "true_rul": latest[config.RUL_COL].to_numpy(),
        "actually_failing": latest[config.LABEL_COL].to_numpy().astype(bool),
    })
    summary["alert"] = summary["failure_probability"] >= alert_threshold
    return summary.sort_values(
        "failure_probability", ascending=False
    ).reset_index(drop=True)


# ------------------------------------------------------- RUL regression ----

LSTM_MODEL_FILE = "lstm_rul.keras"
LSTM_SCALER_FILE = "lstm_scaler.joblib"
LSTM_METADATA_FILE = "lstm_metadata.json"
LSTM_FILES = (LSTM_MODEL_FILE, LSTM_SCALER_FILE, LSTM_METADATA_FILE)


def lstm_paths() -> list[Path]:
    return [config.MODELS_DIR / name for name in LSTM_FILES]


def lstm_artifacts_exist() -> bool:
    return all(path.exists() for path in lstm_paths())


def lstm_signature() -> tuple:
    return tuple(
        (path.name, path.stat().st_mtime_ns if path.exists() else 0)
        for path in lstm_paths()
    )


@dataclass
class RulArtifacts:
    """Everything produced by src.train_lstm, loaded back."""

    model: object
    scaler: object
    metadata: dict
    trained_at: pd.Timestamp

    @property
    def feature_cols(self) -> list[str]:
        return list(self.metadata["feature_cols"])

    @property
    def window_size(self) -> int:
        return int(self.metadata["window_size"])

    @property
    def is_quick_run(self) -> bool:
        return bool(self.metadata.get("quick_run", False))


def load_rul_artifacts() -> RulArtifacts:
    """Load the LSTM. TensorFlow is imported here, not at module import,
    so tools that only need the classifiers do not pay its start-up cost."""
    missing = [path.name for path in lstm_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {config.MODELS_DIR}. "
            "Run: python -m src.train_lstm"
        )
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    metadata_path = config.MODELS_DIR / LSTM_METADATA_FILE
    return RulArtifacts(
        model=tf.keras.models.load_model(config.MODELS_DIR / LSTM_MODEL_FILE),
        scaler=joblib.load(config.MODELS_DIR / LSTM_SCALER_FILE),
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        trained_at=pd.Timestamp(metadata_path.stat().st_mtime, unit="s"),
    )


def predict_rul_every_cycle(rul: RulArtifacts, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Predicted RUL at every cycle of every engine in ``raw_df``.

    Returns columns unit, cycle, predicted_rul. Scales with the LSTM's own
    training-time scaler, as with the classifiers.
    """
    scaled = apply_scaler(raw_df, rul.scaler, rul.feature_cols)
    X, index = make_windows_every_cycle(scaled, rul.feature_cols, rul.window_size)
    raw = rul.model.predict(X, batch_size=2048, verbose=0).reshape(-1)
    clip = float(rul.metadata["rul_clip"])
    index = index.copy()
    index["predicted_rul"] = np.clip(raw * clip, 0, clip)
    return index


def engine_history(scored: pd.DataFrame, engine: int) -> pd.DataFrame:
    """One engine's full recorded run, in cycle order."""
    history = scored[scored[config.UNIT_COL] == engine]
    if history.empty:
        raise ValueError(f"Engine {engine} is not in this split")
    return history.sort_values(config.CYCLE_COL).reset_index(drop=True)
