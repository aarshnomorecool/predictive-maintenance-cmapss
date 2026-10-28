"""Prediction from day one for a new fleet, with no failure history.

A new site or a new engine type has no failures yet, so a model cannot be
trained on it. This module takes the FD001 models, unchanged, to NASA's FD002
fleet, which flies in six different flight conditions. In FD002 the raw
readings jump with the flight condition far more than with wear, so the FD001
models see nonsense.

The adaptation uses only what a new fleet has on day one: its normal running.

1. Identify the six flight conditions from the three operating settings
   (k-means on the settings).
2. For each condition, measure what healthy looks like: the mean and spread
   of every sensor over the first 30 cycles of each FD002 training engine.
   No failure labels and no remaining-life values are used.
3. Translate every FD002 reading into FD001's terms: how many healthy
   spreads it sits from its own condition's healthy level, re-expressed on
   FD001's healthy level and spread.

Scored on the 259 FD002 test engines against NASA's true remaining life:
  * FD001 models on raw FD002 data            (no adaptation)
  * FD001 models on translated data           (zero FD002 labels)
  * an LSTM trained with full FD002 labels    (the realistic best case)

    python -m src.adaptation      # writes models/insights/adaptation.json (a few minutes)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src import config

INSIGHTS_DIR = config.MODELS_DIR / "insights"
RESULT_FILE = INSIGHTS_DIR / "adaptation.json"
NEW_FLEET = "FD002"
CONDITIONS = 6
HEALTHY_CYCLES = 30


class FleetTranslator:
    """Learned from a new fleet's healthy running only."""

    def __init__(self, reference_train: pd.DataFrame, new_train: pd.DataFrame, sensors: list[str]):
        from sklearn.cluster import KMeans
        self.sensors = sensors
        self.kmeans = KMeans(CONDITIONS, n_init=10, random_state=config.RANDOM_SEED).fit(
            new_train[config.SETTING_COLS].to_numpy())
        early = new_train[new_train[config.CYCLE_COL] <= HEALTHY_CYCLES].copy()
        early["condition"] = self.kmeans.predict(early[config.SETTING_COLS].to_numpy())
        self.mean = early.groupby("condition")[sensors].mean()
        self.std = early.groupby("condition")[sensors].std().replace(0, np.nan)
        # FD001's healthy reference (RUL at the 125 ceiling) and its settings
        from src.data_pipeline import add_train_rul
        ref = add_train_rul(reference_train.copy(), clip=config.RUL_CLIP)
        healthy = ref[ref[config.RUL_COL] >= config.RUL_CLIP]
        self.ref_mean = healthy[sensors].mean()
        self.ref_std = healthy[sensors].std().replace(0, np.nan)
        self.ref_settings = reference_train[config.SETTING_COLS].mean()

    def conditions(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(df[config.SETTING_COLS].to_numpy())

    def translate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cond = self.conditions(df)
        mu = self.mean.loc[cond].to_numpy()
        sd = self.std.loc[cond].to_numpy()
        z = (df[self.sensors].to_numpy(float) - mu) / sd
        # A sensor that never varies in a condition (zero spread) carries no wear
        # information there: it reads its reference healthy level, not zero
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        ref_std = self.ref_std.fillna(0).to_numpy()
        out[self.sensors] = self.ref_mean.to_numpy() + z * ref_std
        for c in config.SETTING_COLS:
            out[c] = self.ref_settings[c]
        return out


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.metrics import recall_score, roc_auc_score
    from src.artifact_io import atomic_write_text
    from src.data_pipeline import add_failure_label, add_test_rul, add_train_rul, load_test, load_test_rul, load_train
    from src.inference import load_artifacts, load_rul_artifacts, predict_rul_every_cycle, score_frame
    from src.preprocessing import make_last_windows, make_windows
    from src.train_lstm import build_model, predict_cycles, regression_metrics
    import tensorflow as tf

    artifacts = load_artifacts()
    rul = load_rul_artifacts()
    fd1_train = load_train("FD001")
    new_train = load_train(NEW_FLEET)
    new_test = add_failure_label(add_test_rul(load_test(NEW_FLEET), load_test_rul(NEW_FLEET)))
    truth_end = new_test.sort_values([config.UNIT_COL, config.CYCLE_COL]).groupby(config.UNIT_COL)[config.RUL_COL].last()

    sensors = config.SENSOR_COLS
    translator = FleetTranslator(fd1_train, new_train, sensors)
    translated_test = translator.translate(new_test)

    def lstm_last(df):
        p = predict_rul_every_cycle(rul, df)
        last = p.sort_values([config.UNIT_COL, config.CYCLE_COL]).groupby(config.UNIT_COL)["predicted_rul"].last()
        return regression_metrics(truth_end.loc[last.index].to_numpy(), last.to_numpy())

    def classify(df):
        s = score_frame(artifacts, df)
        y = df[config.LABEL_COL].to_numpy()
        return {"roc_auc": float(roc_auc_score(y, s["prob_xgb"])), "recall": float(recall_score(y, s["prob_xgb"] >= 0.5))}

    raw_lstm, raw_cls = lstm_last(new_test), classify(new_test)
    print(f"\nFD001 models on raw {NEW_FLEET}:        LSTM RMSE {raw_lstm['rmse']:.2f}, "
          f"XGBoost ROC-AUC {raw_cls['roc_auc']:.3f}", flush=True)
    tr_lstm, tr_cls = lstm_last(translated_test), classify(translated_test)
    print(f"FD001 models, translated {NEW_FLEET}:   LSTM RMSE {tr_lstm['rmse']:.2f}, "
          f"XGBoost ROC-AUC {tr_cls['roc_auc']:.3f}", flush=True)

    # Realistic best case: same architecture trained with every FD002 failure label
    feats = rul.feature_cols
    tr_train = add_train_rul(translator.translate(new_train), clip=config.RUL_CLIP)
    scaler = rul.scaler
    from src.preprocessing import apply_scaler
    units = tr_train[config.UNIT_COL].unique()
    rng = np.random.default_rng(config.RANDOM_SEED)
    val_units = set(rng.choice(units, size=int(len(units) * 0.2), replace=False))
    fit_df = apply_scaler(tr_train[~tr_train[config.UNIT_COL].isin(val_units)], scaler, feats)
    val_df = apply_scaler(tr_train[tr_train[config.UNIT_COL].isin(val_units)], scaler, feats)
    X_tr, y_tr, _ = make_windows(fit_df, feats)
    X_va, y_va, _ = make_windows(val_df, feats)
    X_te, _, te_units = make_last_windows(apply_scaler(translated_test, scaler, feats), feats)
    model = build_model(rul.window_size, len(feats), config.RANDOM_SEED)
    model.fit(X_tr, y_tr / config.RUL_CLIP, validation_data=(X_va, y_va / config.RUL_CLIP), epochs=60,
              batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    sup = regression_metrics(truth_end.loc[te_units].to_numpy(), predict_cycles(model, X_te))
    print(f"LSTM trained with all {NEW_FLEET} labels: LSTM RMSE {sup['rmse']:.2f}", flush=True)

    def pack(m):
        return {k: round(float(v), 3) for k, v in m.items() if k in ("rmse", "mae", "nasa_score", "roc_auc", "recall")}

    result = {"new_fleet": NEW_FLEET, "conditions": CONDITIONS, "healthy_cycles_used": HEALTHY_CYCLES,
              "test_engines": int(len(truth_end)),
              "raw": {**pack(raw_lstm), **{f"xgb_{k}": round(v, 4) for k, v in raw_cls.items()}},
              "translated": {**pack(tr_lstm), **{f"xgb_{k}": round(v, 4) for k, v in tr_cls.items()}},
              "supervised": pack(sup),
              "fd001_rmse": round(float(rul.metadata["metrics"]["LSTM"]["test"]["rmse"]), 2)}
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULT_FILE, json.dumps(result, indent=1))
    print(f"Saved {RESULT_FILE}")


if __name__ == "__main__":
    main()
