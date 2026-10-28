"""Which sensors are worth installing? The minimal sensor kit.

Fitting sensors to existing equipment costs money for every channel. This
analysis asks how few sensors the models really need:

1. Backward elimination with XGBoost. Start with all 15 physical sensors the
   models use. Repeatedly remove the sensor whose loss hurts validation
   ROC-AUC least, retraining each time. Choices are made on the 20
   validation engines only; every kit size is then scored on the untouched
   100 test engines.
2. The LSTM is retrained on the kits of a few key sizes to check that
   remaining-life accuracy holds as well.

The two operating settings stay in every kit: they come from the engine's
control system, not from sensors anyone has to buy.

    python -m src.sensor_kit      # writes models/insights/sensor_kit.json (a few minutes)
"""

from __future__ import annotations

import json
import os

import numpy as np

from src import config

INSIGHTS_DIR = config.MODELS_DIR / "insights"
RESULT_FILE = INSIGHTS_DIR / "sensor_kit.json"
LSTM_KIT_SIZES = (15, 8, 5, 3)
WITHIN = 0.01          # "as good as": test ROC-AUC within 0.01 of the full kit


def _xgb(seed=42):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=250, learning_rate=0.08, max_depth=5, subsample=0.8,
                         colsample_bytree=0.9, random_state=seed, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss")


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.metrics import recall_score, roc_auc_score
    from src.artifact_io import atomic_write_text
    from src.data_pipeline import Dataset, build_dataset
    from src.preprocessing import balance_with_smote, prepare_sequences, scale_splits
    from src.sensor_health import SPEC_BY_COLUMN
    from src.train_lstm import build_model, predict_cycles, regression_metrics

    dataset = build_dataset()
    scaled, _ = scale_splits(dataset)
    settings = [c for c in scaled.feature_cols if c in config.SETTING_COLS]
    sensors = [c for c in scaled.feature_cols if c in config.SENSOR_COLS]
    y_tr = scaled.train[config.LABEL_COL].to_numpy()
    y_va = scaled.val[config.LABEL_COL].to_numpy()
    y_te = scaled.test[config.LABEL_COL].to_numpy()
    sym = lambda cols: [SPEC_BY_COLUMN[c].symbol for c in cols]

    def score(kit: list[str]) -> dict:
        cols = settings + kit
        X_tr, y_bal = balance_with_smote(scaled.train[cols].to_numpy(np.float32), y_tr)
        m = _xgb().fit(X_tr, y_bal)
        p_va = m.predict_proba(scaled.val[cols].to_numpy(np.float32))[:, 1]
        p_te = m.predict_proba(scaled.test[cols].to_numpy(np.float32))[:, 1]
        return {"val_auc": float(roc_auc_score(y_va, p_va)), "test_auc": float(roc_auc_score(y_te, p_te)),
                "test_recall": float(recall_score(y_te, p_te >= 0.5))}

    kit = list(sensors)
    steps = [{"size": len(kit), "sensors": sym(kit), "removed": None, **score(kit)}]
    print(f"\nAll {len(kit)} sensors: test ROC-AUC {steps[0]['test_auc']:.4f}", flush=True)
    while len(kit) > 1:
        trials = []
        for c in kit:
            trial = [k for k in kit if k != c]
            trials.append((score(trial)["val_auc"], c))
        best_val, drop = max(trials)
        kit.remove(drop)
        result = score(kit)
        steps.append({"size": len(kit), "sensors": sym(kit), "removed": SPEC_BY_COLUMN[drop].symbol, **result})
        print(f"  {len(kit):>2} sensors (dropped {SPEC_BY_COLUMN[drop].symbol:<8}) test ROC-AUC "
              f"{result['test_auc']:.4f}  recall {result['test_recall']:.3f}", flush=True)

    full = steps[0]["test_auc"]
    smallest = min((s for s in steps if s["test_auc"] >= full - WITHIN), key=lambda s: s["size"])

    # LSTM on a few kit sizes
    import tensorflow as tf
    lstm = []
    for size in LSTM_KIT_SIZES:
        chosen_symbols = next(s["sensors"] for s in steps if s["size"] == size)
        cols = settings + [c for c in sensors if SPEC_BY_COLUMN[c].symbol in chosen_symbols]
        sub = Dataset(subset=scaled.subset, train=scaled.train, val=scaled.val, test=scaled.test,
                      feature_cols=cols, dropped_cols=scaled.dropped_cols)
        seq = prepare_sequences(sub)
        model = build_model(seq.window_size, len(cols), config.RANDOM_SEED)
        model.fit(seq.X_train, seq.y_train / config.RUL_CLIP,
                  validation_data=(seq.X_val, seq.y_val / config.RUL_CLIP),
                  epochs=60, batch_size=256, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                              restore_best_weights=True)])
        m = regression_metrics(seq.y_test, predict_cycles(model, seq.X_test))
        lstm.append({"size": size, "sensors": chosen_symbols, "test_rmse": round(m["rmse"], 2),
                     "test_mae": round(m["mae"], 2), "nasa_score": round(m["nasa_score"], 1)})
        print(f"  LSTM with {size:>2} sensors: test RMSE {m['rmse']:.2f}", flush=True)

    result = {"settings_always_included": settings,
              "steps": [{**s, "val_auc": round(s["val_auc"], 4), "test_auc": round(s["test_auc"], 4),
                         "test_recall": round(s["test_recall"], 3)} for s in steps],
              "full_test_auc": round(full, 4), "within": WITHIN,
              "smallest_equivalent": {"size": smallest["size"], "sensors": smallest["sensors"],
                                      "test_auc": round(smallest["test_auc"], 4)},
              "lstm": lstm}
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULT_FILE, json.dumps(result, indent=1))
    print(f"\nSmallest kit within {WITHIN} ROC-AUC of all {len(sensors)} sensors: {smallest['size']} sensors "
          f"({', '.join(smallest['sensors'])}), test ROC-AUC {smallest['test_auc']:.4f} vs {full:.4f}")


if __name__ == "__main__":
    main()
