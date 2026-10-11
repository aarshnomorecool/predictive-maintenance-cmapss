"""Train the LSTM that predicts remaining useful life (RUL) in cycles.

This is the regression branch: the classifiers answer "will this engine fail
soon?", the LSTM answers "how many cycles does it have left?". It reads the
last 30 cycles of all 17 features as a sequence, so it can see a trend that a
single-cycle model cannot.

Prints live progress every epoch. Every number is a real score on the
validation engines, converted back to cycles so it reads as "off by N cycles".

    python -m src.train_lstm            # full run, a few minutes on CPU
    python -m src.train_lstm --quick    # tiny smoke run, seconds
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src import config
from src.artifact_io import atomic_write_text, atomic_save
from src.inference import LSTM_METADATA_FILE, LSTM_MODEL_FILE, LSTM_SCALER_FILE
from src.preprocessing import prepare_all

RULE = "=" * 78


# --------------------------------------------------------------- metrics ----

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM 2008 challenge score (Saxena et al., eq. 1). Lower is better.

    Asymmetric on purpose: predicting failure too late (d > 0) is penalised
    harder than too early, because a late call means an engine failing in
    service while an early call only wastes some life.
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": nasa_score(y_true, y_pred),
        "late_fraction": float(np.mean(error > 0)),
        "n": int(len(error)),
    }


# ----------------------------------------------------------------- model ----

def build_model(window_size: int, n_features: int, seed: int):
    """Two stacked LSTM layers and a small dense head.

    The target is RUL divided by the 125-cycle ceiling, so the network learns
    a number between 0 and 1. That keeps the loss on the same scale as the
    scaled inputs; predictions are multiplied back to cycles afterwards.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def predict_cycles(model, X: np.ndarray) -> np.ndarray:
    """Model output back in cycles, clamped to the range the target lives in."""
    raw = model.predict(X, batch_size=1024, verbose=0).reshape(-1)
    return np.clip(raw * config.RUL_CLIP, 0, config.RUL_CLIP)


def make_progress_callback(X_val: np.ndarray, y_val: np.ndarray, total_epochs: int):
    """Per-epoch line with validation error in cycles, and the best so far."""
    import tensorflow as tf

    class Progress(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.started = time.perf_counter()
            self.best = np.inf

        def on_epoch_end(self, epoch, logs=None):
            rmse = float(np.sqrt(logs["val_loss"])) * config.RUL_CLIP
            marker = ""
            if rmse < self.best:
                self.best, marker = rmse, "  best"
            print(
                f"  epoch {epoch + 1:>3}/{total_epochs}   "
                f"train RMSE {np.sqrt(logs['loss']) * config.RUL_CLIP:6.2f}   "
                f"val RMSE {rmse:6.2f} cycles"
                f"   [{time.perf_counter() - self.started:5.1f}s]{marker}",
                flush=True,
            )

    return Progress()


# ------------------------------------------------------------------ main ----

def stage(title: str) -> float:
    print(f"\n{RULE}\n  {title}\n{RULE}", flush=True)
    return time.perf_counter()


def done(started: float) -> None:
    print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=config.SUBSET)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12,
                        help="stop after this many epochs without val improvement")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check the script works")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF start-up noise
    import tensorflow as tf

    epochs = 3 if args.quick else args.epochs

    print(f"\n{RULE}")
    print("  PREDICTIVE MAINTENANCE - REMAINING USEFUL LIFE (LSTM)")
    print(f"  subset {args.subset} | window {config.WINDOW_SIZE} cycles | "
          f"up to {epochs} epochs | early stopping after {args.patience}")
    print(RULE, flush=True)

    t = stage("STEP 1/4  Loading, labelling, scaling, windowing")
    dataset, scaler, _, seq = prepare_all(subset=args.subset, use_smote=False)
    X_train, y_train = seq.X_train, seq.y_train
    if args.quick:
        X_train, y_train = X_train[::10], y_train[::10]
    print(f"  train windows {X_train.shape}  from {len(np.unique(seq.units_train))} engines")
    print(f"  val   windows {seq.X_val.shape}  from {len(np.unique(seq.units_val))} engines")
    print(f"  test  windows {seq.X_test.shape}  last window of each test engine")
    done(t)

    t = stage("STEP 2/4  Training")
    model = build_model(seq.window_size, len(seq.feature_cols), config.RANDOM_SEED)
    history = model.fit(
        X_train, y_train / config.RUL_CLIP,
        validation_data=(seq.X_val, seq.y_val / config.RUL_CLIP),
        epochs=epochs,
        batch_size=args.batch,
        shuffle=True,
        verbose=0,
        callbacks=[
            make_progress_callback(seq.X_val, seq.y_val, epochs),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        ],
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"  kept the weights from epoch {best_epoch} "
          f"of {len(history.history['val_loss'])} run")
    done(t)

    t = stage("STEP 3/4  Evaluating")
    val_pred = predict_cycles(model, seq.X_val)
    test_pred = predict_cycles(model, seq.X_test)

    # Baseline: predict the average training RUL for every engine. A model
    # that cannot beat this has learned nothing from the sensors.
    baseline = np.full_like(seq.y_test, float(np.mean(seq.y_train)))

    results = {
        "LSTM": {
            "val_all_windows": regression_metrics(seq.y_val, val_pred),
            "test": regression_metrics(seq.y_test, test_pred),
        },
        "MeanBaseline": {
            "test": regression_metrics(seq.y_test, baseline),
        },
    }
    for model_name, splits in results.items():
        for split, m in splits.items():
            print(f"  {model_name:<13}{split:<16} RMSE {m['rmse']:6.2f}   "
                  f"MAE {m['mae']:6.2f}   NASA score {m['nasa_score']:10.1f}   "
                  f"late {m['late_fraction']:.0%}")
    done(t)

    t = stage("STEP 4/4  Saving")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save(config.MODELS_DIR / LSTM_MODEL_FILE, model.save)
    import joblib
    atomic_save(config.MODELS_DIR / LSTM_SCALER_FILE,
                lambda path: joblib.dump(scaler, path))

    per_engine = [
        {"engine": int(u), "true_rul": float(y), "predicted_rul": round(float(p), 2)}
        for u, y, p in zip(seq.units_test, seq.y_test, test_pred)
    ]
    metadata = {
        "subset": args.subset,
        "feature_cols": seq.feature_cols,
        "window_size": seq.window_size,
        "rul_clip": config.RUL_CLIP,
        "epochs_run": len(history.history["val_loss"]),
        "best_epoch": best_epoch,
        "quick_run": args.quick,
        "metrics": results,
        "test_predictions": per_engine,
        "history": {
            "train_rmse": [float(np.sqrt(v)) * config.RUL_CLIP for v in history.history["loss"]],
            "val_rmse": [float(np.sqrt(v)) * config.RUL_CLIP for v in history.history["val_loss"]],
        },
        "train_units": dataset.train_units,
        "val_units": dataset.val_units,
    }
    atomic_write_text(config.MODELS_DIR / LSTM_METADATA_FILE, json.dumps(metadata, indent=2))
    for name in (LSTM_MODEL_FILE, LSTM_SCALER_FILE, LSTM_METADATA_FILE):
        print(f"  saved  models/{name}", flush=True)
    done(t)

    lstm, base = results["LSTM"]["test"], results["MeanBaseline"]["test"]
    print(f"\n{RULE}\n  FINAL SCORES  (100 test engines, RUL at the last recorded cycle)\n{RULE}")
    print(f"  {'':<16}{'RMSE':>8}{'MAE':>8}{'NASA score':>13}")
    print(f"  {'LSTM':<16}{lstm['rmse']:>8.2f}{lstm['mae']:>8.2f}{lstm['nasa_score']:>13.1f}")
    print(f"  {'Mean baseline':<16}{base['rmse']:>8.2f}{base['mae']:>8.2f}{base['nasa_score']:>13.1f}")
    print("\n  RMSE and MAE are in cycles. True RUL is capped at 125, the same\n"
          "  convention used in training and in most published FD001 results.")
    if args.quick:
        print("\n  NOTE: --quick run. Scores are not meaningful. "
              "Re-run without --quick for real results.")
    print(flush=True)


if __name__ == "__main__":
    main()
