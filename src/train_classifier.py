"""Train the binary failure classifiers: Random Forest and XGBoost.

Prints live progress while training rather than sitting silent. Every number
that scrolls past is a real score from a real partially-trained model, not a
simulated progress bar.

    python -m src.train_classifier            # full run
    python -m src.train_classifier --quick    # small smoke run, seconds
    python -m src.train_classifier --no-smote # skip class balancing
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src import config
from src.preprocessing import prepare_all

RULE = "=" * 78


# ------------------------------------------------------------- reporting ----

@contextmanager
def stage(title: str):
    """Announce a stage and report how long it took, so nothing looks idle."""
    print(f"\n{RULE}\n  {title}\n{RULE}", flush=True)
    start = time.perf_counter()
    yield
    print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Standard binary metrics. Accuracy alone is misleading here."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def print_metrics_table(results: dict[str, dict[str, dict[str, float]]]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'model':<16}{'split':<8}" + "".join(f"{m:>11}" for m in metrics)
    print(f"\n{RULE}\n  FINAL SCORES\n{RULE}")
    print(header)
    print("-" * len(header))
    for model_name, splits in results.items():
        for split_name, scores in splits.items():
            row = f"{model_name:<16}{split_name:<8}"
            row += "".join(f"{scores[m]:>11.4f}" for m in metrics)
            print(row)
    print(
        "\n  Test positives are rare because test engines stop well before "
        "failure.\n  Read recall and ROC-AUC, not accuracy.",
        flush=True,
    )


# --------------------------------------------------------------- models -----

def train_random_forest(
    X_train, y_train, X_val, y_val, n_trees: int, batch: int, seed: int,
) -> RandomForestClassifier:
    """Grow the forest in batches so validation score streams as it improves.

    warm_start keeps the trees already grown and appends more, which lets us
    score an honest partial forest between batches. The final model is
    identical to fitting all the trees in one call.
    """
    model = RandomForestClassifier(
        n_estimators=batch,
        warm_start=True,
        n_jobs=-1,
        random_state=seed,
        min_samples_leaf=2,
    )
    started = time.perf_counter()
    for total in range(batch, n_trees + 1, batch):
        model.n_estimators = total
        model.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        print(
            f"  trees {total:>4}/{n_trees}   val ROC-AUC {val_auc:.4f}"
            f"   [{time.perf_counter() - started:5.1f}s]",
            flush=True,
        )
    return model


def train_xgboost(
    X_train, y_train, X_val, y_val, n_rounds: int, report_every: int, seed: int,
) -> XGBClassifier:
    """XGBoost streams per-round eval metrics natively."""
    model = XGBClassifier(
        n_estimators=n_rounds,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=report_every,
    )
    best = getattr(model, "best_iteration", None)
    if best is not None:
        print(f"  best iteration {best} of {n_rounds}", flush=True)
    return model


# ------------------------------------------------------------------ main ----

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=config.SUBSET)
    parser.add_argument("--trees", type=int, default=300,
                        help="Random Forest size")
    parser.add_argument("--rounds", type=int, default=500,
                        help="XGBoost boosting rounds")
    parser.add_argument("--no-smote", action="store_true",
                        help="train on the real class prior instead")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check the script works")
    args = parser.parse_args()

    n_trees, n_rounds = args.trees, args.rounds
    tree_batch, report_every = 25, 10
    if args.quick:
        n_trees, n_rounds = 20, 30
        tree_batch, report_every = 5, 5

    print(f"\n{RULE}")
    print("  PREDICTIVE MAINTENANCE - FAILURE CLASSIFICATION")
    print(f"  subset {args.subset} | RandomForest {n_trees} trees | "
          f"XGBoost {n_rounds} rounds | SMOTE {not args.no_smote}")
    print(RULE, flush=True)

    with stage("STEP 1/4  Loading, labelling, scaling, balancing"):
        dataset, scaler, tabular, _ = prepare_all(
            subset=args.subset, use_smote=not args.no_smote
        )
        print(f"  features        : {len(tabular.feature_cols)}")
        print(f"  train  {tabular.X_train.shape}  "
              f"positives {tabular.y_train.mean():.2%}")
        print(f"  val    {tabular.X_val.shape}  "
              f"positives {tabular.y_val.mean():.2%}")
        print(f"  test   {tabular.X_test.shape}  "
              f"positives {tabular.y_test.mean():.2%}", flush=True)

    with stage(f"STEP 2/4  Random Forest ({n_trees} trees)"):
        rf = train_random_forest(
            tabular.X_train, tabular.y_train, tabular.X_val, tabular.y_val,
            n_trees, tree_batch, config.RANDOM_SEED,
        )

    with stage(f"STEP 3/4  XGBoost ({n_rounds} rounds, early stopping at 50)"):
        xgb = train_xgboost(
            tabular.X_train, tabular.y_train, tabular.X_val, tabular.y_val,
            n_rounds, report_every, config.RANDOM_SEED,
        )

    with stage("STEP 4/4  Evaluating and saving"):
        results = {
            "RandomForest": {
                "val": evaluate(rf, tabular.X_val, tabular.y_val),
                "test": evaluate(rf, tabular.X_test, tabular.y_test),
            },
            "XGBoost": {
                "val": evaluate(xgb, tabular.X_val, tabular.y_val),
                "test": evaluate(xgb, tabular.X_test, tabular.y_test),
            },
        }

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, config.MODELS_DIR / "rf_classifier.joblib")
        joblib.dump(xgb, config.MODELS_DIR / "xgb_classifier.joblib")
        joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")

        importances = dict(
            zip(tabular.feature_cols, rf.feature_importances_.tolist())
        )
        metadata = {
            "subset": args.subset,
            "feature_cols": tabular.feature_cols,
            "failure_threshold": config.FAILURE_THRESHOLD,
            "rul_clip": config.RUL_CLIP,
            "smote_applied": tabular.smote_applied,
            "n_trees": n_trees,
            "n_rounds": n_rounds,
            "quick_run": args.quick,
            "metrics": results,
            "rf_feature_importance": importances,
            "train_units": dataset.train_units,
            "val_units": dataset.val_units,
        }
        (config.MODELS_DIR / "classifier_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        for name in ("rf_classifier.joblib", "xgb_classifier.joblib",
                     "scaler.joblib", "classifier_metadata.json"):
            print(f"  saved  models/{name}", flush=True)

    print_metrics_table(results)

    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"\n  Top features (Random Forest):")
    for name, value in top:
        print(f"    {name:<12} {value:.4f}")

    if args.quick:
        print("\n  NOTE: --quick run. Scores are not meaningful. "
              "Re-run without --quick for real results.")
    print(flush=True)


if __name__ == "__main__":
    main()
