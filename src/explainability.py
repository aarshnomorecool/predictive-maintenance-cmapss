"""SHAP attribution for the failure classifiers.

Feature importance from a forest tells you which sensors matter *on average*.
It cannot tell an engineer why engine 24 was flagged this morning. SHAP does:
it splits a single prediction into one signed contribution per sensor, and
those contributions sum back to the prediction.

    python -m src.explainability                 # global + top-risk engine
    python -m src.explainability --engine 24     # explain one engine
    python -m src.explainability --model XGBoost

Two things worth knowing before reading the numbers:

* Random Forest contributions are in probability units, so they add up to a
  probability. XGBoost works in log-odds, so its contributions add up to a
  logit that has to go through a sigmoid to become the number the dashboard
  shows. The `units` field on an Explanation says which you are looking at,
  and the two are not comparable side by side.
* Attribution is computed on scaled features, because that is what the model
  was trained on, but the values reported alongside are the raw sensor
  readings. The mapping is one-to-one, so a contribution still belongs to the
  sensor it names.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src import config
from src.inference import (
    PROBABILITY_COLUMNS,
    Artifacts,
    engine_history,
    latest_per_engine,
    load_artifacts,
    load_scored_test,
)
from src.preprocessing import apply_scaler

# Random Forest attribution is far more expensive than XGBoost's, so the
# global view runs on a sample rather than all 13,096 test rows. 400 rows is
# enough for the mean to settle; see --sample to change it.
DEFAULT_SAMPLE = 400

MODEL_UNITS = {"RandomForest": "probability", "XGBoost": "log-odds"}


# ---------------------------------------------------------------- explain ----

@dataclass
class Explanation:
    """SHAP output for a set of rows, with the raw readings that produced it."""

    model_name: str
    feature_names: list[str]
    values: np.ndarray          # (n_rows, n_features), signed contributions
    base_value: float           # model output for an average engine
    feature_values: pd.DataFrame  # raw unscaled readings, same row order
    units: str

    def contributions(self, row: int = 0) -> pd.DataFrame:
        """One row's drivers, strongest first.

        `direction` reads the sign the way an operator would: a sensor either
        pushed this engine towards failure or away from it.
        """
        contribution = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": self.values[row],
            "reading": self.feature_values.iloc[row][self.feature_names].to_numpy(),
        })
        contribution["direction"] = np.where(
            contribution["shap_value"] >= 0, "towards failure", "towards healthy"
        )
        contribution["magnitude"] = contribution["shap_value"].abs()
        return (
            contribution.sort_values("magnitude", ascending=False)
            .drop(columns="magnitude")
            .reset_index(drop=True)
        )

    def global_importance(self) -> pd.Series:
        """Mean absolute contribution per feature, largest first."""
        return pd.Series(
            np.abs(self.values).mean(axis=0), index=self.feature_names
        ).sort_values(ascending=False)

    def prediction(self, row: int = 0) -> float:
        """Base value plus this row's contributions, in the model's own units.

        Recomputing rather than reading `predict_proba` is deliberate: if the
        two disagree, the attribution is wrong and should not be trusted.
        """
        return float(self.base_value + self.values[row].sum())


def build_explainer(artifacts: Artifacts, model_name: str) -> shap.TreeExplainer:
    if model_name not in artifacts.models:
        raise KeyError(
            f"Unknown model {model_name!r}. Trained: {list(artifacts.models)}"
        )
    # No background dataset, so SHAP uses the tree-path-dependent estimator.
    # It needs no reference sample and is exact for tree ensembles.
    return shap.TreeExplainer(artifacts.models[model_name])


def _positive_class(values: np.ndarray) -> np.ndarray:
    """Reduce whatever shape SHAP returned to (n_rows, n_features) for class 1.

    scikit-learn classifiers give one attribution per class; XGBoost gives one
    set for the positive class only. Both shapes reach this function.
    """
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unexpected SHAP value shape {values.shape}")


def _base_value(explainer: shap.TreeExplainer) -> float:
    expected = np.asarray(explainer.expected_value, dtype=float).ravel()
    return float(expected[-1])


def explain(
    artifacts: Artifacts,
    raw_df: pd.DataFrame,
    model_name: str,
) -> Explanation:
    """Attribute the predictions for `raw_df` to individual sensors."""
    feature_cols = artifacts.feature_cols
    missing = [column for column in feature_cols if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Input is missing trained features: {missing}")

    scaled = apply_scaler(raw_df, artifacts.scaler, feature_cols)
    X = scaled[feature_cols].to_numpy(dtype=np.float32)

    explainer = build_explainer(artifacts, model_name)
    values = _positive_class(explainer.shap_values(X, check_additivity=False))

    return Explanation(
        model_name=model_name,
        feature_names=list(feature_cols),
        values=values,
        base_value=_base_value(explainer),
        feature_values=raw_df.reset_index(drop=True),
        units=MODEL_UNITS[model_name],
    )


def explain_engine(
    artifacts: Artifacts,
    scored: pd.DataFrame,
    engine: int,
    model_name: str,
) -> Explanation:
    """Why this engine carries the risk it does, at its most recent cycle."""
    latest = engine_history(scored, engine).tail(1)
    return explain(artifacts, latest, model_name)


def explain_sample(
    artifacts: Artifacts,
    scored: pd.DataFrame,
    model_name: str,
    sample_size: int = DEFAULT_SAMPLE,
    seed: int = config.RANDOM_SEED,
) -> Explanation:
    """Global view over a random sample of scored rows."""
    if sample_size < len(scored):
        rows = scored.sample(n=sample_size, random_state=seed)
    else:
        rows = scored
    return explain(artifacts, rows, model_name)


# -------------------------------------------------------------------- cli ----

def _print_contributions(explanation: Explanation, top: int = 8) -> None:
    frame = explanation.contributions().head(top)
    width = max(len(name) for name in frame["feature"])
    print(f"  {'sensor':<{width}}  {'reading':>10}  {'contribution':>13}  effect")
    print(f"  {'-' * width}  {'-' * 10}  {'-' * 13}  ------")
    for _, row in frame.iterrows():
        arrow = "^ towards failure" if row["shap_value"] >= 0 else "v towards healthy"
        print(
            f"  {row['feature']:<{width}}  {row['reading']:>10.3f}  "
            f"{row['shap_value']:>+13.4f}  {arrow}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="XGBoost", choices=sorted(PROBABILITY_COLUMNS),
        help="Which trained classifier to explain.",
    )
    parser.add_argument(
        "--engine", type=int, default=None,
        help="Engine to explain. Defaults to the highest-risk test engine.",
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help="Rows to sample for the global view.",
    )
    args = parser.parse_args()

    artifacts = load_artifacts()
    scored = load_scored_test(artifacts)
    probability_col = PROBABILITY_COLUMNS[args.model]

    if args.engine is None:
        latest = latest_per_engine(scored)
        engine = int(latest.loc[latest[probability_col].idxmax(), config.UNIT_COL])
    else:
        engine = args.engine

    print(f"\nModel {args.model}  |  contributions in {MODEL_UNITS[args.model]}")

    print(f"\nGlobal importance over {min(args.sample, len(scored))} sampled cycles")
    global_explanation = explain_sample(
        artifacts, scored, args.model, sample_size=args.sample
    )
    importance = global_explanation.global_importance()
    width = max(len(name) for name in importance.index)
    for name, value in importance.head(10).items():
        bar = "#" * max(1, int(round(40 * value / importance.iloc[0])))
        print(f"  {name:<{width}}  {value:>9.4f}  {bar}")

    engine_explanation = explain_engine(artifacts, scored, engine, args.model)
    row = engine_history(scored, engine).iloc[-1]
    print(
        f"\nEngine {engine} at cycle {int(row[config.CYCLE_COL])}  |  "
        f"failure probability {row[probability_col]:.3f}  |  "
        f"true RUL {int(row[config.RUL_COL])}"
    )
    _print_contributions(engine_explanation)
    print(
        f"\n  baseline {engine_explanation.base_value:+.4f} "
        f"+ contributions = {engine_explanation.prediction():+.4f} "
        f"({engine_explanation.units})\n"
    )


if __name__ == "__main__":
    main()
