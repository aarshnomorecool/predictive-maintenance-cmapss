"""Which engines should a technician inspect next?

Inspections are expensive, so each one should teach the model as much as
possible. This module does two things:

1. Proves the idea. A classifier starts from only four inspected moments and
   learns from more inspections, five per round. Three ways of choosing them
   are compared on the 100 held-out test engines:
     * least sure:   where the model's failure probability is closest to 50%
     * random:       any engine, any time
     * riskiest:     the highest probability first, a common practice
   An inspection reveals whether the engine is within 30 cycles of failure at
   that moment. "Least sure, one per engine" also refuses to spend two of a
   round's inspections on the same engine: a weak early model's most uncertain
   moments are often near-duplicates around one engine's transition.
   "Explore, then least sure" (the recommended policy) inspects at random for
   the first rounds, while the model is too weak to judge its own doubt, then
   switches to least sure, one inspection per engine per round.
   Wear only goes one way, so "healthy at cycle t" also labels
   every earlier cycle of that engine, and "failing at cycle t" every later one.

2. Suggests inspections for the live fleet: the engines where the models are
   least sure or disagree with each other.

    python -m src.active_inspection      # writes models/insights/inspection.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config

INSIGHTS_DIR = config.MODELS_DIR / "insights"
RESULT_FILE = INSIGHTS_DIR / "inspection.json"

START_INSPECTIONS = 4
PER_ROUND = 5
ROUNDS = 16
SEEDS = 5
CANDIDATE_EVERY = 10          # an engine can be inspected at every 10th cycle
STRATEGIES = ("explore, then least sure", "least sure", "random", "riskiest")
EXPLORE_ROUNDS = 3           # random inspections first, while the model is too weak to know what it doesn't know


def _model(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, subsample=0.9,
                         colsample_bytree=0.9, random_state=seed, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss")


def _labels_from(inspections: list[tuple[int, int]], pool: pd.DataFrame) -> pd.Series:
    """Rows labelled by the inspections so far, using one-way wear."""
    label = pd.Series(np.nan, index=pool.index)
    for unit, cycle in inspections:
        rows = pool[config.UNIT_COL] == unit
        failing = bool(pool.loc[rows & (pool[config.CYCLE_COL] == cycle), config.LABEL_COL].iloc[0])
        if failing:
            label[rows & (pool[config.CYCLE_COL] >= cycle)] = 1
        else:
            label[rows & (pool[config.CYCLE_COL] <= cycle)] = 0
    return label


def simulate(seed: int, strategy: str, pool: pd.DataFrame, X_pool, X_test, y_test, candidates) -> list[float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    # Starting set: two clearly healthy and two clearly failing moments, so the
    # first model has both classes (a real fleet would have a few known cases)
    healthy = candidates[candidates["label"] == 0].sample(2, random_state=seed)
    failing = candidates[candidates["label"] == 1].sample(2, random_state=seed)
    inspected = [(int(r.unit), int(r.cycle)) for r in pd.concat([healthy, failing]).itertuples()]
    scores = []
    for round_no in range(ROUNDS + 1):
        y = _labels_from(inspected, pool)
        known = y.notna().to_numpy()
        model = _model(seed)
        model.fit(X_pool[known], y[known].astype(int))
        scores.append(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))
        done = set(inspected)
        left = candidates[[(u, c) not in done for u, c in zip(candidates.unit, candidates.cycle)]]
        exploring = strategy == "explore, then least sure" and round_no < EXPLORE_ROUNDS
        if strategy == "random" or exploring:
            pick = left.sample(PER_ROUND, random_state=int(rng.integers(1 << 30)))
        else:
            p = model.predict_proba(X_pool[left["row"].to_numpy()])[:, 1]
            order = np.argsort(-p) if strategy == "riskiest" else np.argsort(np.abs(p - 0.5))
            ranked = left.iloc[order]
            if strategy == "explore, then least sure":
                ranked = ranked.drop_duplicates("unit")
            pick = ranked.iloc[:PER_ROUND]
        inspected += [(int(r.unit), int(r.cycle)) for r in pick.itertuples()]
    return scores


def run_experiment() -> dict:
    from src.data_pipeline import build_dataset
    from src.preprocessing import scale_splits
    dataset = build_dataset()
    scaled, _ = scale_splits(dataset)
    pool = pd.concat([scaled.train, scaled.val], ignore_index=True)      # all 100 training engines
    features = scaled.feature_cols
    X_pool = pool[features].to_numpy(np.float32)
    X_test = scaled.test[features].to_numpy(np.float32)
    y_test = scaled.test[config.LABEL_COL].to_numpy()
    cand = pool[pool[config.CYCLE_COL] % CANDIDATE_EVERY == 0]
    candidates = pd.DataFrame({"unit": cand[config.UNIT_COL].to_numpy(), "cycle": cand[config.CYCLE_COL].to_numpy(),
                               "label": cand[config.LABEL_COL].to_numpy(), "row": cand.index.to_numpy()})

    curves = {s: [] for s in STRATEGIES}
    for seed in range(SEEDS):
        for s in STRATEGIES:
            curves[s].append(simulate(seed, s, pool, X_pool, X_test, y_test, candidates))
            print(f"  seed {seed + 1}/{SEEDS} {s:<27} final ROC-AUC {curves[s][-1][-1]:.4f}", flush=True)
    inspections = [START_INSPECTIONS + k * PER_ROUND for k in range(ROUNDS + 1)]
    mean = {s: np.mean(curves[s], axis=0).round(4).tolist() for s in STRATEGIES}
    target = 0.98

    def needed(series):
        hit = next((n for n, v in zip(inspections, series) if v >= target), None)
        return hit

    return {"inspections": inspections, "mean_auc": mean, "seeds": SEEDS, "target_auc": target,
            "inspections_to_target": {s: needed(mean[s]) for s in STRATEGIES},
            "full_data_auc": None, "per_round": PER_ROUND, "start": START_INSPECTIONS}


# ------------------------------------------------- live suggestions ----

def suggest(latest: pd.DataFrame, threshold: float, predicted: dict | None = None, k: int = 3) -> list[dict]:
    """Engines where an inspection would settle the most doubt, from each engine's
    latest cycle: probability close to the alert level, and the two classifiers
    (or the LSTM) disagreeing."""
    rows = []
    for _, r in latest.iterrows():
        unit = int(r[config.UNIT_COL])
        p_rf, p_xgb = float(r["prob_rf"]), float(r["prob_xgb"])
        mean = (p_rf + p_xgb) / 2
        closeness = 1 - min(abs(mean - threshold) / max(threshold, 1 - threshold), 1)
        disagree = abs(p_rf - p_xgb)
        reasons = []
        lstm_conflict = 0.0
        if predicted is not None and unit in predicted:
            says_failing = predicted[unit] <= config.FAILURE_THRESHOLD
            if says_failing != (mean >= threshold):
                lstm_conflict = 1.0
                reasons.append(f"the LSTM predicts {predicted[unit]:.0f} cycles left but the classifiers average "
                               f"{mean:.0%} risk")
        if disagree >= 0.15:
            reasons.append(f"Random Forest says {p_rf:.0%}, XGBoost says {p_xgb:.0%}")
        if closeness >= 0.6:
            reasons.append(f"risk {mean:.0%} is close to the {threshold:.0%} alert level")
        score = 0.5 * closeness + 0.3 * disagree + 0.4 * lstm_conflict
        rows.append({"engine": unit, "score": round(score, 3), "probability": round(mean, 3),
                     "reasons": reasons or [f"risk {mean:.0%}"]})
    rows.sort(key=lambda d: d["score"], reverse=True)
    return rows[:k]


def load_result() -> dict | None:
    try:
        return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    from src.artifact_io import atomic_write_text
    print(f"\nStart from {START_INSPECTIONS} inspections, add {PER_ROUND} per round for {ROUNDS} rounds, "
          f"{SEEDS} random starts, scored on the 100 test engines\n")
    result = run_experiment()
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULT_FILE, json.dumps(result, indent=1))
    names = {"explore, then least sure": "explore+sure", "least sure": "least sure",
             "random": "random", "riskiest": "riskiest"}
    print(f"\n  {'inspections':<13}" + "".join(f"{names[s]:>14}" for s in STRATEGIES))
    for k, n in enumerate(result["inspections"]):
        if k % 2 == 0 or k == len(result["inspections"]) - 1:
            print(f"  {n:<13}" + "".join(f"{result['mean_auc'][s][k]:>14.4f}" for s in STRATEGIES))
    print(f"\n  Inspections to reach ROC-AUC {result['target_auc']}: " +
          ", ".join(f"{s} {v if v is not None else 'not reached'}" for s, v in result["inspections_to_target"].items()))


if __name__ == "__main__":
    main()
