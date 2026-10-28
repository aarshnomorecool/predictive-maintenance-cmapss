"""Group workshop visits: when one engine goes in, which others should join it?

Every workshop visit has a fixed cost (taking engines off service, setting up
the crew and tooling). Servicing each engine separately, at the last safe
moment, wastes little life but needs many visits. Grouping engines that are
all close to failure into one visit saves visits at the price of retiring
some a little early.

The planner uses the LSTM's predicted remaining life. It is checked on the
20 validation engines, which ran to failure and were never used to fit the
LSTM's weights, so the model's real prediction exists at every cycle of their
lives. The engines are placed on one shared calendar with staggered start
dates, like a fleet bought over time, and the fleet is run forward: at every
step each engine's current prediction is the one the LSTM actually made at
that point of its life. Repeated over many random staggers.

Policies compared:
  * individual:  each engine is serviced alone once its predicted life
                 reaches SAFETY cycles
  * grouped:     same trigger, but every visit also takes any engine
                 predicted to reach SAFETY within the next HORIZON cycles

    python -m src.maintenance_plan      # writes models/insights/repairs.json
"""

from __future__ import annotations

import json

import numpy as np

from src import config

INSIGHTS_DIR = config.MODELS_DIR / "insights"
RESULT_FILE = INSIGHTS_DIR / "repairs.json"

SAFETY = 15                      # service when predicted life is down to this
HORIZONS = (10, 20, 30, 40)      # how far ahead a visit looks for companions
DEFAULT_HORIZON = 20


def run_policy(lives: list[tuple[np.ndarray, np.ndarray]], offsets: np.ndarray,
               horizon: int | None) -> dict:
    """lives: per engine, (predicted life at each cycle, true life at each cycle).
    offsets: calendar time at which each engine enters service."""
    n = len(lives)
    state = np.zeros(n, dtype=int)            # 0 waiting, 1 running, 2 serviced, 3 failed
    life_left_at_service = []
    visits = failures = 0
    end = int(offsets.max() + max(len(p) for p, _ in lives) + 1)
    for t in range(end):
        age = t - offsets
        state[(state == 0) & (age >= 0)] = 1
        running = np.flatnonzero(state == 1)
        if not len(running):
            continue
        pred_now = np.array([lives[i][0][min(age[i], len(lives[i][0]) - 1)] for i in running])
        true_now = np.array([lives[i][1][min(age[i], len(lives[i][1]) - 1)] for i in running])
        failed = running[true_now <= 0]
        failures += len(failed)
        state[failed] = 3
        ok = true_now > 0
        running, pred_now, true_now = running[ok], pred_now[ok], true_now[ok]
        due = pred_now <= SAFETY
        if due.any():
            visits += 1
            take = due if horizon is None else pred_now <= SAFETY + horizon
            state[running[take]] = 2
            life_left_at_service.extend(true_now[take].tolist())
    return {"visits": visits, "failures_in_service": failures,
            "mean_life_left_at_service": round(float(np.mean(life_left_at_service)), 1)
            if life_left_at_service else None,
            "engines_serviced": len(life_left_at_service)}


def evaluate(lives, repeats: int = 30, spread: int = 150, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    offsets_list = [rng.integers(0, spread, len(lives)) for _ in range(repeats)]
    rows = []
    for horizon in (None, *HORIZONS):
        runs = [run_policy(lives, off, horizon) for off in offsets_list]
        rows.append({"policy": "individual" if horizon is None else "grouped", "horizon": horizon,
                     "visits": round(float(np.mean([r["visits"] for r in runs])), 1),
                     "failures_in_service": round(float(np.mean([r["failures_in_service"] for r in runs])), 2),
                     "mean_life_left_at_service": round(float(np.mean(
                         [r["mean_life_left_at_service"] for r in runs if r["mean_life_left_at_service"] is not None])), 1)})
    return {"safety": SAFETY, "default_horizon": DEFAULT_HORIZON, "policies": rows,
            "engines": len(lives), "repeats": repeats, "stagger_cycles": spread}


def plan(predicted: dict[int, float], anchor: int | None = None, horizon: int = DEFAULT_HORIZON) -> dict:
    """The next workshop visit for the live fleet.

    If ``anchor`` is given (an engine already going in), list the engines that
    should join it; otherwise the engines that are due now plus their companions."""
    due = sorted((e for e, p in predicted.items() if p <= SAFETY), key=lambda e: predicted[e])
    window = SAFETY + horizon
    if anchor is not None:
        companions = sorted((e for e, p in predicted.items() if e != anchor and p <= window),
                            key=lambda e: predicted[e])
        return {"anchor": anchor, "horizon": horizon, "join": companions}
    companions = sorted((e for e, p in predicted.items() if SAFETY < p <= window), key=lambda e: predicted[e])
    return {"due": due, "join": companions, "horizon": horizon, "safety": SAFETY}


def load_result() -> dict | None:
    try:
        return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src.artifact_io import atomic_write_text
    from src.data_pipeline import load_train
    from src.inference import load_rul_artifacts, predict_rul_every_cycle

    rul = load_rul_artifacts()
    val_units = json.loads((config.MODELS_DIR / "classifier_metadata.json").read_text())["val_units"]
    train = load_train(rul.metadata["subset"])
    val = train[train[config.UNIT_COL].isin(val_units)]
    pred = predict_rul_every_cycle(rul, val)
    lives = []
    for unit, g in pred.groupby(config.UNIT_COL):
        g = g.sort_values(config.CYCLE_COL)
        true_left = (g[config.CYCLE_COL].max() - g[config.CYCLE_COL]).to_numpy(float) + 1
        lives.append((g["predicted_rul"].to_numpy(), true_left))
    result = evaluate(lives)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RESULT_FILE, json.dumps(result, indent=1))

    print(f"\n{result['engines']} validation engines on a shared calendar, start dates staggered over "
          f"{result['stagger_cycles']} cycles, {result['repeats']} random fleets. Service trigger: predicted "
          f"life <= {SAFETY} cycles.\n")
    print(f"  {'policy':<22}{'visits':>8}{'failures':>10}{'life left at service':>23}")
    for r in result["policies"]:
        name = "individual" if r["horizon"] is None else f"grouped, {r['horizon']} ahead"
        print(f"  {name:<22}{r['visits']:>8.1f}{r['failures_in_service']:>10.2f}{r['mean_life_left_at_service']:>20.1f} cyc")


if __name__ == "__main__":
    main()
