"""Grounding test: Bob on every FD001 test engine, at its latest cycle.

For each engine the findings come from the trained models (XGBoost risk, LSTM
remaining life with its likely range, SHAP drivers, sensor drift, the sensor
guard and the unfamiliar-wear flag). Bob writes a work order, and the
guardrail checks it. The template writer is always tested; the local language
model is tested too when Ollama is running with the configured model, and its
replies are checked *before* the guardrail replaces a failed one.

    python -m src.bob                       # writes models/insights/bob_grounding.json
    BOB_MODEL=qwen2.5:7b python -m src.bob  # test a different local model
"""

from __future__ import annotations

import json
import os


def fleet_findings(model: str = "XGBoost", threshold: float = 0.5):
    """Findings for every test engine at its latest cycle, from the trained models."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src import config
    from src.bob.findings import Findings, status_from_rule
    from src.explainability import explain
    from src.inference import (PROBABILITY_COLUMNS, latest_per_engine, load_artifacts, load_rul_artifacts,
                               load_scored_test, predict_rul_every_cycle, score_frame)
    from src.profiles import load_profile
    from src.sensor_health import calibrate, score_history
    from src.sensor_integrity import assess, fit_integrity
    from src.trust import likely_range, load_trust
    from src.unfamiliar_wear import pattern_flags

    profile = load_profile("turbofan")
    artifacts = load_artifacts()
    rul = load_rul_artifacts()
    trust = load_trust()
    scored = load_scored_test(artifacts).sort_values([config.UNIT_COL, config.CYCLE_COL]).reset_index(drop=True)
    cals = calibrate(artifacts.feature_cols)
    integrity = fit_integrity(cals)
    raw_cols = [config.UNIT_COL, config.CYCLE_COL] + config.SETTING_COLS + config.SENSOR_COLS
    res = assess(scored[raw_cols], integrity)
    _, unfamiliar = pattern_flags(res)
    scored["unfamiliar"] = unfamiliar.to_numpy()
    flags = res.flags.reset_index(drop=True)
    # score repaired readings where the guard flagged a sensor, as the dashboard does
    corrected = res.corrected.reset_index(drop=True)
    pred = predict_rul_every_cycle(rul, corrected).set_index([config.UNIT_COL, config.CYCLE_COL])["predicted_rul"]
    drift = score_history(corrected, cals).set_index([config.UNIT_COL, config.CYCLE_COL])
    col = PROBABILITY_COLUMNS[model]
    out = []
    for unit, g in scored.groupby(config.UNIT_COL):
        i = g.index[-1]
        cycle = int(g[config.CYCLE_COL].iloc[-1])
        flagged = [c for c in flags.columns if bool(flags.loc[i, c])]
        row = corrected.loc[[i]] if flagged else scored.loc[[i], raw_cols]
        prob = float(score_frame(artifacts, row)[col].iloc[0]) if flagged else float(scored.loc[i, col])
        exp = explain(artifacts, row, model).contributions()
        shap = {r.feature: float(r.shap_value) for r in exp.itertuples() if profile.sensor(r.feature)}
        r_now = float(pred.loc[(unit, cycle)])
        rng = likely_range(r_now, len(g), trust) if trust else None
        d = {c: float(drift.loc[(unit, cycle), c]) for c in cals}
        status = status_from_rule(prob, threshold, r_now, profile.labelling.get("failure_window"), d)
        out.append(Findings(machine="turbofan", unit=str(int(unit)), cycle=cycle, status=status, probability=prob,
                            threshold=threshold, model=model, rul=r_now, rul_range=rng, shap=shap, drift=d,
                            flagged=flagged, unfamiliar=bool(scored.loc[i, "unfamiliar"])))
    return profile, out


def main() -> None:
    from src import config
    from src.artifact_io import atomic_write_text
    from src.bob.findings import make_plan
    from src.bob.guardrail import check
    from src.bob.providers import OllamaProvider, TemplateProvider
    from src.bob.workorder import make_work_order, render_text

    profile, findings = fleet_findings()
    template, llm = TemplateProvider(), OllamaProvider()

    def evaluate(provider):
        n = valid = agree = 0
        for f in findings:
            plan = make_plan(f, profile)
            try:
                reply = provider.write(f, plan, profile)
                g = check(reply, plan, profile)
            except Exception:
                n += 1
                continue                      # an unusable reply counts as neither valid nor agreeing
            n += 1
            valid += g.valid
            agree += g.ok
        return {"replies": n, "valid_pct": 100 * valid / n, "agree_pct": 100 * agree / n,
                "blocked_pct": 100 * (n - agree) / n}

    result = {"engines": len(findings), "template": evaluate(template)}
    if llm.available():
        result["llm"] = {**evaluate(llm), "model": llm.model}
        result["llm_status"] = f"tested with {llm.model}"
    else:
        result["llm"] = None
        result["llm_status"] = (f"not tested: Ollama with {llm.model} is not running on this computer "
                                f"(install Ollama, run 'ollama pull {llm.model}', then run python -m src.bob again)")
    priorities = {}
    for f in findings:
        p = make_plan(f, profile).priority
        priorities[p] = priorities.get(p, 0) + 1
    result["priorities"] = priorities
    example = next(f for f in findings if f.unit == "34")
    result["example"] = make_work_order(example, profile, template)
    t = result["template"]
    result["summary"] = (f"{t['valid_pct']:.0f}% of {t['replies']} template orders valid and agreeing"
                         + (f"; LLM {result['llm']['agree_pct']:.0f}% before the guardrail" if result["llm"] else
                            "; LLM not tested here"))
    out = config.MODELS_DIR / "insights" / "bob_grounding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(result, indent=1, ensure_ascii=False))

    print(f"\nBob on {len(findings)} test engines (latest cycle)")
    print(f"  priorities: {priorities}")
    print(f"  template writer: valid {t['valid_pct']:.0f}%, agree with findings {t['agree_pct']:.0f}%, "
          f"blocked {t['blocked_pct']:.0f}%")
    print(f"  local LLM: {result['llm_status']}")
    if result["llm"]:
        l = result["llm"]
        print(f"             valid {l['valid_pct']:.0f}%, agree {l['agree_pct']:.0f}%, blocked {l['blocked_pct']:.0f}%")
    print("\nExample, engine 34:\n")
    print(render_text(result["example"]))


if __name__ == "__main__":
    main()
