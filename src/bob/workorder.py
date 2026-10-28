"""Builds a work order from findings, and renders it as text or a printable page."""

from __future__ import annotations

import html
from datetime import datetime

from src.bob.findings import Findings, make_plan
from src.bob.guardrail import check
from src.bob.providers import TemplateProvider, get_provider
from src.profiles import Profile


def evidence_lines(f: Findings, profile: Profile, drivers: list[str]) -> list[str]:
    lines = [f"Failure risk {f.probability:.1%} ({f.model}); alert level {f.threshold:.0%}."]
    if f.rul is not None:
        rng = f", likely {f.rul_range[0]:.0f} to {f.rul_range[1]:.0f}" if f.rul_range else ""
        lines.append(f"Predicted remaining life {f.rul:.0f} {profile.time_unit}s{rng}"
                     f"{' (not reliable: unfamiliar wear)' if f.unfamiliar else ''}.")
    for s in drivers[:4]:
        comp = profile.component(profile.component_of(s) or "")
        where = f" ({comp['name']})" if comp else ""
        d = f.drift.get(s)
        drift = f", {d:.0%} of the way to its failure level" if d is not None else ""
        lines.append(f"{profile.label_of(s)}{where}: pushes the risk up by {f.shap[s]:+.2f}{drift}.")
    for s in f.flagged:
        lines.append(f"{profile.label_of(s)} disagrees with the other cross-checked sensors: suspected sensor fault; "
                     f"its reading was rebuilt before scoring.")
    if f.unfamiliar:
        lines.append("Two or more sensors disagree with the learned wear pattern at once: unfamiliar wear.")
    return lines


def make_work_order(f: Findings, profile: Profile, provider=None) -> dict:
    plan = make_plan(f, profile)
    provider = provider or get_provider()
    template = TemplateProvider()
    blocked, problems = False, []
    if provider.name != "template":
        try:
            reply = provider.write(f, plan, profile)
            g = check(reply, plan, profile)
            if not g.ok:
                blocked, problems = True, g.problems
                reply = template.write(f, plan, profile)
        except Exception as exc:          # model not running, timeout, bad JSON: fall back, never fail
            blocked, problems = True, [f"language model unavailable: {exc}"]
            reply = template.write(f, plan, profile)
    else:
        reply = template.write(f, plan, profile)
    final = check(reply, plan, profile)
    written_by = provider.label if (provider.name != "template" and not blocked) else template.label
    order = {
        "id": f"WO-{profile.id}-{f.unit}-{f.cycle if f.cycle is not None else 'now'}",
        "machine": profile.id,
        "machine_name": profile.name,
        "unit_label": profile.unit_label,
        "unit": f.unit,
        "cycle": f.cycle,
        "time_unit": profile.time_unit,
        "created": datetime.now().isoformat(timespec="seconds"),
        "priority": reply.priority,
        "status": f.status,
        "risk": round(f.probability, 4),
        "rul": None if f.rul is None else round(f.rul, 1),
        "rul_range": list(f.rul_range) if f.rul_range else None,
        "rul_reliable": not f.unfamiliar,
        "summary": reply.summary,
        "sensors": [profile.label_of(s) for s in f.flagged] + [profile.label_of(s) for s in plan.drivers[:4]],
        "components": [{"id": c, "name": profile.component(c)["name"]} for c in reply.components],
        "steps": [{"component": s["component"], "component_name": profile.component(s["component"])["name"],
                   "step_id": s["step_id"], "text": s["text"]} for s in reply.steps],
        "evidence": evidence_lines(f, profile, plan.drivers),
        "notes": plan.notes,
        "provider": written_by,
        "from_template": written_by == template.label,
        "guardrail": {"valid": final.valid, "agrees": final.agrees, "blocked_reply": blocked,
                      "problems": problems},
        "profile_note": profile.data.get("note", ""),
    }
    return order


def render_text(wo: dict) -> str:
    out = [f"WORK ORDER {wo['id']}", f"{wo['machine_name']}, {wo['unit_label']} {wo['unit']}"
           + (f", {wo['time_unit']} {wo['cycle']}" if wo["cycle"] is not None else ""),
           f"Priority: {wo['priority']}    Status from the models: {wo['status']}", "", wo["summary"], ""]
    if wo["steps"]:
        out.append("Steps:")
        out += [f"  {k}. [{s['component_name']}] {s['text']}" for k, s in enumerate(wo["steps"], 1)]
        out.append("")
    out.append("Evidence:")
    out += [f"  - {e}" for e in wo["evidence"]]
    for n in wo["notes"]:
        out.append(f"Note: {n}")
    out.append(f"Written by: {wo['provider']}")
    return "\n".join(out)


def render_html(wo: dict) -> str:
    e = html.escape
    steps = "".join(f"<li><b>{e(s['component_name'])}:</b> {e(s['text'])} <span class='box'></span></li>"
                    for s in wo["steps"]) or "<li>No component action needed now.</li>"
    evidence = "".join(f"<li>{e(x)}</li>" for x in wo["evidence"])
    notes = "".join(f"<p class='note'>{e(n)}</p>" for n in wo["notes"])
    life = "" if wo["rul"] is None else (
        f"<div><dt>Predicted life</dt><dd>{wo['rul']:.0f} {e(wo['time_unit'])}s"
        + (f" (likely {wo['rul_range'][0]:.0f} to {wo['rul_range'][1]:.0f})" if wo["rul_range"] else "")
        + ("" if wo["rul_reliable"] else " — not reliable") + "</dd></div>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Work order {e(wo['id'])}</title>
<style>
body {{ font: 15px/1.45 "Segoe UI", system-ui, sans-serif; color: #1b2631; margin: 0; padding: 28px; background: #fff; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} .sub {{ color: #5b6873; margin: 0 0 18px; }}
dl {{ display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 0 0 18px; }} dt {{ color: #5b6873; font-size: 13px; }}
dd {{ margin: 0; font-weight: 600; }} h2 {{ font-size: 16px; margin: 20px 0 6px; }}
ol, ul {{ padding-left: 22px; margin: 0; }} li {{ margin: 4px 0; }}
.box {{ display: inline-block; width: 12px; height: 12px; border: 1px solid #1b2631; margin-left: 6px; vertical-align: middle; }}
.note {{ background: #f6f8f9; border-left: 3px solid #e08a00; padding: 6px 10px; margin: 8px 0; }}
.foot {{ color: #5b6873; font-size: 12px; margin-top: 22px; }} .sign {{ margin-top: 26px; display: flex; gap: 60px; }}
@media print {{ body {{ padding: 0; }} button {{ display: none; }} }}
</style></head><body>
<button onclick="window.print()" style="float:right">Print</button>
<h1>Work order {e(wo['id'])}</h1>
<p class="sub">{e(wo['machine_name'])} · {e(wo['unit_label'].capitalize())} {e(str(wo['unit']))}{'' if wo['cycle'] is None else f" · {e(wo['time_unit'])} {wo['cycle']}"} · {e(wo['created'].replace('T', ' '))}</p>
<dl><div><dt>Priority</dt><dd>{e(wo['priority'])}</dd></div><div><dt>Status from the models</dt><dd>{e(wo['status'])}</dd></div>
<div><dt>Failure risk</dt><dd>{wo['risk']:.1%}</dd></div>{life}</dl>
<p>{e(wo['summary'])}</p>{notes}
<h2>Steps</h2><ol>{steps}</ol>
<h2>Evidence</h2><ul>{evidence}</ul>
<div class="sign"><span>Inspected by: ____________________</span><span>Outcome: fault confirmed / no fault found</span></div>
<p class="foot">Written by: {e(wo['provider'])}. Bob explains the models' findings and phrases a fixed checklist; he does not
diagnose, and the models, not Bob, set the status, risk and remaining life. {e(wo['profile_note'])}</p>
</body></html>"""
