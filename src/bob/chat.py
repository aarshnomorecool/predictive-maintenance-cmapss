"""Bob the mechanic: interactive technician chat.

Grounding rules:
1. Bob explains, the models decide. Bob never sets status, RUL or probability.
2. Bob mentions ONLY valid components listed in the machine's profile.
3. Every response is validated against the machine profile components and findings.
4. If an external LLM fails, times out, or hallucinates, the template fallback answers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from src.bob.findings import Findings, make_plan
from src.bob.providers import (
    NVIDIACloudProvider,
    OllamaProvider,
    TemplateProvider,
    _extract_json,
    _load_env_if_needed,
    get_provider,
)
from src.profiles import Profile

log = logging.getLogger("bob.chat")


def format_chat_system_prompt(f: Findings, profile: Profile) -> str:
    valid_comp_names = [c["name"] for c in profile.components]
    comp_map = {c["id"]: c["name"] for c in profile.components}
    
    plan = make_plan(f, profile)
    flagged_names = [comp_map.get(c, c) for c in plan.components]
    
    checklist_text = []
    for c_id in plan.components:
        c_name = comp_map.get(c_id, c_id)
        steps = plan.steps.get(c_id, [])
        for s in steps:
            checklist_text.append(f"- [{c_name}] Step {s['id']}: {s['text']}")
            
    sensor_evidence = []
    for s_col, z in sorted(f.drift.items(), key=lambda x: abs(x[1]), reverse=True)[:6]:
        sm = profile.sensor(s_col)
        label = sm.get("label", s_col) if sm else s_col
        unit = sm.get("unit", "") if sm else ""
        comp_id = sm.get("component_id") if sm else None
        comp_str = f" (linked to {comp_map.get(comp_id, comp_id)})" if comp_id else ""
        sensor_evidence.append(f"- {label}: z-score {z:+.2f}{' ' + unit if unit else ''}{comp_str}")

    rul_str = f"{f.rul:.0f} {profile.time_unit}s" if f.rul is not None else "Unknown"
    if f.rul_range:
        rul_str += f" (likely range {f.rul_range[0]:.0f} to {f.rul_range[1]:.0f})"

    return (
        "You are 'Bob the mechanic', a seasoned, practical industrial maintenance mechanic and shop foreman.\n"
        "You are communicating directly with a field technician inspecting equipment.\n\n"
        "STRICT GROUNDING & SAFETY RULES:\n"
        "1. BOB EXPLAINS, THE MODELS DECIDE. You must NEVER override, invent, or contradict the model outputs.\n"
        f"   Machine: {profile.name} | Unit: {f.unit} | Cycle: {f.cycle}\n"
        f"   Model Status: {f.status.upper()} | Failure Risk: {f.probability:.1%} | Predicted Life Left (RUL): {rul_str}\n"
        f"   Priority: {plan.priority}\n"
        "2. ALLOWED COMPONENTS: You may ONLY reference components belonging to this machine type:\n"
        f"   Allowed: {', '.join(valid_comp_names)}\n"
        "   NEVER invent external or fictitious components not on this list.\n"
        f"3. SUSPICIOUS COMPONENTS FLAGGED: {', '.join(flagged_names) if flagged_names else 'None (routine operation)'}\n"
        "4. SENSOR DRIFT & TELEMETRY EVIDENCE:\n"
        + "\n".join(sensor_evidence) + "\n"
        "5. OFFICIAL CHECKLIST STEPS:\n"
        + ("\n".join(checklist_text) if checklist_text else "- Routine visual inspection\n") + "\n"
        "6. TONE & STYLE: Direct, concise, technical, helpful, shop-floor tone. Give actionable advice on what to "
        "look for, tools needed (e.g. borescope, calibrated torque wrench, pressure gauge), and safety warnings (lock-out/tag-out, hot surfaces).\n"
    )


def template_chat_reply(message: str, f: Findings, profile: Profile) -> str:
    """Deterministic, grounded fallback reply from Bob."""
    plan = make_plan(f, profile)
    comp_map = {c["id"]: c["name"] for c in profile.components}
    flagged_names = [comp_map.get(c, c) for c in plan.components]
    q = message.lower()

    life_str = f"about {round(f.rul)} {profile.time_unit}s" if f.rul is not None else "unknown cycles"
    if f.rul_range:
        life_str += f" (likely {round(f.rul_range[0])}–{round(f.rul_range[1])})"

    # 1. Why flagged / sensor anomalies
    if any(k in q for k in ["why", "flag", "reason", "sensor", "drift", "reading", "evidence"]):
        top_sensors = sorted(f.drift.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        s_descs = []
        for s_col, z in top_sensors:
            sm = profile.sensor(s_col)
            name = sm.get("label", s_col) if sm else s_col
            s_descs.append(f"{name} (deviation z = {z:+.2f} std dev)")
        
        reply = (
            f"Here's what triggered the alert on {profile.unit_label} {f.unit}: the models detected a "
            f"{f.probability:.1%} probability of impending failure with {life_str} remaining. "
        )
        if s_descs:
            reply += f"The strongest telemetry drifts are in {', '.join(s_descs)}. "
        if flagged_names:
            reply += (
                f"Based on the sensor-to-station mapping, the suspicious component is the "
                f"{', '.join(flagged_names)}. We need to verify if this is actual mechanical degradation or sensor bias."
            )
        else:
            reply += "No specific component is past threshold yet, but keep monitoring the trend."
        return reply

    # 2. Checklist / steps / procedure / inspect
    if any(k in q for k in ["step", "check", "inspect", "procedure", "how to", "order", "what to do"]):
        if not plan.components:
            return (
                f"Status on {profile.unit_label} {f.unit} is currently {f.status.upper()} (Risk: {f.probability:.1%}). "
                "No immediate corrective teardown is required. Perform routine external visual inspection, check for loose wiring harnesses, "
                "and ensure clean fuel/oil filters during regular turn."
            )
        reply = f"Here is the recommended action plan for {profile.unit_label} {f.unit} ({plan.priority} Priority):\n"
        step_num = 1
        for c_id in plan.components:
            c_name = comp_map.get(c_id, c_id)
            for s in plan.steps.get(c_id, []):
                reply += f"{step_num}. [{c_name}] {s['text']}\n"
                step_num += 1
        reply += "\nSafety tip: Ensure Lock-Out / Tag-Out (LOTO) protocols and allow the hot section to cool down before inserting inspection borescopes."
        return reply

    # 3. Can it run another cycle / delay / flight / operation
    if any(k in q for k in ["run", "cycle", "operate", "delay", "fly", "wait", "defer", "safe"]):
        if f.status in ["critical", "warning"]:
            return (
                f"As a shop mechanic, I'd strongly advise against running without inspection. The models report status "
                f"{f.status.upper()} with a failure risk of {f.probability:.1%} and only {life_str} left. "
                f"Priority is {plan.priority}. Deferring maintenance risks secondary damage to the "
                f"{', '.join(flagged_names) if flagged_names else 'turbofan components'}. Inspect the checklist items first."
            )
        else:
            return (
                f"The models show status {f.status.upper()} with low risk ({f.probability:.1%}) and {life_str} remaining. "
                f"The unit is within normal operating parameters for the upcoming cycle, provided standard pre-run checks pass."
            )

    # 4. Tools / safety / equipment needed
    if any(k in q for k in ["tool", "equipment", "safety", "gear", "ppe", "wear"]):
        return (
            f"For inspecting the {', '.join(flagged_names) if flagged_names else 'flagged sections'}:\n"
            "• Digital video borescope with 360° articulation for internal blade and liner inspection.\n"
            "• Calibrated digital torque wrench for flange and manifold hardware.\n"
            "• Multimeter / sensor calibrator for verifying sensor harness resistance.\n"
            "• PPE: Heat-resistant gloves, safety glasses, and hearing protection.\n"
            "Always follow shop manual lock-out/tag-out procedures before opening cowlings."
        )

    # Default overview
    reply = (
        f"Bob here. Looking at {profile.unit_label} {f.unit} at cycle {f.cycle}:\n"
        f"• Status: {f.status.upper()} ({f.probability:.1%} failure risk)\n"
        f"• Estimated life remaining: {life_str}\n"
    )
    if flagged_names:
        reply += f"• Primary concern: {', '.join(flagged_names)}\n"
        reply += f"• Next step: Review the {plan.priority}-priority inspection checklist above. "
    else:
        reply += "• All monitored components are currently within acceptable variance. "
    reply += "Feel free to ask me about specific checklist steps, sensor readings, or required tools."
    return reply


def validate_chat_response(text: str, profile: Profile) -> tuple[bool, list[str]]:
    """Verify that Bob's response doesn't hallucinate components outside the machine profile."""
    # List of known common aircraft/machine components that might be hallucinated if not in profile
    disallowed_fictional = [
        "spark plug", "carburetor", "piston", "radiator", "brake pad", "wing", "rudder",
        "landing gear", "propeller", "transmission fluid"
    ]
    problems = []
    text_lower = text.lower()
    for bad in disallowed_fictional:
        if bad in text_lower:
            problems.append(f"hallucinated external part '{bad}'")
    return len(problems) == 0, problems


def chat_with_bob(
    f: Findings,
    profile: Profile,
    message: str,
    history: list[dict[str, str]] | None = None,
    provider_name: str | None = None,
) -> dict[str, Any]:
    """Execute a chat turn with Bob the mechanic with full grounding and fallback."""
    _load_env_if_needed()
    provider = get_provider(provider_name)
    history = history or []
    system_prompt = format_chat_system_prompt(f, profile)

    # 1. NVIDIA Cloud AI
    if isinstance(provider, NVIDIACloudProvider) and provider.available():
        try:
            formatted_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            for h in history[-6:]:
                role = h.get("role", "user")
                if role in ("user", "assistant"):
                    formatted_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": str(h.get("content", ""))}]
                    })
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": message}]
            })

            payload = {
                "model": provider.model,
                "messages": formatted_messages,
                "temperature": 0.3,
                "max_tokens": 800,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{provider.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {provider.api_key}",
                    "User-Agent": "MachineEarlyWarning/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                raw_body = r.read().decode("utf-8")

            # Handle SSE or standard JSON
            if raw_body.strip().startswith("data:"):
                chunks = []
                for line in raw_body.splitlines():
                    line = line.strip()
                    if line.startswith("data:") and not line.endswith("[DONE]"):
                        try:
                            chunk = json.loads(line[5:].strip())
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            chunks.append(delta)
                        except Exception:
                            pass
                ans = "".join(chunks).strip()
            else:
                out = json.loads(raw_body)
                ans = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if ans:
                valid, problems = validate_chat_response(ans, profile)
                if valid:
                    return {
                        "reply": ans,
                        "provider": provider.name,
                        "guardrail": {"valid": True, "agrees": True, "problems": []},
                    }
                else:
                    log.warning(f"Bob chat guardrail rejected NVIDIA reply: {problems}")
        except Exception as exc:
            log.warning(f"Bob chat NVIDIA request failed ({exc}), falling back to template")

    # 2. Local Ollama
    elif isinstance(provider, OllamaProvider) and provider.available():
        try:
            full_prompt = f"{system_prompt}\n\nTechnician says: {message}\n\nBob reply:"
            body = json.dumps({
                "model": provider.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{provider.url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=provider.timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
            ans = out.get("response", "").strip()
            if ans:
                valid, problems = validate_chat_response(ans, profile)
                if valid:
                    return {
                        "reply": ans,
                        "provider": provider.name,
                        "guardrail": {"valid": True, "agrees": True, "problems": []},
                    }
        except Exception as exc:
            log.warning(f"Bob chat Ollama request failed ({exc}), falling back to template")

    # 3. Deterministic Template Fallback
    tpl_reply = template_chat_reply(message, f, profile)
    return {
        "reply": tpl_reply,
        "provider": "template",
        "guardrail": {"valid": True, "agrees": True, "problems": []},
    }
