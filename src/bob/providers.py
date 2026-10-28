"""Who phrases the work order: a template writer, or a local language model.

Both return the same Reply shape. The template writer copies the plan word for
word and needs nothing installed. The Ollama provider asks a model running on
this computer (http://127.0.0.1:11434 by default) to phrase and order the
planned steps; its reply still has to pass the guardrail.

Configuration (environment variables):
    BOB_PROVIDER   template (default) or ollama
    BOB_MODEL      Ollama model name, default llama3.1:8b (qwen2.5:7b and mistral:7b also work)
    OLLAMA_URL     default http://127.0.0.1:11434
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from src.bob.findings import Findings, Plan
from src.profiles import Profile


@dataclass
class Reply:
    summary: str
    priority: str
    components: list[str]
    steps: list[dict] = field(default_factory=list)        # {"component", "step_id", "text"}
    provider: str = "template"


def _fmt_life(f: Findings, time_unit: str) -> str:
    if f.rul is None:
        return ""
    if f.rul_range:
        return f" with about {f.rul:.0f} {time_unit}s left (likely {f.rul_range[0]:.0f} to {f.rul_range[1]:.0f})"
    return f" with about {f.rul:.0f} {time_unit}s left"


class TemplateProvider:
    name = "template"
    label = "Template writer (no language model)"

    def available(self) -> bool:
        return True

    def write(self, f: Findings, plan: Plan, profile: Profile) -> Reply:
        unit = f"{profile.unit_label.capitalize()} {f.unit}"
        names = [profile.component(c)["name"] for c in plan.components]
        risk = f"failure risk {f.probability:.0%}{_fmt_life(f, profile.time_unit)}"
        if not plan.components:
            summary = f"{unit}: {risk}. No component action is needed now."
        else:
            where = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            verb = {"Urgent": "Inspect before the next run", "High": "Schedule an inspection soon",
                    "Medium": "Check at the next convenient stop", "Routine": "Check during routine maintenance"}
            summary = f"{unit}: {risk}. The readings point to the {where}. {verb[plan.priority]}."
        if f.unfamiliar:
            summary += " The wear pattern is unfamiliar to the models, so the remaining-life number is not reliable."
        steps = [{"component": c, "step_id": item["id"], "text": item["text"]}
                 for c in plan.components for item in plan.steps.get(c, [])]
        return Reply(summary=summary, priority=plan.priority, components=list(plan.components), steps=steps,
                     provider=self.name)


class OllamaProvider:
    def __init__(self, model: str | None = None, url: str | None = None, timeout: float = 60):
        self.model = model or os.environ.get("BOB_MODEL", "llama3.1:8b")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{self.model}"
        self.label = f"Local language model ({self.model}), checked by the guardrail"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=1.5) as r:
                tags = json.loads(r.read().decode("utf-8"))
            names = {m.get("name", "") for m in tags.get("models", [])}
            return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def prompt(self, f: Findings, plan: Plan, profile: Profile) -> str:
        allowed = {c: [{"step_id": i["id"], "text": i["text"]} for i in plan.steps.get(c, [])] for c in plan.components}
        names = {c: profile.component(c)["name"] for c in plan.components}
        return (
            "You write maintenance work orders. Use ONLY the components and steps given below. Do not add, "
            "rename or remove components. Keep every step_id exactly as given. You may rephrase each step as a "
            "short, clear instruction for a technician and order steps by importance. Keep the priority.\n"
            f"Machine: {profile.name}. {profile.unit_label.capitalize()}: {f.unit}.\n"
            f"Status from the models: {f.status}; failure risk {f.probability:.0%}; "
            f"remaining life {'unknown' if f.rul is None else round(f.rul)}.\n"
            f"Priority: {plan.priority}\n"
            f"Components (id: name): {json.dumps(names)}\n"
            f"Allowed steps per component: {json.dumps(allowed)}\n"
            f"Notes: {json.dumps(plan.notes)}\n"
            'Reply with JSON only: {"summary": "...", "priority": "...", "components": ["id", ...], '
            '"steps": [{"component": "id", "step_id": "id", "text": "..."}]}'
        )

    def write(self, f: Findings, plan: Plan, profile: Profile) -> Reply:
        body = json.dumps({"model": self.model, "prompt": self.prompt(f, plan, profile), "format": "json",
                           "stream": False, "options": {"temperature": 0.2}}).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        data = json.loads(out.get("response", "{}"))
        steps = [{"component": str(s.get("component", "")), "step_id": str(s.get("step_id", "")),
                  "text": str(s.get("text", "")).strip()} for s in data.get("steps", []) if isinstance(s, dict)]
        return Reply(summary=str(data.get("summary", "")).strip(), priority=str(data.get("priority", "")),
                     components=[str(c) for c in data.get("components", [])], steps=steps, provider=self.name)


class NVIDIACloudProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 30):
        _load_env_if_needed()
        self.model = model or os.environ.get("BOB_NVIDIA_MODEL") or "minimaxai/minimax-m3.1-flash"
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "").strip()
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        self.timeout = timeout
        self.name = f"nvidia:{self.model}"
        self.label = f"NVIDIA Cloud AI ({self.model}), checked by the guardrail"

    def available(self) -> bool:
        return bool(self.api_key)

    def prompt(self, f: Findings, plan: Plan, profile: Profile) -> str:
        return _format_llm_prompt(f, plan, profile)

    def write(self, f: Findings, plan: Plan, profile: Profile) -> Reply:
        if not self.available():
            raise RuntimeError("NVIDIA API key not configured. Set NVIDIA_API_KEY in .env or environment.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": self.prompt(f, plan, profile)}]}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "MachineEarlyWarning/1.0",
            }
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw_body = r.read().decode("utf-8")

        # Parse response (standard JSON or SSE stream lines)
        raw_text = ""
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
            raw_text = "".join(chunks)
        else:
            try:
                out = json.loads(raw_body)
                raw_text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                raw_text = raw_body

        data = _extract_json(raw_text)
        steps = [{"component": str(s.get("component", "")), "step_id": str(s.get("step_id", "")),
                  "text": str(s.get("text", "")).strip()} for s in data.get("steps", []) if isinstance(s, dict)]
        return Reply(summary=str(data.get("summary", "")).strip(), priority=str(data.get("priority", "")),
                     components=[str(c) for c in data.get("components", [])], steps=steps, provider=self.name)



def _load_env_if_needed():
    if "NVIDIA_API_KEY" in os.environ:
        return
    try:
        from pathlib import Path
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _extract_json(raw: str) -> dict:
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(raw)


def _format_llm_prompt(f: Findings, plan: Plan, profile: Profile) -> str:
    allowed = {c: [{"step_id": i["id"], "text": i["text"]} for i in plan.steps.get(c, [])] for c in plan.components}
    names = {c: profile.component(c)["name"] for c in plan.components}
    return (
        "You write maintenance work orders. Use ONLY the components and steps given below. Do not add, "
        "rename or remove components. Keep every step_id exactly as given. You may rephrase each step as a "
        "short, clear instruction for a technician and order steps by importance. Keep the priority.\n"
        f"Machine: {profile.name}. {profile.unit_label.capitalize()}: {f.unit}.\n"
        f"Status from the models: {f.status}; failure risk {f.probability:.0%}; "
        f"remaining life {'unknown' if f.rul is None else round(f.rul)}.\n"
        f"Priority: {plan.priority}\n"
        f"Components (id: name): {json.dumps(names)}\n"
        f"Allowed steps per component: {json.dumps(allowed)}\n"
        f"Notes: {json.dumps(plan.notes)}\n"
        'Reply with JSON only: {"summary": "...", "priority": "...", "components": ["id", ...], '
        '"steps": [{"component": "id", "step_id": "id", "text": "..."}]}'
    )


def list_providers() -> list[dict]:
    tpl = TemplateProvider()
    nv = NVIDIACloudProvider()
    ol = OllamaProvider()
    return [
        {"id": "template", "name": tpl.name, "label": tpl.label, "available": True, "type": "rule-based"},
        {"id": "nvidia", "name": nv.name, "label": nv.label, "available": nv.available(), "type": "cloud-llm", "model": nv.model},
        {"id": "ollama", "name": ol.name, "label": ol.label, "available": ol.available(), "type": "local-llm", "model": ol.model},
    ]


def get_provider(name: str | None = None):
    _load_env_if_needed()
    name = (name or os.environ.get("BOB_PROVIDER", "template")).lower()
    if name.startswith("nvidia"):
        model = name.split(":", 1)[1] if ":" in name else None
        return NVIDIACloudProvider(model=model)
    if name.startswith("ollama"):
        model = name.split(":", 1)[1] if ":" in name else None       # "ollama:qwen2.5:7b" -> "qwen2.5:7b"
        return OllamaProvider(model=model)
    return TemplateProvider()

