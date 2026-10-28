"""The guardrail: a reply is shown only if it stays inside the machine's tables.

valid   every component in the reply exists in the machine's component table,
        and every step is an item of that component's checklist
agrees  the reply also matches the plan built from the findings: the same
        components, the same priority, and every suspected faulty sensor listed;
        its free text does not name any component outside the plan

A reply that is not valid, or does not agree, is blocked and the template
writer's reply is used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.bob.findings import Plan
from src.bob.providers import Reply
from src.profiles import INSTRUMENT_PREFIX, Profile


@dataclass
class Check:
    valid: bool
    agrees: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.valid and self.agrees


def check(reply: Reply, plan: Plan, profile: Profile) -> Check:
    problems = []
    for c in reply.components:
        if profile.component(c) is None:
            problems.append(f"names a component that is not in the profile: {c!r}")
    for s in reply.steps:
        c = s.get("component", "")
        if profile.component(c) is None:
            problems.append(f"step for an unknown component {c!r}")
            continue
        ids = {i["id"] for i in profile.checklist(c)}
        if s.get("step_id") not in ids:
            problems.append(f"step {s.get('step_id')!r} is not in the checklist of {c}")
        if not s.get("text"):
            problems.append(f"step {s.get('step_id')!r} has no text")
    valid = not problems

    agree_problems = []
    if set(reply.components) != set(plan.components):
        agree_problems.append(f"components {sorted(reply.components)} differ from the plan {sorted(plan.components)}")
    if reply.priority != plan.priority:
        agree_problems.append(f"priority {reply.priority!r} differs from {plan.priority!r}")
    for c in plan.components:
        if c.startswith(INSTRUMENT_PREFIX) and not any(s.get("component") == c for s in reply.steps):
            agree_problems.append(f"leaves out the suspected faulty sensor {c[len(INSTRUMENT_PREFIX):]}")
    if plan.components and not reply.steps:
        agree_problems.append("has no steps")
    summary_lower = reply.summary.lower()
    for comp in profile.components:
        if comp["id"] in plan.components:
            continue
        if re.search(r"\b" + re.escape(comp["name"].lower()) + r"\b", summary_lower):
            agree_problems.append(f"mentions {comp['name']}, which is not in the plan")
    return Check(valid=valid, agrees=valid and not agree_problems, problems=problems + agree_problems)

