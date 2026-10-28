"""Bob the mechanic: turns the system's findings into a grounded work order.

Bob explains, the models decide. Bob never sets an engine's status, risk or
remaining life; he reads them from a findings object, maps flagged sensors to
components through the machine profile's fixed table, takes inspection steps
from the profile's fixed checklists, and (optionally) lets a local language
model phrase them. A guardrail rejects any reply that names a component or a
step outside the profile, and the template writer, which needs no language
model, is always available.

    python -m src.bob              # grounding test on the 100 FD001 test engines
"""

from src.bob.findings import Findings, Plan, make_plan
from src.bob.workorder import make_work_order, render_html, render_text

__all__ = ["Findings", "Plan", "make_plan", "make_work_order", "render_html", "render_text"]
