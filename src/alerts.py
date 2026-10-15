"""Failure alerts: decide when an engine needs attention, and email someone.

The rule was chosen by measurement on the 20 real validation engines, which
ran to failure and which the classifiers never trained on, and on simulated
engines (see README, "Alert rule"):

    alert when  the 5-cycle average failure risk >= the alert level
           or   the LSTM predicts <= 30 cycles of life left

Averaging five cycles stops a single noisy reading from firing an alarm. The
LSTM clause catches fast-developing faults earlier, because the classifier
alone reacts late when degradation runs faster than in training.

Each engine alerts once, then stays quiet until it has clearly recovered,
so an engine sitting in the red does not send an email every cycle.

Email settings come from a .env file in the project root (never committed):

    ALERT_SMTP_HOST=smtp.gmail.com
    ALERT_SMTP_PORT=587
    ALERT_SMTP_USER=you@gmail.com
    ALERT_SMTP_PASSWORD=your 16-character app password
    ALERT_EMAIL_TO=team@example.com, guide@example.com
    DASHBOARD_URL=http://127.0.0.1:8050

Without them, alerts are still raised and logged in the dashboard; they are
marked "email not configured" instead of being sent.

    python -m src.alerts --test     # send one test email with the current settings
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from src import config
from src.artifact_io import atomic_write_text

ENV_FILE = config.PROJECT_ROOT / ".env"
LOG_FILE = config.MODELS_DIR / "alert_log.json"
LOG_LIMIT = 200

SMOOTHING_CYCLES = 5
RUL_ALERT = config.FAILURE_THRESHOLD          # 30 cycles
# Recovery needs a clear margin below the alert line, so a value hovering at
# the threshold does not flip between alert and normal every cycle.
RECOVERY_MARGIN = 0.10
RUL_RECOVERY = RUL_ALERT + 10
SMTP_TIMEOUT = 12                              # seconds; fail fast on blocked networks


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=value lines; # comments and blank lines ignored. No dependency."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# ----------------------------------------------------------------- email ----

@dataclass
class EmailSettings:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()
    dashboard_url: str = "http://127.0.0.1:8050"

    @classmethod
    def from_env(cls) -> "EmailSettings":
        env = {**load_env(), **{k: v for k, v in os.environ.items() if k.startswith("ALERT_")}}
        user = env.get("ALERT_SMTP_USER", "")
        return cls(
            host=env.get("ALERT_SMTP_HOST", "smtp.gmail.com" if user.endswith("@gmail.com") else ""),
            port=int(env.get("ALERT_SMTP_PORT", "587") or 587),
            user=user,
            password=env.get("ALERT_SMTP_PASSWORD", "").replace(" ", ""),
            sender=env.get("ALERT_EMAIL_FROM", user),
            recipients=tuple(r.strip() for r in env.get("ALERT_EMAIL_TO", "").split(",") if r.strip()),
            dashboard_url=env.get("DASHBOARD_URL", load_env().get("DASHBOARD_URL", "http://127.0.0.1:8050")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.recipients)

    def missing(self) -> list[str]:
        needed = {"ALERT_SMTP_USER": self.user, "ALERT_SMTP_PASSWORD": self.password,
                  "ALERT_EMAIL_TO": self.recipients, "ALERT_SMTP_HOST": self.host}
        return [k for k, v in needed.items() if not v]

    def public(self) -> dict:
        """What the dashboard may show. Never the password."""
        return {"configured": self.configured, "host": self.host, "port": self.port,
                "sender": self.sender, "recipients": list(self.recipients),
                "missing": self.missing()}


def send_email(settings: EmailSettings, subject: str, text: str, html: str) -> None:
    """Send one message over STARTTLS. Raises on any failure."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.sender
    message["To"] = ", ".join(settings.recipients)
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    context = ssl.create_default_context()
    if settings.port == 465:
        with smtplib.SMTP_SSL(settings.host, settings.port, timeout=SMTP_TIMEOUT, context=context) as smtp:
            smtp.login(settings.user, settings.password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(settings.host, settings.port, timeout=SMTP_TIMEOUT) as smtp:
            smtp.starttls(context=context)
            smtp.login(settings.user, settings.password)
            smtp.send_message(message)


def explain_smtp_error(exc: Exception) -> str:
    """Turn the usual failures into an instruction, not a stack trace."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return ("Login refused. For Gmail, ALERT_SMTP_PASSWORD must be an app password "
                "(Google Account > Security > App passwords), not your normal password.")
    if isinstance(exc, (TimeoutError, OSError)) and not isinstance(exc, smtplib.SMTPException):
        return (f"Could not reach the mail server ({exc.__class__.__name__}). The network may "
                "block outgoing mail on port 587; try another network or a phone hotspot.")
    return f"{exc.__class__.__name__}: {exc}"


# ------------------------------------------------------------- messages ----

def compose(kind: str, engine_label: str, cycle: int, risk: float | None, model: str,
            predicted_rul: float | None, drivers: list[dict], reason: str, link: str
            ) -> tuple[str, str, str]:
    """Subject, plain text and HTML for an alert, a recovery or a test."""
    life = "unknown" if predicted_rul is None else f"{predicted_rul:.0f} cycles"
    risk_text = "unknown" if risk is None else f"{risk:.0%}"
    if kind == "alert":
        subject = f"[Inspect] {engine_label}: {risk_text} failure risk, about {life} left"
        lead = f"{engine_label} needs inspection."
    elif kind == "recovered":
        subject = f"[Back to normal] {engine_label}"
        lead = f"{engine_label} is back below the alert level."
    else:
        subject = "[Test] Turbofan health monitor email alerts work"
        lead = "This is a test message. Email alerts are set up correctly."

    driver_lines = [f"  - {d['symbol']}: {d['description']} (reading {d['reading']:g})"
                    for d in drivers[:3]]
    text = "\n".join([
        lead, "",
        f"Why: {reason}" if reason else "",
        f"Cycle: {cycle}" if cycle else "",
        f"Failure risk ({model}, 5-cycle average): {risk_text}" if risk is not None else "",
        f"Predicted life left (LSTM): {life}" if predicted_rul is not None else "",
        "",
        "Sensors pushing the risk up most:" if driver_lines else "",
        *driver_lines,
        "",
        f"Open the dashboard: {link}",
        "",
        "Sent by the Turbofan Health Monitor (NASA C-MAPSS predictive maintenance project).",
    ]).replace("\n\n\n", "\n\n")

    colour = {"alert": "#d7263d", "recovered": "#1e9e5a"}.get(kind, "#1f5fad")
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#5b6873'>{k}</td><td style='padding:4px 0'><b>{v}</b></td></tr>"
        for k, v in (("Cycle", cycle if cycle else None),
                     (f"Failure risk ({model}, 5-cycle average)", risk_text if risk is not None else None),
                     ("Predicted life left (LSTM)", life if predicted_rul is not None else None))
        if v is not None)
    driver_html = "".join(f"<li>{d['symbol']}: {d['description']} (reading {d['reading']:g})</li>"
                          for d in drivers[:3])
    html = f"""<div style="font-family:Segoe UI,Arial,sans-serif;color:#1b2631;max-width:560px">
      <div style="border-left:4px solid {colour};padding:4px 0 4px 12px;margin-bottom:12px">
        <div style="font-size:18px;font-weight:600">{lead}</div>
        {f'<div style="color:#5b6873">{reason}</div>' if reason else ''}
      </div>
      <table style="border-collapse:collapse;font-size:14px">{rows}</table>
      {f'<p style="margin:14px 0 4px">Sensors pushing the risk up most:</p><ul style="margin:0">{driver_html}</ul>' if driver_html else ''}
      <p style="margin:18px 0"><a href="{link}" style="background:#1b2631;color:#fff;padding:8px 14px;
        border-radius:6px;text-decoration:none">Open in the dashboard</a></p>
      <p style="color:#8c99a3;font-size:12px">Turbofan Health Monitor, NASA C-MAPSS predictive maintenance project.</p>
    </div>"""
    return subject, text, html


# --------------------------------------------------------------- manager ----

class AlertManager:
    """Applies the alert rule cycle by cycle and keeps the alert log."""

    def __init__(self, log_file: Path = LOG_FILE):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.log: deque[dict] = deque(self._read_log(), maxlen=LOG_LIMIT)
        self.status: dict[str, str] = {}            # engine key -> normal | alert
        self.risk_window: dict[str, deque] = {}
        self.threshold = 0.5
        self.model = "XGBoost"

    def _read_log(self) -> list[dict]:
        try:
            return json.loads(self.log_file.read_text(encoding="utf-8"))[-LOG_LIMIT:]
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        try:
            atomic_write_text(self.log_file, json.dumps(list(self.log), indent=1))
        except OSError:
            pass                                     # the log is a convenience

    def rule(self) -> dict:
        return {"threshold": self.threshold, "model": self.model,
                "smoothing_cycles": SMOOTHING_CYCLES, "rul_alert": RUL_ALERT,
                "text": (f"Alert when the {SMOOTHING_CYCLES}-cycle average {self.model} risk reaches "
                         f"{self.threshold:.0%}, or the LSTM predicts {RUL_ALERT} cycles or fewer.")}

    def set_rule(self, threshold: float | None = None, model: str | None = None) -> None:
        with self.lock:
            if threshold is not None:
                self.threshold = min(max(float(threshold), 0.01), 0.99)
            if model is not None:
                self.model = model

    def reset_engine(self, key: str) -> None:
        with self.lock:
            self.status.pop(key, None)
            self.risk_window.pop(key, None)

    def entries(self) -> list[dict]:
        with self.lock:
            return list(reversed(self.log))

    def last_id(self) -> int:
        with self.lock:
            return self.log[-1]["id"] if self.log else 0

    def evaluate(self, key: str, label: str, cycle: int, probability: float,
                 predicted_rul: float | None, drivers_fn, link: str) -> dict | None:
        """Feed one cycle. Returns the new log entry if an alert or recovery fired.

        ``drivers_fn`` is only called when something fires, because SHAP
        attribution is not free.
        """
        with self.lock:
            window = self.risk_window.setdefault(key, deque(maxlen=SMOOTHING_CYCLES))
            window.append(probability)
            risk = sum(window) / len(window)
            state = self.status.get(key, "normal")
            risk_hit = risk >= self.threshold
            rul_hit = predicted_rul is not None and predicted_rul <= RUL_ALERT

            kind = reason = None
            if state == "normal" and (risk_hit or rul_hit):
                kind = "alert"
                reasons = []
                if risk_hit:
                    reasons.append(f"average failure risk {risk:.0%} reached the {self.threshold:.0%} alert level")
                if rul_hit:
                    reasons.append(f"the LSTM predicts only {predicted_rul:.0f} cycles left")
                reason = " and ".join(reasons)
                reason = reason[0].upper() + reason[1:] + "."
            elif (state == "alert" and risk < self.threshold - RECOVERY_MARGIN
                  and (predicted_rul is None or predicted_rul > RUL_RECOVERY)):
                kind = "recovered"
                reason = (f"Average risk fell to {risk:.0%} and predicted life rose to "
                          f"{predicted_rul:.0f} cycles." if predicted_rul is not None
                          else f"Average risk fell to {risk:.0%}.")
            if kind is None:
                return None
            self.status[key] = "alert" if kind == "alert" else "normal"
            model = self.model

        drivers = drivers_fn() if kind == "alert" else []
        return self._record(kind, label, cycle, risk, model, predicted_rul, drivers, reason, link)

    def send_test(self, link: str) -> dict:
        return self._record("test", "Test", 0, None, self.model, None, [], "", link)

    def _record(self, kind, label, cycle, risk, model, predicted_rul, drivers, reason, link) -> dict:
        settings = EmailSettings.from_env()
        subject, text, html = compose(kind, label, cycle, risk, model, predicted_rul, drivers, reason, link)
        with self.lock:
            entry = {
                "id": (self.log[-1]["id"] + 1) if self.log else 1,
                "time": datetime.now().isoformat(timespec="seconds"),
                "kind": kind, "engine": label, "cycle": cycle,
                "risk": None if risk is None else round(risk, 4),
                "model": model,
                "predicted_rul": None if predicted_rul is None else round(predicted_rul, 1),
                "drivers": drivers[:3], "reason": reason,
                "subject": subject, "body": text,
                "recipients": list(settings.recipients),
                "status": "sending" if settings.configured else "not_configured",
                "error": None if settings.configured else
                         "Email is not set up, so this alert was only logged. Missing in .env: "
                         + ", ".join(settings.missing()),
            }
            self.log.append(entry)
            self._save()
        if settings.configured:
            threading.Thread(target=self._deliver, args=(entry, settings, subject, text, html),
                             daemon=True).start()
        return entry

    def _deliver(self, entry, settings, subject, text, html) -> None:
        started = time.perf_counter()
        try:
            send_email(settings, subject, text, html)
            status, error = "sent", None
        except Exception as exc:
            status, error = "failed", explain_smtp_error(exc)
        with self.lock:
            entry["status"] = status
            entry["error"] = error
            entry["send_seconds"] = round(time.perf_counter() - started, 1)
            self._save()


# ------------------------------------------------------------------- cli ----

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Email alert settings check")
    parser.add_argument("--test", action="store_true", help="send a test email now")
    args = parser.parse_args()

    settings = EmailSettings.from_env()
    print(f"\n.env file: {ENV_FILE} ({'found' if ENV_FILE.exists() else 'not found'})")
    for key, value in settings.public().items():
        print(f"  {key:<11} {value}")
    if not settings.configured:
        print("\nEmail is not configured. Copy .env.example to .env and fill it in.")
        return
    if args.test:
        subject, text, html = compose("test", "Test", 0, None, "XGBoost", None, [], "",
                                      settings.dashboard_url)
        try:
            send_email(settings, subject, text, html)
            print(f"\nTest email sent to {', '.join(settings.recipients)}.")
        except Exception as exc:
            print(f"\nSending failed. {explain_smtp_error(exc)}")


if __name__ == "__main__":
    main()
