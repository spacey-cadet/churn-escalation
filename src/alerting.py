"""
Free-tier stand-in for a PagerDuty/Slack integration: a plain HTTP POST to a
webhook URL. Both Slack and Discord issue free incoming-webhook URLs with no paid
tier required -- set ALERT_WEBHOOK_URL in the environment to wire this up for real.
With no URL configured, alerts just print, so the pipeline never breaks in a
sandbox/CI environment that has no webhook configured.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def send_alert(message: str, severity: str = "warning") -> None:
    prefix = {"warning": "[ALERT]", "critical": "[CRITICAL ALERT]"}.get(severity, "[ALERT]")
    line = f"{prefix} {message}"
    print(line)

    if not config.ALERT_WEBHOOK_URL:
        return

    try:
        import requests
        # Slack and Discord webhooks both accept a JSON body with a "text"/"content" key.
        payload = {"text": line, "content": line}
        requests.post(config.ALERT_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:  # never let alerting itself take down the pipeline
        print(f"[alerting] failed to deliver webhook alert: {e}")
