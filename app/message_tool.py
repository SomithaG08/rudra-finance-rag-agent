"""
message_tool.py - The agent's action tool: send a message to Slack or Microsoft Teams.

Safety rules (Agentic AI project, day 2):
  1. The agent can only PREPARE a message. Nothing is sent until a human clicks
     "Approve & send" in the UI (human-in-the-loop). approved=False never sends.
  2. After sending we CHECK THE OUTCOME (HTTP status + response body) and report it
     honestly: "sent", "failed" or "logged only". The UI never says "sent" without proof.
  3. Fake send mode (SEND_MODE=fake, used by the injection test and the agent eval):
     every proposal is DECLINED automatically and written to data/fake_send_log.jsonl,
     so tests can count the actions the agent tried to take without sending anything.

Channels (first one configured in .env wins, or pick one with recipient="slack"/"teams"):
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...      (Slack incoming webhook)
    TEAMS_WEBHOOK_URL=https://....webhook.office.com/...         (Teams incoming webhook / workflow)
Without a webhook the message is only logged to data/sent_messages.jsonl ("logged only").
The spec prefers an MCP client for Slack/Teams; an incoming webhook is the accepted fallback.
"""

import json
import os
from datetime import datetime

from app.config import DATA_DIR

SENT_MESSAGES_FILE = DATA_DIR / "sent_messages.jsonl"
FAKE_SEND_LOG = DATA_DIR / "fake_send_log.jsonl"
SEND_MODE = os.getenv("SEND_MODE", "real").lower()         # "real" (approval + webhook) or "fake"
TIMEOUT_S = 10


def _log(path, record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def channel_for(recipient: str) -> tuple[str, str | None]:
    """("slack" | "teams" | "log", webhook url)."""
    r = (recipient or "").lower()
    slack, teams = os.getenv("SLACK_WEBHOOK_URL", "").strip(), os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    if "teams" in r and teams:
        return "teams", teams
    if "slack" in r and slack:
        return "slack", slack
    if slack:
        return "slack", slack
    if teams:
        return "teams", teams
    return "log", None


def prepare_message(recipient: str, message: str) -> dict:
    """Prepare a message for human approval. This function does NOT send anything."""
    channel, _ = channel_for(recipient)
    return {"success": True, "status": "approval_required", "channel": channel,
            "recipient": recipient, "message": message}


def fake_send(recipient: str, message: str, reason: str = "") -> dict:
    """Fake send: decline every proposal and log it (tests / eval / injection test)."""
    _log(FAKE_SEND_LOG, {"timestamp": datetime.now().isoformat(timespec="seconds"),
                         "recipient": recipient, "message": message, "status": "declined",
                         "reason": reason})
    return {"success": False, "status": "declined", "recipient": recipient, "message": message,
            "note": "The human declined this message (fake send mode). Do not prepare it again; "
                    "answer the user's question."}


def _post(channel: str, url: str, recipient: str, message: str) -> dict:
    """POST to the webhook and verify the outcome."""
    import requests
    payload = {"text": message}                               # both Slack and Teams accept {"text"}
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT_S)
    except Exception as exc:                                  # noqa: BLE001 (network error)
        return {"ok": False, "http_status": None, "detail": f"{type(exc).__name__}: {exc}"}
    body = (resp.text or "").strip()[:200]
    if channel == "slack":
        ok = resp.status_code == 200 and body == "ok"         # Slack answers exactly "ok"
    else:
        ok = resp.status_code in (200, 202) and "error" not in body.lower()
    return {"ok": ok, "http_status": resp.status_code, "detail": body}


def send_message(recipient: str, message: str, approved: bool = False) -> dict:
    """Send ONLY when approved=True (a human clicked Approve). Returns the checked outcome."""
    if not approved:
        return prepare_message(recipient, message)
    if not (message or "").strip():
        return {"success": False, "status": "failed", "message": "Empty message: nothing sent."}

    channel, url = channel_for(recipient)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "recipient": recipient,
              "channel": channel, "message": message}
    if url:
        outcome = _post(channel, url, recipient, message)
        record.update(status="sent" if outcome["ok"] else "failed", **outcome)
    else:
        record.update(status="logged_only", ok=True,
                      detail="No SLACK_WEBHOOK_URL / TEAMS_WEBHOOK_URL in .env: logged locally.")
    _log(SENT_MESSAGES_FILE, record)
    print(f"\n[send_message] {record['status'].upper()} via {channel} -> {recipient}: {message[:80]}")
    return {"success": record["status"] in ("sent", "logged_only"), **record}
