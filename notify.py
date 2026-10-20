"""Telegram alerts. Uses TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (GitHub secrets), or a local
telegram.json (never committed). Silently does nothing if neither is set.

Messages are sent as HTML, so anything wrapped with code() appears as tap-to-copy text."""
import html
import json
import os
from pathlib import Path

import requests

LOCAL = Path(__file__).parent / "telegram.json"


def _settings():
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) and LOCAL.exists():
        data = json.loads(LOCAL.read_text(encoding="utf-8"))
        token, chat = data.get("token"), data.get("chat_id")
    return token, chat


def esc(text):
    """Make plain text safe inside an HTML message."""
    return html.escape(str(text), quote=False)


def code(text):
    """Tap-to-copy monospace text."""
    return f"<code>{esc(text)}</code>"


def block(text):
    """Tap-to-copy monospace block, for longer text like an expression."""
    return f"<pre>{esc(text)}</pre>"


def telegram(text):
    """Send an HTML message. Plain parts must be passed through esc()."""
    import time
    token, chat = _settings()
    if not (token and chat):
        return False
    for attempt in range(3):  # a missed pass alert is costly: retry network blips
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                                    "disable_web_page_preview": True}, timeout=30)
            if r.ok:
                return True
            if r.status_code == 400:
                return False  # bad message (e.g. formatting): retrying won't help
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return False


OFFSET = Path(__file__).parent / "telegram_offset.json"


def commands():
    """Text messages you sent to the bot since the last call (e.g. "/login"), only from your own chat.
    The read position is kept in telegram_offset.json so each message is handled once."""
    token, chat = _settings()
    if not (token and chat):
        return []
    try:
        offset = json.loads(OFFSET.read_text(encoding="utf-8"))["offset"]
    except (OSError, ValueError, KeyError):
        offset = 0
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                         params={"offset": offset, "timeout": 0}, timeout=30)
        updates = r.json().get("result", []) if r.ok else []
    except (requests.RequestException, ValueError):
        return []
    texts = []
    for u in updates:
        offset = max(offset, u["update_id"] + 1)
        m = u.get("message") or {}
        if str((m.get("chat") or {}).get("id")) == str(chat) and m.get("text"):
            texts.append(m["text"].strip())
    if updates:
        tmp = OFFSET.with_name(OFFSET.name + ".tmp")
        tmp.write_text(json.dumps({"offset": offset}), encoding="utf-8")
        os.replace(tmp, OFFSET)
    return texts
