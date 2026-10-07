"""Telegram alerts. Uses TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (GitHub secrets), or a local
telegram.json (never committed). Silently does nothing if neither is set."""
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


def telegram(text):
    token, chat = _settings()
    if not (token and chat):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "disable_web_page_preview": True}, timeout=20)
        return r.ok
    except requests.RequestException:
        return False
