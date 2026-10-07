"""Small client for the WorldQuant BRAIN API (login, simulate, data fields, submit)."""
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

API = "https://api.worldquantbrain.com"
HERE = Path(__file__).parent


class BrainError(Exception):
    pass


class BrainBusy(BrainError):
    """BRAIN was overloaded or unreachable; the request itself may be fine to retry later."""


def table(recordset):
    """Turn BRAIN's {schema, records} format into a list of dicts."""
    if not recordset or "records" not in recordset:
        return []
    names = [p["name"] for p in recordset["schema"]["properties"]]
    return [dict(zip(names, rec)) for rec in recordset["records"]]


def load_credentials():
    """Read credentials from BRAIN_EMAIL / BRAIN_PASSWORD, or from credentials.json next to this file."""
    email, password = os.environ.get("BRAIN_EMAIL"), os.environ.get("BRAIN_PASSWORD")
    if email and password:
        return email, password
    path = HERE / "credentials.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["email"], data["password"]
    raise BrainError(
        "No credentials found. Create credentials.json with "
        '{"email": "...", "password": "..."} or set BRAIN_EMAIL and BRAIN_PASSWORD.'
    )


class Brain:
    def __init__(self):
        self.s = requests.Session()
        self._login_lock = threading.Lock()
        self.login()

    def login(self):
        with self._login_lock:
            self.s.auth = load_credentials()
            r = self.s.post(f"{API}/authentication", timeout=60)
            if r.status_code == 401 and r.headers.get("WWW-Authenticate") == "persona":
                url = urljoin(r.url, r.headers["Location"])
                if sys.stdin and sys.stdin.isatty():
                    input(f"BRAIN asks for a biometric check. Open this link, finish it, then press Enter:\n{url}\n")
                    r = self.s.post(url, timeout=60)
                else:
                    # Unattended (GitHub Actions): send the link by Telegram and wait for it to be completed.
                    from notify import telegram
                    telegram(f"\U0001F510 BRAIN wants an identity check before the alpha tool can log in.\n"
                             f"Open this link and complete it within 20 minutes:\n{url}")
                    for _ in range(40):
                        time.sleep(30)
                        r = self.s.post(url, timeout=60)
                        if r.status_code in (200, 201):
                            telegram("✅ Identity check done - the alpha tool is running.")
                            break
            if r.status_code not in (200, 201):
                raise BrainError(f"Login failed ({r.status_code}): {r.text[:300]}")

    def request(self, method, path, **kwargs):
        """HTTP call that re-logs in on 401 and backs off on rate limits / server errors."""
        url = path if path.startswith("http") else API + path
        last = None
        # Up to ~30 minutes of waiting: a 429 usually just means our other simulations are still running.
        for attempt in range(40):
            try:
                r = self.s.request(method, url, timeout=60, **kwargs)
            except requests.RequestException as e:
                last = str(e)
                time.sleep(min(5 * (attempt + 1), 60))
                continue
            if r.status_code == 401:
                self.login()
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"{r.status_code} {r.text[:200]}"
                time.sleep(float(r.headers.get("Retry-After") or min(10 * (attempt + 1), 60)))
                continue
            return r
        raise BrainBusy(f"{method} {url} kept failing: {last}")

    def _wait(self, url, max_seconds=None):
        """Poll a progress URL until BRAIN stops sending Retry-After (or max_seconds passes)."""
        deadline = time.time() + max_seconds if max_seconds else None
        while True:
            r = self.request("GET", url)
            wait = float(r.headers.get("Retry-After", 0))
            if wait == 0:
                return r
            if deadline and time.time() + wait > deadline:
                raise BrainBusy(f"{url} still not finished after {max_seconds}s")
            time.sleep(wait)

    def simulate(self, expression, settings):
        """Run one simulation and return the resulting alpha record (with IS stats and checks)."""
        payload = {"type": "REGULAR", "settings": settings, "regular": expression}
        r = self.request("POST", "/simulations", json=payload)
        if r.status_code != 201:
            raise BrainError(f"Simulation rejected ({r.status_code}): {r.text[:300]}")
        result = self._wait(r.headers["Location"]).json()
        if "alpha" not in result:
            raise BrainError(result.get("message") or f"status {result.get('status')}")
        return self.alpha(result["alpha"])

    def alpha(self, alpha_id):
        r = self.request("GET", f"/alphas/{alpha_id}")
        if r.status_code != 200:
            raise BrainError(f"Could not load alpha {alpha_id} ({r.status_code}): {r.text[:200]}")
        return r.json()

    def check(self, alpha_id):
        """BRAIN's own pre-submission check, including self-correlation.

        Returns (checks, self_correlated) where self_correlated lists your most correlated alphas.
        """
        r = self._wait(f"{API}/alphas/{alpha_id}/check", max_seconds=90)
        if r.status_code != 200:
            raise BrainError(f"Check failed for {alpha_id} ({r.status_code}): {r.text[:200]}")
        result = r.json()["is"]
        return result.get("checks", []), table(result.get("selfCorrelated"))

    def yearly_stats(self, alpha_id):
        r = self._wait(f"{API}/alphas/{alpha_id}/recordsets/yearly-stats")
        if r.status_code != 200:
            raise BrainError(f"No yearly stats for {alpha_id} ({r.status_code}): {r.text[:200]}")
        return table(r.json())

    def submitted_alphas(self):
        alphas, params = [], {"stage": "OS", "limit": 100, "offset": 0}
        while True:
            r = self.request("GET", "/users/self/alphas", params=params)
            if r.status_code != 200:
                raise BrainError(f"Could not list submitted alphas ({r.status_code}): {r.text[:200]}")
            data = r.json()
            alphas += data["results"]
            params["offset"] += params["limit"]
            if not data["results"] or params["offset"] >= data["count"]:
                return alphas

    def data_fields(self, region, universe, delay, dataset=None, search=None):
        params = {"instrumentType": "EQUITY", "region": region, "universe": universe,
                  "delay": delay, "limit": 50, "offset": 0}
        if dataset:
            params["dataset.id"] = dataset
        if search:
            params["search"] = search
        fields = []
        while True:
            r = self.request("GET", "/data-fields", params=params)
            if r.status_code != 200:
                raise BrainError(f"Could not list data fields ({r.status_code}): {r.text[:200]}")
            data = r.json()
            fields += data["results"]
            params["offset"] += params["limit"]
            if not data["results"] or params["offset"] >= data["count"]:
                return fields

    def submit(self, alpha_id):
        r = self.request("POST", f"/alphas/{alpha_id}/submit")
        if r.status_code not in (200, 201):
            raise BrainError(f"Submit rejected ({r.status_code}): {r.text[:300]}")
        self._wait(f"{API}/alphas/{alpha_id}/submit")
        return self.alpha(alpha_id)
