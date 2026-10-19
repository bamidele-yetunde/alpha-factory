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


class BrainLoginBlocked(BrainError):
    """BRAIN refuses to log in until a person completes a check on the website (captcha or biometric
    verification, or BIOMETRICS_THROTTLED after too many attempts). Retrying only prolongs it."""


SESSION = HERE / "brain_session.json"
PERSONA = HERE / "brain_face_id.json"   # the face ID check waiting for you, if any
PERSONA_GAP_HOURS = 6                     # never open a new face ID check more often than this


class BrainStuck(BrainError):
    """A simulation BRAIN never finished (it kept saying "retry later"). Recorded as an error so the
    same simulation isn't retried every cycle: on 2026-10-07 one froze a whole run for 2+ hours."""


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
        if not self._resume_session():
            self.login()

    def _resume_session(self):
        """Reuse the saved session instead of logging in again. Logging in every 40-minute chunk, each
        from a different GitHub server, is the likely reason BRAIN started demanding biometric checks
        and then throttled the account (2026-10-17)."""
        try:
            cookies = json.loads(SESSION.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        self.s.cookies.update(cookies)
        try:
            r = self.s.get(f"{API}/authentication", timeout=60)
        except requests.RequestException:
            return False
        if r.status_code != 200:
            self.s.cookies.clear()
            return False
        try:
            expiry = float((r.json().get("token") or {}).get("expiry", 3600))
        except ValueError:
            expiry = 3600
        self.expires_at = time.time() + expiry
        return expiry > 600  # at least 10 minutes left, otherwise log in afresh

    # ---------- renewing before the 4-hour session runs out (face ID is required for every login) ----------

    def renew_if_needed(self, minutes_before=25):
        """Called between cycles. When the session ends within `minutes_before`, open ONE face ID check in
        a separate connection and send you the link, so you can renew before the bot stops. Also completes a
        renewal you have just finished. Never blocks."""
        if getattr(self, "_renewal", None):
            self._poll_renewal()
            return
        left = getattr(self, "expires_at", 0) - time.time()
        if left > minutes_before * 60 or self._pending_face_id():
            return
        from notify import esc, telegram
        s = requests.Session()
        s.auth = load_credentials()
        try:
            r = s.post(f"{API}/authentication", timeout=60)
        except requests.RequestException:
            return
        if r.status_code in (200, 201):  # BRAIN didn't ask for face ID this time
            self.s.cookies.update(s.cookies)
            self._finish_login("renewed, no face ID needed")
            return
        if r.status_code == 401 and r.headers.get("WWW-Authenticate") == "persona":
            url = urljoin(r.url, r.headers["Location"])
            self._remember_face_id(url, s)
            self._renewal = (s, url)
            ends = time.strftime("%H:%M", time.localtime(getattr(self, "expires_at", time.time())))
            telegram(f"\U0001F510 The bot's BRAIN session ends at {ends}. Renew it now with your face ID so the "
                     f"bot keeps working without a gap:\n{esc(url)}")

    def _poll_renewal(self):
        s, url = self._renewal
        try:
            r = s.post(url, timeout=60)
        except requests.RequestException:
            return
        if r.status_code in (200, 201):
            self.s.cookies.update(s.cookies)
            self._renewal = None
            self._finish_login("renewed with face ID")

    def _wait_for_face_id(self, url, minutes):
        """True as soon as BRAIN confirms the face ID check (asked every 3 minutes for `minutes`)."""
        deadline = time.time() + minutes * 60
        while True:
            try:
                r = self.s.post(url, timeout=60)
                if r.status_code in (200, 201):
                    return True
            except requests.RequestException:
                pass
            if time.time() + 180 > deadline:
                return False
            time.sleep(180)

    def _remember_face_id(self, url, session):
        tmp = PERSONA.with_name(PERSONA.name + ".tmp")
        tmp.write_text(json.dumps({"url": url, "opened_at": time.time(),
                                   "cookies": requests.utils.dict_from_cookiejar(session.cookies)}),
                       encoding="utf-8")
        os.replace(tmp, PERSONA)

    def _save_session(self):
        try:
            tmp = SESSION.with_name(SESSION.name + ".tmp")
            tmp.write_text(json.dumps(requests.utils.dict_from_cookiejar(self.s.cookies)), encoding="utf-8")
            os.replace(tmp, SESSION)
        except OSError:
            pass

    def _pending_face_id(self):
        """The identity check (face ID) still waiting for you, if one was opened less than
        `PERSONA_GAP_HOURS` ago: {"url", "cookies", "opened_at"} or None."""
        try:
            p = json.loads(PERSONA.read_text(encoding="utf-8"))
            if time.time() - p["opened_at"] < PERSONA_GAP_HOURS * 3600:
                return p
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _finish_login(self, how):
        """Save the session and report how long BRAIN will honour it (it decides how often you need
        face ID: a new login from GitHub's changing servers asks for it)."""
        self._save_session()
        PERSONA.unlink(missing_ok=True)
        hours = None
        try:
            r = self.s.get(f"{API}/authentication", timeout=60)
            hours = round(float((r.json().get("token") or {}).get("expiry")) / 3600, 1)
        except (requests.RequestException, ValueError, TypeError):
            pass
        self.session_hours = hours
        self.expires_at = time.time() + (hours or 4) * 3600
        from notify import telegram
        telegram(f"✅ Logged in to BRAIN ({how}). Session valid for about {hours if hours is not None else '?'} h; "
                 "no face ID needed until it runs out.")

    def login(self):
        with self._login_lock:
            from notify import esc, telegram
            # A face ID check you haven't finished yet: only look whether it is done now. Opening a new
            # check on every attempt is what got the account throttled (BIOMETRICS_THROTTLED, 2026-10-17).
            pending = self._pending_face_id()
            if pending:
                self.s.cookies.update(pending["cookies"])
                self.s.auth = load_credentials()
                # Look every 3 minutes for up to 50 minutes, so the bot starts soon after you finish.
                if self._wait_for_face_id(pending["url"], minutes=50):
                    self._finish_login("face ID done")
                    return
                raise BrainLoginBlocked(
                    f"waiting for you to complete the face ID check sent on Telegram "
                    f"{(time.time() - pending['opened_at']) / 3600:.1f} h ago")

            self.s.auth = load_credentials()
            r = self.s.post(f"{API}/authentication", timeout=60)
            if r.status_code in (400, 429) and any(k in r.text for k in ("captcha", "BIOMETRICS")):
                raise BrainLoginBlocked(f"Login blocked by BRAIN ({r.status_code}): {r.text[:200]}")
            if r.status_code == 401 and r.headers.get("WWW-Authenticate") == "persona":
                url = urljoin(r.url, r.headers["Location"])
                if sys.stdin and sys.stdin.isatty():
                    input(f"BRAIN asks for a face ID check. Open this link, finish it, then press Enter:\n{url}\n")
                    r = self.s.post(url, timeout=60)
                else:
                    # Unattended: remember this check (so no new one is opened for PERSONA_GAP_HOURS),
                    # send the link once, and look every 3 minutes for up to 50 minutes.
                    self._remember_face_id(url, self.s)
                    telegram(f"\U0001F510 BRAIN wants your face ID before the bot can log in.\n"
                             f"Open this link and complete it:\n{esc(url)}\n"
                             f"The bot looks every few minutes and starts as soon as you're done; it won't "
                             f"send a new link for {PERSONA_GAP_HOURS} h.")
                    if not self._wait_for_face_id(url, minutes=50):
                        raise BrainLoginBlocked("face ID check not completed yet (link sent on Telegram)")
                    self._finish_login("face ID done")
                    return
                if r.status_code in (200, 201):
                    self._finish_login("face ID done")
                    return
            if r.status_code not in (200, 201):
                raise BrainError(f"Login failed ({r.status_code}): {r.text[:300]}")
            self._finish_login("no face ID needed")

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

    def _wait(self, url, max_seconds=None, on_timeout=BrainBusy):
        """Poll a progress URL until BRAIN stops sending Retry-After (or max_seconds passes,
        then raise `on_timeout`)."""
        deadline = time.time() + max_seconds if max_seconds else None
        while True:
            r = self.request("GET", url)
            wait = float(r.headers.get("Retry-After", 0))
            if wait == 0:
                return r
            if deadline and time.time() + wait > deadline:
                raise on_timeout(f"{url} still not finished after {max_seconds}s")
            time.sleep(wait)

    def simulate(self, expression, settings):
        """Run one simulation and return the resulting alpha record (with IS stats and checks)."""
        payload = {"type": "REGULAR", "settings": settings, "regular": expression}
        r = self.request("POST", "/simulations", json=payload)
        if r.status_code != 201:
            raise BrainError(f"Simulation rejected ({r.status_code}): {r.text[:300]}")
        # A simulation stuck on BRAIN's side must not hang the whole run (it did on 2026-10-07).
        result = self._wait(r.headers["Location"], max_seconds=20 * 60, on_timeout=BrainStuck).json()
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
        r = self._wait(f"{API}/alphas/{alpha_id}/recordsets/yearly-stats", max_seconds=5 * 60)
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
