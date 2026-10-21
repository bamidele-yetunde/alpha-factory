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
PERSONA_GAP_HOURS = 6                     # a pending check is looked at for at most this long
NIGHT_END_HOUR = 6                        # no new face ID checks opened by the bot itself 00:00-06:00
LOG = print                               # alpha_tool sets its log() here, so login events reach activity_log.txt


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
        # Renew while the current, face-ID-verified session is still valid: the new request carries that
        # session's cookie. If BRAIN accepts it, sessions can be chained (4 h each, BRAIN's maximum) and face ID
        # is only needed when the chain breaks. A copy of the cookies is used, so a refusal can't touch the
        # session the bot is working with; if BRAIN refuses, its reply is the face ID link we send you anyway.
        s = requests.Session()
        s.cookies.update(self.s.cookies)
        s.auth = load_credentials()
        try:
            r = s.post(f"{API}/authentication", json={"expiry": 14400}, timeout=60)
        except requests.RequestException as e:
            LOG(f"RENEW: renewal request failed ({e}); trying again next round")
            return
        LOG(f"RENEW: session ends in {left / 60:.0f} min; BRAIN answered the renewal with {r.status_code} "
            f"{r.headers.get('WWW-Authenticate', '')}".rstrip())
        if r.status_code in (200, 201):  # BRAIN didn't ask for face ID this time
            self.s.cookies.update(s.cookies)
            self.renewed_silently = True
            self._finish_login("session renewed without face ID")
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
        if r.status_code == 410:  # the renewal link expired unused: stop looking; the normal login takes over
            LOG("RENEW: the renewal face ID link expired (410 Gone)")
            self._renewal = None
            self._mark_expired()
            return
        if r.status_code in (200, 201):
            self.s.cookies.update(s.cookies)
            self._renewal = None
            self._finish_login("renewed with face ID")

    def _wait_for_face_id(self, url, minutes):
        """Ask BRAIN every minute, for up to `minutes`, whether the face ID check is done. Returns True (done),
        "EXPIRED" (BRAIN answers 410 Gone: the link is dead, shown to you as "details gone"), "REQUESTED"
        (you sent /login to the Telegram bot for a fresh link) or False (still waiting)."""
        from notify import commands
        deadline = time.time() + minutes * 60
        while True:
            try:
                r = self.s.post(url, timeout=60)
                if r.status_code in (200, 201):
                    return True
                if r.status_code == 410:
                    return "EXPIRED"
            except requests.RequestException:
                pass
            if any(c.lower().lstrip("/").startswith("login") for c in commands()):
                return "REQUESTED"
            if time.time() + 60 > deadline:
                return False
            time.sleep(60)

    @staticmethod
    def _may_open_face_id():
        """Open a new face ID check by itself only outside the night (00:00-06:00 local) and at most every
        30 minutes: every new check is a sign-in attempt, and too many got the account throttled."""
        try:
            last = json.loads(PERSONA.read_text(encoding="utf-8")).get("opened_at", 0)
        except (OSError, ValueError):
            last = 0
        night = time.localtime().tm_hour < NIGHT_END_HOUR
        return not night and time.time() - last > 30 * 60

    @staticmethod
    def _expired_message():
        # In the daytime the wait is only the 30-minute spacing between new checks (said "after 06:00" before).
        when = "after 06:00" if time.localtime().tm_hour < NIGHT_END_HOUR else "within 30 minutes"
        return f"face ID link expired; a fresh one comes {when}, or send /login to the Telegram bot for one now"

    @staticmethod
    def _mark_expired():
        """Keep the time the dead check was opened (for the 30-minute spacing) but stop looking at it."""
        try:
            p = json.loads(PERSONA.read_text(encoding="utf-8"))
            p["opened_at"] = min(p.get("opened_at", 0), time.time())
            p["url"] = ""
            PERSONA.write_text(json.dumps(p), encoding="utf-8")
        except (OSError, ValueError):
            pass

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
            if p.get("url") and time.time() - p["opened_at"] < PERSONA_GAP_HOURS * 3600:
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
            requested = False
            pending = self._pending_face_id()
            if pending:
                self.s.cookies.update(pending["cookies"])
                self.s.auth = load_credentials()
                # Look every minute for up to 50 minutes, so the bot starts soon after you finish.
                result = self._wait_for_face_id(pending["url"], minutes=50)
                if result is True:
                    self._finish_login("face ID done")
                    return
                if result == "REQUESTED":
                    requested = True  # you asked for a fresh link: open one now, whatever the time
                elif result == "EXPIRED":
                    if not self._may_open_face_id():
                        self._mark_expired()
                        raise BrainLoginBlocked(self._expired_message())
                else:
                    raise BrainLoginBlocked(
                        f"waiting for you to complete the face ID check sent on Telegram "
                        f"{(time.time() - pending['opened_at']) / 3600:.1f} h ago")
                self.s = requests.Session()  # the old check is dead: start a clean sign-in
            elif PERSONA.exists() and not self._may_open_face_id():
                # Last link expired at night: wait for 06:00 or for /login.
                from notify import commands
                if any(c.lower().lstrip("/").startswith("login") for c in commands()):
                    requested = True
                else:
                    raise BrainLoginBlocked(self._expired_message())

            self.s.auth = load_credentials()
            r = self.s.post(f"{API}/authentication", json={"expiry": 14400}, timeout=60)  # BRAIN's maximum: 4 h
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
                             "The bot starts within a minute of you finishing. If the link says \"details "
                             "gone\" (expired), send /login here for a fresh one.")
                    result = self._wait_for_face_id(url, minutes=50)
                    if result is not True:
                        if result == "EXPIRED":
                            self._mark_expired()
                        raise BrainLoginBlocked("face ID check not completed yet (link sent on Telegram)"
                                                if result is False else f"face ID link {result.lower()}")
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
        # BRAIN's daily simulation quota (consultants have one): remembered so the bot can stay under it.
        self.rate_limit = {k: r.headers.get(k) for k in ("X-Ratelimit-Limit", "X-Ratelimit-Remaining")}
        if r.status_code != 201:
            raise BrainError(f"Simulation rejected ({r.status_code}): {r.text[:300]}")
        # A simulation stuck on BRAIN's side must not hang the whole run (it did on 2026-10-07).
        result = self._wait(r.headers["Location"], max_seconds=20 * 60, on_timeout=BrainStuck).json()
        if "alpha" not in result:
            raise BrainError(result.get("message") or f"status {result.get('status')}")
        return self.alpha(result["alpha"])

    def simulate_many(self, items):
        """Simulate up to 10 regular alphas of one region in one BRAIN multi-simulation (the account's
        MULTI_SIMULATION permission): they share one simulation slot, so a 4-hour session tests several times
        more ideas. `items` is [(expression, settings)]. Returns, in order, each alpha record or the
        BrainError saying why that one failed. Tested 2026-10-20: 2 alphas in 180 s; regions can't be mixed."""
        def one_by_one():
            out = []
            for expression, settings in items:
                try:
                    out.append(self.simulate(expression, settings))
                except (BrainLoginBlocked, BrainBusy):
                    raise
                except BrainError as e:
                    out.append(e)
            return out

        if len(items) == 1:
            return one_by_one()
        payload = [{"type": "REGULAR", "settings": s, "regular": e} for e, s in items]
        r = self.request("POST", "/simulations", json=payload)
        self.rate_limit = {k: r.headers.get(k) for k in ("X-Ratelimit-Limit", "X-Ratelimit-Remaining")}
        if r.status_code == 400:
            return one_by_one()  # one invalid idea rejects the whole batch: run them singly, only it fails
        if r.status_code != 201:
            raise BrainError(f"Multi-simulation rejected ({r.status_code}): {r.text[:300]}")
        result = self._wait(r.headers["Location"], max_seconds=40 * 60, on_timeout=BrainStuck).json()
        children = result.get("children") or []
        if len(children) != len(items):
            raise BrainError(f"multi-simulation gave {len(children)} results for {len(items)} ideas: "
                             f"{result.get('message') or result.get('status')}")
        out = []
        for child in children:
            try:
                c = self._wait(f"/simulations/{child}", max_seconds=10 * 60, on_timeout=BrainStuck).json()
                out.append(self.alpha(c["alpha"]) if "alpha" in c
                           else BrainError(c.get("message") or f"status {c.get('status')}"))
            except (BrainLoginBlocked, BrainBusy):
                raise
            except BrainError as e:
                out.append(e)
        # One invalid idea (e.g. an unknown field) makes BRAIN cancel the rest of the batch (tested
        # 2026-10-20): run the cancelled ones again without it.
        cancelled = [i for i, o in enumerate(out) if isinstance(o, BrainError) and str(o) == "status CANCELLED"]
        if cancelled and len(cancelled) < len(items):
            for i, o in zip(cancelled, self.simulate_many([items[i] for i in cancelled])):
                out[i] = o
        return out

    def prod_corr(self, alpha_id, max_seconds=60):
        """Highest correlation of the alpha with all alphas in BRAIN's production pool (the consultant
        PROD_CORRELATION test, limit 0.7). About 5-8 s per alpha (2026-10-20), but minutes when BRAIN is busy
        (2026-10-21): raises BrainBusy after `max_seconds`. None if BRAIN gives none."""
        r = self._wait(f"/alphas/{alpha_id}/correlations/prod", max_seconds=max_seconds)
        if r.status_code != 200 or not r.text:
            raise BrainError(f"production correlation of {alpha_id}: {r.status_code} {r.text[:120]}")
        return r.json().get("max")

    def simulate_super(self, selection, combo, settings):
        """Simulate a SuperAlpha (a weighted combination of your submitted alphas) and return its record.
        SuperAlphas are slower than regular alphas, so the wait is longer (60 minutes)."""
        payload = {"type": "SUPER", "settings": settings, "selection": selection, "combo": combo}
        r = self.request("POST", "/simulations", json=payload)
        self.rate_limit = {k: r.headers.get(k) for k in ("X-Ratelimit-Limit", "X-Ratelimit-Remaining")}
        if r.status_code != 201:
            raise BrainError(f"SuperAlpha simulation rejected ({r.status_code}): {r.text[:300]}")
        result = self._wait(r.headers["Location"], max_seconds=60 * 60, on_timeout=BrainStuck).json()
        if "alpha" not in result:
            raise BrainError(result.get("message") or f"status {result.get('status')}")
        return self.alpha(result["alpha"])

    def describe(self, alpha_id, regular=None, selection=None, combo=None, name=None):
        """Attach descriptions (and a name) to an alpha: regular alphas have one description, SuperAlphas
        one for the selection and one for the combo (both required to submit)."""
        body = {}
        if regular is not None:
            body["regular"] = {"description": regular}
        if selection is not None:
            body["selection"] = {"description": selection}
        if combo is not None:
            body["combo"] = {"description": combo}
        if name:
            body["name"] = name
        r = self.request("PATCH", f"/alphas/{alpha_id}", json=body)
        if r.status_code >= 300:
            raise BrainError(f"Could not set description on {alpha_id} ({r.status_code}): {r.text[:200]}")

    def alpha(self, alpha_id):
        r = self.request("GET", f"/alphas/{alpha_id}")
        if r.status_code != 200:
            raise BrainError(f"Could not load alpha {alpha_id} ({r.status_code}): {r.text[:200]}")
        return r.json()

    def check(self, alpha_id, max_seconds=90):
        """BRAIN's own pre-submission check, including self-correlation.

        Returns (checks, self_correlated) where self_correlated lists your most correlated alphas.
        """
        r = self._wait(f"{API}/alphas/{alpha_id}/check", max_seconds=max_seconds)
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
        self._wait(f"{API}/alphas/{alpha_id}/submit", max_seconds=15 * 60)
        return self.alpha(alpha_id)
