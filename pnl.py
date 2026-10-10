"""Daily profit records from BRAIN, cached, and correlations computed the way BRAIN's
self-correlation check does it: correlation of daily PnL changes over the last 4 years of the
in-sample period. Checked on 2026-10-10: zqbd6xrd vs 58gjmP81 gave 0.8205 here and 0.8205 on BRAIN.

The cache (pnl_cache.json.gz) only grows: an alpha's in-sample PnL never changes.
"""
import gzip
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "pnl_cache.json.gz"


class PnlBook:
    def __init__(self, client_factory):
        """client_factory: returns a logged-in brain.Brain (called only when a fetch is needed)."""
        self._client_factory = client_factory
        self._daily = {}   # alpha_id -> {date: daily pnl change}, last-4-years window
        self._missing = set()
        self._dirty = False
        try:
            with gzip.open(CACHE, "rt", encoding="utf-8") as f:
                raw = json.load(f)
            self._daily = {a: dict(zip(v["d"], v["p"])) for a, v in raw.items()}
        except (OSError, ValueError, KeyError):
            pass

    def save(self):
        if not self._dirty:
            return
        raw = {a: {"d": list(v), "p": list(v.values())} for a, v in self._daily.items()}
        with gzip.open(CACHE, "wt", encoding="utf-8") as f:
            json.dump(raw, f, separators=(",", ":"))
        self._dirty = False

    def daily(self, alpha_id):
        """{date: daily PnL change} over the last 4 years, or None if BRAIN has no record."""
        if alpha_id in self._daily:
            return self._daily[alpha_id]
        if alpha_id in self._missing:
            return None
        from brain import API, BrainError, table
        try:
            r = self._client_factory()._wait(f"{API}/alphas/{alpha_id}/recordsets/pnl", max_seconds=120)
            rows = table(r.json()) if r.status_code == 200 else []
        except (BrainError, ValueError):
            rows = []
        rows = [x for x in rows if x.get("pnl") is not None]
        if len(rows) < 250:
            self._missing.add(alpha_id)
            return None
        start = f"{int(rows[-1]['date'][:4]) - 3}-01-01"  # e.g. 2020-01-01 .. 2023-12-29
        daily = {rows[i]["date"]: rows[i]["pnl"] - rows[i - 1]["pnl"]
                 for i in range(1, len(rows)) if rows[i]["date"] >= start}
        self._daily[alpha_id] = daily
        self._dirty = True
        return daily

    def corr(self, a, b):
        x, y = self.daily(a), self.daily(b)
        if not x or not y:
            return None
        days = [d for d in x if d in y]
        if len(days) < 250:
            return None
        try:
            return statistics.correlation([x[d] for d in days], [y[d] for d in days])
        except statistics.StatisticsError:
            return None  # a flat PnL (no trades) has no correlation

    def blend(self, ids):
        """Approximate daily PnL of rank(A) + rank(B) + ...: each part scaled to the same risk, summed."""
        parts = [self.daily(a) for a in ids]
        if not all(parts):
            return None
        days = [d for d in parts[0] if all(d in p for p in parts[1:])]
        if len(days) < 250:
            return None
        out = dict.fromkeys(days, 0.0)
        for p in parts:
            sd = statistics.pstdev([p[d] for d in days]) or 1.0
            for d in days:
                out[d] += p[d] / sd
        return out

    def blend_signed(self, parts):
        """Like blend, for [(alpha_id, +1 or -1), ...]: -1 for a part used as rank(-(...))."""
        base = self.blend([a for a, _ in parts])
        if base is None or all(sign > 0 for _, sign in parts):
            return base
        days = list(base)
        out = dict.fromkeys(days, 0.0)
        for a, sign in parts:
            p = self.daily(a)
            sd = statistics.pstdev([p[d] for d in days]) or 1.0
            for d in days:
                out[d] += sign * p[d] / sd
        return out

    def series_max_corr(self, series, others):
        """Highest correlation of a PnL series (e.g. a blend) with any of `others`."""
        best, best_id = None, ""
        for o in others:
            y = self.daily(o)
            if not y:
                continue
            days = [d for d in series if d in y]
            if len(days) < 250:
                continue
            try:
                c = statistics.correlation([series[d] for d in days], [y[d] for d in days])
            except statistics.StatisticsError:
                continue
            if best is None or c > best:
                best, best_id = c, o
        return best, best_id

    def max_corr(self, alpha_id, others):
        """(highest correlation, with which alpha) against `others`; (None, "") if nothing to compare."""
        best, best_id = None, ""
        for o in others:
            if o == alpha_id:
                continue
            c = self.corr(alpha_id, o)
            if c is not None and (best is None or c > best):
                best, best_id = c, o
        return best, best_id
