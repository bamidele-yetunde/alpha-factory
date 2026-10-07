"""Learns from results.csv which ideas, settings and repairs lead to passing alphas.

Everything is recomputed on every cycle from results.csv (simulations), checks.csv (BRAIN's
submission check + robustness), submissions.csv and submitted_alphas.csv, so those files are
the memory: delete rows from them to make the system forget.

An alpha only counts as passed when it passed the in-sample checks AND the verification gate
(self-correlation + robustness). In-sample passes not verified yet are "Pending".
"""
import json
import random
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timedelta

CHECK_HELP = {
    "LOW_SHARPE": "Signal too weak or noisy. Try other neutralization, longer windows, or a different field.",
    "LOW_FITNESS": "Returns too small for the turnover. Raise decay or smooth (ts_mean) to cut turnover.",
    "HIGH_TURNOVER": "Trades too much. Raise decay, use longer windows, or wrap in ts_mean/ts_decay_linear.",
    "LOW_TURNOVER": "Barely trades (slow data). Lower decay or use shorter windows.",
    "CONCENTRATED_WEIGHT": "Too much money in a few stocks. Lower truncation, wrap in rank(), backfill gaps.",
    "LOW_SUB_UNIVERSE_SHARPE": "Only works in small stocks. Neutralize by subindustry/industry or rank().",
    "SELF_CORRELATION": "Too similar to an alpha you already submitted. Use a different field or idea.",
    "PROD_CORRELATION": "Too similar to alphas already on the platform. Use a less popular field or idea.",
    "NOT_ROBUST": "Profit comes from only some years (losing years / weak recent year). Likely to fail out-of-sample.",
}

# Failures found by the verification gate; settings tweaks don't fix these, so they are not repaired.
GATE_CHECKS = {"SELF_CORRELATION", "PROD_CORRELATION", "NOT_ROBUST"}
GROUPS = {"market", "sector", "industry", "subindustry"}

# Repairs to try for each failing check, in default order. Learned success rates re-rank them.
REPAIRS = {
    "HIGH_TURNOVER": ["decay_up5", "decay_up15", "smooth", "decay_linear"],
    "LOW_FITNESS": ["decay_up5", "smooth", "neut_subindustry", "decay_up15", "neut_industry", "neut_market"],
    "LOW_SHARPE": ["neut_subindustry", "neut_industry", "neut_sector", "neut_market", "neut_none", "smooth"],
    "LOW_TURNOVER": ["decay_down"],
    "CONCENTRATED_WEIGHT": ["decay_up5", "decay_up15", "trunc_05", "rank"],  # decay spreads holdings over more stocks
    "LOW_SUB_UNIVERSE_SHARPE": ["neut_subindustry", "neut_industry", "rank"],
}

# Fixes for alphas that pass everything except self-correlation (or robustness): change what the
# alpha holds so its P&L stops tracking your submitted alphas. Learned success rates re-rank them.
CORR_REPAIRS = ["subtract_corr", "neut_none", "neut_market", "neut_sector", "neut_industry",
                "neut_subindustry", "field_swap", "group_swap", "universe_swap", "size_neutral"]
NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
UNIVERSES = ["TOP3000", "TOP2000", "TOP1000", "TOP500"]

TURNOVER_CUTTERS = {"decay_up5", "decay_up15", "smooth", "decay_linear"}

DIMENSIONS = ["template", "field", "dataset", "window", "group", "decay", "neutralization", "universe",
              "truncation"]
BANNABLE = {"template", "field", "dataset"}
NOTES_HEADING = "## My notes"


def parse_time(text):
    """ISO time (BRAIN's with offset, or ours in local time) as an aware datetime, or None."""
    if not text:
        return None
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        return None
    return t if t.tzinfo else t.astimezone()


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def placeholders(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def strip_calls(expr, name):
    """Replace every name(x, ...) with x, e.g. drop ts_backfill wrappers."""
    token = name + "("
    while (start := expr.find(token)) != -1:
        open_at = start + len(token) - 1
        depth, first_comma, end = 0, None, None
        for k in range(open_at, len(expr)):
            ch = expr[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
            elif ch == "," and depth == 1 and first_comma is None:
                first_comma = k
        if end is None:
            break
        expr = expr[:start] + expr[open_at + 1:first_comma or end] + expr[end + 1:]
    return expr


def skeleton(expr):
    """The shape of an expression with data fields removed: alphas with the same shape on
    similar data (e.g. two quarterly fundamentals) tend to be highly correlated."""
    s = re.sub(r"\s+", "", expr.lower())
    s = strip_calls(s, "ts_backfill")
    return re.sub(r"\b[a-z_][a-z0-9_]*\b(?!\()",
                  lambda m: "G" if m.group(0) in GROUPS else "F", s)


def fields_in(expr):
    """Identifiers in an expression that are not operators or group names."""
    return {t for t in re.findall(r"\b[a-z_][a-z0-9_]*\b(?!\s*[(=])", expr.lower()) if t not in GROUPS}


def invalid_names(expr, field_ids, operators):
    """Names in an expression that BRAIN won't recognise (catches typos before wasting a simulation)."""
    calls = {t for t in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expr.lower())} - operators
    unknown = fields_in(expr) - field_ids - {"true", "false", "nan"}
    return calls | unknown


def one_sided(row):
    """Mostly long (or mostly short): a market bet, not a stock-picking signal. Its Sharpe comes
    from the market's direction, and BRAIN flags it as concentrated."""
    longs, shorts = num(row.get("long_count")), num(row.get("short_count"))
    return longs + shorts > 0 and min(longs, shorts) < 0.1 * max(longs, shorts)


def reward(row):
    """1 for a verified pass, 0.6 for an unverified one, 0.15 for one that failed the
    correlation/robustness gate, otherwise up to 0.6 for how close Sharpe and fitness got."""
    if row["passed"] == "True":
        return 1.0
    if row["passed"] == "Pending":
        return 0.6
    if set(row["failed_checks"].split(";")) & GATE_CHECKS:
        return 0.15
    closeness = min(abs(num(row["sharpe"])) / 1.25, abs(num(row["fitness"])) / 1.0)
    return 0.6 * max(0.0, min(1.0, closeness))


def components(row):
    settings = json.loads(row["settings"] or "{}")
    values = {"template": row["template"], "field": row["field"], "dataset": row["dataset"],
              "window": row["window"], "group": row["group"],
              "decay": settings.get("decay", ""), "neutralization": settings.get("neutralization", ""),
              "universe": settings.get("universe", ""), "truncation": settings.get("truncation", "")}
    return [f"{dim}:{value}" for dim, value in values.items() if value != ""]


def apply_repair(kind, expr, overrides, base):
    """Return (expression, overrides) after one repair, or None if it would change nothing."""
    full = {**base, **overrides}
    new = dict(full)
    if kind == "flip":
        expr = f"-({expr})"
    elif kind == "smooth":
        expr = f"ts_mean({expr}, 5)"
    elif kind == "decay_linear":
        expr = f"ts_decay_linear({expr}, 10)"
    elif kind == "rank":
        if expr.startswith("rank("):
            return None  # already ranked: another rank() only makes a near-copy of the same alpha
        expr = f"rank({expr})"
    elif kind == "size_neutral":
        # Remove the part of the signal explained by company size (a common source of overlap).
        expr = f'group_neutralize({expr}, bucket(rank(cap), range="0.1,1,0.1"))'
    elif kind == "decay_up5":
        new["decay"] = full["decay"] + 5
    elif kind == "decay_up15":
        new["decay"] = full["decay"] + 15
    elif kind == "decay_down":
        new["decay"] = full["decay"] // 2
    elif kind == "trunc_05":
        new["truncation"] = 0.05
    elif kind.startswith("neut_"):
        new["neutralization"] = kind[5:].upper()
    if new == full and kind not in ("flip", "smooth", "decay_linear", "rank", "size_neutral"):
        return None
    return expr, {k: v for k, v in new.items() if k in overrides or v != base.get(k)}


class Knowledge:
    def __init__(self, rows, submissions, cfg, checks=(), submitted_alphas=()):
        self.cfg = cfg
        # BRAIN takes a while to include a new submission in its correlation checks, so checks
        # made shortly after a submission can't be trusted for alphas sharing data with it.
        common = {"cap", "returns", "close", "volume", "open", "high", "low", "vwap"}
        self.recent_submissions = [(parse_time(a.get("submitted_at")), fields_in(a["code"]) - common)
                                   for a in submitted_alphas if parse_time(a.get("submitted_at"))]
        # The latest verification of each alpha decides whether its in-sample pass is genuine.
        self.checks = {}
        for c in checks:
            self.checks[c["alpha_id"]] = c
        rows = [self._apply_check(r) for r in rows]
        self.rows = rows
        self.sims = [r for r in rows if r["alpha_id"]]
        by_id = {r["alpha_id"]: r for r in self.sims}
        self.children = Counter(r["parent"] for r in rows if r["parent"])
        self.child_repairs = defaultdict(set)
        for r in rows:
            if r["parent"]:
                self.child_repairs[r["parent"]].add(r["repair"])
        self.submitted = {s["alpha_id"] for s in submissions} | {a["id"] for a in submitted_alphas}
        # Shapes and fields of everything already submitted: new alphas copying them get correlated.
        self.submitted_shapes = {skeleton(a["code"]) for a in submitted_alphas}
        self.submitted_code = {a["id"]: a["code"] for a in submitted_alphas}
        self.submitted_fields = set().union(*(fields_in(a["code"]) for a in submitted_alphas))

        # A new idea is judged by the best result its whole repair family reached.
        def root(r):
            for _ in range(10):
                if not r["parent"] or r["parent"] not in by_id:
                    break
                r = by_id[r["parent"]]
            return id(r)

        family_reward, family_pass = defaultdict(float), defaultdict(bool)
        for r in rows:
            family_reward[root(r)] = max(family_reward[root(r)], reward(r))
            family_pass[root(r)] |= r["passed"] == "True"

        # stats[key] = [tries, total reward, passes, total |Sharpe|]
        self.stats = defaultdict(lambda: [0, 0.0, 0, 0.0])
        for r in rows:
            if r["parent"]:
                continue
            for key in components(r):
                s = self.stats[key]
                s[0] += 1
                s[1] += family_reward[id(r)]
                s[2] += family_pass[id(r)]
                s[3] += abs(num(r["sharpe"]))

        # repair_stats[(repair, check it was fixing)] = [tries, improved, passed]
        self.repair_stats = defaultdict(lambda: [0, 0, 0])
        for r in self.sims:
            parent = by_id.get(r["parent"])
            if not (r["repair"] and parent):
                continue
            for check in filter(None, parent["failed_checks"].split(";")):
                s = self.repair_stats[(r["repair"], check)]
                s[0] += 1
                s[1] += reward(r) > reward(parent) + 0.02
                s[2] += r["passed"] == "True"

        # Fields that already gave passes, or failed correlation at submission, are "used up".
        self.field_passes = Counter(r["field"] for r in self.sims if r["passed"] == "True" and r["field"])
        for s in submissions:
            if "CORRELATION" in s["failed_checks"]:
                self.field_passes[s["field"]] += cfg["max_passes_per_field"]
        for f in self.submitted_fields:
            self.field_passes[f] += cfg["max_passes_per_field"]
        # Every data field used anywhere in a verified pass (combinations use several).
        common = {"cap", "returns", "close", "volume"}
        self.component_passes = Counter(f for r in self.sims if r["passed"] == "True"
                                        for f in fields_in(r["expression"]) - common)

    def check_is_stale(self, row, check):
        """True if this PASS was checked before BRAIN caught up with a submission sharing its data."""
        checked = parse_time(check.get("checked_at"))
        if not checked:
            return False
        lag = timedelta(minutes=self.cfg.get("submission_lag_minutes", 20))
        mine = fields_in(row["expression"])
        return any(checked < when + lag and mine & theirs for when, theirs in self.recent_submissions)

    def _apply_check(self, row):
        if row["passed"] != "True":
            return row
        row = dict(row)
        check = self.checks.get(row["alpha_id"])
        if (check is None or check["verdict"] == "RETRY"
                or (check["verdict"] == "PASS" and self.check_is_stale(row, check))):
            row["passed"] = "Pending"
        elif check["verdict"] != "PASS":
            row["passed"] = "False"
            row["failed_checks"] = check["failed"]
            row["check_details"] = check["detail"]
        return row

    # ---------- generating new ideas ----------

    def banned(self):
        out = set(self.cfg.get("exclude", []))
        for key, (n, _, passes, abs_sharpe) in self.stats.items():
            if (key.split(":", 1)[0] in BANNABLE and n >= self.cfg["dead_after_trials"]
                    and passes == 0 and abs_sharpe / n < 0.5):
                out.add(key)
        return out

    def pick(self, dim, values, penalty=None):
        """Thompson sampling: proven options win more often, untried ones still get explored."""
        def draw(v):
            n, r, _, _ = self.stats.get(f"{dim}:{v}", (0, 0.0, 0, 0.0))
            return random.betavariate(1 + r, 1 + n - r) * (penalty or {}).get(v, 1.0)
        return max(values, key=draw)

    def propose(self, templates, fields, done, n, operators=None, known_fields=None):
        """fields: (field id, dataset id, "MATRIX"/"VECTOR") the generator may pick from.
        known_fields: every field id that exists (for checking names written directly in ideas)."""
        cfg, search = self.cfg, self.cfg["search"]
        banned = self.banned()

        def allowed(dim, values):
            return [v for v in values if f"{dim}:{v}" not in banned] or values

        pools = defaultdict(lambda: defaultdict(list))  # pools[type][dataset] -> field ids
        for field_id, dataset, kind in fields:
            pools[kind][dataset].append(field_id)
        field_ids = known_fields or {f for f, _, _ in fields}
        saturated = {f: 0.2 for f, k in self.field_passes.items() if k >= cfg["max_passes_per_field"]}
        per_field = Counter()

        def pick_field(kind, exclude_dataset=None):
            datasets = [d for d in pools[kind] if d != exclude_dataset]
            if not datasets:
                return None, None
            dataset = self.pick("dataset", allowed("dataset", sorted(datasets)))
            options = [f for f in allowed("field", pools[kind][dataset])
                       if per_field[f] < cfg["max_per_field_per_cycle"]]
            return (self.pick("field", options, saturated), dataset) if options else (None, None)

        out, seen = [], set(done)
        for _ in range(n * 50):
            if len(out) >= n:
                break
            template = self.pick("template", allowed("template", templates))
            names = placeholders(template)
            row = {"template": template, "field": "", "dataset": "", "window": "", "group": ""}
            fill = {}
            if "field" in names or "vfield" in names:
                kind, slot = ("MATRIX", "field") if "field" in names else ("VECTOR", "vfield")
                fill[slot], row["dataset"] = pick_field(kind)
                row["field"] = fill[slot]
                if not row["field"]:
                    continue
            if "field2" in names:
                # Second signal from a different dataset: combinations rarely match anything submitted.
                fill["field2"], _ = pick_field("MATRIX", exclude_dataset=row["dataset"])
                if not fill["field2"]:
                    continue
            # Universe first: smaller universes get their own groups, neutralisation, decay and truncation.
            universe = self.pick("universe", search["universe"]) if search.get("universe") else None
            prof = {**search, **cfg.get("universe_profiles", {}).get(universe or "", {})}
            if "window" in names:
                row["window"] = fill["window"] = self.pick("window", prof["windows"])
            if "group" in names:
                row["group"] = fill["group"] = self.pick("group", prof["groups"])
            settings = {"decay": self.pick("decay", prof["decay"]),
                        "neutralization": self.pick("neutralization", prof["neutralization"])}
            if universe:
                settings["universe"] = universe
            if prof.get("truncation"):
                settings["truncation"] = self.pick("truncation", prof["truncation"])
            expr = template.format(**fill)
            key = (expr, json.dumps(settings, sort_keys=True))
            if key in seen or skeleton(expr) in self.submitted_shapes:
                continue  # already tried, or same shape as something you submitted
            if operators and invalid_names(expr, field_ids, operators):
                continue  # unknown field or operator: BRAIN would reject it
            seen.add(key)
            per_field[row["field"]] += 1
            out.append({**row, "expression": expr, "settings": settings, "repair": "", "parent": "", "depth": 0})
        return out

    # ---------- repairing near-misses ----------

    def rank_repairs(self, checks):
        order = []
        for check in checks:
            order += [k for k in REPAIRS.get(check, []) if k not in order]

        def score(kind):
            tries = improved = passed = 0
            for check in checks:
                t, i, p = self.repair_stats.get((kind, check), (0, 0, 0))
                tries, improved, passed = tries + t, improved + i, passed + p
            return (improved + passed + 1) / (2 * tries + 3)
        return sorted(order, key=lambda k: (-score(k), order.index(k)))

    def corr_variants(self, r, kind, fields_by_dataset):
        """Variants of an alpha that failed self-correlation / robustness. Yields (expr, overrides, field)."""
        base = self.cfg["settings"]
        overrides = json.loads(r["settings"] or "{}")
        full = {**base, **overrides}
        expr, field = r["expression"], r["field"]
        if kind.startswith("neut_"):
            result = apply_repair(kind, expr, overrides, base)
            if result:
                yield result[0], result[1], field
        elif kind == "size_neutral":
            if "bucket(rank(cap)" not in expr:
                yield apply_repair(kind, expr, overrides, base) + (field,)
        elif kind == "subtract_corr":
            # Remove the overlap directly: subtract the submitted alpha this one is too similar to.
            other = self.submitted_code.get(self.checks.get(r["alpha_id"], {}).get("corr_with", ""), "")
            if other and ";" not in other and not re.search(r"\b[a-z_]+\s*=(?!=)", other.lower()):
                other = " ".join(other.split())
                for w in (0.5, 0.3, 0.7):
                    yield f"rank({expr}) - {w} * rank({other})", overrides, field
        elif kind == "universe_swap":
            for u in UNIVERSES:
                if u != full.get("universe"):
                    yield expr, {**overrides, "universe": u}, field
        elif kind == "group_swap" and r["group"]:
            for g in sorted(GROUPS - {"market", r["group"]}):
                yield re.sub(rf"\b{r['group']}\b", g, expr), overrides, field
        elif kind == "field_swap" and field and r["dataset"] in fields_by_dataset:
            # Related fields first (sharing a word with the original), then any unused field.
            words = set(field.lower().split("_")) - {"fnd6", "fn", "anl4", "a", "q", "value", "v1300"}
            pool = [f for f in fields_by_dataset[r["dataset"]]
                    if f != field and f not in self.submitted_fields
                    and self.field_passes[f] < self.cfg["max_passes_per_field"]]
            related = [f for f in pool if words & set(f.lower().split("_"))]
            random.shuffle(related)
            others = [f for f in pool if f not in related]
            random.shuffle(others)
            for f in (related[:2] + others[:1]):
                yield re.sub(rf"\b{re.escape(field)}\b", f, expr), overrides, f

    def rank_corr_repairs(self, checks):
        def score(kind):
            t, i, p = [sum(self.repair_stats.get((kind, c), (0, 0, 0))[k] for c in checks) for k in range(3)]
            return (i + 2 * p + 1) / (3 * t + 3)
        return sorted(CORR_REPAIRS, key=lambda k: (-score(k), CORR_REPAIRS.index(k)))

    def plan_corr_repairs(self, seen, budget, fields_by_dataset, focus=None):
        """Alphas that passed in-sample but failed self-correlation/robustness: change field,
        neutralisation, group, universe or size exposure until one gets through.
        focus: only repair these alpha ids (sweep mode), with a deeper repair limit."""
        cfg = self.cfg
        windows = {str(w) for w in cfg["search"]["windows"]}
        max_depth = cfg.get("sweep_max_depth", 4) if focus else cfg.get("max_corr_repair_depth", 2)
        candidates = [r for r in self.sims if r["passed"] == "False" and not one_sided(r)
                      and (focus is None or r["alpha_id"] in focus)
                      and set(r["failed_checks"].split(";")) & GATE_CHECKS
                      and (focus or not r["window"] or r["window"] in windows)
                      and int(r["depth"] or 0) < max_depth
                      and "subtract_corr" not in self.child_repairs[r["alpha_id"]]]
        # Correlation-only failures first (subtraction can't fix weak years), then least-correlated:
        # they need the smallest change to get under 0.7.
        candidates.sort(key=lambda r: ("NOT_ROBUST" in r["failed_checks"],
                                       num(self.checks.get(r["alpha_id"], {}).get("self_corr"), 1.0)))
        jobs = []
        for r in candidates:
            checks = [c for c in r["failed_checks"].split(";") if c in GATE_CHECKS]
            added = 0
            for kind in self.rank_corr_repairs(checks):
                for expr, settings, field in self.corr_variants(r, kind, fields_by_dataset):
                    if added >= cfg.get("corr_repairs_per_alpha", 10) or len(jobs) >= budget:
                        break
                    key = (expr, json.dumps(settings, sort_keys=True))
                    if key in seen or skeleton(expr) in self.submitted_shapes:
                        continue
                    seen.add(key)
                    added += 1
                    jobs.append({**{k: r[k] for k in ("template", "dataset", "window", "group")}, "field": field,
                                 "expression": expr, "settings": settings, "repair": kind,
                                 "parent": r["alpha_id"], "depth": int(r["depth"] or 0) + 1})
            if len(jobs) >= budget:
                break
        return jobs

    def plan_combos(self, seen, budget):
        """Add up two near-miss signals from different ideas/datasets. Two weakly related signals
        with Sharpe ~1 each combine to ~1.4, and the mix rarely matches anything already submitted."""
        cfg = self.cfg
        floor = cfg.get("combo_min_sharpe", 0.85)
        windows = {str(w) for w in cfg["search"]["windows"]}
        ok_checks = {"LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"}
        combo_universes = set(cfg.get("combo_universes", ["TOP3000"]))
        universe_of = lambda r: json.loads(r["settings"] or "{}").get("universe", "TOP3000")
        pool = [r for r in self.sims if r["passed"] == "False" and not one_sided(r)
                and abs(num(r["sharpe"])) >= floor
                and set(filter(None, r["failed_checks"].split(";"))) <= ok_checks
                and (not r["window"] or r["window"] in windows)
                and r["field"] not in self.submitted_fields
                and json.loads(r["settings"] or "{}").get("universe", "TOP3000") in combo_universes
                # Components already in enough verified passes would only make more of the same family.
                and not any(self.component_passes[f] >= cfg["max_passes_per_field"]
                            for f in fields_in(r["expression"]))
                and 0.01 <= num(r["turnover"]) <= 0.6]
        pool.sort(key=lambda r: abs(num(r["sharpe"])), reverse=True)
        best = {}  # one per idea+field: repaired copies of the same alpha would only duplicate it
        for r in pool:
            best.setdefault((r["template"], r["field"]), r)
        pool = list(best.values())[:cfg.get("combo_pool", 25)]
        jobs = []
        # Strongest pairs first, but spread out: pair (0,1), (0,2), (1,2), (0,3), (1,3)...
        pairs = sorted(((i, j) for i in range(len(pool)) for j in range(i + 1, len(pool))),
                       key=lambda p: (p[1], p[0]))
        used = Counter()
        for i, j in pairs:
            a, b = pool[i], pool[j]
            if len(jobs) >= budget:
                return jobs
            cap = cfg.get("combo_uses_per_alpha", 3)
            if used[i] >= cap or used[j] >= cap:
                continue  # each ingredient in at most `cap` combinations, so passes are distinct families
            # Different idea and different data, or the two halves would just duplicate each other.
            if a["template"] == b["template"] or (a["field"] and a["field"] == b["field"]):
                continue
            # Same universe only, and the combination runs there: a signal tested on TOP500 doesn't
            # carry over to TOP3000 (mixed combinations averaged Sharpe 0.89 vs 1.42, 2026-10-07).
            if universe_of(a) != universe_of(b):
                continue
            part = lambda r: f"rank({r['expression']})" if num(r["sharpe"]) > 0 else f"rank(-({r['expression']}))"
            expr = f"{part(a)} + {part(b)}"
            settings = json.loads(a["settings"] or "{}")
            key = (expr, json.dumps(settings, sort_keys=True))
            if key in seen or skeleton(expr) in self.submitted_shapes:
                continue
            seen.add(key)
            used[i] += 1
            used[j] += 1
            jobs.append({"template": "COMBINE", "field": a["field"], "dataset": a["dataset"],
                         "window": "", "group": "", "expression": expr, "settings": settings,
                         "repair": "combine", "parent": a["alpha_id"],
                         "depth": max(int(a["depth"] or 0), int(b["depth"] or 0)) + 1})
        return jobs

    def plan_repairs(self, done, budget, fields=(), focus=None):
        """focus: in sweep mode, only repair this set of alpha ids (one idea's versions), with a
        lower Sharpe threshold and deeper limit, and no combinations."""
        cfg, base = self.cfg, self.cfg["settings"]
        seen = set(done)
        fields_by_dataset = defaultdict(list)
        for field_id, dataset, kind in fields:
            if kind == "MATRIX":
                fields_by_dataset[dataset].append(field_id)
        # Correlation repairs first: these alphas already pass every in-sample check.
        jobs = self.plan_corr_repairs(seen, budget, fields_by_dataset, focus)
        if focus is None:
            jobs += self.plan_combos(seen, min(budget - len(jobs), cfg.get("combos_per_cycle", 12)))
        windows = {str(w) for w in cfg["search"]["windows"]}
        max_depth = cfg.get("sweep_max_depth", 4) if focus else cfg["max_repair_depth"]
        min_sharpe = cfg.get("sweep_repair_min_sharpe", 0.8) if focus else cfg["repair_min_sharpe"]
        candidates = [r for r in self.sims if r["passed"] == "False" and not one_sided(r)
                      and (focus is None or r["alpha_id"] in focus)
                      and not set(r["failed_checks"].split(";")) & GATE_CHECKS
                      and (focus or not r["window"] or r["window"] in windows)  # skip retired windows
                      and int(r["depth"] or 0) < max_depth
                      and abs(num(r["sharpe"])) >= min_sharpe
                      and not self.children[r["alpha_id"]]]
        candidates.sort(key=lambda r: (reward(r), abs(num(r["sharpe"]))), reverse=True)  # strongest first
        if len(jobs) >= budget:
            return jobs
        for r in candidates:
            checks = [c for c in r["failed_checks"].split(";") if c]
            # A strongly negative Sharpe means the idea works backwards: flip it first.
            kinds = ["flip"] if num(r["sharpe"]) < 0 else self.rank_repairs(checks)
            if num(r["turnover"]) < 0.125:
                # Fitness treats turnover below 12.5% as 12.5%, so cutting it further cannot help;
                # a faster-reacting signal (lower decay) has a better chance of lifting returns.
                kinds = ["decay_down"] + [k for k in kinds if k not in TURNOVER_CUTTERS]
            overrides = json.loads(r["settings"] or "{}")
            added = 0
            for kind in kinds:
                if added >= cfg["repairs_per_alpha"] or len(jobs) >= budget:
                    break
                if kind == r["repair"] and kind in ("flip", "smooth", "decay_linear", "rank"):
                    continue  # wrapping the same operator twice adds nothing
                result = apply_repair(kind, r["expression"], overrides, base)
                if not result:
                    continue
                expr, settings = result
                key = (expr, json.dumps(settings, sort_keys=True))
                if key in seen or skeleton(expr) in self.submitted_shapes:
                    continue
                seen.add(key)
                added += 1
                jobs.append({**{k: r[k] for k in ("template", "field", "dataset", "window", "group")},
                             "expression": expr, "settings": settings, "repair": kind,
                             "parent": r["alpha_id"], "depth": int(r["depth"] or 0) + 1})
            if len(jobs) >= budget:
                break
        return jobs

    # ---------- reporting ----------

    def shortlist(self, n=10, all_passes=False):
        """Verified alphas not yet submitted. By default the best one per family; with all_passes,
        every one, each tagged with its family (alphas sharing a data field are likely to be
        correlated, so submitting one can make the others fail)."""
        passing = [r for r in self.sims if r["passed"] == "True" and r["alpha_id"] not in self.submitted]
        passing.sort(key=lambda r: num(r["fitness"]), reverse=True)
        common = {"cap", "returns", "close", "volume", "open", "high", "low", "vwap"}
        families = []  # list of sets of fields
        out = []
        for r in passing:
            fields = (fields_in(r["expression"]) - common) or {r["expression"]}
            family = next((i for i, fs in enumerate(families) if fs & fields), None)
            if family is None:
                families.append(set(fields))
                family = len(families) - 1
                best = True
            else:
                families[family] |= fields
                best = False
            if best or all_passes:
                out.append({**r, "self_corr": self.checks[r["alpha_id"]]["self_corr"],
                            "family": family + 1, "best_in_family": best})
        return out[:n]

    def suggestions(self, cycles, failures):
        out = []
        ordered = sorted(cycles)
        if len(ordered) >= 2:
            first, last = cycles[ordered[0]], cycles[ordered[-1]]
            out.append(f"Pass rate went from {first[1]}/{first[0]} in cycle {ordered[0]} "
                       f"to {last[1]}/{last[0]} in cycle {ordered[-1]}.")
        if failures:
            top, count = failures.most_common(1)[0]
            out.append(f"Most common failure is {top} ({count}x). {CHECK_HELP.get(top, '')}")
        for key, (n, _, passes, _) in self.stats.items():
            dim, value = key.split(":", 1)
            if dim == "template" and n >= 20 and passes == 0:
                out.append(f"Template `{value}` has 0 passes in {n} tries: rework or remove it from templates.txt.")
        for (kind, check), (n, improved, passed) in self.repair_stats.items():
            if n >= 10 and improved == 0:
                out.append(f"Repair `{kind}` never helped {check} in {n} tries; the system now tries it last.")
        used = [f for f, k in self.field_passes.items() if k >= self.cfg["max_passes_per_field"]]
        if len(used) >= 10:
            out.append(f"{len(used)} fields are used up. Download a new dataset: "
                       "`python alpha_tool.py fields <dataset id>`.")
        for dim in ("decay", "neutralization"):
            ranked = self.top(dim, min_tries=10)
            if ranked:
                out.append(f"Best {dim} so far is {ranked[0][0]}; it is being picked more often automatically.")
        return out

    def top(self, dim, min_tries=3, k=8):
        rows = []
        for key, (n, total, passes, abs_sharpe) in self.stats.items():
            d, value = key.split(":", 1)
            if d == dim and n >= min_tries:
                rows.append((value, n, passes, total / n, abs_sharpe / n))
        return sorted(rows, key=lambda x: (x[2] / x[1], x[3]), reverse=True)[:k]

    def write_insights(self, path):
        notes = ""
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if NOTES_HEADING in text:
                notes = text.split(NOTES_HEADING, 1)[1].strip()

        cycles = defaultdict(lambda: [0, 0])
        for r in self.sims:
            c = cycles[int(r["cycle"] or 0)]
            c[0] += 1
            c[1] += r["passed"] == "True"
        failures = Counter(c for r in self.sims for c in r["failed_checks"].split(";") if c)
        passed = sum(r["passed"] == "True" for r in self.sims)
        errors = sum(1 for r in self.rows if r["error"])

        lines = ["# Alpha insights", "",
                 "Rewritten after every cycle from results.csv. Only the **My notes** section at the "
                 "bottom is kept, so write your own observations there.", "",
                 f"**{len(self.sims)} simulated, {passed} passed "
                 f"({100 * passed / max(len(self.sims), 1):.0f}%), {errors} errors.**", "",
                 "## Progress by cycle", "", "| Cycle | Simulated | Passed | Pass rate |", "|---|---|---|---|"]
        for c in sorted(cycles):
            n, p = cycles[c]
            lines.append(f"| {c} | {n} | {p} | {100 * p / n:.0f}% |")

        lines += ["", "## What works best", "",
                  "Pass rate counts an idea as passed if it or any of its repairs passed.", ""]
        for dim in DIMENSIONS:
            ranked = self.top(dim)
            if not ranked:
                continue
            lines += [f"**{dim}**", "", "| Value | Tries | Passed | Pass rate | Avg score | Avg abs Sharpe |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| `{v}` | {n} | {p} | {100 * p / n:.0f}% | {s:.2f} | {sh:.2f} |"
                      for v, n, p, s, sh in ranked]
            lines.append("")

        banned = sorted(self.banned())
        lines += ["## Dead ends (skipped automatically)", ""]
        lines += [f"- `{b}`" for b in banned] or ["- none yet"]

        lines += ["", "## Why alphas fail", "", "| Check | Count | Fix |", "|---|---|---|"]
        lines += [f"| {c} | {n} | {CHECK_HELP.get(c, '')} |" for c, n in failures.most_common()]

        is_passes = [r for r in self.sims if r["alpha_id"] in self.checks or r["passed"] == "Pending"]
        gate_fails = Counter(c for r in is_passes for c in r["failed_checks"].split(";") if c in GATE_CHECKS)
        corr_with = Counter(self.checks[r["alpha_id"]]["corr_with"] for r in is_passes
                            if "SELF_CORRELATION" in r["failed_checks"] and r["alpha_id"] in self.checks)
        lines += ["", "## Verification gate (self-correlation + robustness)", "",
                  f"{len(is_passes)} alphas passed the in-sample checks; "
                  f"{sum(r['passed'] == 'True' for r in is_passes)} also passed verification, "
                  f"{sum(r['passed'] == 'Pending' for r in is_passes)} still waiting.", ""]
        lines += [f"- {c}: {n} failed. {CHECK_HELP.get(c, '')}" for c, n in gate_fails.most_common()]
        if corr_with:
            lines.append("- Your submitted alphas they were too similar to: "
                         + ", ".join(f"{a} ({n}x)" for a, n in corr_with.most_common(5)))

        lines += ["", "## Which repairs work", "", "| Repair | Fixing | Tries | Improved | Passed |",
                  "|---|---|---|---|---|"]
        for (kind, check), (n, improved, ok) in sorted(self.repair_stats.items(), key=lambda x: -x[1][2]):
            lines.append(f"| {kind} | {check} | {n} | {improved} | {ok} |")

        lines += ["", "## Ready to submit (verified, best per field and shape, not yet submitted)", "",
                  "Run `python alpha_tool.py best` before submitting: it re-checks them live.", "",
                  "| Alpha | Sharpe | Fitness | Turnover | Max self-corr | Expression | Settings |",
                  "|---|---|---|---|---|---|---|"]
        for r in self.shortlist():
            lines.append(f"| {r['alpha_id']} | {r['sharpe']} | {r['fitness']} | {r['turnover']} | "
                         f"{r['self_corr']} | `{r['expression']}` | `{r['settings']}` |")

        lines += ["", "## Suggestions", ""]
        lines += [f"- {s}" for s in self.suggestions(cycles, failures)] or ["- run more cycles first"]
        lines += ["", NOTES_HEADING, "", notes or "_Write your own observations and ideas here._", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
