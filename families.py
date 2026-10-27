"""Hypothesis-driven idea families (2026-10-21).

Why: the template sweeps (any field into a generic shape) gave 0 regular alphas in 1,845 consultant-era
simulations: strong results were copies of production alphas, unique ones were weak. Published consultant
experience says the same: 13,319 unedited sweep runs gave 0 submittable alphas, while hand-built ideas with
an economic reason passed (Shrijalexpo/brain-alpha-atlas). The passing ones were:
- profitability or value RATIOS of two related accounting fields (ebitda / enterprise value, cash flow from
  operations / assets, operating income / equity), not raw accounting levels (those gave Sharpe < 0.5);
- often plus a smoothed 5-day reversal leg, rank(ts_mean(-returns, 5)), which lifted fitness;
- option-implied volatility spreads, short interest against its own history, analyst EPS revisions;
- country x industry neutralisation in multi-country regions (EUR, GLB, ASI, AMR);
- CROWDING neutralisation, which removes platform-wide crowded positioning (lower production correlation).

Each family names the ROLE of each field (a profit measure, a scale measure, an estimate...), and fields are
found by meaning (their description) in every region, so the same economic idea is tried across regions and
datasets. Families are learned like templates: the family pattern is the job's "template".
"""
import random
import re
from collections import defaultdict

from learner import region_of

# Field roles, found by the field's description (MATRIX fields only).
ROLES = {
    "profit": r"\bebitda\b|operating income|^net income|(operating|operations).{0,20}cash ?flow|cash ?flow.{0,20}operat|"
              r"^(net )?sales|^revenue|total revenue|gross profit",
    "scale": r"^total assets|assets - total|^assets$|total (shareholders'? )?equity|common equity|"
             r"stockholders'? equity|enterprise value",
    "estimate": r"(eps|earnings per share).{0,40}(estimate|consensus|forecast|mean)|"
                r"(estimate|consensus|forecast).{0,40}(eps|earnings per share)",
    "short": r"short interest",
}
# Leave out statistics of a field (an estimate's high/low/spread...), counts and dates: they are not amounts.
NOT_AMOUNT = r"(\b|_)(high|low|std|stddev|standard deviation|number of|count|date|flag|ratio|growth|change|chg|"\
             r"percent|pct|rank|score|surprise|revision|num)(\b|_)"
ROLE_DATASETS = {"profit": r"fund|fnd", "scale": r"fund|fnd"}  # accounting amounts from fundamental datasets

MULTI_COUNTRY = {"EUR", "GLB", "ASI", "AMR"}
REV = "rank(ts_mean(-returns, 5))"
RATIO = "ts_backfill({num}, 63) / ts_backfill({den}, 63)"

# (pattern, decays). {group} is filled per region; {ctry} only in multi-country regions.
FAMILIES = [
    (f"group_rank(ts_rank({RATIO}, 126), {{group}})", [0, 2, 4]),
    (f"rank(ts_rank({RATIO}, 126)) + {REV}", [2, 4, 6]),
    (f"group_rank(ts_delta({RATIO}, 63), {{group}}) + 0.5 * {REV}", [2, 4, 6]),
    (f"vector_neut(group_rank(ts_rank({RATIO}, 126), {{group}}), rank(ts_std_dev(returns, 120)))", [0, 2, 4]),
    (f"group_neutralize(ts_rank({RATIO}, 126), group_cartesian_product(country, industry))", [0, 2, 4]),
    ("group_rank(ts_delta(ts_backfill({est}, 21), 63) / (abs(ts_mean(ts_backfill({est}, 21), 63)) + 0.001), {group})",
     [0, 2, 4]),
    (f"rank(ts_delta(ts_backfill({{est}}, 21), 63) / (abs(ts_mean(ts_backfill({{est}}, 21), 63)) + 0.001)) + {REV}",
     [2, 4]),
    ("group_rank(ts_backfill({ivc}, 5) - ts_backfill({ivp}, 5), {group})", [0, 2]),
    ("-group_rank(ts_backfill({ivc}, 5) - ts_std_dev(returns, 20) * 15.87, {group})", [0, 2, 4]),
    (f"-group_rank(ts_backfill({{short}}, 21) / (ts_mean(ts_backfill({{short}}, 21), 252) + 0.001), {{group}}) + 0.3 * {REV}",
     [2, 4]),
]
GROUPS = ["subindustry", "industry", "sector"]


def placeholders(pattern):
    return set(re.findall(r"\{(\w+)\}", pattern))


def field_roles(fields_rows):
    """{region: {role: [(field id, dataset label)]}} plus option IV pairs {region: [(call, put, dataset)]}."""
    roles = defaultdict(lambda: defaultdict(list))
    calls = defaultdict(dict)
    puts = defaultdict(dict)
    for f in fields_rows:
        if f.get("type") != "MATRIX":
            continue
        region = f.get("region") or "USA"
        label = f["dataset"] if region == "USA" else f"{f['dataset']}@{region}"
        desc = f.get("description") or ""
        for role, pattern in ROLES.items():
            if not re.search(pattern, desc, re.I):
                continue
            if role in ("profit", "scale", "estimate") and re.search(NOT_AMOUNT, desc + " " + f["id"], re.I):
                continue
            if role in ROLE_DATASETS and not re.search(ROLE_DATASETS[role], f["dataset"], re.I):
                continue
            roles[region][role].append((f["id"], label))
        m = re.match(r"implied_volatility_(call|put)_(\d+)$", f["id"])
        if m:
            (calls if m.group(1) == "call" else puts)[(region, f["dataset"])][m.group(2)] = f["id"]
    iv = defaultdict(list)
    for (region, dataset), by_tenor in calls.items():
        for tenor, call in by_tenor.items():
            put = puts.get((region, dataset), {}).get(tenor)
            if put and 30 <= int(tenor) <= 270:
                iv[region].append((call, put, dataset if region == "USA" else f"{dataset}@{region}"))
    return roles, iv


def propose(know, fields_rows, done, n, cfg):
    """Up to n family ideas, spread over regions by the learner's region weights and over families by
    Thompson sampling on each family's results (its pattern is the job's template)."""
    roles, iv = field_roles(fields_rows)
    weights = know.region_weights()
    regions = [g for g in weights if roles.get(g) or iv.get(g)]
    if not regions:
        return []
    seen, out = set(done), []
    tried = {(r["template"], r["field"]) for r in know.rows if r.get("field")}
    for _ in range(n * 30):
        if len(out) >= n:
            break
        region = random.choices(regions, weights=[weights[g] for g in regions])[0]
        prof = {**cfg["search"], **cfg.get("regions", {}).get(region, {})}
        usable = [p for p, _ in FAMILIES if family_ok(p, region, roles[region], iv.get(region))]
        if not usable:
            continue
        pattern = know.pick("template", usable)
        decays = dict(FAMILIES)[pattern]
        fill, field, dataset = {}, "", ""
        names = placeholders(pattern)
        if "num" in names:
            num, dataset = choose(roles[region]["profit"], know)
            dens = [d for d in roles[region]["scale"] if d[1] == dataset] or roles[region]["scale"]
            den, _ = random.choice(dens)
            fill.update(num=num, den=den)
            field = f"{num}/{den}"
        if "est" in names:
            fill["est"], dataset = choose(roles[region]["estimate"], know)
            field = fill["est"]
        if "short" in names:
            fill["short"], dataset = choose(roles[region]["short"], know)
            field = fill["short"]
        if "ivc" in names:
            fill["ivc"], fill["ivp"], dataset = choose(iv[region], know)
            field = fill["ivc"]
        if (pattern, field) in tried:
            continue
        fill["group"] = know.pick("group", [g for g in GROUPS if g in prof.get("groups", GROUPS)] or GROUPS)
        expr = pattern.format(**fill)
        neutral = list(prof.get("neutralization", ["SUBINDUSTRY"]))
        if region != "AMR" and "CROWDING" not in neutral:
            neutral.append("CROWDING")  # orthogonal to platform-wide crowding: lower production correlation
        settings = {"decay": random.choice(decays), "neutralization": know.pick("neutralization", neutral)}
        if region == "USA":
            settings["universe"] = "TOP3000"  # the published passes and our own all came from TOP3000
        else:
            settings.update(region=region, universe=prof["universe"])
            if prof.get("maxTrade"):
                settings["maxTrade"] = prof["maxTrade"]
        import json
        key = (expr, json.dumps(settings, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        tried.add((pattern, field))
        out.append({"template": pattern, "field": field, "dataset": dataset, "window": "", "group": fill["group"],
                    "expression": expr, "settings": settings, "repair": "family", "parent": "", "depth": 0})
    return out


def choose(options, know):
    """A random option (its dataset label last), weighted towards pyramids one or two submissions short."""
    boost = getattr(know, "pyramid_boost", None) or {}
    return random.choices(options, weights=[boost.get(o[-1], 1.0) for o in options])[0]


def family_ok(pattern, region, roles, iv_pairs):
    names = placeholders(pattern)
    if "group_cartesian_product(country" in pattern and region not in MULTI_COUNTRY:
        return False
    if "num" in names and not (roles.get("profit") and roles.get("scale")):
        return False
    if "est" in names and not roles.get("estimate"):
        return False
    if "short" in names and not roles.get("short"):
        return False
    if "ivc" in names and not iv_pairs:
        return False
    return True
