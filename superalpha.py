"""SuperAlphas: the user's main target since 2026-10-19 (they carry more weight than regular alphas).

A SuperAlpha combines the user's own submitted, ACTIVE alphas (BRAIN docs, "SuperAlpha" section):
  * the SELECTION expression scores every submitted alpha on its properties (prod_correlation,
    self_correlation, turnover, long_count, datacategories, ...); the top `selectionLimit` are used;
  * the COMBO expression gives each selected alpha a daily weight (equal weight "1", inverse correlation,
    combo_a, ...).
Submission rules are the regular consultant tests except turnover, which must be 2-40%, and BOTH a selection
and a combo description are required.

Each recipe below is a documented, economically motivated combination of BRAIN's own example selections and
combos, not a blind search: the user's guidelines ask for alphas with an economic foundation and warn against
using automation as the source of ideas. The bot only chooses which recipe to test next, learning from
results which selections, combos and settings work for this pool.
"""
import csv
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
SUPERS = HERE / "supers.csv"
COLUMNS = ["created", "alpha_id", "recipe", "selection", "combo", "settings", "sharpe", "fitness", "turnover",
           "returns", "drawdown", "failed_checks", "check_details", "error", "verdict", "self_corr",
           "corr_with", "verify_detail", "described"]

# (key, selection expression, selection handling, what it is for)
SELECTIONS = [
    ("low_prod_corr", "(1 - prod_correlation) * (turnover < 0.3)", "POSITIVE",
     "components least like the production pool, tradable turnover"),
    ("low_self_corr", "(1 - self_correlation) * (turnover < 0.3)", "POSITIVE",
     "components least like each other"),
    ("breadth", "(long_count + short_count) / sqrt(universe_size(universe)) * (turnover < 0.25)", "POSITIVE",
     "broad, diversified components"),
    ("balanced_low_corr", "(1 - prod_correlation) * (1 - self_correlation) * (turnover < 0.35)", "POSITIVE",
     "distinct from both the platform and my pool"),
    ("smooth", "(1 - prod_correlation) * (decay >= 4) * (turnover < 0.3)", "POSITIVE",
     "components that already smooth their positions"),
    ("net_long_decay", "(long_count - short_count) * decay", "NON_NAN",
     "the selection of the user's first passing SuperAlpha (RRA2Ppxb)"),
    # Production correlation is what blocks most SuperAlphas (2026-10-19: 0.71-0.86 on the first tests), because
    # averaging many alphas drifts towards the factors everyone holds. These keep only the most unique components.
    ("unique_only", "(prod_correlation < 0.5) * (1 - prod_correlation) * (1 - self_correlation)", "POSITIVE",
     "only components clearly unlike the production pool"),
    ("unique_lowturn", "(prod_correlation < 0.55) * (turnover < 0.2) * (1 - prod_correlation)", "POSITIVE",
     "unique, slow-trading components"),
    # 2026-10-24: four unique_only recipes (MARKET neutralisation) were refused at production correlation
    # 0.76-0.83: keep only the clearly unique components.
    ("ultra_unique", "(prod_correlation < 0.4) * (1 - prod_correlation)", "POSITIVE",
     "only the components least like the production pool (production correlation below 0.4)"),
    # A region with just 10 submitted alphas: filters leave fewer than BRAIN's minimum of 10, so take them all
    # (YPxeQew6, IND, 2026-10-25: Sharpe 4.85, production correlation 0.69, accepted).
    ("all", "1", "POSITIVE", "every submitted alpha in the region (a young, unique pool)"),
    # Rankings rather than filters, so a small regional pool still yields 10 components, and each picks a
    # different 10: N1NXWe7L (IND, slowest-trading 10, SLOW_AND_FAST) was accepted on 2026-10-26 at 0.696
    # with YPxeQew6, while rankings close to its own gave 0.89-0.91.
    ("rank_low_turnover", "1 - turnover", "POSITIVE", "the slowest-trading components"),
    ("rank_low_self_corr", "1 - self_correlation", "POSITIVE", "the components least like the rest of my pool"),
    ("rank_low_prod_corr", "1 - prod_correlation", "POSITIVE", "the components least like the production pool"),
    # 2026-10-29: every new SuperAlpha was refused at 0.71-0.94 self-correlation with an earlier one, because each
    # picks mostly the same components. The most recently submitted alphas are the ones no SuperAlpha holds yet.
    ("rank_recent", "os_start_date * (turnover < 0.4)", "POSITIVE",
     "the most recently submitted components with tradable turnover"),
    ("recent_unique", "os_start_date * (prod_correlation < 0.6)", "POSITIVE",
     "the most recent components that are clearly unlike the production pool"),
]
# Recipes already submitted as SuperAlphas (never re-test them: they would only copy the submission).
SUBMITTED_RECIPES = {"net_long_decay|combo_a_30_scaled|STATISTICAL|d4|n10",  # RRA2Ppxb, 2026-10-18
                     "balanced_low_corr|combo_a_250|STATISTICAL|d4|n10",  # omo0QEqJ, 2026-10-24 (prod corr 0.61)
                     "all|combo_a_250|STATISTICAL|d4|n10|IND"}  # YPxeQew6, 2026-10-25 (IND, Sharpe 4.85, 0.69)
CATEGORY_SELECTION = ('in(datacategories, "{cat}") * (1 - prod_correlation)', "POSITIVE")

INV_CORR = ("stats = generate_stats(alpha); innerCorr = self_corr(stats.returns, 500); "
            "ic = if_else(innerCorr == 1.0, nan, innerCorr); maxCorr = reduce_max(ic); 1 - maxCorr")
COMBOS = [
    ("equal", "1"),
    ("inverse_corr", INV_CORR),
    ("combo_a_250", "combo_a(alpha, nlength = 250, mode = 'algo1')"),
    ("combo_a_60", "combo_a(alpha, nlength = 60, mode = 'algo1')"),
    ("combo_a_30_scaled", "scale(combo_a(alpha, nlength = 30, mode = 'algo1'), scale=3, longscale=4, shortscale=3)"),
]

BASE = {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1, "truncation": 0.08,
        "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "OFF", "maxTrade": "OFF",
        "language": "FASTEXPR", "visualization": False, "componentActivation": "IS"}


def load():
    try:
        with SUPERS.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def append(row):
    new = not SUPERS.exists()
    with SUPERS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def rewrite(rows):
    tmp = SUPERS.with_name(SUPERS.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(SUPERS)


def recipes(cfg, region_categories):
    """Every recipe the bot may try, per eligible region: selections (incl. one per data category in that region's
    pool) x combos x settings. A SuperAlpha only selects alphas of its own region and delay (BRAIN docs), so a
    region qualifies once it has `super.min_region_alphas` (10) submitted alphas. USA keys have no region suffix."""
    sc = cfg.get("super", {})
    out = []
    for region, categories in region_categories.items():
        prof = cfg.get("regions", {}).get(region, {})
        sels = list(SELECTIONS) + [(f"cat_{c}", CATEGORY_SELECTION[0].format(cat=c), CATEGORY_SELECTION[1],
                                    f"{c} alphas least like the production pool") for c in sorted(categories)]
        for skey, sel, handling, _ in sels:
            for ckey, combo in COMBOS:
                for neut in sc.get("neutralizations", ["STATISTICAL", "MARKET"]):
                    for decay in sc.get("decays", [2, 4]):
                        for limit in sc.get("limits", [10, 20, 30]):
                            settings = {**BASE, "neutralization": neut, "decay": decay,
                                        "selectionHandling": handling, "selectionLimit": limit}
                            if region != "USA":
                                settings.update(region=region, universe=prof.get("universe", BASE["universe"]))
                                if prof.get("maxTrade"):
                                    settings["maxTrade"] = prof["maxTrade"]
                            key = f"{skey}|{ckey}|{neut}|d{decay}|n{limit}" + ("" if region == "USA" else f"|{region}")
                            out.append({"recipe": key, "selection": sel, "combo": combo, "settings": settings})
    return out


def plan(cfg, region_categories, n):
    """Pick `n` untried recipes. Each recipe part (selection, combo, neutralisation, decay, size) is scored by the
    best Sharpe it has reached so far plus an exploration bonus for parts with few tries, so the bot moves towards
    what works for this pool without stopping to explore."""
    done = {r["recipe"] for r in load()} | SUBMITTED_RECIPES
    # A selection that left fewer than BRAIN's minimum of 10 components in a region (74 IND tries on 2026-10-29)
    # is skipped there for a week, until the region's pool has grown.
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    skip = {(r["recipe"].split("|")[0], (r["recipe"].split("|") + ["USA"] * 6)[5]) for r in load()
               if "At least 10 component" in (r["error"] or "") and r["created"] >= week_ago}
    # Any other simulation error means BRAIN rejects the expression itself: never try that selection again there.
    skip |= {(r["recipe"].split("|")[0], (r["recipe"].split("|") + ["USA"] * 6)[5]) for r in load()
                if r["error"] and "At least 10 component" not in r["error"] and "busy" not in r["error"].lower()}
    stats = {}
    # A part's value is the Sharpe it reached, discounted by what BRAIN's checks said: a high Sharpe that fails
    # production correlation is worth little, a verified pass is worth full value.
    weight = {"PASS": 1.0, "CHECKING": 0.6, "": 0.6, "FAIL": 0.25}
    for r in load():
        if r["error"] or not r["sharpe"]:
            continue
        value = float(r["sharpe"]) * (0.25 if r["failed_checks"] else weight.get(r["verdict"], 0.6))
        for part in r["recipe"].split("|"):
            stats.setdefault(part, []).append(value)
    # An untried part (e.g. a new neutralisation) is valued like the best part in its position, so it is
    # tried soon; with a fixed 1.2 the proven parts (Sharpe 4+) kept it out for good (2026-10-29).
    best_at = {}
    for r in load():
        for i, part in enumerate(r["recipe"].split("|")):
            if stats.get(part):
                best_at[i] = max(best_at.get(i, 0), max(stats[part]))

    def score(rec):
        s = 0.0
        for i, part in enumerate(rec["recipe"].split("|")):
            v = stats.get(part, [])
            s += (max(v) if v else best_at.get(i, 1.2)) + 0.6 / (1 + len(v)) ** 0.5 + random.random() * 0.2
        return s
    todo = [r for r in recipes(cfg, region_categories) if r["recipe"] not in done
            and (r["recipe"].split("|")[0], r["settings"].get("region", "USA")) not in skip]
    todo.sort(key=score, reverse=True)
    # At most one recipe per selection in a round, so each round explores different component sets.
    out, seen = [], set()
    for r in todo:
        skey = (r["recipe"].split("|")[0], r["settings"].get("region", "USA"))
        if skey in seen:
            continue
        seen.add(skey)
        out.append(r)
        if len(out) >= n:
            break
    return out


def pool_categories(submitted_codes, field_dataset):
    """BRAIN data categories used by your submitted alphas (dataset id without digits, e.g. analyst4 -> analyst)."""
    valid = {"analyst", "broker", "earnings", "fundamental", "imbalance", "insiders", "institutions", "macro",
             "model", "news", "option", "other", "pv", "risk", "sentiment", "shortinterest", "socialmedia"}
    cats = {}
    for code in submitted_codes:
        for token in set(re.findall(r"\b[a-z_][a-z0-9_]*\b", code.lower())):
            ds = field_dataset.get(token)
            if ds:
                c = re.sub(r"\d+$", "", ds)
                if c in valid:
                    cats[c] = cats.get(c, 0) + 1
    return {c for c, n in cats.items() if n >= 10}  # need at least 10 alphas to fill a selection


def summarize(alpha):
    i = alpha.get("is") or {}
    checks = i.get("checks", [])
    failed = [c for c in checks if c.get("result") == "FAIL"]
    return {"alpha_id": alpha["id"], "sharpe": i.get("sharpe"), "fitness": i.get("fitness"),
            "turnover": i.get("turnover"), "returns": i.get("returns"), "drawdown": i.get("drawdown"),
            "failed_checks": ";".join(c["name"] for c in failed),
            "check_details": "; ".join(f"{c['name']} value={c.get('value')} limit={c.get('limit')}" for c in failed),
            "pending": any(c.get("result") in ("PENDING", "ERROR") for c in checks)}
