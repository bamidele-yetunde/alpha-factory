"""Self-improving alpha factory for WorldQuant BRAIN.

Each cycle:
  1. sync your submitted alphas, so new ideas don't copy their shapes or fields
  2. propose new alphas, favouring templates/fields/settings that passed before
  3. simulate them on BRAIN
  4. repair near-misses with the fixes that have worked best for their failed checks
  5. verify every in-sample pass: BRAIN's submission check (self-correlation) + robustness by year
  6. rewrite insights.md with what was learned

Only alphas that pass step 5 count as PASS.

Usage:
  python alpha_tool.py fields fundamental6 analyst4   # add datasets' fields to fields.csv
  python alpha_tool.py cycle --dry-run                # preview the next cycle
  python alpha_tool.py auto --cycles 5                # run 5 learning cycles back to back (0 = forever)
  python alpha_tool.py watch                          # every 10 min: keep READY_TO_SUBMIT.csv current
  python alpha_tool.py insights                       # rebuild and show insights.md
  python alpha_tool.py verdicts                       # PASS/FAIL for each result (--cycle N)
  python alpha_tool.py recheck                        # verify in-sample passes not verified yet (--all)
  python alpha_tool.py best                           # re-verify live and list what is safe to submit
  python alpha_tool.py submit <ALPHA_ID>              # verify again, then submit

Files: config.json (settings and search space), templates.txt (idea shapes),
results.csv (every simulation = the memory), checks.csv (verification results),
submitted_alphas.csv (your submitted alphas), insights.md (feedback + your notes).
config.json and templates.txt are re-read every cycle, so you can edit them while `auto` runs.
"""
import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from learner import Knowledge

HERE = Path(__file__).parent
CONFIG = HERE / "config.json"
TEMPLATES = HERE / "templates.txt"
FIELDS = HERE / "fields.csv"
RESULTS = HERE / "results.csv"
SUBMISSIONS = HERE / "submissions.csv"
INSIGHTS = HERE / "insights.md"
CHECKS = HERE / "checks.csv"
SUBMITTED = HERE / "submitted_alphas.csv"
READY = HERE / "READY_TO_SUBMIT.csv"
IDEAS = HERE / "ideas.txt"
NOTIFIED = HERE / "notified.json"
OPERATORS = HERE / "operators.json"

RESULT_COLUMNS = ["cycle", "expression", "settings", "template", "field", "dataset", "window", "group",
                  "repair", "parent", "depth", "alpha_id", "passed", "sharpe", "fitness", "turnover",
                  "returns", "drawdown", "margin", "long_count", "short_count", "failed_checks",
                  "check_details", "error"]
SUBMISSION_COLUMNS = ["alpha_id", "field", "status", "failed_checks"]
FIELD_COLUMNS = ["id", "dataset", "type", "coverage", "alpha_count", "description"]
CHECK_COLUMNS = ["alpha_id", "checked_at", "verdict", "failed", "self_corr", "corr_with", "yearly_sharpe", "detail"]
SUBMITTED_COLUMNS = ["id", "code", "universe", "decay", "neutralization", "submitted_at", "sharpe"]

ACTIVITY_LOG = HERE / "activity_log.txt"

_brain = None
_book = None
_log_lock = threading.Lock()


def log(message):
    """Print and immediately append a timestamped line to activity_log.txt."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    with _log_lock:
        if not os.environ.get("QUIET"):  # QUIET in GitHub Actions: public run logs must not show alphas
            print(line, flush=True)
        with ACTIVITY_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def brain():
    global _brain
    if _brain is None:
        from brain import Brain
        _brain = Brain()
    return _brain


def pnl_book():
    """Daily profit records (cached in pnl_cache.json.gz) for exact self-correlation checks."""
    global _book
    if _book is None:
        from pnl import PnlBook
        _book = PnlBook(brain)
    return _book


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path, columns, rows):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def load_templates():
    """Expression shapes from templates.txt plus research ideas from ideas.txt."""
    out = []
    for path in (TEMPLATES, IDEAS):
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            out += [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return list(dict.fromkeys(out))


def load_fields(cfg):
    """List of (field id, dataset id, type) the generator may pick for {field}/{field2}/{vfield}."""
    if cfg["fields"]:
        return [(f, "custom", "MATRIX") for f in cfg["fields"]]
    rows = load_csv(FIELDS)
    if not rows:
        sys.exit("No fields to use. Run `python alpha_tool.py fields <dataset id>` "
                 "or list field names under \"fields\" in config.json.")
    excluded = {e.split(":", 1)[1] for e in cfg.get("exclude", []) if e.startswith("field:")}
    excluded_datasets = set(cfg.get("exclude_datasets_from_blanks", []))
    return [(r["id"], r["dataset"], r["type"]) for r in rows
            if r["type"] in ("MATRIX", "VECTOR") and r["id"] not in excluded
            and r["dataset"] not in excluded_datasets
            and float(r["coverage"] or 0) >= cfg["min_field_coverage"]]


def known_names():
    """(all field ids, all operator names), for rejecting expressions BRAIN would not understand."""
    fields = {r["id"].lower() for r in load_csv(FIELDS)}
    operators = {o["name"] for o in json.loads(OPERATORS.read_text(encoding="utf-8"))} if OPERATORS.exists() else None
    return fields, operators


def report_invalid_ideas(templates):
    fields, operators = known_names()
    if not operators:
        return
    from learner import invalid_names, placeholders
    for t in templates:
        if not placeholders(t) - {"window", "group"}:
            bad = invalid_names(t.format(window=66, group="industry"), fields, operators)
            if bad:
                log(f"IDEA SKIPPED (unknown names {sorted(bad)}): {t}")


def knowledge(cfg):
    know = Knowledge(load_csv(RESULTS), load_csv(SUBMISSIONS), cfg, load_csv(CHECKS), load_csv(SUBMITTED))
    # Exact correlations from daily profits (fetched lazily, only when a plan asks for them).
    subs = [s["id"] for s in load_csv(SUBMITTED)]
    know.corr_to_submitted = lambda alpha_id: pnl_book().max_corr(alpha_id, subs)[0]
    know.predict_combo_corr = lambda parts: (lambda blend: pnl_book().series_max_corr(blend, subs)[0]
                                             if blend else None)(pnl_book().blend_signed(parts))
    return know


def build_basket(candidates, cfg):
    """Strongest first, keep an alpha only if it is below `basket_corr_limit` (0.65) with every alpha
    already kept and passes the exact check against all your submissions: everything in the basket
    can be submitted together, in any order. Returns (basket, backups, now_failing)."""
    book, limit = pnl_book(), cfg.get("basket_corr_limit", 0.65)
    basket, backups, failing = [], {}, {}
    for r in candidates:  # shortlist order = best fitness first
        corr, closest, blocked = corr_screen(r["alpha_id"], cfg)
        if blocked:
            failing[r["alpha_id"]] = (corr, closest)
            continue
        clash = max(((book.corr(r["alpha_id"], b["alpha_id"]) or 0, b["alpha_id"]) for b in basket),
                    default=(0, ""))
        if clash[0] >= limit:
            backups[r["alpha_id"]] = clash
            continue
        basket.append({**r, "self_corr": round(corr, 4) if corr is not None else r["self_corr"]})
    return basket, backups, failing


def write_reports(cfg):
    """Rewrite insights.md and READY_TO_SUBMIT.csv: the basket of verified alphas that can all be
    submitted together (each below 0.7 with your submissions and below 0.65 with each other)."""
    know = knowledge(cfg)
    know.write_insights(INSIGHTS)
    before = {r["alpha_id"] for r in load_csv(READY)}
    ready, backups, failing = build_basket(know.shortlist(200, all_passes=True), cfg)
    now = datetime.now().isoformat(timespec="seconds")
    if failing:
        # A new submission made these too similar: record it so they leave the list for good.
        append_csv(CHECKS, CHECK_COLUMNS, [
            {"alpha_id": a, "checked_at": now, "verdict": "FAIL", "failed": "SELF_CORRELATION",
             "self_corr": round(c, 4), "corr_with": w,
             "detail": f"SELF_CORRELATION value={round(c, 4)} limit=0.7 with {w} (exact PnL check after a submission)"}
            for a, (c, w) in failing.items()])
    from notify import code, esc, telegram
    for r in ready:
        if r["alpha_id"] not in before:
            log(f"READY TO SUBMIT: {r['alpha_id']} added (sharpe={r['sharpe']} fitness={r['fitness']} "
                f"self-corr={r['self_corr']}; basket now {len(ready)}) {r['expression']} {r['settings']}")
            risk = f"\n⚠️ RISKY: {esc(r['risky'])}" if r.get("risky") else ""
            telegram("\n".join([
                f"✅ NEW PASS: {code(r['alpha_id'])}{risk}",
                f"Sharpe {esc(r['sharpe'])} | Fitness {esc(r['fitness'])} | Self-corr {esc(r['self_corr'])}",
                f"Ready list: {len(ready)} - all can be submitted together, in any order."]))
    for alpha_id in before - {r["alpha_id"] for r in ready}:
        if alpha_id in know.submitted:
            reason = "submitted"
        elif alpha_id in failing:
            c, w = failing[alpha_id]
            reason = f"now too similar to your submission {w} ({c:.2f})"
        elif alpha_id in backups:
            c, w = backups[alpha_id]
            reason = f"replaced by {w}, a stronger alpha making the same bet ({c:.2f}); kept as a backup"
        elif know.checks.get(alpha_id, {}).get("verdict") == "PASS":
            reason = "being re-checked against a submission you just made"
        else:
            reason = "no longer passes verification"
        log(f"READY TO SUBMIT: {alpha_id} removed ({reason})")
        if reason != "submitted":
            telegram(f"⚠️ {code(alpha_id)} removed from the ready list: {esc(reason)}. Don't submit it now.")
    if backups and set(backups) != set(json.loads(load_notified().get("backups_logged", "[]"))):
        log(f"BASKET: {len(ready)} alphas submittable together; {len(backups)} backups making the same bet as "
            f"one of them: {', '.join(f'{a} ({w} {c:.2f})' for a, (c, w) in list(backups.items())[:8])}")
        state = load_notified()
        state["backups_logged"] = json.dumps(sorted(backups))
        save_notified(state)
    with READY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_id", "family", "best_in_family", "risky", "sharpe", "fitness", "turnover",
                         "self_corr", "yearly_sharpe", "verified_at", "expression", "settings"])
        for n, r in enumerate(ready, 1):
            check = know.checks[r["alpha_id"]]
            writer.writerow([r["alpha_id"], n, "yes", r["risky"], r["sharpe"], r["fitness"], r["turnover"],
                             r["self_corr"], check["yearly_sharpe"], check["checked_at"], r["expression"],
                             r["settings"]])
    pnl_book().save()
    return know


def sync_submitted():
    """Save your submitted alphas locally; new ideas are compared against them."""
    from brain import BrainError
    try:
        alphas = brain().submitted_alphas()
    except BrainError as e:
        log(f"Could not refresh submitted alphas, using the saved list: {e}")
        return
    known = {r["id"] for r in load_csv(SUBMITTED)}
    if known:
        for a in alphas:
            if a["id"] not in known:
                log(f"NEW SUBMISSION detected on BRAIN: {a['id']} {(a.get('regular') or {}).get('code', '')[:120]}")
    with SUBMITTED.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMITTED_COLUMNS)
        writer.writeheader()
        for a in alphas:
            s = a.get("settings") or {}
            writer.writerow({"id": a["id"], "code": (a.get("regular") or {}).get("code", ""),
                             "universe": s.get("universe"), "decay": s.get("decay"),
                             "neutralization": s.get("neutralization"),
                             "submitted_at": a.get("dateSubmitted") or "",
                             "sharpe": (a.get("is") or {}).get("sharpe", "")})


def corr_screen(alpha_id, cfg):
    """Exact self-correlation against every submitted alpha, from daily profits (matches BRAIN's
    number). Returns (highest corr, with which alpha, blocked). Not blocked at >= 0.7 only when BRAIN's
    exception might apply (Sharpe >= 10% above every submitted alpha it correlates with); BRAIN decides."""
    from learner import num
    subs = {s["id"]: num(s.get("sharpe"), 0) for s in load_csv(SUBMITTED)}
    book, limit = pnl_book(), cfg.get("corr_limit", 0.7)
    corrs = {o: book.corr(alpha_id, o) for o in subs if o != alpha_id}
    corrs = {o: c for o, c in corrs.items() if c is not None}
    if not corrs:
        return None, "", False
    closest = max(corrs, key=corrs.get)
    over = [o for o, c in corrs.items() if c >= limit]
    if not over:
        return corrs[closest], closest, False
    mine = abs(next((num(r["sharpe"]) for r in load_csv(RESULTS) if r["alpha_id"] == alpha_id), 0))
    exception = all(subs[o] > 0 and mine >= 1.1 * subs[o] for o in over)
    return corrs[closest], closest, not exception


def robustness(alpha_id, cfg):
    """Year-by-year in-sample Sharpe and our robustness problems with it."""
    rules = cfg["robustness"]
    sharpes = [y["sharpe"] for y in brain().yearly_stats(alpha_id)
               if y.get("stage", "IS") == "IS" and y.get("sharpe") is not None]
    problems = []
    losing = sum(s < 0 for s in sharpes)
    if losing > rules["max_losing_years"]:
        problems.append(f"{losing} losing years")
    if sharpes and min(sharpes) < rules["min_year_sharpe"]:
        problems.append(f"worst year Sharpe {min(sharpes)}")
    if sharpes and sharpes[-1] < rules["min_last_year_sharpe"]:
        problems.append(f"last year Sharpe {sharpes[-1]}")
    return sharpes, problems


def verify(alpha_id, cfg, brain_allowed=True):
    """Self-correlation (exact, from daily profits) and robustness first; BRAIN's own submission
    check only for alphas that pass both, since BRAIN's check is slow (often minutes).

    Returns a checks.csv row, None if BRAIN has not finished checking yet, or "WAIT" when it passed
    the fast checks but BRAIN may not be asked yet (brain_allowed=False: backing off after timeouts).
    """
    now = datetime.now().isoformat(timespec="seconds")
    corr, closest, blocked = corr_screen(alpha_id, cfg)
    sharpes, problems = robustness(alpha_id, cfg)
    local = round(corr, 4) if corr is not None else ""
    failed, details = [], []
    if blocked:
        failed.append("SELF_CORRELATION")
        details.append(f"SELF_CORRELATION value={local} limit={cfg.get('corr_limit', 0.7)} with {closest} "
                       "(exact PnL check; BRAIN check skipped)")
    if problems and (blocked or cfg.get("robustness_mode", "block") != "warn"):
        failed.append("NOT_ROBUST")
        details.append("NOT_ROBUST " + ", ".join(problems))
    if failed:
        return {"alpha_id": alpha_id, "checked_at": now, "verdict": "FAIL", "failed": ";".join(failed),
                "self_corr": local, "corr_with": closest, "yearly_sharpe": json.dumps(sharpes),
                "detail": "; ".join(details)}

    # Passed BRAIN before and only re-checked because you submitted something since: the exact
    # PnL check against every submission (incl. the new one) is enough, no need to wait for BRAIN.
    earlier = [c for c in load_csv(CHECKS) if c["alpha_id"] == alpha_id and c["verdict"] in ("PASS", "FAIL")]
    if earlier and earlier[-1]["verdict"] == "PASS" and corr is not None:
        return {"alpha_id": alpha_id, "checked_at": now, "verdict": "PASS", "failed": "",
                "self_corr": local, "corr_with": closest, "yearly_sharpe": json.dumps(sharpes),
                "detail": "re-confirmed by exact PnL check against all submissions"}

    if not brain_allowed:
        return "WAIT"
    checks, correlated = brain().check(alpha_id)
    if any(c.get("result") in ("PENDING", "ERROR") for c in checks):
        return None  # BRAIN hasn't finished, or its check errored: not verified, retry later
    self_check = next((c for c in checks if c["name"] == "SELF_CORRELATION"), None)
    if self_check is None or self_check.get("value") is None:
        return None  # no correlation value: not verified, retry later
    failed = [c["name"] for c in checks if c.get("result") == "FAIL"]
    details = [f"{c['name']} value={c.get('value')} limit={c.get('limit')}"
               for c in checks if c.get("result") == "FAIL"]
    self_corr = self_check.get("value")
    top = max(correlated, key=lambda a: a.get("correlation") or -1, default={})
    if problems:
        failed.append("NOT_ROBUST")
        details.append("NOT_ROBUST " + ", ".join(problems))
    return {"alpha_id": alpha_id, "checked_at": now,
            "verdict": "FAIL" if failed else "PASS", "failed": ";".join(failed),
            "self_corr": self_corr, "corr_with": top.get("id", ""),
            "yearly_sharpe": json.dumps(sharpes), "detail": "; ".join(details)}


def verify_and_record(alpha_ids, cfg, sync_each=False, deadline=None, local_only=()):
    """Verify alphas, save each result to checks.csv, return {alpha_id: row}.

    sync_each: refresh your submissions and the ready list after every check, so a submission
    shows up within one check (about a minute) even when BRAIN is slow.
    """
    from brain import BrainBusy, BrainError
    results = {}
    import time
    for i, alpha_id in enumerate(alpha_ids, 1):
        if deadline and time.time() > deadline:
            log(f"  Verification time budget used up; {len(alpha_ids) - i + 1} alphas wait for the next round")
            break
        safe_heartbeat(cfg)
        if sync_each and i > 1:
            sync_submitted()
            write_reports(cfg)
        try:
            row = verify(alpha_id, cfg, brain_allowed=alpha_id not in local_only)
        except BrainBusy:
            row = None  # BRAIN took too long: treat like an errored check and retry later
        except BrainError as e:
            log(f"  [{i}/{len(alpha_ids)}] {alpha_id}: could not verify yet ({str(e)[:80]})")
            continue
        if row == "WAIT":
            continue  # passed the fast checks; BRAIN's check comes after the back-off
            # Unverified until a complete check comes back: keep it off the ready list meanwhile.
            append_csv(CHECKS, CHECK_COLUMNS, [{"alpha_id": alpha_id, "verdict": "RETRY",
                                                "checked_at": datetime.now().isoformat(timespec="seconds"),
                                                "detail": "BRAIN check errored or unfinished"}])
            log(f"  [{i}/{len(alpha_ids)}] {alpha_id}: BRAIN check errored or unfinished - "
                f"off the ready list until a complete check passes")
            continue
        append_csv(CHECKS, CHECK_COLUMNS, [row])
        results[alpha_id] = row
        why = f"  {row['detail']}" if row["detail"] else ""
        log(f"  [{i}/{len(alpha_ids)}] VERIFY {alpha_id}: {row['verdict']}  self-corr={row['self_corr']}"
              f"  yearly Sharpe={row['yearly_sharpe']}{why}")
        if row["verdict"] == "PASS":
            write_reports(cfg)  # show it in READY_TO_SUBMIT.csv straight away
    if results:
        write_reports(cfg)
    pnl_book().save()
    return results


def verify_pending(cfg, sync_each=False):
    """Verify every in-sample pass that has not been verified yet."""
    know = knowledge(cfg)
    pending = list(dict.fromkeys(r["alpha_id"] for r in know.sims  # an ID can appear in several rows
                                 if r["passed"] == "Pending" and r["alpha_id"] not in know.submitted))
    # Back off on alphas whose BRAIN check keeps erroring: 30 min after the first failed check,
    # doubling up to 8 h. On 2026-10-07, 339 failed re-checks of 40 alphas took ~3 of 12 hours.
    streak, last_try = {}, {}
    for c in load_csv(CHECKS):
        streak[c["alpha_id"]] = streak.get(c["alpha_id"], 0) + 1 if c["verdict"] == "RETRY" else 0
        last_try[c["alpha_id"]] = c["checked_at"]
    now = datetime.now()

    def due(alpha_id):
        n = streak.get(alpha_id, 0)
        if not n or not last_try.get(alpha_id):
            return True
        wait = min(cfg.get("retry_check_minutes", 30) * 2 ** (n - 1), 8 * 60)
        try:
            return (now - datetime.fromisoformat(last_try[alpha_id])).total_seconds() >= wait * 60
        except ValueError:
            return True

    # Backing off only delays BRAIN's slow check: the fast exact checks still run for everyone.
    waiting = {a for a in pending if not due(a)}
    if waiting:
        log(f"  ({len(waiting)} alphas whose BRAIN check timed out before get the fast checks only this round)")
    # Earlier passes waiting for a re-check after a submission go first (re-confirmed in seconds),
    # then the newest in-sample passes. A time budget keeps a slow BRAIN from eating the whole cycle.
    passed_before = {c["alpha_id"] for c in load_csv(CHECKS) if c["verdict"] == "PASS"}
    order = {a: i for i, a in enumerate(pending)}
    pending.sort(key=lambda a: (a not in passed_before, -order[a]))
    if pending:
        import time
        log(f"Verifying {len(pending)} in-sample passes (self-correlation + robustness)")
        verify_and_record(pending, cfg, sync_each=sync_each, local_only=waiting,
                          deadline=time.time() + cfg.get("verify_budget_minutes", 15) * 60)


def done_keys():
    return {(r["expression"], r["settings"]) for r in load_csv(RESULTS)}


def summarize(alpha):
    stats = alpha.get("is") or {}
    failed = [c for c in stats.get("checks", []) if c.get("result") == "FAIL"]
    return {
        "alpha_id": alpha["id"],
        "passed": not failed,
        "sharpe": stats.get("sharpe"),
        "fitness": stats.get("fitness"),
        "turnover": stats.get("turnover"),
        "returns": stats.get("returns"),
        "drawdown": stats.get("drawdown"),
        "margin": stats.get("margin"),
        "long_count": stats.get("longCount"),
        "short_count": stats.get("shortCount"),
        "failed_checks": ";".join(c["name"] for c in failed),
        "check_details": "; ".join(f"{c['name']} value={c.get('value')} limit={c.get('limit')}" for c in failed),
    }


def simulate_jobs(jobs, cfg, cycle):
    """Simulate jobs concurrently, appending each result to results.csv as soon as it finishes."""
    from brain import BrainBusy, BrainError
    client = brain()
    lock = threading.Lock()

    def run_one(job):
        row = {**job, "cycle": cycle, "settings": json.dumps(job["settings"], sort_keys=True)}
        try:
            row.update(summarize(client.simulate(job["expression"], {**cfg["settings"], **job["settings"]})))
        except BrainBusy as e:
            # Not the alpha's fault: don't record it, so it is retried and not learned from.
            row["error"] = "BRAIN busy, will retry next cycle: " + str(e)[:200]
            return row
        except BrainError as e:
            row["error"] = str(e)[:300]
        with lock:
            append_csv(RESULTS, RESULT_COLUMNS, [row])
        return row

    with ThreadPoolExecutor(max_workers=cfg["concurrent_simulations"]) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            safe_heartbeat(cfg)  # long batches must not delay the hourly "alive" message
            if row.get("error"):
                status = "ERROR " + row["error"][:80]
            else:
                status = (f"SIM {row['alpha_id']} {'in-sample PASS' if row['passed'] else 'fail'}  "
                          f"sharpe={row['sharpe']}  fitness={row['fitness']}")
                if row["failed_checks"]:
                    status += f"  [{row['failed_checks']}]"
            fix = f" (repair: {row['repair']})" if row["repair"] else ""
            log(f"  [{i}/{len(jobs)}] {status}  {row['expression']} {row['settings']}{fix}")
            if row.get("passed") is True:
                # Verify straight away so a genuine pass reaches READY_TO_SUBMIT.csv within seconds.
                verify_and_record([row["alpha_id"]], cfg)


def print_jobs(title, jobs):
    print(f"{title}: {len(jobs)}")
    for j in jobs[:15]:
        fix = f"  <- {j['repair']} of {j['parent']}" if j["repair"] else ""
        print(f"  {j['expression']} {json.dumps(j['settings'], sort_keys=True)}{fix}")


def run_cycle(cfg, dry_run=False):
    """One learning cycle. Returns False when there is nothing left to try."""
    rows = load_csv(RESULTS)
    cycle = max((int(r["cycle"] or 0) for r in rows), default=0) + 1
    if not dry_run:
        sync_submitted()
        verify_pending(cfg)
    templates = load_templates()
    if not dry_run:
        report_invalid_ideas(templates)
    field_ids, operators = known_names()
    new = knowledge(cfg).propose(templates, load_fields(cfg), done_keys(), cfg["new_alphas_per_cycle"],
                                 operators=operators, known_fields=field_ids)
    if dry_run:
        print_jobs(f"Cycle {cycle} new ideas", new)
        print_jobs("Repairs waiting from earlier cycles", knowledge(cfg).plan_repairs(done_keys(), cfg["repairs_per_cycle"], load_fields(cfg)))
        return True

    # Repairs waiting from earlier cycles go first: correlation repairs are the closest to a pass.
    k = knowledge(cfg)
    early = k.plan_repairs(done_keys(), cfg["repairs_per_cycle"], load_fields(cfg))
    log(f"=== Cycle {cycle}: simulating {len(early)} repairs waiting from earlier cycles"
        f" ({getattr(k, 'combo_skipped', 0)} combinations skipped: predicted too similar to your submissions)")
    if early:
        simulate_jobs(early, cfg, cycle)
    explore = [j for j in new if j["repair"] == "explore"]
    log(f"=== Cycle {cycle}: simulating {len(new)} new ideas ({len(explore)} on never-tried fields: "
        f"{', '.join(sorted({j['dataset'] for j in explore})) or 'none left'})")
    if new:
        simulate_jobs(new, cfg, cycle)
    verify_pending(cfg)
    repairs = knowledge(cfg).plan_repairs(done_keys(), cfg["repairs_per_cycle"], load_fields(cfg))
    log(f"=== Cycle {cycle}: simulating {len(repairs)} repairs of this cycle's results")
    if repairs:
        simulate_jobs(repairs, cfg, cycle)
    verify_pending(cfg)
    repairs = early + repairs

    know = write_reports(cfg)
    exploration_report(know, cfg)
    this = [r for r in know.sims if r["cycle"] == str(cycle)]
    passed = sum(r["passed"] == "True" for r in this)
    log(f"=== Cycle {cycle} done: {passed}/{len(this)} verified passes. Feedback written to {INSIGHTS.name}")
    return bool(new or repairs)


SWEEP_STATE = HERE / "sweep_state.json"
SWEEP_UNIVERSES = ["TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "TOPSP500"]
SWEEP_NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]


def load_sweep_state():
    try:
        return json.loads(SWEEP_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"current": None, "done": []}


def save_sweep_state(state):
    SWEEP_STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def sweep_family(know, seed_id):
    """Every alpha descended from a sweep seed: its grid versions and all their repairs."""
    children = {}
    for r in know.sims:
        children.setdefault(r["parent"], []).append(r)
    # BRAIN returns the same alpha ID for an identical simulation, so a row can be its own parent:
    # track visited IDs to avoid going round in circles.
    family, todo, seen = [], list(children.get(seed_id, [])), {seed_id}
    while todo:
        r = todo.pop()
        if r["alpha_id"] in seen:
            continue
        seen.add(r["alpha_id"])
        family.append(r)
        todo += children.get(r["alpha_id"], [])
    return family


def choose_seed(know, cfg, state):
    """Strongest unswept idea: |Sharpe| >= 1, not built on data you've already submitted."""
    from learner import fields_in, num, one_sided
    swept = set(state["done"]) | {state["current"]["seed"]} if state.get("current") else set(state["done"])
    seen_shapes = set()
    for r in sorted(know.sims, key=lambda r: abs(num(r["sharpe"])), reverse=True):
        if (abs(num(r["sharpe"])) < cfg.get("sweep_seed_min_sharpe", 1.0) or r["alpha_id"] in swept
                or r["passed"] == "True" or r["repair"].startswith("sweep") or one_sided(r)
                or know.burned(r["expression"])
                or (r["window"] and int(r["window"]) not in cfg["search"]["windows"])  # retired short windows
                or num(know.checks.get(r["alpha_id"], {}).get("self_corr"), 0) >= 0.8
                or fields_in(r["expression"]) & know.submitted_fields):
            continue
        shape = (r["template"], r["field"])
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        expr = r["expression"] if num(r["sharpe"]) > 0 else f"-({r['expression']})"
        return {"seed": r["alpha_id"], "expression": expr, "template": r["template"],
                "field": r["field"], "dataset": r["dataset"],
                "decay": json.loads(r["settings"] or "{}").get("decay", cfg["settings"]["decay"])}
    return None


def run_sweep_cycle(cfg):
    """One idea, all universes x neutralisations, then repairs from each version's errors.
    Moves to the next idea once it has `sweep_target_passes` passes or used `sweep_max_sims`."""
    sync_submitted()
    verify_pending(cfg)
    state = load_sweep_state()
    know = knowledge(cfg)
    cur = state.get("current")
    if cur:
        family = sweep_family(know, cur["seed"])
        passes = [r["alpha_id"] for r in family if r["passed"] == "True"]
        if len(passes) >= cfg.get("sweep_target_passes", 5) or len(family) >= cfg.get("sweep_max_sims", 60):
            log(f"SWEEP DONE {cur['seed']}: {len(passes)} verified passes from {len(family)} simulations "
                f"{passes}  idea: {cur['expression'][:120]}")
            state["done"].append(cur["seed"])
            cur = state["current"] = None
    if not cur:
        cur = state["current"] = choose_seed(know, cfg, state)
        if not cur:
            log("SWEEP: no unswept idea with |Sharpe| >= 1 left; running a normal cycle to find more")
            save_sweep_state(state)
            return run_cycle(cfg)
        log(f"SWEEP START {cur['seed']}: testing across {len(SWEEP_UNIVERSES)} universes x "
            f"{len(SWEEP_NEUTRALIZATIONS)} neutralisations  idea: {cur['expression'][:150]}")
    save_sweep_state(state)

    cycle = max((int(r["cycle"] or 0) for r in load_csv(RESULTS)), default=0) + 1
    done = done_keys()
    grid = []
    for universe in SWEEP_UNIVERSES:
        for neut in SWEEP_NEUTRALIZATIONS:
            settings = {"decay": cur["decay"], "neutralization": neut, "universe": universe}
            if (cur["expression"], json.dumps(settings, sort_keys=True)) not in done:
                grid.append({"template": cur["template"], "field": cur["field"], "dataset": cur["dataset"],
                             "window": "", "group": "", "expression": cur["expression"],
                             "settings": settings, "repair": "sweep", "parent": cur["seed"], "depth": 1})
    if grid:
        log(f"=== Sweep {cur['seed']}: simulating {len(grid)} universe x neutralisation versions")
        simulate_jobs(grid, cfg, cycle)
        verify_pending(cfg)

    # Use each version's errors to tweak it (decay, truncation, group, subtraction, ...).
    know = knowledge(cfg)
    focus = {r["alpha_id"] for r in sweep_family(know, cur["seed"])}
    repairs = know.plan_repairs(done_keys(), cfg.get("sweep_repairs_per_round", 15), load_fields(cfg), focus=focus)
    log(f"=== Sweep {cur['seed']}: simulating {len(repairs)} tweaks based on the versions' errors")
    if repairs:
        simulate_jobs(repairs, cfg, cycle)
        verify_pending(cfg)
    know = write_reports(cfg)
    family = sweep_family(know, cur["seed"])
    passes = [r["alpha_id"] for r in family if r["passed"] == "True"]
    log(f"=== Sweep {cur['seed']}: {len(passes)} verified passes so far from {len(family)} simulations {passes}")
    if not repairs and not grid:
        state = load_sweep_state()
        log(f"SWEEP DONE {cur['seed']}: nothing left to tweak; {len(passes)} passes from {len(family)} simulations")
        state["done"].append(cur["seed"])
        state["current"] = None
        save_sweep_state(state)
    return True


def cmd_fields(args):
    add_datasets(args.datasets)


def add_datasets(datasets):
    """Download the data fields of these datasets into fields.csv. Returns {dataset: field count}."""
    cfg = load_config()["settings"]
    existing = {r["id"]: r for r in load_csv(FIELDS)}
    counts = {}
    for dataset in datasets:
        fields = brain().data_fields(cfg["region"], cfg["universe"], cfg["delay"], dataset)
        for fd in fields:
            existing[fd["id"]] = {"id": fd["id"], "dataset": (fd.get("dataset") or {}).get("id", dataset),
                                  "type": fd.get("type"), "coverage": fd.get("coverage"),
                                  "alpha_count": fd.get("alphaCount"), "description": fd.get("description")}
        log(f"{dataset}: {len(fields)} fields")
        counts[dataset] = len(fields)
    with FIELDS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(existing.values())
    log(f"{FIELDS.name} now has {len(existing)} fields")
    return counts


def load_notified():
    try:
        return json.loads(NOTIFIED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_notified(state):
    NOTIFIED.write_text(json.dumps(state), encoding="utf-8")


def discover_datasets(cfg):
    """Once a day (`dataset_check_hours`): ask BRAIN which datasets your account can use and add any
    new ones to fields.csv, so new families get explored as soon as BRAIN unlocks them."""
    from brain import BrainError
    from notify import code, esc, telegram
    state = load_notified()
    last = state.get("datasets_checked_at")
    if last and (datetime.now() - datetime.fromisoformat(last)).total_seconds() < cfg.get("dataset_check_hours", 24) * 3600:
        return
    st = cfg["settings"]
    try:
        found, params = [], {"instrumentType": st["instrumentType"], "region": st["region"], "delay": st["delay"],
                             "universe": st["universe"], "limit": 50, "offset": 0}
        while True:
            r = brain().request("GET", "/data-sets", params=params)
            if r.status_code != 200:
                raise BrainError(f"data-sets {r.status_code}")
            data = r.json()
            found += data["results"]
            params["offset"] += params["limit"]
            if not data["results"] or params["offset"] >= data["count"]:
                break
        known = {r["dataset"] for r in load_csv(FIELDS)}
        new = [d["id"] for d in found if d["id"] not in known]
        if new:
            counts = add_datasets(new)
            log(f"NEW DATASETS from BRAIN: {counts} - their fields are now explored")
            telegram("🆕 New BRAIN data available to your account: "
                     + ", ".join(f"{code(d)} ({n} fields)" for d, n in counts.items())
                     + "\nThe bot has started exploring it.")
        else:
            log(f"Dataset check: no new datasets ({len(found)} available, all known)")
    except (BrainError, KeyError, ValueError) as e:
        log(f"Dataset check failed, will retry next run: {str(e)[:120]}")
        return
    state = load_notified()
    state["datasets_checked_at"] = datetime.now().isoformat(timespec="seconds")
    save_notified(state)


def exploration_report(know, cfg):
    """Log fields that became burned (worn out by self-correlation) since the last report."""
    state = load_notified()
    before = set(state.get("burned", []))
    newly = sorted(know.burned_fields - before)
    for f in newly:
        n, k = know.field_corr[f]
        log(f"BURNED FIELD: {f} - {k} of {n} checked alphas using it were too similar to your submissions; "
            "no longer used in new ideas, combinations or repairs")
    if newly or set(state.get("burned", [])) != know.burned_fields:
        state["burned"] = sorted(know.burned_fields)
        save_notified(state)


def cmd_cycle(args):
    run_cycle(load_config(), args.dry_run)


def cmd_auto(args):
    done = 0
    while args.cycles <= 0 or done < args.cycles:
        cfg = load_config()
        step = run_sweep_cycle if cfg.get("mode") == "sweep" else run_cycle
        if not step(cfg):
            log("Nothing left to try. Add templates, datasets or search options in config.json.")
            break
        done += 1
    cmd_best(args)


def heartbeat(cfg):
    """Every `heartbeat_hours` (default 3), send a Telegram 'still alive' message with a short status.
    If these stop arriving, the tool has stopped running."""
    from notify import code, esc, telegram
    try:
        state = json.loads(NOTIFIED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    now = datetime.now()
    hours = cfg.get("heartbeat_hours", 3)
    last = state.get("heartbeat_at")
    if last and (now - datetime.fromisoformat(last)).total_seconds() < hours * 3600:
        return
    know = knowledge(cfg)
    verified = {c["alpha_id"] for c in load_csv(CHECKS) if c["verdict"] == "PASS" and c["self_corr"]}
    ready = [r["alpha_id"] for r in load_csv(READY)]
    submitted = {s["id"] for s in load_csv(SUBMITTED)} & {r["alpha_id"] for r in know.sims}
    sweep = load_sweep_state().get("current") if cfg.get("mode") == "sweep" else None
    all_fields = {f for f, _, _ in load_fields(cfg)}
    working_on = f"sweep of {sweep['expression'][:60]}..." if sweep else f"{cfg.get('mode', 'normal')} cycles"
    lines = [f"\U0001F493 Alpha bot alive - {now:%H:%M}"]
    if last:
        lines.append(f"Last {hours}h: {len(know.sims) - state.get('sims', 0)} simulations, "
                     f"{len(verified) - state.get('verified', 0)} new verified passes")
    lines += [f"Ready to submit: {len(ready)} (all can be submitted together)"
              + (f" ({', '.join(code(a) for a in ready[:6])})" if ready else ""),
              f"Working on: {esc(working_on)}",
              f"Explored: {len(know.tried_fields & all_fields)} of {len(all_fields)} data fields; "
              f"{len(know.burned_fields)} worn-out fields skipped",
              f"Total submitted from the tool: {len(submitted)}"]
    if telegram("\n".join(lines)):
        state.update(heartbeat_at=now.isoformat(timespec="seconds"), sims=len(know.sims), verified=len(verified))
        NOTIFIED.write_text(json.dumps(state), encoding="utf-8")
        log("HEARTBEAT sent")


def safe_heartbeat(cfg):
    """Heartbeat from inside long loops; a Telegram or file problem must never stop the work."""
    try:
        heartbeat(cfg)
    except Exception as e:
        log(f"Heartbeat failed: {str(e)[:120]}")


def watch_once(cfg):
    """Sync your submissions, re-check the ready list, then alphas waiting for a check."""
    from brain import BrainError
    safe_heartbeat(cfg)
    try:
        sync_submitted()
        # Drop what you just submitted, and re-test every ready alpha against all your submissions
        # with the exact PnL check (seconds, no slow BRAIN re-check needed).
        write_reports(cfg)
        # Then alphas waiting for a check (incl. ones hidden after a related submission).
        verify_pending(cfg, sync_each=True)
        write_reports(cfg)
    except BrainError as e:
        log(f"Watcher: BRAIN unavailable, will try again: {e}")


def cmd_watch(args):
    """Keep READY_TO_SUBMIT.csv current: drop what you submitted and anything now too correlated."""
    import time
    while True:
        watch_once(load_config())
        if args.once:
            return
        time.sleep(args.minutes * 60)


def cmd_run_for(args):
    """For GitHub Actions: watcher duties, then learning cycles until about `minutes` have passed
    (a cycle that has started is always finished, so allow some extra time)."""
    import time
    end = time.time() + args.minutes * 60
    log(f"RUN: GitHub run started ({args.minutes} min)")
    discover_datasets(load_config())
    watch_once(load_config())
    while time.time() < end:
        cfg = load_config()
        (run_sweep_cycle if cfg.get("mode") == "sweep" else run_cycle)(cfg)
        watch_once(cfg)
    log("RUN: GitHub run finished; state saved")


def cmd_insights(args):
    sync_submitted()
    write_reports(load_config())
    print(INSIGHTS.read_text(encoding="utf-8"))


def cmd_best(args):
    cfg = load_config()
    sync_submitted()
    shortlist = knowledge(cfg).shortlist(200, all_passes=True)
    if not shortlist:
        log("No verified alphas waiting to be submitted yet.")
        return
    # Re-verify live: anything you submitted since the last check may have made these correlated.
    log(f"Re-checking {len(shortlist)} verified alphas against your current submissions...")
    fresh = verify_and_record([r["alpha_id"] for r in shortlist], cfg)
    safe = [r for r in shortlist if fresh.get(r["alpha_id"], {}).get("verdict") == "PASS"]
    write_reports(cfg)
    log(f"{len(safe)} alphas safe to submit right now (submit one, then run `best` again):")
    for r in safe:
        log(f"  {r['alpha_id']}  sharpe={r['sharpe']:>6}  fitness={r['fitness']:>6}  "
              f"self-corr={fresh[r['alpha_id']]['self_corr']}  {r['expression']} {r['settings']}")


def cmd_recheck(args):
    cfg = load_config()
    sync_submitted()
    if args.all:
        know = knowledge(cfg)
        ids = [r["alpha_id"] for r in know.sims
               if r["passed"] in ("True", "Pending") or r["alpha_id"] in know.checks]
        log(f"Re-verifying {len(ids)} alphas")
        verify_and_record(ids, cfg)
    else:
        verify_pending(cfg)
    write_reports(cfg)
    log(f"Verified alphas ready to submit are listed in {READY.name}")


def cmd_verdicts(args):
    rows = knowledge(load_config()).rows
    if args.cycle:
        rows = [r for r in rows if r["cycle"] == str(args.cycle)]
    rows = rows[-args.last:]
    counts = {"PASS": 0, "FAIL": 0, "PENDING": 0, "ERROR": 0}
    for r in rows:
        verdict = ("ERROR" if r["error"] else "PASS" if r["passed"] == "True"
                   else "PENDING" if r["passed"] == "Pending" else "FAIL")
        counts[verdict] += 1
        if r["error"]:
            reason = r["error"][:60]
        elif verdict == "PENDING":
            reason = "passed in-sample, waiting for correlation/robustness check"
        else:
            reason = r["check_details"] if set(r["failed_checks"].split(";")) & {"SELF_CORRELATION", "NOT_ROBUST"} \
                else r["failed_checks"].replace(";", ", ")
        fix = f"  (repair: {r['repair']} of {r['parent']})" if r["repair"] else ""
        print(f"{verdict:7}  {r['alpha_id'] or '-':9} sharpe={r['sharpe'] or '-':>6} fitness={r['fitness'] or '-':>6}  "
              f"{reason}\n         {r['expression']} {r['settings']}{fix}")
    print(f"\n{counts['PASS']} PASS (verified), {counts['FAIL']} FAIL, {counts['PENDING']} PENDING, "
          f"{counts['ERROR']} ERROR")


def cmd_submit(args):
    cfg = load_config()
    sync_submitted()
    check = verify_and_record([args.alpha_id], cfg).get(args.alpha_id)
    if not check or check["verdict"] != "PASS":
        log(f"Not submitting {args.alpha_id}: it does not pass verification right now.")
        return
    alpha = brain().submit(args.alpha_id)
    failed = [c for c in (alpha.get("is") or {}).get("checks", []) if c.get("result") == "FAIL"]
    log(f"{args.alpha_id}: status {alpha.get('status')}")
    for c in failed:
        log(f"  FAIL: {c['name']} value={c.get('value')} limit={c.get('limit')}")
    field = next((r["field"] for r in load_csv(RESULTS) if r["alpha_id"] == args.alpha_id), "")
    append_csv(SUBMISSIONS, SUBMISSION_COLUMNS, [{
        "alpha_id": args.alpha_id, "field": field, "status": alpha.get("status"),
        "failed_checks": ";".join(c["name"] for c in failed)}])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fields", help="add datasets' fields to fields.csv")
    f.add_argument("datasets", nargs="+", help="dataset ids, e.g. fundamental6 analyst4 news12")
    c = sub.add_parser("cycle", help="run one learning cycle")
    c.add_argument("--dry-run", action="store_true", help="only show what would be simulated")
    a = sub.add_parser("auto", help="run several learning cycles")
    a.add_argument("--cycles", type=int, default=5, help="0 = keep going until stopped")
    w = sub.add_parser("watch", help="keep READY_TO_SUBMIT.csv up to date in the background")
    w.add_argument("--minutes", type=int, default=10)
    w.add_argument("--once", action="store_true")
    rf = sub.add_parser("run-for", help="GitHub Actions: watcher duties + cycles for N minutes")
    rf.add_argument("--minutes", type=int, default=40)
    sub.add_parser("insights", help="rebuild and show insights.md")
    sub.add_parser("best", help="re-verify live and list alphas safe to submit")
    rc = sub.add_parser("recheck", help="verify in-sample passes (correlation + robustness)")
    rc.add_argument("--all", action="store_true", help="re-verify everything verified before too")
    v = sub.add_parser("verdicts", help="show PASS/FAIL for each simulated alpha")
    v.add_argument("--cycle", type=int, help="only this cycle")
    v.add_argument("--last", type=int, default=50, help="how many recent results to show")
    s = sub.add_parser("submit", help="submit one alpha by id")
    s.add_argument("alpha_id")
    args = p.parse_args()
    commands = {"fields": cmd_fields, "cycle": cmd_cycle, "auto": cmd_auto, "watch": cmd_watch,
                "run-for": cmd_run_for,
                "insights": cmd_insights,
                "best": cmd_best, "recheck": cmd_recheck, "verdicts": cmd_verdicts, "submit": cmd_submit}
    commands[args.command](args)


if __name__ == "__main__":
    main()
