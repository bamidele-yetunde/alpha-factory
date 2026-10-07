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
SUBMITTED_COLUMNS = ["id", "code", "universe", "decay", "neutralization", "submitted_at"]

ACTIVITY_LOG = HERE / "activity_log.txt"

_brain = None
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
    return Knowledge(load_csv(RESULTS), load_csv(SUBMISSIONS), cfg, load_csv(CHECKS), load_csv(SUBMITTED))


def write_reports(cfg):
    """Rewrite insights.md and READY_TO_SUBMIT.csv (verified alphas not yet submitted)."""
    know = knowledge(cfg)
    know.write_insights(INSIGHTS)
    before = {r["alpha_id"] for r in load_csv(READY)}
    ready = know.shortlist(200, all_passes=True)
    from notify import telegram
    for r in ready:
        if r["alpha_id"] not in before:
            log(f"READY TO SUBMIT: {r['alpha_id']} added (sharpe={r['sharpe']} fitness={r['fitness']} "
                f"self-corr={r['self_corr']}) {r['expression']} {r['settings']}")
            telegram("\n".join([
                f"✅ NEW PASS: {r['alpha_id']}",
                f"Sharpe {r['sharpe']} | Fitness {r['fitness']} | Self-corr {r['self_corr']} | Family {r['family']}",
                f"Submit on BRAIN: Alphas > Unsubmitted > search {r['alpha_id']}",
                "(Submit one per family at a time.)"]))
    for alpha_id in before - {r["alpha_id"] for r in ready}:
        if alpha_id in know.submitted:
            reason = "submitted"
        elif know.checks.get(alpha_id, {}).get("verdict") == "RETRY":
            reason = "BRAIN's check errored; hidden until a complete check passes"
        elif know.checks.get(alpha_id, {}).get("verdict") == "PASS":
            reason = (f"hidden for {cfg.get('submission_lag_minutes', 45)} min because you submitted a related "
                      "alpha and BRAIN is slow to count it; re-checked after that")
        else:
            reason = "no longer passes verification"
        log(f"READY TO SUBMIT: {alpha_id} removed ({reason})")
        if reason != "submitted":
            telegram(f"⚠️ {alpha_id} removed from the ready list: {reason}. Don't submit it now.")
    with READY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_id", "family", "best_in_family", "sharpe", "fitness", "turnover", "self_corr",
                         "yearly_sharpe", "verified_at", "expression", "settings"])
        for r in sorted(ready, key=lambda r: (r["family"], not r["best_in_family"])):
            check = know.checks[r["alpha_id"]]
            writer.writerow([r["alpha_id"], r["family"], "yes" if r["best_in_family"] else "no",
                             r["sharpe"], r["fitness"], r["turnover"], check["self_corr"],
                             check["yearly_sharpe"], check["checked_at"], r["expression"], r["settings"]])
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
                             "submitted_at": a.get("dateSubmitted") or ""})


def verify(alpha_id, cfg):
    """BRAIN's submission check (incl. self-correlation) plus a year-by-year robustness test.

    Returns a checks.csv row, or None if BRAIN has not finished checking yet.
    """
    checks, correlated = brain().check(alpha_id)
    if any(c.get("result") in ("PENDING", "ERROR") for c in checks):
        return None  # BRAIN hasn't finished, or its check errored: not verified, retry later
    self_check = next((c for c in checks if c["name"] == "SELF_CORRELATION"), None)
    if self_check is None or self_check.get("value") is None:
        return None  # no correlation value: not verified, retry later
    failed = [c["name"] for c in checks if c.get("result") == "FAIL"]
    details = [f"{c['name']} value={c.get('value')} limit={c.get('limit')}"
               for c in checks if c.get("result") == "FAIL"]
    self_corr = next((c.get("value") for c in checks if c["name"] == "SELF_CORRELATION"), None)
    top = max(correlated, key=lambda a: a.get("correlation") or -1, default={})

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
    if problems:
        failed.append("NOT_ROBUST")
        details.append("NOT_ROBUST " + ", ".join(problems))

    return {"alpha_id": alpha_id, "checked_at": datetime.now().isoformat(timespec="seconds"),
            "verdict": "FAIL" if failed else "PASS", "failed": ";".join(failed),
            "self_corr": self_corr, "corr_with": top.get("id", ""),
            "yearly_sharpe": json.dumps(sharpes), "detail": "; ".join(details)}


def verify_and_record(alpha_ids, cfg, sync_each=False):
    """Verify alphas, save each result to checks.csv, return {alpha_id: row}.

    sync_each: refresh your submissions and the ready list after every check, so a submission
    shows up within one check (about a minute) even when BRAIN is slow.
    """
    from brain import BrainBusy, BrainError
    results = {}
    for i, alpha_id in enumerate(alpha_ids, 1):
        if sync_each and i > 1:
            sync_submitted()
            write_reports(cfg)
        try:
            row = verify(alpha_id, cfg)
        except BrainBusy:
            row = None  # BRAIN took too long: treat like an errored check and retry later
        except BrainError as e:
            log(f"  [{i}/{len(alpha_ids)}] {alpha_id}: could not verify yet ({str(e)[:80]})")
            continue
        if row is None:
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
    return results


def verify_pending(cfg, sync_each=False):
    """Verify every in-sample pass that has not been verified yet."""
    know = knowledge(cfg)
    pending = [r["alpha_id"] for r in know.sims
               if r["passed"] == "Pending" and r["alpha_id"] not in know.submitted]
    if pending:
        log(f"Verifying {len(pending)} in-sample passes (self-correlation + robustness)")
        verify_and_record(pending, cfg, sync_each=sync_each)


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
    early = knowledge(cfg).plan_repairs(done_keys(), cfg["repairs_per_cycle"], load_fields(cfg))
    log(f"=== Cycle {cycle}: simulating {len(early)} repairs waiting from earlier cycles")
    if early:
        simulate_jobs(early, cfg, cycle)
    log(f"=== Cycle {cycle}: simulating {len(new)} new ideas")
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
    cfg = load_config()["settings"]
    existing = {r["id"]: r for r in load_csv(FIELDS)}
    for dataset in args.datasets:
        fields = brain().data_fields(cfg["region"], cfg["universe"], cfg["delay"], dataset)
        for fd in fields:
            existing[fd["id"]] = {"id": fd["id"], "dataset": (fd.get("dataset") or {}).get("id", dataset),
                                  "type": fd.get("type"), "coverage": fd.get("coverage"),
                                  "alpha_count": fd.get("alphaCount"), "description": fd.get("description")}
        log(f"{dataset}: {len(fields)} fields")
    with FIELDS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(existing.values())
    log(f"{FIELDS.name} now has {len(existing)} fields")


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
    from notify import telegram
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
    working_on = f"sweep of {sweep['expression'][:60]}..." if sweep else f"{cfg.get('mode', 'normal')} cycles"
    lines = [f"\U0001F493 Alpha bot alive - {now:%H:%M}"]
    if last:
        lines.append(f"Last {hours}h: {len(know.sims) - state.get('sims', 0)} simulations, "
                     f"{len(verified) - state.get('verified', 0)} new verified passes")
    lines += [f"Ready to submit: {len(ready)}" + (f" ({', '.join(ready[:5])})" if ready else ""),
              f"Working on: {working_on}",
              f"Total submitted from the tool: {len(submitted)}"]
    if telegram("\n".join(lines)):
        state.update(heartbeat_at=now.isoformat(timespec="seconds"), sims=len(know.sims), verified=len(verified))
        NOTIFIED.write_text(json.dumps(state), encoding="utf-8")
        log("HEARTBEAT sent")


def watch_once(cfg):
    """Sync your submissions, re-check the ready list, then alphas waiting for a check."""
    from brain import BrainError
    heartbeat(cfg)
    try:
        sync_submitted()
        write_reports(cfg)  # drop what you just submitted straight away
        ready = [r["alpha_id"] for r in knowledge(cfg).shortlist(200, all_passes=True)]
        log(f"Watcher: re-checking {len(ready)} ready alphas")
        verify_and_record(ready, cfg, sync_each=True)
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
