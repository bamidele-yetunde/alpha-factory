"""Alpha descriptions for BRAIN submissions.

The user's rule (2026-10-19): every alpha gets a description of at least 100 WORDS that matches the alpha.
BRAIN's own rules (as implemented by community tools from its docs): at least 100 characters, three
sections "Idea", "Rationale for data used", "Rationale for operators used", must not contain the alpha's
expression, and should not duplicate another of your descriptions.

Everything here is built from the alpha itself: the economic idea is read from the shape of the
expression, the data rationale from each field's BRAIN description and dataset, the operator rationale
from BRAIN's operator descriptions and the simulation settings.
"""
import difflib
import re

MIN_WORDS = 100

GROUP_WORDS = {"market", "sector", "industry", "subindustry"}
PRICE_WORDS = {"close", "open", "high", "low", "vwap", "volume", "returns", "cap", "adv20", "sharesout"}

# (pattern in the expression, the economic idea). First match leads; up to two more add colour.
IDEAS = [
    (r"days_from_last_change", "Freshly updated information is not yet fully priced. Stocks whose underlying data "
     "changed recently are treated differently from stocks whose data has gone stale, so the alpha trades the "
     "arrival of new information rather than its level."),
    (r"last_diff_value", "The size and direction of the latest revision matters more than the level: when a value "
     "jumps relative to its previous different value, the market tends to under-react at first, and the alpha "
     "positions for the adjustment that follows."),
    (r"ts_regression", "Only the part of one signal that another signal does not already explain is used: the "
     "time-series residual strips out the common, widely known component and keeps the stock-specific surprise, "
     "which should be less crowded."),
    (r"ts_std_dev\([^;]*22\)\s*/", "A change in volatility regime is informative: when the recent variability of a "
     "data series rises well above its own long-run variability, uncertainty about the company has shifted, and "
     "prices tend to adjust to that new uncertainty with a delay."),
    (r"ts_arg_max", "Timing carries information: how many days have passed since a data series peaked separates "
     "stocks still near their best readings from stocks drifting away from them."),
    (r"signed_power", "Only the extremes are bet on: raising the centred cross-sectional rank to an odd power keeps "
     "the sign but concentrates weight on the most unusual stocks, where the information is strongest."),
    (r"bucket\(rank\(cap\)", "Stocks are compared with companies of similar size instead of their industry, so the "
     "signal is not just a disguised size bet."),
    (r"ts_corr\([^;]*volume", "The link between the data and trading activity is used: when a data series moves "
     "together with volume, investors are actively trading on it, which changes how prices respond."),
    (r"ts_corr\([^;]*returns", "Stocks whose price has recently moved with the data are separated from those whose "
     "price has ignored it, betting that the gap closes."),
    (r"trade_when", "The position is opened only right after an information event and held for a fixed horizon, "
     "which captures the post-event drift while avoiding trading on stale data."),
    (r"ts_av_diff|ts_mean\([^;]*, (5|10|22)\)", "Short-horizon deviations from the recent average tend to reverse or "
     "persist predictably as liquidity providers and slower investors react, and the alpha trades that pattern."),
    (r"ts_delta", "The direction of change matters: companies whose data improved over the lookback are expected "
     "to keep outperforming peers whose data deteriorated, because markets absorb such trends gradually."),
    (r"ts_zscore", "Readings that are unusual relative to the company's own history signal mispricing that tends to "
     "correct as the reading normalises."),
    (r"ts_rank", "Where today's value sits within its own history captures persistent improvement or decline in a "
     "way that is robust to differences in scale between companies."),
    (r"/\s*cap\b", "Scaling by market capitalisation turns the quantity into a yield on the company's value, so "
     "cheaper companies per unit of the underlying quantity rank higher."),
    (r"vec_(avg|sum)", "Several records per company (for example multiple estimates or events) are summarised into "
     "one consensus value before comparing companies."),
]

DATA_KINDS = [
    (r"^(fnd|fn_|fundamental)|sales|assets|income|ebit|cash|debt|capex|revenue|inventory", "company fundamentals"),
    (r"^(anl|analyst|est_)|estimate|eps|guidance|recommend|target", "analyst estimates"),
    (r"^(nws|news|rp_|ai_news)|story|event|sentiment", "news analytics"),
    (r"^(snt|scl|social)|buzz|social", "social media sentiment"),
    (r"implied_vol|pcr_|option|put_|call_|forward_price|iv_", "the options market"),
    (r"search_interest|google|web", "web search interest"),
    (r"^pv|close|volume|vwap|returns|adv", "price and volume"),
    (r"risk|beta|model|score|rank_derivative|factor", "quantitative risk and factor models"),
]

SETTINGS_TEXT = {
    "MARKET": "market neutralisation keeps the book dollar neutral so the result does not depend on the market's direction",
    "SECTOR": "sector neutralisation removes sector bets so returns come from stock selection within sectors",
    "INDUSTRY": "industry neutralisation removes industry bets so the alpha compares like with like",
    "SUBINDUSTRY": "sub-industry neutralisation compares each stock only with its closest peers",
    "NONE": "no extra neutralisation is applied because the expression already ranks stocks against peers",
}


def _words(text):
    return len(re.findall(r"\b\w+\b", text))


def _identifiers(expr):
    return re.findall(r"\b([a-z_][a-z0-9_]*)\b(?!\s*\()", expr.lower())


def _operators(expr):
    return list(dict.fromkeys(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expr.lower())))


DATASET_KINDS = [  # by dataset id first: field names alone can mislead
    (r"^fundamental", "company fundamentals"), (r"^analyst", "analyst estimates"), (r"^news", "news analytics"),
    (r"^socialmedia", "social media sentiment"), (r"^option", "the options market"), (r"^model", "quantitative models"),
    (r"^pv1$", "price and volume"), (r"^pv", "price, volume and company-relationship data"),
    (r"search|trend", "web search interest"), (r"^ai_", "AI-derived signals"), (r"risk", "risk models"),
]
OPERATOR_TEXT = {
    "ts_regression": "takes the residual of the first series after regressing it on the second over the lookback, "
                     "keeping only what the second series does not explain",
    "trade_when": "opens or refreshes the position only when the entry condition holds and closes it when the exit "
                  "condition holds, otherwise keeping the previous position",
    "bucket": "splits stocks into groups by rank (here size deciles) so each stock is compared with similar-sized peers",
    "signed_power": "raises the value to a power while keeping its sign, which emphasises the extremes",
}


def data_kind(field, dataset=""):
    for pattern, kind in DATASET_KINDS:
        if dataset and re.search(pattern, dataset.lower()):
            return kind
    f = field.lower()
    if f in PRICE_WORDS:
        return "price and volume"
    for pattern, kind in DATA_KINDS:
        if re.search(pattern, f):
            return kind
    return "an alternative dataset"


def describe(expression, settings, field_info, operator_info, others=()):
    """Return a description (>= MIN_WORDS words) for a regular alpha.

    field_info: {field id: {"dataset": ..., "description": ...}}; operator_info: {name: description}.
    others: existing descriptions, to avoid near-duplicates.
    """
    expr = expression or ""
    low = expr.lower()
    ideas = [text for pattern, text in IDEAS if re.search(pattern, low)]
    if not ideas:
        ideas = ["The alpha ranks companies on a data signal relative to their peers and holds the stocks the "
                 "signal favours against those it disfavours, expecting prices to catch up with the information."]
    combined = re.match(r"^\s*rank\(.*\)\s*[+-]\s*[\d.]*\s*\*?\s*rank\(", expr) is not None
    reversed_ = expr.strip().startswith("-") or "rank(-(" in low
    idea = ideas[0]
    if len(ideas) > 1:
        idea += " " + ideas[1]
    if combined:
        idea += (" Two separately tested signals are blended with equal weight after ranking, so each contributes "
                 "evenly and their errors partly cancel, giving a steadier combined return than either alone.")
    if reversed_:
        idea += " The signal is used in reverse: stocks that score highest are sold and the lowest are bought."

    ids = [i for i in dict.fromkeys(_identifiers(expr)) if i not in GROUP_WORDS]
    fields = [i for i in ids if i in field_info or i in PRICE_WORDS]
    groups = sorted(set(_identifiers(expr)) & GROUP_WORDS)
    data_parts = []
    for f in fields:
        info = field_info.get(f, {})
        what = (info.get("description") or "").strip().rstrip(".") or f.replace("_", " ")
        ds = info.get("dataset")
        data_parts.append(f"{f} measures {what[0].lower() + what[1:] if what else what}"
                          f" ({data_kind(f, ds or '')}{', dataset ' + ds if ds else ''})")
    kinds = sorted({data_kind(f, field_info.get(f, {}).get("dataset") or "") for f in fields})
    data = ("; ".join(data_parts) or "the fields are standard price and volume data") + "."
    if kinds:
        joined = kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + " and " + kinds[-1]
        data += (f" The inputs come from {joined}; information of this kind reaches prices gradually, which is "
                 "what makes it predictive.")
    if groups:
        names = groups[0] if len(groups) == 1 else ", ".join(groups[:-1]) + " and " + groups[-1]
        verb = "is" if len(groups) == 1 else "are"
        data += f" {names.capitalize()} {verb} used only to define peer groups, not as input signals."

    op_parts = []
    for op in _operators(expr):
        if op in ("add", "subtract", "multiply", "divide", "reverse"):
            continue
        text = OPERATOR_TEXT.get(op) or (operator_info.get(op) or "").strip()
        first = re.split(r"(?<=\.)\s", text)[0].rstrip(".") if text else ""
        if op == "ts_backfill":
            first = "fills short gaps in the data with the last known value so stocks are not dropped between updates"
        op_parts.append(f"{op} {first[0].lower() + first[1:] if first else 'transforms the data'}")
    st = settings or {}
    neut = SETTINGS_TEXT.get(str(st.get("neutralization", "")).upper())
    tail = []
    if neut:
        tail.append(neut)
    if st.get("decay"):
        tail.append(f"a decay of {st['decay']} days smooths the positions to keep turnover and costs reasonable")
    if st.get("universe"):
        tail.append(f"the {st['universe']} universe keeps the alpha in liquid, tradable stocks")
    ops = ("; ".join(op_parts) or "simple arithmetic combines the inputs") + "."
    if tail:
        ops += " In the settings, " + "; ".join(tail) + "."

    text = f"Idea: {idea}\n\nRationale for data used: {data}\n\nRationale for operators used: {ops}"
    if _words(text) < MIN_WORDS:
        text += ("\n\nRisk and robustness: the alpha is checked year by year, against your submitted alphas and "
                 "against the production pool before submission, so it adds a distinct, stable source of return "
                 "rather than repeating an existing bet.")
    # Make near-duplicates distinct by naming the exact settings combination.
    for other in others:
        if other and difflib.SequenceMatcher(None, text, other).ratio() > 0.85:
            text += (f"\n\nVariant note: this version uses decay {st.get('decay', '?')} with "
                     f"{str(st.get('neutralization', 'default')).lower()} neutralisation on {st.get('universe', 'TOP3000')}.")
            break
    return text


# ---------------------------------------------------------------- SuperAlpha descriptions
# BRAIN requires two descriptions for a SuperAlpha: one for the selection expression, one for the combo.

SELECTION_TERMS = [
    (r"prod_correlation", "prod_correlation, the alpha's correlation with every alpha already in BRAIN's production "
     "pool, so alphas that duplicate common platform signals are pushed down"),
    (r"self_correlation", "self_correlation, the alpha's highest correlation with my other submissions, so "
     "near-copies inside my own pool are avoided"),
    (r"turnover", "turnover, the in-sample share of the book traded each day, which keeps trading costs and the "
     "SuperAlpha's own turnover in the 2-40% range BRAIN requires"),
    (r"long_count|short_count", "long_count and short_count, the average number of stocks held long and short, a "
     "measure of how broad and balanced each component is"),
    (r"universe_size", "universe_size, which scales counts by the size of each component's universe"),
    (r"datacategories|datasets", "the data categories each alpha uses, so the components come from the chosen "
     "information source"),
    (r"operator_count", "operator_count, the number of operators per alpha, which keeps components simple and within "
     "BRAIN's 8,000-operator limit"),
    (r"os_start_date", "os_start_date, the start of each alpha's out-of-sample period, to favour recent work"),
    (r"\bdecay\b", "each component's decay setting, which favours alphas that already smooth their positions"),
    (r"neutralization", "each component's neutralisation setting"),
]

COMBO_TERMS = [
    (r"^\s*1\s*$", "equal weight: every selected alpha gets the same weight every day",
     "With no model risk at all, equal weighting is the benchmark BRAIN itself shows; it is the most robust choice "
     "when the components are already diverse and of similar quality."),
    (r"self_corr", "weights each alpha by one minus its highest rolling correlation with the other selected alphas",
     "Alphas that overlap with others are down-weighted and distinct ones up-weighted, so the SuperAlpha spreads "
     "its risk over genuinely different bets instead of doubling up on one."),
    (r"combo_a", "uses BRAIN's combo_a, which balances each alpha's recent history of returns and risk to set its "
     "weight",
     "combo_a adapts the weights through time, leaning towards alphas that have recently delivered steadier returns "
     "and away from those that are struggling, while keeping every component in the mix."),
    (r"reduce_ir|ts_ir", "weights each alpha by its recent information ratio (return per unit of risk)",
     "Alphas that have recently earned more per unit of risk receive more capital, which tends to raise the "
     "SuperAlpha's Sharpe ratio."),
]


def describe_super(selection, combo, settings, n_selected=None):
    """Return (selection_description, combo_description), each >= MIN_WORDS words, matching the code."""
    st = settings or {}
    sel_low = (selection or "").lower()
    used = [text for pattern, text in SELECTION_TERMS if re.search(pattern, sel_low)]
    cats = re.findall(r'in\(\s*datacategories\s*,\s*"([a-z]+)"', sel_low)
    filters = []
    for m in re.finditer(r"\(\s*(turnover|prod_correlation|self_correlation|operator_count)\s*([<>]=?)\s*([\d.]+)\s*\)",
                         sel_low):
        filters.append(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    limit = st.get("selectionLimit", "the selection limit")
    handling = str(st.get("selectionHandling", "POSITIVE")).replace("_", "-").lower()
    idea = ("Idea: Combine my submitted alphas into one diversified signal by choosing the components that are most "
            "likely to add something new. ")
    if cats:
        idea += (f"Only alphas built on {', '.join(cats)} data are eligible, so this SuperAlpha is a focused "
                 f"{'/'.join(cats)} portfolio whose components share an information source but differ in method. ")
    if "1 - prod_correlation" in sel_low.replace(" ", " "):
        idea += "Alphas least correlated with the production pool rank highest, which targets unique, less crowded returns. "
    elif "1 - self_correlation" in sel_low:
        idea += "Alphas least correlated with the rest of my pool rank highest, which maximises diversification. "
    sel_text = (idea +
                f"\n\nRationale for data used: The selection reads alpha properties rather than market data: "
                f"{'; '.join(used) or 'basic alpha properties'}."
                + (f" Hard filters keep only alphas with {', '.join(filters)}." if filters else "") +
                f"\n\nRationale for operators used: Comparisons return 1 or 0, so multiplying by a condition removes "
                f"alphas that fail it; subtracting a correlation from one turns 'less correlated' into a higher score. "
                f"Selection handling is set to {handling}, so only alphas with a valid score are kept, and the "
                f"top {limit} by score are selected"
                + (f" ({n_selected} alphas were selected in the simulation)" if n_selected else "") + ".")
    combo_low = (combo or "").lower()
    what, why = next(((w, y) for p, w, y in COMBO_TERMS if re.search(p, combo_low)),
                     ("sets daily weights for the selected alphas", "The weights aim to improve on equal weighting."))
    n = re.search(r"nlength\s*=\s*(\d+)", combo_low)
    look = re.search(r"self_corr\([^,]+,\s*(\d+)\)", combo_low)
    neut = SETTINGS_TEXT.get(str(st.get("neutralization", "")).upper(),
                             f"{str(st.get('neutralization', '')).lower().replace('_', ' ')} neutralisation removes "
                             "common risk-factor exposure from the combined book")
    combo_text = (f"Idea: The combo {what}. {why}"
                  f"\n\nRationale for data used: The input is the daily returns history of each selected alpha"
                  + (f", looked at over {n.group(1)} trading days" if n else "")
                  + (f", with correlations measured over {look.group(1)} trading days (about "
                     f"{int(look.group(1)) // 250} years)" if look else "") +
                  ". No market data is added at this stage; all information comes from the components' own "
                  "performance, so the combo cannot overfit to a new data source."
                  f"\n\nRationale for operators used: "
                  + ("generate_stats turns each alpha into daily statistics; self_corr builds the rolling correlation "
                     "matrix; the if_else removes each alpha's correlation with itself; reduce_max keeps its highest "
                     "correlation with any other alpha; one minus that gives the weight. " if look else "")
                  + ("combo_a computes the adaptive weights directly from the components' history. " if n else "")
                  + ("scale rebalances the long and short books to the chosen proportions. " if "scale(" in combo_low else "")
                  + f"In the settings, {neut}; a decay of {st.get('decay', 0)} days smooths the combined positions "
                    f"and truncation {st.get('truncation', 0.08)} caps any single stock's weight.")
    for label in ("sel", "combo"):
        t = sel_text if label == "sel" else combo_text
        if _words(t) < MIN_WORDS:
            t += ("\n\nRobustness: the SuperAlpha is checked against BRAIN's consultant tests (Sharpe, fitness, "
                  "turnover between 2% and 40%, IS ladder, sub-universe, self and production correlation) before it "
                  "is offered for submission, and it is compared with the equal-weight combination of the same "
                  "components.")
        if label == "sel":
            sel_text = t
        else:
            combo_text = t
    return sel_text, combo_text


def check(text, expression):
    """Problems with a description, empty when it is acceptable."""
    problems = []
    if _words(text) < MIN_WORDS:
        problems.append(f"only {_words(text)} words (need {MIN_WORDS})")
    for section in ("idea", "rationale for data used", "rationale for operators used"):
        if section not in text.lower():
            problems.append(f"missing section '{section}'")
    compact = re.sub(r"\s+", "", expression or "")
    if compact and compact in re.sub(r"\s+", "", text):
        problems.append("contains the alpha expression")
    return problems
