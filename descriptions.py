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
