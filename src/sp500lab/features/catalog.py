"""What every feature means, as data rather than as a docstring.

The family modules explain *why* a feature earns its place; this file makes the same
information machine-readable, so a report can describe the feature layer without anybody
opening the code. That is the whole reason it exists: the reports are meant to be the
documentation for somebody who does not want to read Python, and a report that lists 75
column names and no explanations is a directory listing.

Four things per feature, and each one answers a question a reader will actually have:

    family      where it comes from, so the list groups into something readable
    what        one sentence. What is this number
    reading     which end is "good", or that nobody knows - the sign is a MODELLING
                choice and this file refuses to make it. `accruals` is stored raw and
                Sloan's finding is that low is good; a feature file that had already
                negated it would be smuggling a hypothesis into the data layer.
    source      which table or computation it came from, for anyone who does want to check

`FEATURE_DOCS` is asserted complete against the built panel in
`tests/test_features.py::test_every_feature_is_documented`. A feature added without an
entry fails the suite, which is the only way a catalogue like this stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The families, in the order reports should present them: price first because it needs
#: nothing but the panel, macro last because it does not discriminate between stocks.
FAMILIES = ("Momentum & trend", "Risk & tail", "Liquidity & size", "Index membership",
            "Dividends", "Valuation", "Profitability & quality", "Growth",
            "Filing behaviour", "Macro", "Market state")

#: How each family behaves as a group, in a sentence a report can print above its table.
FAMILY_NOTES = {
    "Momentum & trend": (
        "Past returns, measured several ways. The disagreements between them are the "
        "point: 12-1 momentum and 1-month reversal predict in opposite directions, and "
        "`info_discreteness` is about HOW a return arrived rather than how large it was."),
    "Risk & tail": (
        "Volatility, beta, and the shape of the tail. The low-risk anomaly says the "
        "safest names have historically been the best risk-adjusted buys, which no risk "
        "model predicts and which is the single most robust effect in this repository's "
        "own results."),
    "Liquidity & size": (
        "What it costs to get out, and how large the company is. `half_spread_bp` is the "
        "cost model's own input, exposed so a strategy can decline to trade what it "
        "cannot afford to trade."),
    "Index membership": (
        "Reconstructed point-in-time (ADR-001), at MONTHLY granularity. Not computable "
        "from an ordinary price feed at all."),
    "Dividends": (
        "Built from discrete dated payments, not from an adjusted-close column - "
        "adjustment dissolves dividends into the price and the event stops existing. "
        "Available from 2007, unlike everything fundamental."),
    "Valuation": (
        "Price relative to a fundamental. Every one divides by market capitalisation or "
        "by assets, so a wrong share count shows up here first - see ADR-030."),
    "Profitability & quality": (
        "How much the business earns on what it owns, and how much of that is cash. "
        "Sloan (1996) and Novy-Marx (2013) are the references; the genetic algorithm "
        "rediscovered this family without being told either exists."),
    "Growth": (
        "Year-on-year change, measured against the value as ORIGINALLY REPORTED a year "
        "ago rather than as later restated - a growth rate computed against a number "
        "revised two years later is one nobody could have computed at the time."),
    "Filing behaviour": (
        "How a company files, which is information about the company. These three cannot "
        "be computed from a single-vintage fundamentals feed at all, because a single "
        "vintage IS the restatement."),
    "Macro": (
        "One number per date, broadcast across every security. ONLY series that are "
        "never revised: 7 of 18 FRED series in this project are restated after "
        "publication and using them at face value is a lookahead leak (ADR-011). Every "
        "one is lagged a session, because even an unrevised daily series is published "
        "after the session it describes."),
    "Market state": (
        "Computed from the same equal-weighted point-in-time cross-section the betas use, "
        "so the market definition is consistent across the whole layer and "
        "survivorship-free in the same way the universe is."),
}


@dataclass(frozen=True)
class FeatureDoc:
    family: str
    what: str
    reading: str
    source: str


def _d(family: str, what: str, reading: str, source: str) -> FeatureDoc:
    return FeatureDoc(family=family, what=what, reading=reading, source=source)


_PRICE = "adjusted close panel"
_VOL = "adjusted close panel (daily returns)"
_MKT = "equal-weighted point-in-time index"
_XBRL = "xbrl_facts, filed_date <= as-of"
_FRED = "fred_series (unrevised only), lagged one session"

FEATURE_DOCS: dict[str, FeatureDoc] = {
    # ---------------------------------------------------- momentum & trend
    "mom_12_1": _d("Momentum & trend",
                   "Total return over twelve months, skipping the most recent one.",
                   "High is the classic buy signal (Jegadeesh & Titman). The skip is not "
                   "decoration: the last month reverses and including it dilutes the "
                   "signal.", _PRICE),
    "mom_6_1": _d("Momentum & trend",
                  "The same idea over six months instead of twelve.",
                  "High. A shorter horizon, which reacts faster and is noisier.", _PRICE),
    "mom_1m": _d("Momentum & trend", "Return over the last month.",
                 "LOW has historically been the buy signal - one-month returns reverse. "
                 "Stored unnegated; `rev_1m` is the flipped version.", _PRICE),
    "rev_1m": _d("Momentum & trend", "Minus the last month's return.",
                 "High. The short-term reversal effect, pointing the usual way.", _PRICE),
    "ret_3m": _d("Momentum & trend", "Total return over three months.",
                 "Ambiguous - it sits between the reversal and momentum horizons, which "
                 "is exactly why it is worth having as a separate input.", _PRICE),
    "ret_12m": _d("Momentum & trend",
                  "Total return over twelve months, INCLUDING the last one.",
                  "High, weakly. The contrast with `mom_12_1` measures how much the skip "
                  "is worth.", _PRICE),
    "mom_on_12_1": _d("Momentum & trend",
                      "12-1 momentum computed from the OVERNIGHT legs only - every "
                      "close-to-open return over the year, compounded, skipping the "
                      "last month.",
                      "High. Lou, Polk & Skouras (2019): momentum profits accrue almost "
                      "entirely overnight, so this is the component of `mom_12_1` doing "
                      "the work.", "adjusted open + close panels"),
    "mom_id_12_1": _d("Momentum & trend",
                      "12-1 momentum computed from the INTRADAY legs only - every "
                      "open-to-close return over the year, compounded, skipping the "
                      "last month.",
                      "Ambiguous by design. LPS find intraday momentum reverses; this "
                      "exists mostly as the control for `mom_on_12_1`.",
                      "adjusted open + close panels"),
    "on_minus_id_252d": _d("Momentum & trend",
                           "Trailing-year overnight log return minus intraday log "
                           "return - the tug-of-war spread, no skip.",
                           "Contested. Positive means the name earns its keep while "
                           "the market is closed, which LPS link to institutional "
                           "demand at the open. Stored raw; the sign is a hypothesis, "
                           "not a fact.", "adjusted open + close panels"),
    "trend_200d": _d("Momentum & trend",
                     "Price divided by its own 200-day average, minus one.",
                     "High. A time-series trend measure rather than a cross-sectional "
                     "one - it asks whether the stock is above its own history, not "
                     "whether it beat its peers.", _PRICE),
    "high_52w_ratio": _d("Momentum & trend",
                         "Price as a fraction of its own 52-week high.",
                         "High. George & Hwang: nearness to the 52-week high predicts, "
                         "and it is not the same information as the return that produced "
                         "it.", _PRICE),
    "resid_mom_12_1": _d("Momentum & trend",
                         "12-1 momentum with market beta removed, standardised by the "
                         "volatility of its own residual.",
                         "High. Blitz, Huij & Martens: raw momentum is partly a bet on "
                         "beta and crashes when the market rebounds; the residual is the "
                         "part specific to the company.", f"{_PRICE} + {_MKT}"),
    "info_discreteness": _d("Momentum & trend",
                            "sign(12-month return) x (share of down days - share of up "
                            "days) over the same window.",
                            "LOW - strongly negative - means the return arrived "
                            "continuously in many small pieces. Da, Gurun & Warachka: "
                            "continuous information is under-reacted to, so the drift "
                            "afterwards is stronger.", _VOL),

    # ------------------------------------------------------------ risk & tail
    "vol_21d": _d("Risk & tail", "Annualised volatility of the last month's returns.",
                  "LOW. Fast-moving, so it reacts to a shock within weeks.", _VOL),
    "vol_126d": _d("Risk & tail", "Annualised volatility of the last six months.",
                   "LOW. The workhorse low-risk signal, and the best-performing single "
                   "feature in this repository's own scoreboard.", _VOL),
    "vol_of_vol_252d": _d("Risk & tail",
                          "Standard deviation of the six-month volatility over a year.",
                          "LOW, probably. Uncertainty about risk, which is a different "
                          "thing from risk.", _VOL),
    "skew_252d": _d("Risk & tail", "Skewness of daily returns over a year.",
                    "LOW. Positive skew is a lottery ticket, and lottery tickets are "
                    "systematically overpriced.", _VOL),
    "max_ret_21d": _d("Risk & tail", "The largest single daily return of the last month.",
                      "LOW. Bali, Cakici & Whitelaw's MAX effect - the same behavioural "
                      "preference as skew, measured more bluntly.", _VOL),
    "beta_252d": _d("Risk & tail",
                    "Trailing one-year beta against the equal-weighted index.",
                    "LOW. The low-beta anomaly: high-beta names have not been paid for "
                    "the risk they carry.", f"{_VOL} + {_MKT}"),
    "corr_mkt_252d": _d("Risk & tail",
                        "Correlation with the market over a year.",
                        "Ambiguous. A diversification signal rather than a return "
                        "signal - a low-correlation name changes what a PORTFOLIO does "
                        "more than what a position does.", f"{_VOL} + {_MKT}"),
    "idio_vol_252d": _d("Risk & tail",
                        "Annualised volatility of the residual after removing the market.",
                        "LOW. Ang et al. found idiosyncratic volatility is priced "
                        "NEGATIVELY, which no risk model predicts and which is why it is "
                        "interesting rather than obvious.", f"{_VOL} + {_MKT}"),

    # ------------------------------------------------------ liquidity & size
    "amihud_illiq": _d("Liquidity & size",
                       "log(1 + average |return| per billion dollars traded) over "
                       "63 sessions.",
                       "High is the documented premium - you are paid for holding what "
                       "is hard to sell. It is also exactly what the cost model charges "
                       "you for, which makes this the most honest feature in the file.",
                       f"{_VOL} + trailing dollar volume"),
    "log_dollar_volume": _d("Liquidity & size",
                            "log of trailing median daily dollar volume.",
                            "Ambiguous. A size proxy that needs no shares outstanding, so "
                            "it works back to 2000 where market cap starts in 2010.",
                            "panel dollar volume"),
    "half_spread_bp": _d("Liquidity & size",
                         "Estimated proportional half-spread, in basis points.",
                         "LOW is cheaper to trade. This is the cost model's own input "
                         "(ADR-020) and the weakest number in the chain - it is "
                         "estimated, because quote data costs more than this project's "
                         "budget.", "gold_half_spread"),
    "log_market_cap": _d("Liquidity & size", "log of market capitalisation in dollars.",
                         "LOW is the classic size premium, though it has been weak for "
                         "decades. Watch the coverage: it starts in 2010 and needs a "
                         "share count, which is where ADR-030's bad data lived.",
                         f"{_XBRL} share counts x as-traded price"),

    # ------------------------------------------------------ index membership
    "months_in_index": _d("Index membership",
                          "Consecutive month-ends this security has been in the index.",
                          "LOW means newly added. Index funds must buy a new constituent "
                          "regardless of price, which is a demand shock. Resets on "
                          "removal and re-entry, because the shock happens again.",
                          "sp500_membership_intervals"),
    "new_member": _d("Index membership",
                     "1 if the security has been in the index three months or less.",
                     "The flag version of the above, for a strategy that wants a "
                     "universe rather than a ranking.", "sp500_membership_intervals"),
    "log_tenure": _d("Index membership", "log(1 + months in the index).",
                     "The compressed version, so a 20-year member and a 19-year member "
                     "are not treated as meaningfully different.",
                     "sp500_membership_intervals"),

    # ----------------------------------------------------------- dividends
    "div_yield": _d("Dividends",
                    "Dividends paid over the last 372 days, divided by the price.",
                    "High is the classic value/income tilt. Over 2007-2021 it was "
                    "largely a bet on financials and energy, which is worth knowing "
                    "before reading too much into it.", "corporate_actions"),
    "div_growth_1y": _d("Dividends",
                        "This year's trailing dividend against last year's.",
                        "High. Raising a dividend commits future cash, which makes it "
                        "one of the few corporate statements that costs money to make.",
                        "corporate_actions"),
    "div_cut": _d("Dividends", "1 if the trailing dividend fell more than 5% year on year.",
                  "AVOID. A cut is an admission that management ran out of alternatives, "
                  "and it is one of the most reliably negative corporate events there is.",
                  "corporate_actions"),
    "pays_dividend": _d("Dividends", "1 if the security paid anything in the last year.",
                        "Neither. An eligibility flag, so a dividend strategy is not "
                        "silently ranking non-payers at zero.", "corporate_actions"),
    "div_due_1m": _d("Dividends",
                     "1 if the security's own payment cadence predicts an ex-dividend "
                     "within roughly the coming month; NaN until three payments have "
                     "established a cadence.",
                     "High. Hartzmark & Solomon's dividend-month premium: prices drift "
                     "up in months where a payment is expected. Predicted from past "
                     "ex-dates only, so it is the conservative version of what a real "
                     "trader would know.", "corporate_actions"),

    # ----------------------------------------------------------- valuation
    "book_to_market": _d("Valuation", "Shareholders' equity divided by market cap.",
                         "High is the textbook value signal. It was a poor bet over "
                         "2010-2021, and the genetic algorithm independently ranked it "
                         "NEGATIVELY while ranking earnings yield positively.", _XBRL),
    "earnings_yield": _d("Valuation", "Annual net income divided by market cap.",
                         "High. Cheap on earnings, which over this window behaved very "
                         "differently from cheap on assets.", _XBRL),
    "cf_yield": _d("Valuation", "Annual operating cash flow divided by market cap.",
                   "High. The same idea measured in cash, which is harder to manage than "
                   "earnings.", _XBRL),
    "buyback_yield": _d("Valuation",
                        "Cash spent repurchasing stock over a year, divided by market cap.",
                        "High is a shareholder-return signal, though it is also a "
                        "leverage signal - some of the largest buybacks were debt-funded.",
                        _XBRL),

    # ------------------------------------------------ profitability & quality
    "gross_profitability": _d("Profitability & quality",
                              "Gross profit divided by total assets.",
                              "High. Novy-Marx's 'other side of value': it predicts about "
                              "as well as book-to-market and is nearly uncorrelated with "
                              "it. Only ~338 companies tag GrossProfit, so read the "
                              "coverage.", _XBRL),
    "roe": _d("Profitability & quality",
              "Annual net income divided by shareholders' equity.",
              "High. The most familiar quality measure, and the one most distorted by "
              "leverage - a heavily indebted company can post a fine ROE.", _XBRL),
    "accruals": _d("Profitability & quality",
                   "(net income - operating cash flow) divided by assets.",
                   "LOW. Sloan (1996): the part of earnings that is not cash reverses, "
                   "while the cash part persists, and the market prices the two as if "
                   "they were the same. Stored RAW - the negation is the strategy's "
                   "choice, not the data layer's.", _XBRL),
    "cash_ratio": _d("Profitability & quality", "Cash and equivalents divided by assets.",
                     "Ambiguous. Balance-sheet safety, or capital nobody has found a use "
                     "for.", _XBRL),
    "current_ratio": _d("Profitability & quality",
                        "Current assets divided by current liabilities.",
                        "High is short-term solvency. A blunt instrument, and it means "
                        "very different things across industries.", _XBRL),
    "leverage": _d("Profitability & quality", "1 - (shareholders' equity / assets).",
                   "LOW is the safety leg of quality. Defined this way rather than as "
                   "liabilities/assets because only 489 of 649 companies tag "
                   "`Liabilities` and the identity holds for all of them.", _XBRL),
    "debt_to_assets": _d("Profitability & quality",
                         "Long-term non-current debt divided by assets.",
                         "LOW. The narrower leverage measure - actual borrowing rather "
                         "than every liability.", _XBRL),
    "capex_intensity": _d("Profitability & quality",
                          "Cash spent on property, plant and equipment, over assets.",
                          "LOW, weakly - heavy investment has historically preceded weak "
                          "returns. It is also mostly an industry classification in "
                          "disguise.", _XBRL),
    "rnd_intensity": _d("Profitability & quality",
                        "Research and development spending divided by assets.",
                        "High, weakly, and only within an industry. Fewer than 300 "
                        "companies report it at all, which is most of what this feature "
                        "measures.", _XBRL),

    # -------------------------------------------------------------- growth
    "asset_growth": _d("Growth", "Total assets against the same quarter a year earlier.",
                       "LOW. Cooper, Gulen & Schill: companies that grow their balance "
                       "sheet fast subsequently underperform, which is the opposite of "
                       "the intuitive reading.", _XBRL),
    "sales_growth": _d("Growth", "Annual revenue against the previous year's.",
                       "Ambiguous. Growth is priced, so the signal is in the surprise "
                       "rather than the level.", _XBRL),
    "earnings_growth": _d("Growth",
                          "Change in annual net income, scaled by the magnitude of the "
                          "prior year (so a swing through zero is meaningful).",
                          "High, weakly. Scaled rather than a ratio because a plain "
                          "ratio is meaningless when the denominator changes sign.",
                          _XBRL),
    "eps_surprise": _d("Growth",
                       "Standardised unexpected earnings: this quarter's diluted EPS "
                       "minus the same quarter a year ago, over the volatility of that "
                       "difference.",
                       "High. Post-earnings-announcement drift, the most robust anomaly "
                       "after momentum. NaN in fiscal Q4, which is normally filed only "
                       "inside the annual figure - reconstructing it is one sign error "
                       "away from a fake surprise every year.", _XBRL),
    "eps_change_yoy": _d("Growth", "The unstandardised version of the above, in dollars.",
                         "High. Kept alongside because the standardisation divides by a "
                         "volatility that is itself estimated from eight observations.",
                         _XBRL),

    # ---------------------------------------------------- filing behaviour
    "days_since_filing": _d("Filing behaviour",
                            "Days since this company last filed anything.",
                            "LOW means the fundamentals above are fresh. Everything in "
                            "the valuation and quality families is exactly this stale.",
                            _XBRL),
    "filing_lag_days": _d("Filing behaviour",
                          "Days between the period closing and the filing appearing.",
                          "LOW. A company drifting later than its own habit is a "
                          "documented red flag - late filings cluster with restatements "
                          "and with bad news.", _XBRL),
    "restatement_rate": _d("Filing behaviour",
                           "Share of this company's published facts that it has since "
                           "revised, counted only from revisions that had ALREADY "
                           "happened by the as-of date.",
                           "LOW is the accounting-quality reading. Treat with suspicion: "
                           "a company with more subsidiaries files more facts and has "
                           "more to revise, so part of this may be a size and complexity "
                           "tilt wearing a governance costume.", _XBRL),

    # ---------------------------------------------------------------- macro
    "vix": _d("Macro", "CBOE volatility index level.",
              "Context, not a ranking. High is fear, and fear has historically been a "
              "better time to buy than to sell.", _FRED),
    "vix_chg_63d": _d("Macro", "Change in VIX over roughly a quarter.",
                      "Context. Rising is the regime shift; the level alone does not say "
                      "whether it is arriving or leaving.", _FRED),
    "vix_relative": _d("Macro", "VIX divided by its own trailing one-year median.",
                       "Context, and the stationary version - a VIX of 20 meant "
                       "something different in 2008 and in 2017.", _FRED),
    "term_spread": _d("Macro", "10-year minus 2-year Treasury yield.",
                      "Context. Inversion is the most-watched recession signal there is, "
                      "with a lead time long enough to be almost untradable.", _FRED),
    "term_spread_chg_63d": _d("Macro", "Change in the 10y-2y spread over a quarter.",
                              "Context. Steepening and flattening are different regimes.",
                              _FRED),
    "term_spread_3m": _d("Macro", "10-year minus 3-month Treasury yield.",
                         "Context. The other standard inversion measure, and the one with "
                         "the better historical record.", _FRED),
    "term_spread_3m_chg_63d": _d("Macro", "Change in the 10y-3m spread over a quarter.",
                                 "Context.", _FRED),
    "ust10y": _d("Macro", "10-year Treasury yield.",
                 "Context. The discount rate every valuation ratio is implicitly "
                 "measured against.", _FRED),
    "ust10y_chg_63d": _d("Macro", "Change in the 10-year yield over a quarter.",
                         "Context. A rate shock hits long-duration equities hardest, "
                         "which is a cross-sectional consequence of a macro fact.",
                         _FRED),
    "fed_funds": _d("Macro", "Effective federal funds rate.",
                    "Context. Policy, at the short end.", _FRED),
    "fed_funds_chg_63d": _d("Macro", "Change in the fed funds rate over a quarter.",
                            "Context. The hiking/cutting cycle, which is what actually "
                            "moves risk appetite.", _FRED),
    "hy_spread": _d("Macro", "ICE BofA US high-yield option-adjusted spread.",
                    "Context. Credit turns before equity does. ONLY ~11% POPULATED - "
                    "FRED's keyless endpoint returns about three years of this licensed "
                    "series, so it is unusable before 2023.", _FRED),
    "hy_spread_chg_63d": _d("Macro", "Change in the high-yield spread over a quarter.",
                            "Context. Same coverage limit.", _FRED),
    "ig_spread": _d("Macro", "ICE BofA US corporate (investment grade) spread.",
                    "Context. Same ~11% coverage limit as the high-yield series.", _FRED),
    "ig_spread_chg_63d": _d("Macro", "Change in the investment-grade spread.",
                            "Context. Same coverage limit.", _FRED),
    "dollar_index": _d("Macro", "Broad trade-weighted US dollar index.",
                       "Context. A strong dollar is a headwind for multinational "
                       "earnings, which is a sector rotation in disguise.", _FRED),
    "dollar_index_chg_63d": _d("Macro", "Change in the dollar index over a quarter.",
                               "Context.", _FRED),
    "oil": _d("Macro", "WTI crude spot price.",
              "Context. An input cost for most of the index and a revenue line for a "
              "small part of it.", _FRED),
    "oil_chg_63d": _d("Macro", "Change in the oil price over a quarter.",
                      "Context.", _FRED),

    # --------------------------------------------------------- market state
    "mkt_trend_200d": _d("Market state",
                         "The equal-weighted index against its own 200-day average.",
                         "Context. Negative is the classic risk-off trigger, and it is "
                         "the switch both the `defensive_regime` strategy and the "
                         "evolved regime gate actually read.", _MKT),
    "mkt_drawdown": _d("Market state",
                       "How far the equal-weighted index is below its running peak.",
                       "Context. The one macro feature with no lag and no revision "
                       "question at all - it is arithmetic on prices.", _MKT),
    "mkt_vol_21d": _d("Market state", "Realised volatility of the index over a month.",
                      "Context. The fast measure of stress.", _MKT),
    "mkt_vol_252d": _d("Market state", "Realised volatility of the index over a year.",
                       "Context. The slow measure, and the denominator of the ratio "
                       "below.", _MKT),
    "mkt_vol_ratio": _d("Market state",
                        "One-month realised volatility over its own one-year level.",
                        "Context. High and rising is a different regime from high and "
                        "falling, and the ratio is what separates them.", _MKT),
    "mkt_breadth_200d": _d("Market state",
                           "Share of index members trading above their own 200-day "
                           "average.",
                           "Context. It falls before the index does, because the index is "
                           "dominated by its largest names long after the median name "
                           "has rolled over.", _MKT),
}


def describe(name: str) -> FeatureDoc:
    """The catalogue entry for a feature, or a loud placeholder if it has none."""
    return FEATURE_DOCS.get(name, FeatureDoc(
        family="Undocumented",
        what="No catalogue entry. Add one in features/catalog.py.",
        reading="unknown",
        source="unknown"))


def by_family() -> dict[str, list[str]]:
    """{family: [feature names]}, in the presentation order reports use."""
    out: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for name, doc in FEATURE_DOCS.items():
        out.setdefault(doc.family, []).append(name)
    return {k: v for k, v in out.items() if v}


def undocumented(names) -> list[str]:
    """Features present in a built panel with no catalogue entry. Should be empty."""
    return [n for n in names if n not in FEATURE_DOCS]
