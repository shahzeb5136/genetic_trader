# What else can we build

Fourteen candidate algorithm families, ranked by expected value per unit of effort and
risk. Written before the holdout is spent, deliberately — every family added here is one
more thing the single out-of-sample test will eventually have to absorb.

No code changes. This is the plan.

> **Status update, 2026-08-30 (ADR-036/037/038).** Three entries from this list are now
> built and measured: **#2 volatility capping** (`vol_managed` — the only new entrant to
> beat the index on ΔSharpe, +0.08), **#9 ensembling** (`ensemble_rank` — it lost to
> several of its own members, which is itself a finding about how correlated the twelve
> hypotheses are), and **#8 the shallow neural net** (`shallow_mlp` — research Sharpe
> 0.38, study deflates below the 0.95 bar, forward 0.33: mediocre both ways. Its first
> forward record printed 2.20, which was a stale-state leak the harness caught — see
> ADR-037's postscript and the §7 lesson it now joins). Two families this list did not
> anticipate were added alongside:
> the **calendar lab** (a daily leg engine for overnight/weekend/turn-of-month claims,
> TIMING.md) and the **overnight/dividend feature set** with a third GA search over it
> (`ga-night-1` — deflated 0.983 in research, decayed to 0.20 forward, the third GA
> winner in a row to decay). The warning in "How not to fool ourselves" below performed
> exactly as written: `selection_bar()` now counts 38 forward-tested candidates and the
> bar rose accordingly. #1 no-trade bands, #3 HRP, #4 sectors, #5 short interest and the
> walk-forward remain open, and the walk-forward remains the most important.
>
> **Status update, 2026-09-02 (ADR-041/042).** Two data additions change the inputs
> available to several entries without building any of them: daily **Fama-French factor
> returns** (`factors/fama_french_daily`, from 1926) and **24 more benchmark series** — the
> eleven GICS sector SPDRs, the VIX term structure, and a cross-asset regime set. #4's
> sector *rotation* half can now be expressed on ETFs before point-in-time constituent
> sectors exist; #10 and #11 have external series to validate against; and every family
> can be regressed on the factors to ask whether it is a factor bet in disguise. Every
> strategy also now passes a no-lookahead check (ADR-041), which is what any new family
> will be held to on arrival.
>
> **Status update, 2026-09-04 (ADR-048/049/050).** The genetic algorithm was rebuilt
> after its three winners went three for three on decay: the search now moves through
> **nine prior-signed feature families with at most three live**, is scored on the
> **worst quarter of twelve random sub-periods** net of pessimistic costs with a charge
> per rule, and hands on the **ensemble of its 30 best survivors** across seeds rather
> than its champion. That is #9 ensembling applied inside the search rather than across
> the roster, and it changes what #4 the walk-forward should re-run: the whole
> multi-seed search, evaluating each window's ensemble on the next. `ga-families-1` is
> the first search over it: quality plus low risk, 17.75%/yr at 1.15 in research, and
> **5.73%/yr at 0.46 against the index's 13.52%/0.82 forward — decayed, four for four**,
> though with the drawdown halved and the turnover check held. The walk-forward remains
> the most important open item, and the fourth decay is the argument for it.

---

## Read this first: three measurements that decide the ranking

I measured these on the current panel rather than assuming them, and two of them are not
what you would guess.

### 1. There are 177 independent observations, not 63,403

| unit | count |
|---|---:|
| tradable (stock, month) rows, 2007-04 → 2021-12 | **63,403** |
| month-ends in that window | **177** |
| median names per month | 348 |
| daily (stock, session) bars available | 2,053,601 |

The 63,403 looks like a machine-learning dataset. It is not, quite. Those rows share 177
dates, and stocks within a month are heavily correlated — when the market falls, almost
everything falls. **The effective sample for anything predicting the market is 177. The
effective sample for anything predicting the *cross-section* — which stock beats which,
within a month — is much larger, because that comparison nets out the common move.**

This single distinction determines what can work here:

- **Cross-sectional models** (rank stocks against each other): supported. This is where all
  fourteen candidates below should aim.
- **Time-series models** (predict the market's direction or level): not supported. 177
  observations and roughly three genuine regime episodes. Anything that appears to work
  here is fitting 2008, 2020 and the 2022 rate cycle.

### 2. Almost no row has a complete feature vector

| | rows |
|---|---:|
| tradable rows with **all 75** features present | **0** |
| tradable rows with all 71 (excluding the sparse credit spreads), 2010-07+ | 1,819 of 52,034 — **3.5%** |
| features that are ≥90% populated | **55 of 75** |
| tradable rows complete across those 55 | 39,247 of 52,034 — **75%** |

The blockers are sparsely-tagged XBRL fields: `rnd_intensity` 30%, `gross_profitability`
33%, `sales_growth` 49%, `debt_to_assets` 52%, `capex_intensity` 58%.

**This is the most practically important number in this document.** A standard
scikit-learn or PyTorch pipeline drops rows with missing values. Do that here and you train
on 3.5% of the data — and that 3.5% is systematically the largest, most thoroughly-audited
companies. It is a survivorship filter hiding inside `dropna()`.

It has a direct consequence for model choice, and it is specific to this dataset rather
than a generic preference: **gradient-boosted trees handle missingness natively** (they
learn which side of a split a NaN belongs on), **neural nets do not**. That is a real
argument for trees first, here, on measured grounds.

### 3. Costs are already 0.5%–4.2% a year

From the current scoreboard: turnover runs 130%–1,000%/yr and cost drag 0.5%–4.2%/yr.

That matters because it means **there is a guaranteed improvement available that requires
no forecasting at all.** Every candidate below that predicts something has to overcome the
177-observation problem. Reducing turnover does not.

---

## The ranking

| # | family | adds | effort | risk of fooling ourselves | priority |
|---|---|---|---|---|---|
| 1 | No-trade bands | — (cuts cost) | S | **very low** | **do first** |
| 2 | Volatility capping | — (cuts risk) | S | low | **do first** |
| 3 | Covariance-aware weighting (HRP) | — (cuts risk) | M | low | high |
| 4 | Point-in-time sectors | information | M | low | **high** |
| 5 | FINRA short interest | information | M | low | **high** |
| 6 | Strategy ensembling / stacking | — (combines) | S | medium | high |
| 7 | Gradient-boosted trees | capacity | M | medium | high |
| 8 | Shallow neural net | capacity | M | medium-high | medium |
| 9 | 10-K text similarity | information | **L** | low | medium |
| 10 | Uncertainty-aware position sizing | — (sizing) | M | low | medium |
| 11 | Hidden-Markov regime model | structure | M | **high** | medium-low |
| 12 | Genetic programming (expression trees) | capacity | M | **very high** | low |
| 13 | Sequence models (LSTM / Transformer) | capacity | L | **very high** | low |
| 14 | Reinforcement learning | capacity | **XL** | **very high** | not yet |

The ordering principle: **information beats capacity.** Adding a new column that nobody has
looked at is worth more than adding a model that looks harder at the columns we already
have — because capacity is what overfits, and 177 months does not support much of it.

---

## Tier 1 — Improvements that require no forecast

The highest-confidence work in this document. None of them predicts anything, so none of
them can be wrong about the future.

### 1. No-trade bands

Only trade a name when the gap between its target and current weight exceeds a band
(say 20% of the target weight). Everything inside the band drifts.

Strategies here pay 130%–1,000% annual turnover, much of it re-buying names they already
hold at a slightly different weight. At $100k with a $1 per-order minimum, commission is
effectively a flat fee per name traded, so a trade that moves a position from 2.0% to 2.1%
costs the same as one that opens it.

Expected: 30–60% turnover reduction for a small tracking error against the unbanded
strategy. On `lottery_averse` (492% turnover) that is plausibly 40–80bp/yr of recovered
return, with **no forecasting risk whatsoever**.

*Fits*: `Construction` in `portfolio.py`, one parameter, applies to all twenty strategies
at once. Also becomes a gene the GA can tune.

### 2. Volatility capping

Scale gross exposure down when trailing portfolio volatility exceeds a target.

Be precise about the constraint: the mandate is long-only with gross ≤ 1.0, so this can
only **de-risk**, never lever up. That halves the effect reported in the literature
(Moreira & Muir 2017), where the gains come substantially from levering up in calm periods.
It is still worth doing — volatility is the most autocorrelated quantity in finance, far
more predictable than returns.

⚠️ `defensive_regime` and the evolved `ga-price-1` winner already do a crude, discrete
version of this, and it is where their drawdown advantage comes from. A continuous version
is better behaved and has fewer free parameters than a threshold switch.

### 3. Covariance-aware weighting — hierarchical risk parity

Currently: equal, score-rank, or inverse-volatility weights. All three ignore
**correlations** — inverse-vol will happily hand a portfolio to fifty low-volatility
utilities that all move together.

Mean-variance optimisation is the textbook answer and is the wrong one at this sample size:
estimating a 348×348 covariance from 60 months produces a near-singular matrix and an
optimiser that puts everything in whichever name has the noisiest estimate. **Hierarchical
Risk Parity** (López de Prado 2016) avoids matrix inversion entirely — cluster the
correlation matrix, then allocate down the tree. Roughly 80 lines, robust by construction.

*Cost*: a hand-rolled single-linkage clustering, or a scipy dependency. Given the project's
four-library discipline, hand-rolled.

---

## Tier 2 — New information

Genuinely additive: things not currently in the 79 features, all obtainable free.

### 4. Point-in-time GICS sectors — TODO-6, already scoped

**The Wikipedia revision wikitext for all 227 monthly snapshots is already cached on disk.**
Zero network calls. Revisions from ~2008 carry a `GICS Sector` column;
`parse_constituents()` already does header-driven column detection and currently extracts
only the ticker.

This unlocks a whole family rather than one strategy:

- **Sector-neutral ranking** — rank within sector instead of across the whole market. This
  is standard practice and frequently the single largest improvement to a factor's Sharpe,
  because most raw factor portfolios are accidental sector bets. `div_yield` over 2007–2021
  is largely a bet on financials and energy; the GA found this and ranked it −1.00.
- **Sector-relative features** — a company's ROE against its own sector's median is a
  different and better signal than its ROE against a bank's.
- **Concentration constraints** — cap sector exposure.
- **Sector momentum / rotation** as a strategy in its own right.

Effort is small, network cost is zero, and every existing strategy could be re-run in a
sector-neutral variant. **This is the highest value-per-hour item in the document.**

### 5. FINRA short interest

`cdn.finra.org` is already in `config/settings.toml` rate limits — somebody anticipated
this — but there is no ingester. FINRA publishes bi-monthly short interest per security,
free.

Two derived signals, both well documented:

- **Short interest ratio** (shorted shares / shares outstanding) — negative predictor.
  Short sellers are, on average, informed (Boehmer, Jones & Zhang 2008).
- **Days to cover** (short interest / average daily volume) — a crowding measure, and the
  input to a squeeze.

Genuinely new information: nothing in the current 79 features overlaps with it. Bi-monthly
frequency suits a monthly rebalance. Coverage is good for index constituents.

*Effort*: one ingester following the existing `base.py` pattern, plus two features.

### 6. 10-K text similarity — "Lazy Prices"

Cohen, Malloy & Nguyen (2020): companies that **change the language of their annual filing**
year over year subsequently underperform, and the effect is large and slow to arbitrage
away. Managers who change nothing copy-paste; a change means something happened.

The project already talks to `data.sec.gov` and holds every filing's accession number. The
signal is a cosine similarity between consecutive 10-K texts, computed at the filing date.

*Effort is real*: downloading and parsing ~10,000 filings, several GB of bronze storage,
and text extraction from messy HTML. Call it a weekend, not an afternoon. But it is the
highest information content available for free anywhere in this document, it is
point-in-time by construction (a filing has a `filed_date`), and almost nobody at retail
scale has it.

---

## Tier 3 — More capacity on the same features

This is where "neural nets" belongs, and it is the third tier for a reason.

### 7. Gradient-boosted trees

Predict next-month cross-sectional return rank from the 79 features. Refit on an expanding
window at each rebalance — the discipline is already implemented in
`strategies/learned.py::RollingRidge`, where the hard part (no training label reaching the
as-of date) is solved.

Why trees first, on this data specifically:

- **They handle NaN natively.** Given that 3.5% of rows are complete, this is not a
  preference, it is the difference between training on 52,000 rows and 1,800.
- **Monotonic constraints.** LightGBM can be told that lower volatility must never score
  worse. That injects economics as a regulariser rather than hoping the model learns it,
  and it is the most effective anti-overfitting device available for tabular finance.
- **Feature importance is readable**, which keeps the project's "a winner nobody can read is
  a winner nobody can check" rule intact.

*Cost*: a real dependency (LightGBM), which breaks the current four-library discipline.
That is a deliberate trade to make explicitly rather than drift into.

*Realistic expectation*: published cross-sectional GBDT results add perhaps 0.1–0.3 Sharpe
over a linear model on the same features — before costs, on larger universes, with better
data. Expect less here.

### 8. Shallow neural net

Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, is the reference and
its headline finding is worth quoting plainly: **two or three hidden layers performed best,
and deeper networks performed worse.** The gains came from allowing feature interactions,
not from depth.

So the right build is deliberately small: 55 dense features → 32 → 16 → 1, ReLU, dropout,
early stopping, and an **ensemble over 5–10 random seeds** (they found the seed ensemble
matters more than the architecture). That is roughly 50 lines of numpy with Adam — no
PyTorch dependency needed at this size.

Use the 55 features that are ≥90% populated, giving 39,247 complete rows. Do not impute the
sparse fundamentals; imputation on a 30%-populated column invents 70% of a feature.

### 9. Ensembling and stacking

The cheapest idea in this document and possibly the best.

Twenty strategies already produce a score per stock per month. Rank-average them. Or weight
them by trailing performance. Or stack: fit a model whose *inputs are the twenty scores*
rather than the 79 features.

Ensembles beat their members almost universally, and this one requires no new data, no new
dependency and no new features — the scores already exist. It is also the most honest use
of the twelve hand-written strategies: each encodes a different economic story, so
averaging them is averaging over model uncertainty rather than over noise.

---

## Tier 4 — Structure

### 10. Uncertainty-aware position sizing

Every model above produces a point estimate. None produces a confidence. Size positions by
the model's own uncertainty — larger where the cross-sectional spread of predictions is
wide, smaller where it is narrow — or use conformal prediction to get calibrated intervals
without distributional assumptions.

Underused, cheap, and it addresses a real failure mode: the current construction bets the
same amount whether the top-ranked name beats the median by 5 percentile points or 50.

### 11. Hidden-Markov regime model

Fit a 2–3 state HMM to market returns and volatility, then condition strategy choice on the
inferred state. This is the principled version of what `defensive_regime` does with two
hand-set thresholds.

⚠️ **This is the highest-risk item in Tier 4 and the risk is structural.** 177 months
contain roughly three regime episodes. An HMM with 3 states has enough parameters to
memorise them. It will produce a beautiful in-sample fit and there is very little
independent evidence available to contradict it. If built, it should be trained on daily
data (6,702 observations) rather than monthly, and its states should be *validated against
something external* — NBER recession dates, the VIX term structure — rather than judged by
the returns of the strategy that uses them.

### 12. Genetic programming — evolved expression trees

The obvious next step from the current GA: search over `(mom_12_1 / vol_126d) > median` and
similar, rather than over weighted sums.

ADR-031 rejected it and the reasoning still holds — an unconstrained expression search finds
spectacular nonsense on generation one. A defensible version exists: **typed** GP with a
grammar of six operators, depth ≤ 3, and the same deflation discipline. Even then the trial
count explodes, which makes the deflated Sharpe bar much higher, which is the correct
consequence rather than an obstacle.

Build this only after Tier 1 and 2 are exhausted.

---

## Tier 5 — The ones people ask for, and why they rank last

Worth addressing directly rather than omitting.

### 13. Sequence models — LSTM, Transformer

The instinct is that 2 million daily bars is plenty of data for a sequence model. The
instinct is wrong in a specific way: **the inputs are 2 million, but the labels are still
177 months × 348 names.** A model's capacity to overfit is governed by the labels it is
fitted against, not by the length of the sequences it reads.

Gu–Kelly–Xiu found deeper is worse on exactly this problem. Every reasonably-controlled
published comparison on tabular cross-sectional equity data reaches the same conclusion.

There is one version that could work and is worth naming: use a **fixed, untrained** or
lightly-trained sequence encoder to compress each stock's daily path into a handful of
numbers, then feed those as additional *features* to a tree model. That separates
representation from prediction and keeps the fitted parameter count small.

### 14. Reinforcement learning

177 monthly decisions. RL needs orders of magnitude more interaction than that, and unlike a
game it cannot generate more by playing again — the environment is a fixed history.

The narrow defensible use is **execution**: learning when to trade toward a target weight,
where the decisions are daily (6,702 of them), the reward is immediate and measurable
(realised cost versus a benchmark fill), and the action space is small. That is a real
problem this project has — it currently trades to target every month regardless — and it is
a much better-posed RL problem than portfolio choice.

Not yet, either way.

---

## What the harness would need

Most of these plug into what already exists. Honest accounting of what does not:

| family | needs building |
|---|---|
| 1, 2, 3 | new `Construction` options; nothing else |
| 4 | extend `parse_constituents()`; a `reference/sp500_sector_history` table; a sector-neutral variant of `rank_pct` |
| 5 | one ingester (`ingest/finra_short_interest.py`) + 2 features |
| 6 | an ingester, bronze storage for ~10k filings, text extraction |
| 7, 8 | a `learned.py` sibling; the refit discipline already exists |
| 9 | a meta-strategy that instantiates others and averages their scores |
| 10 | `Construction` gains a per-name scaling input |
| 11 | a standalone regime module + external validation |
| 12 | a second genome type in `evolve/` |
| 13, 14 | substantial, and a heavy dependency |

Two things every one of them gets for free, and they are the reason this list is short
rather than a rewrite: **the point-in-time context** (a strategy physically cannot see the
future) and **the trial registry** (every run is counted, so the deflated Sharpe stays
meaningful however many families are added).

---

## Suggested order

1. **No-trade bands and volatility capping.** One week. Applies to all twenty existing
   strategies at once, requires no forecast, and is the only work here whose benefit is
   near-certain.
2. **Point-in-time sectors.** Zero network cost, the data is already on disk, and it
   unlocks sector-neutral variants of everything.
3. **Ensemble the existing twenty.** No new data, no new dependency.
4. **The walk-forward harness** — already the stated next step. Build it *before* the model
   families below, because it is the only way to evaluate them without spending the holdout.
   Note what it is not: the **forward-testing harness** (`sp500lab forward`,
   [FORWARD_TEST.md](FORWARD_TEST.md), ADR-033/034) is built, and it evaluates a *fixed*
   candidate on 2022 onward. A walk-forward re-runs the *search* inside the research
   window. The walk-forward narrows the candidates; the forward test spends the holdout on
   the survivor. Building families 5-14 below without the first means having nothing but
   the second to check them with, and there is only one of those.
5. **Short interest**, then **gradient-boosted trees** on the enlarged feature set.
6. **Then** the neural net, and only then consider Tier 4.

Note where the walk-forward sits. Steps 1–3 are safe to do without it because they involve
almost no fitting. Everything after step 4 involves fitting, and fitting without
out-of-sample evaluation just moves the overfitting somewhere harder to see.

---

## How not to fool ourselves

Four rules that apply to every family above, and one prediction.

**Every new family is a multiple-testing cost.** Twenty strategies and two searches already
exist. Adding a GBDT, an MLP and an ensemble means the eventual holdout test is choosing
from a larger menu, and the deflated Sharpe correction grows accordingly. That is the
correct accounting, not an argument against building them — but it means each family should
be added because there is a reason to believe in it, not because it is available.

That cost now lands twice, and the second one is newly visible. `store.selection_bar()` in
the forward harness applies the same best-of-N correction to the *forward* window: with
twelve candidates forward-tested, the luckiest of twelve worthless ones posts a Sharpe of
about 0.62 over 54 months. Every family added here is one more candidate that will
eventually want a look, and the bar rises with each.

**A new model must beat `multi_factor`, not SPY.** `multi_factor` averages four factor
families with no tuning at all. Any model that cannot beat the untuned average of the
obvious thing has found nothing, whatever it does against the index.

**Report all three cost settings, always.** `random_weight` — which has no signal — posts
the second-best Sharpe in the suite under optimistic costs.

**Compare over the same window.** Anything using fundamentals starts in 2010, where SPY
returned 15.66%/yr against 10.42% from 2007. `backtest suite` already enforces this.

**And the prediction:** most of these will not beat the index. Three of twenty currently do.
That is the correct null result on daily bars over large-cap US equities with free data, it
is what the literature would predict, and a new family that appears to beat it comfortably
on its first run is far more likely to be a bug than an edge. The value of building them is
partly in the ones that fail, because a hypothesis that fails cleanly on an honest harness
is knowledge — and that is what this harness was built to produce.

---

**See also:** [HOW_THE_GA_WORKS.md](HOW_THE_GA_WORKS.md) · [FEATURES.md](FEATURES.md) for
what the 79 features already cover · [STRATEGIES.md](STRATEGIES.md) for the current bar ·
[ADDING_A_STRATEGY.md](ADDING_A_STRATEGY.md) for the interface any of these would implement
· [FORWARD_TEST.md](FORWARD_TEST.md) for how any of them eventually gets checked out of
sample, and why 54 months can refute one but not confirm it
· HANDOFF.md §5b for TODO-6 (sectors) and the walk-forward specification.
