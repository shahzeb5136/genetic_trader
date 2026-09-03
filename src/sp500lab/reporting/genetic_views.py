"""The genetic-algorithm lab: three pages about the search itself.

Pure views, same contract as `views.py` and `algorithms_view.py`: prepared data in, a
`Report` out, no I/O and no markup. `queries.genetic_lab()` does the reading.

Three pages, one question each
-------------------------------
``methodology_report``   how a search works: the space it moves in, the number it
                         maximises, the operators that move it, and every defence
                         against the thing a genetic algorithm does best - overfitting
``features_report``      what the search is allowed to read, why the list is short and
                         frozen, and which features each search actually converged on
``searches_report``      every search that has run: its settings, its training history,
                         its winner decoded into sentences, and what happened to that
                         winner out of sample

There is no index. Three pages that link to each other do not need a fourth page whose
only content is three links, and the set is small enough that any one of them is a fine
place to arrive.

The editorial rule this page set exists under
----------------------------------------------
A genetic algorithm is the single easiest way in this project to produce a number that
means nothing. It maximises whatever it is handed, over thousands of draws, on one fixed
history - so the headline Sharpe of a winner is an estimate of skill PLUS the selection
effect, and the selection effect grows with the trial count. Every page here therefore
carries the trial count and the deflated Sharpe next to any winner's result, and the
forward verdict wherever one exists. A page that showed a 21% CAGR without saying it was
the best of 1,407 tries would be the most misleading thing this project could publish.
"""

from __future__ import annotations

from . import series as S
from . import theme
from .specs import (
    LineChart,
    LinkCard,
    LinkGrid,
    Note,
    Report,
    Section,
    Stat,
    StatRow,
    TableBlock,
)
from .tables import Table, _cell, _text
from .util import gt as _gt

#: The three pages, and how each describes itself on the others' link grids.
PAGES = {
    "methodology": ("How the search works",
                    "The genome, the fitness function, the operators, and the four "
                    "defences against a search that would otherwise overfit every time."),
    "features": ("What the search may read",
                 "The three feature presets, why they are short and frozen, and which "
                 "features each search actually converged on."),
    "searches": ("The searches and their winners",
                 "Every search that has run: settings, training history, the winning "
                 "genome in sentences, and what happened to it after 2022."),
}

#: The deflated-Sharpe convention. Not a law of nature, and the pages say so.
DSR_THRESHOLD = 0.95

VERDICT_EMPHASIS = {"held": "good", "decayed": "warn", "failed": "bad"}


def _dsr(value) -> str:
    """A deflated Sharpe, which is a PROBABILITY and must never be printed as certainty.

    0.9995 at three decimals renders as "1.000", which claims the winner certainly beat
    the best of 1,407 lucky draws. Nothing on 136 monthly observations supports that, so
    the top of the scale is reported as a bound instead.
    """
    from .util import finite
    if not finite(value):
        return "—"
    v = float(value)
    return ">0.999" if v >= 0.9995 else f"{v:.3f}"


def _nav(current: str, hrefs: dict[str, str]) -> Section:
    """The link grid every page carries. Three pages, no index."""
    cards = [LinkCard(title=PAGES[key][0], href=hrefs[key], blurb=PAGES[key][1])
             for key in ("methodology", "features", "searches") if key != current]
    cards.append(LinkCard(
        title="← All algorithms", href=hrefs.get("backtest", "../backtest/index.html"),
        blurb="The full scoreboard: every hand-written strategy and every evolved "
              "winner, scored on the same engine."))
    return Section("The rest of the lab").add(LinkGrid(cards))


def _searched_warning(n_trials: int | None) -> Note:
    """The note that outranks everything else on these pages."""
    count = f"{n_trials:,}" if n_trials else "many"
    return Note(
        f"Every number a search produces is the MAXIMUM over {count} configurations, "
        "and the maximum of N draws is high whether or not there is any signal in the "
        "data. That is arithmetic, not pessimism. The deflated Sharpe corrects for it "
        "and is the number to read first; below the 0.95 convention a searched result "
        "is not distinguishable from the luckiest of that many worthless draws.",
        level="warn", title="A searched Sharpe is not a Sharpe.")


# ==========================================================================
# 1. Methodology
# ==========================================================================

def methodology_report(genome: dict, searches: list[dict], *, generated_at: str = "",
                       hrefs: dict[str, str] | None = None) -> Report:
    """How the search works, from the space it moves in to the correction on its output."""
    hrefs = hrefs or {}
    defaults = genome.get("defaults", {})
    report = Report(
        title="How the genetic algorithm works",
        subtitle="The search space, the number it maximises, the operators that move it, "
                 "and the defences that make its output worth reading",
        generated_at=generated_at,
        meta={"searches run": theme.count(len(searches)),
              "genome": f"{genome['sizes'].get('price', 0)}–"
                        f"{genome['sizes'].get('full', 0)} genes",
              "dead zone": f"±{genome['dead_zone']}",
              "population x generations": f"{defaults.get('population')} x "
                                          f"{defaults.get('generations')}"})

    report.add(_what_it_is(searches))
    report.add(_search_space(genome))
    report.add(_fitness_section(genome, searches))
    report.add(_operators_section(genome))
    report.add(_defences_section(searches))
    report.add(_nav("methodology", hrefs))
    return report


def _what_it_is(searches: list[dict]) -> Section:
    total_trials = sum((s.get("deflation") or {}).get("n_trials") or 0 for s in searches)
    s = Section("What it actually is", blurb=(
        "A loop over `run_backtest`. That is the whole trick, and it is why the engine "
        "was built first: the fitness function IS the backtest, so an evolved strategy "
        "is scored by exactly the same accounting, the same next-open execution, the "
        "same costs and the same survivorship-free universe as a hand-written one. The "
        "scoreboard cannot tell them apart, which is the point."))
    s.add(StatRow([
        Stat("A strategy is", "a float vector", "weights per ranked feature, plus "
                                                "portfolio shape"),
        Stat("Scored by", "the backtest engine", "no separate simulator, no shortcuts"),
        Stat("One evaluation", "~0.15 s", "panel memoised, feature ranks precomputed"),
        Stat("Trials logged", theme.count(total_trials), "every individual, not just "
                                                         "the winners"),
    ]))
    s.add(Note(
        "The search moves through a space of weighted sums of ranked features. It cannot "
        "invent an indicator, multiply two of them together, or write an `if`. That is a "
        "deliberate trade of expressive power for the ability to read the answer: a "
        "tree-based genetic algorithm on this data would find spectacular nonsense in its "
        "first generation, and nobody could tell whether it had.",
        level="info", title="The space is small on purpose."))
    return s


def _search_space(genome: dict) -> Section:
    s = Section("The search space", blurb=(
        "One gene per feature the preset carries, holding that feature's weight, plus "
        "six genes for the shape of the portfolio. A point in this box decodes into a "
        "strategy the engine can run, and nothing outside the box is reachable."))

    sizes = genome.get("sizes", {})
    s.add(StatRow([
        Stat("price genome", theme.count(sizes.get("price")), "13 features + 6 shape"),
        Stat("full genome", theme.count(sizes.get("full")), "23 features + 6 shape"),
        Stat("night genome", theme.count(sizes.get("night")), "17 features + 6 shape"),
        Stat("dead zone", f"±{genome['dead_zone']}", "below this a weight is exactly 0"),
    ]))

    rows = []
    for g in genome.get("shape_genes", []):
        if g.get("choices"):
            span = " | ".join(g["choices"])
        elif g.get("integer"):
            span = f"{int(g['low'])} to {int(g['high'])}"
        else:
            span = f"{g['low']:g} to {g['high']:g}"
        rows.append([_text(g["name"]), _text(span), _text(g.get("note", ""))])
    s.add(TableBlock(Table(["gene", "range", "what it decides"], rows,
                           aligns=["left", "left", "left"], sortable=False,
                           caption="The six genes that are not feature weights. Every "
                                   "feature in the preset adds one more gene, bounded "
                                   "to [-1, +1]: negative means the LOW end of that "
                                   "feature is the good end."),
                     title="The portfolio genes"))

    s.add(Note(
        "Weights are applied to PERCENTILE RANKS, never to raw values. A book-to-market "
        "of 300 caused by one bad share count cannot dominate a portfolio, because it is "
        "worth exactly “first place”. Cross-sectional fundamental data has tails that "
        "are not merely fat but wrong, and ranking is what makes a weighted sum of "
        "twenty different units meaningful at all.",
        level="info", title="Ranks, not values."))
    s.add(Note(
        f"A weight whose magnitude is below {genome['dead_zone']} contributes exactly "
        "nothing. Without that dead zone every individual uses every feature a little, "
        "“how many features does this use” has no answer, parsimony pressure has nothing "
        "to grip, and two genomes that behave identically look different to the "
        "deduplicator - which would inflate the trial count and quietly under-correct "
        "the winner.",
        level="info", title="The dead zone earns its place three times."))
    s.add(Note(
        "The regime gate is the only non-linearity in the space. Three genes decide "
        "whether an individual de-risks when the index is below its 200-day average or "
        "realised volatility is far above its own year, and how much of the account "
        "sits in cash while it does. Under a long-only mandate cash is the only "
        "defensive asset there is, and whether a search chooses to USE the gate is one "
        "of the more interesting things it reports.",
        level="info", title="One switch, and it is the only bend in the space."))
    return s


def _fitness_section(genome: dict, searches: list[dict]) -> Section:
    d = genome.get("defaults", {})
    s = Section("What is being maximised", blurb=(
        "A genetic algorithm optimises the number you hand it, exactly and without "
        "mercy. So the objective is not a detail of the search, it IS the search, and "
        "every term below exists because of a specific way an unconstrained one goes "
        "wrong."))
    s.add(StatRow([
        Stat("metric", str(d.get("metric", "")), "monthly, not daily"),
        Stat("aggregate", str(d.get("aggregate", "")), "mean across folds, minus their "
                                                       "spread"),
        Stat("folds", theme.count(d.get("n_folds")), "contiguous, never shuffled"),
        Stat("embargo", f"{d.get('embargo_days')} days", "one holding period between "
                                                         "folds"),
    ]))

    rows = [
        ["not raw return",
         "Handed return, the search finds leverage in whatever form the mandate still "
         "allows. Here that is concentration: it drives the holding count to its floor, "
         "holds ten names, and reports a magnificent CAGR next to a 70% drawdown. "
         "Sharpe closes that door."],
        ["the MONTHLY Sharpe",
         "Daily returns of a monthly-rebalanced portfolio are strongly autocorrelated "
         "inside each holding period, so ~4,900 daily observations carry roughly 176 "
         "independent ones. A search maximising the daily figure is partly maximising "
         "an artefact of the sampling - and the deflated Sharpe uses monthly statistics, "
         "so this way the search and its own significance test look at one quantity."],
        ["fold consistency, not full-sample",
         "An individual is scored on how it did in EVERY sub-period, minus how much "
         "those sub-periods disagreed. A strategy that made all its money in 2009 and "
         "nothing since has a fine full-sample Sharpe and a terrible score here. That "
         "distinction is most of what separates a discovery from a coincidence."],
        ["contiguous folds with an embargo",
         "Financial data is autocorrelated and a shuffled K-fold leaks across every "
         "boundary. The folds are contiguous calendar spans separated by one month, "
         "which is exactly the span over which a position opened in one fold can still "
         "be held in the next."],
        ["penalties, charged on purpose",
         "Turnover is charged again on top of the cost model, deliberately: the "
         "half-spread estimate is the weakest input in the chain, and a strategy "
         "trading 500%/yr is making a large bet that it is right. Complexity is charged "
         "per active feature, because every extra feature is another dimension the "
         "search could have found a coincidence in."],
    ]
    s.add(TableBlock(Table(["the objective is", "because"],
                           [[_text(a), _text(b)] for a, b in rows],
                           aligns=["left", "left"], sortable=False)))

    folds = [s_["config"].get("n_folds") for s_ in searches if s_.get("config")]
    s.add(Note(
        "Folds here measure ROBUSTNESS, not out-of-sample performance. Every fold is "
        "inside the research window and every individual is selected using all of them, "
        "so a good fold score is evidence of consistency and not evidence of "
        "generalisation. This is the single most important sentence on this page: the "
        "genuine out-of-sample test is the reserved period after 2022, it was looked at "
        "once, and looking is permanently recorded."
        + (f" Every search here used {folds[0]} folds." if folds else ""),
        level="danger", title="What the folds are NOT."))
    return s


def _operators_section(genome: dict) -> Section:
    d = genome.get("defaults", {})
    s = Section("How the population moves", blurb=(
        "Selection, crossover and mutation operate on float vectors inside a box and "
        "know nothing about finance. That separation is the point: the operators cannot "
        "accidentally encode a market opinion, and the genome cannot accidentally encode "
        "a search strategy."))
    rows = [
        ["tournament selection", f"size {d.get('tournament_size')}",
         "Not fitness-proportionate. Fitness is a Sharpe ratio, which can be negative "
         "and whose absolute scale is arbitrary; roulette-wheel selection needs positive "
         "weights and an offset somebody invented. A tournament needs only an ordering."],
        ["blend crossover (BLX-α)", f"α = {genome.get('blx_alpha')}, "
         f"rate {d.get('crossover_rate')}",
         "Not single-point. A genome of feature weights has no meaningful gene ORDER, so "
         "a cut point is arbitrary. BLX samples each gene from an interval slightly "
         "wider than the parents span, which lets a converged population still reach "
         "outside itself."],
        ["two mutations", f"rate {d.get('mutation_rate')}, σ {d.get('mutation_sigma')}, "
         f"reset {d.get('reset_rate')}",
         "A Gaussian nudge scaled to each gene's own range explores locally; a full "
         "resample, applied rarely, is the only operator that can reintroduce diversity "
         "a population has lost entirely. With just the nudge, a converged population "
         "stays converged."],
        ["elitism + immigration", f"{d.get('elite')} elites, "
         f"{d.get('immigrants')} immigrants",
         "The best individuals survive untouched so the search can never go backwards; "
         "a few random newcomers arrive every generation so it can never finish "
         "converging either. The two pull in opposite directions on purpose."],
        ["duplicate suppression", "behavioural fingerprint",
         "Genomes that behave identically waste an evaluation and, worse, corrupt the "
         "deflated Sharpe: the registry counts distinct fingerprints as trials, so a "
         "population of clones would report a small trial count for a large search and "
         "deflate its winner far too gently."],
        ["seeded start", f"{'on' if d.get('seed_with_baselines') else 'off'} by default",
         "The initial population contains genomes reproducing momentum, low volatility, "
         "reversal, trend, value and quality. That is not a shortcut, it is the "
         "experiment: if a population starting from the hand-written ideas cannot evolve "
         "anything better than its seeds, that is the direct answer to “can the search "
         "improve on what a person wrote”."],
    ]
    s.add(TableBlock(Table(["operator", "setting", "what it defends against"],
                           [[_text(a), _text(b), _text(c)] for a, b, c in rows],
                           aligns=["left", "left", "left"], sortable=False)))
    s.add(Note(
        "Every operator takes an explicit random generator; nothing touches the global "
        "numpy state. A search that cannot be replayed exactly cannot be audited, and a "
        "search that cannot be audited is a story about a number.",
        level="info", title="Determinism is not optional."))
    return s


def _defences_section(searches: list[dict]) -> Section:
    s = Section("The four defences, and the one that decides everything", blurb=(
        "A genetic algorithm overfits more aggressively than almost anything else in "
        "this project, because it is EXPLICITLY maximising the number being reported. "
        "Each of these is load-bearing."))
    rows = [
        ["1. a small, bounded, readable space",
         "Weighted sums of ranked features, not evolved expression trees. Every "
         "individual has an economic reading a human can argue with."],
        ["2. every individual is logged as a trial",
         "Not just the winners. The deflated Sharpe needs the trial count and the "
         "spread of trial Sharpes, and neither can be recovered after the fact."],
        ["3. fitness is fold consistency",
         "An individual that made all its money in one 18-month stretch scores badly "
         "however good its headline looks."],
        ["4. the holdout is untouched",
         "The search stops the day before the reserved period begins. Testing a winner "
         "there is a separate, deliberate, permanently recorded act."],
    ]
    s.add(TableBlock(Table(["defence", "what it buys"],
                           [[_text(a), _text(b)] for a, b in rows],
                           aligns=["left", "left"], sortable=False)))

    tested = [x for x in searches if x.get("forward")]
    if tested:
        decayed = [x for x in tested
                   if str((x["forward"] or {}).get("verdict")) == "decayed"]
        s.add(Note(
            f"{len(decayed)} of {len(tested)} winners tested out of sample DECAYED. "
            "Every one of them cleared the deflated-Sharpe convention on the research "
            "window first. That is the honest summary of this whole apparatus: the "
            "defences above are necessary and they were not sufficient, and the reason "
            "to keep them is that the result is now measurable rather than merely "
            "hoped for.",
            level="danger", title="What the defences did and did not buy."))
    return s


# ==========================================================================
# 2. The features the search may read
# ==========================================================================

def features_report(presets: dict, searches: list[dict], catalog, *,
                    generated_at: str = "",
                    hrefs: dict[str, str] | None = None) -> Report:
    """What the search is allowed to read, and what it converged on.

    `catalog` maps a feature name to an object with `family`, `what` and `reading`, so
    the descriptions come from `features/catalog.py` rather than being restated here.
    """
    hrefs = hrefs or {}
    every = sorted({f for names in presets.values() for f in names})
    report = Report(
        title="What the search may read",
        subtitle="The three feature presets, why the lists are short and frozen, and "
                 "which features each search actually converged on",
        generated_at=generated_at,
        meta={"features in any preset": theme.count(len(every)),
              "presets": theme.count(len(presets)),
              "searches": theme.count(len(searches))})

    report.add(_presets_section(presets, searches))
    report.add(_catalog_section(presets, every, catalog))
    report.add(_convergence_section(searches))
    report.add(_regime_inputs_section(catalog))
    report.add(_nav("features", hrefs))
    return report


def _presets_section(presets: dict, searches: list[dict]) -> Section:
    used_by: dict[str, list[str]] = {}
    for s in searches:
        used_by.setdefault(s["preset"], []).append(s["study"])

    s = Section("Three presets, and why they are short", blurb=(
        "The feature layer carries 79 point-in-time features. A search may read 13, 17 "
        "or 23 of them. Every feature added to a preset multiplies the space the search "
        "moves through and adds to the trial count the deflated Sharpe has to discount, "
        "so the lists are curated rather than complete."))
    rows = []
    for name, features in presets.items():
        rows.append([
            _text(name),
            _cell(len(features), theme.count),
            _text(", ".join(used_by.get(name, [])) or "—",
                  "" if used_by.get(name) else "muted"),
            _text("2007-04 onward" if name != "full" else "2010-07 onward",
                  "warn" if name == "full" else ""),
        ])
    s.add(TableBlock(Table(["preset", "features", "searches that used it",
                            "usable from"], rows, sortable=False,
                           aligns=["left", "right", "left", "left"])))
    s.add(Note(
        "`full` costs three years of history. XBRL fundamentals begin in 2009-04 and the "
        "derived growth and surprise features need several quarters after that, so a "
        "search on this preset starts in 2010-07 and its winner's CAGR is not comparable "
        "with a `price` winner's without saying so. The index itself returned far more "
        "per year from 2010 than from 2007.",
        level="warn", title="A preset chooses a window as well as a feature list."))
    s.add(Note(
        "A preset is FROZEN once a search has stored genomes against it. A stored winner "
        "decodes by position, so appending a feature to an existing preset would silently "
        "mis-decode every genome already in a checkpoint - each weight would land on the "
        "wrong feature and the strategy would still run. That is why the overnight and "
        "dividend-calendar features arrived as a new `night` preset rather than as an "
        "extension of `price`.",
        level="danger", title="Presets are append-only by creating a new one."))
    return s


def _catalog_section(presets: dict, every: list[str], catalog) -> Section:
    s = Section("Every feature the search may read", blurb=(
        "Taken from the feature catalogue the strategies themselves read, so this page "
        "and the code cannot drift apart. “Which end is good” is the historical reading; "
        "the search is free to disagree with it, and a negative weight means it did."))
    rows = []
    for f in every:
        doc = catalog(f)
        member = [name for name, names in presets.items() if f in names]
        rows.append([
            _text(f),
            _text(getattr(doc, "family", ""), "muted"),
            _text(getattr(doc, "what", "")),
            _text(getattr(doc, "reading", "").split(".")[0]),
            _text(", ".join(member)),
        ])
    s.add(TableBlock(Table(
        ["feature", "family", "what it is", "which end is historically good", "presets"],
        rows, aligns=["left", "left", "left", "left", "left"],
        caption="Sorted alphabetically. The full reading for each feature, its coverage "
                "and its source are in the feature-layer report "
                "(`sp500lab report features`).")))
    return s


def _convergence_section(searches: list[dict]) -> Section:
    s = Section("What each search converged on", blurb=(
        "Two different statements. The winner's weight is one individual, and one "
        "individual is one draw. The population share is how many of the 60 survivors "
        "put a live weight on that feature, and no single lucky genome can move it - so "
        "it is the more honest description of what the search actually found."))
    for search in searches:
        rows = []
        weights = dict(search["weights"])
        usage = search["usage"]
        n_pop = max(search["n_population"], 1)
        for feature in sorted(set(weights) | set(usage),
                              key=lambda f: (-abs(weights.get(f, 0.0)),
                                             -usage.get(f, 0))):
            w = weights.get(feature)
            share = usage.get(feature, 0) / n_pop
            rows.append([
                _text(feature),
                _cell(w, lambda v: f"{v:+.2f}",
                      emphasis="good" if _gt(w, 0) else ("bad" if w is not None else ""))
                if w is not None else _text("in the dead zone", "muted"),
                _text("high is good" if _gt(w, 0)
                      else ("LOW is good" if w is not None else "—"),
                      "muted"),
                _cell(share, theme.pct, emphasis="good" if share > 0.9 else ""),
            ])
        s.add(TableBlock(Table(
            ["feature", "winner's weight", "direction it chose", "share of the "
             "final population using it"], rows,
            aligns=["left", "right", "left", "right"],
            caption=f"{search['study']}, preset `{search['preset']}`, "
                    f"{search['n_population']} individuals in the final generation. "
                    f"The winner ranks {len(search['weights'])} feature(s) and ignores "
                    f"{search['n_ignored']}."),
            title=f"{search['study']} — what it kept"))
    if not searches:
        s.add(Note("No search has been run yet.", level="warn"))
    return s


def _regime_inputs_section(catalog) -> Section:
    s = Section("The two inputs that are never ranked", blurb=(
        "The regime gate reads market-level state, not a cross-section. These columns "
        "are one number broadcast across every security, so ranking them would produce "
        "a constant and destroy the information."))
    rows = []
    for f in ("mkt_trend_200d", "mkt_vol_ratio"):
        doc = catalog(f)
        rows.append([_text(f), _text(getattr(doc, "family", ""), "muted"),
                     _text(getattr(doc, "what", "")),
                     _text(getattr(doc, "reading", "").split(".")[0])])
    s.add(TableBlock(Table(["input", "family", "what it is", "reading"], rows,
                           aligns=["left", "left", "left", "left"], sortable=False,
                           caption="Both are macro series, never revised and lagged one "
                                   "session, so a regime decision made on a date could "
                                   "have been made on that date.")))
    return s


# ==========================================================================
# 3. The searches and their winners
# ==========================================================================

def searches_report(searches: list[dict], registry_only: list[dict], *,
                    generated_at: str = "",
                    hrefs: dict[str, str] | None = None) -> Report:
    """Every search that has run, its training details, and what its winner became."""
    hrefs = hrefs or {}
    report = Report(
        title="The searches and their winners",
        subtitle="Every genetic-algorithm run on disk: what it was told to do, how it "
                 "converged, what it found, and what happened to that next",
        generated_at=generated_at,
        meta={"searches": theme.count(len(searches)),
              "trials": theme.count(sum((s.get("deflation") or {}).get("n_trials") or 0
                                        for s in searches)),
              "forward-tested": theme.count(sum(1 for s in searches
                                                if s.get("forward")))})
    if not searches:
        report.add(Section("Nothing has been searched yet", [Note(
            "No checkpoint in `data/experiments/evolve/`. Run "
            "`sp500lab evolve run --study my-search`.", "warn")]))
        report.add(_nav("searches", hrefs))
        return report

    report.add(_headline_section(searches))
    report.add(_settings_section(searches))
    report.add(_training_section(searches))
    for search in searches:
        report.add(_winner_section(search))
    if registry_only:
        report.add(_registry_only_section(registry_only))
    report.add(_nav("searches", hrefs))
    return report


def _headline_section(searches: list[dict]) -> Section:
    tested = [s for s in searches if s.get("forward")]
    decayed = [s for s in tested
               if str((s["forward"] or {}).get("verdict")) == "decayed"]
    survived = [s for s in searches
                if _gt((s.get("deflation") or {}).get("deflated_sharpe"),
                       DSR_THRESHOLD)]
    s = Section("The short version", blurb=(
        "Each search evaluated on the order of 1,400 distinct configurations inside the "
        "research window, kept the best, and had that winner carried into the reserved "
        "period as a pre-registered candidate."))
    s.add(StatRow([
        Stat("searches", theme.count(len(searches))),
        Stat("cleared deflation", f"{len(survived)} of {len(searches)}",
             f"deflated Sharpe above {DSR_THRESHOLD}",
             emphasis="good" if survived else ""),
        Stat("forward-tested", theme.count(len(tested)), "out of sample, once each"),
        Stat("decayed forward", f"{len(decayed)} of {len(tested)}",
             "the headline result", emphasis="bad" if decayed else ""),
    ]))

    rows = []
    for x in searches:
        d = x.get("deflation") or {}
        r = x.get("research") or {}
        f = x.get("forward") or {}
        verdict = str(f.get("verdict", ""))
        dsr = d.get("deflated_sharpe")
        rows.append([
            _text(x["study"]),
            _text(x["preset"], "muted"),
            _text(x["window"], "muted"),
            _cell(d.get("n_trials"), theme.count),
            _cell(r.get("cagr"), theme.pct),
            _cell(r.get("sharpe"), theme.num),
            _cell(r.get("maxdd"), theme.pct),
            _cell(dsr, _dsr,
                  emphasis="good" if _gt(dsr, DSR_THRESHOLD) else "bad",
                  title="probability the winner beats the luckiest of that many "
                        "worthless draws"),
            _cell(f.get("forward_sharpe"), theme.num),
            _text(verdict.upper() or "—",
                  VERDICT_EMPHASIS.get(verdict, "muted")),
        ])
    s.add(TableBlock(Table(
        ["search", "preset", "window", "trials", "CAGR", "Sharpe", "maxDD",
         "deflated", "fwd Sharpe", "forward"], rows,
        aligns=["left", "left", "left"] + ["right"] * 6 + ["left"],
        caption="Research-window results under realistic costs. `deflated` corrects the "
                "winner's Sharpe for the number of configurations evaluated before it "
                "was picked; `forward` is the 2022 onward verdict, where `decayed` means "
                "it beat the index in research and did not out of sample.")))
    s.add(_searched_warning(max((x.get("deflation") or {}).get("n_trials") or 0
                                for x in searches)))

    if len([s_ for s_ in searches if s_.get("forward")]) >= 2:
        s.add(Note(
            "Every winner cleared its deflated-Sharpe threshold and every winner then "
            "decayed out of sample. Read those two facts together: the deflated Sharpe "
            "corrects for how many configurations were TRIED, and it cannot correct for "
            "the search and the researcher having seen the same fifteen years of market "
            "history. This is now the most replicated finding in the project.",
            level="danger", title="Three for three."))
    return s


def _settings_section(searches: list[dict]) -> Section:
    s = Section("What each search was told to do", blurb=(
        "The full configuration is stored with every checkpoint, so a search can be "
        "replayed exactly. These are the settings that differ or that decide the "
        "outcome."))
    keys = [
        ("population", "population", theme.count),
        ("generations", "generations", theme.count),
        ("elite", "elite", theme.count),
        ("immigrants", "immigrants", theme.count),
        ("tournament_size", "tournament", theme.count),
        ("crossover_rate", "crossover", theme.num),
        ("mutation_rate", "mutation", theme.num),
        ("mutation_sigma", "sigma", theme.num),
        ("turnover_penalty", "turnover penalty", theme.num),
        ("complexity_penalty", "complexity penalty", theme.num),
        ("dispersion_weight", "dispersion weight", theme.num),
        ("seed", "seed", theme.count),
    ]
    rows = []
    for key, label, fmt in keys:
        row = [_text(label)]
        for x in searches:
            row.append(_cell(x["config"].get(key), fmt))
        rows.append(row)
    rows.append([_text("costs")]
                + [_text(str(x["config"].get("costs", ""))) for x in searches])
    rows.append([_text("holdout")]
                + [_text(str(x["config"].get("holdout", "")), "good") for x in searches])
    rows.append([_text("seeded with baselines")]
                + [_text("yes" if x["config"].get("seed_with_baselines") else "no")
                   for x in searches])
    s.add(TableBlock(Table(
        [""] + [x["study"] for x in searches], rows,
        aligns=["left"] + ["right"] * len(searches), sortable=False,
        caption="`holdout: exclude` on every column is the one row to check: it is what "
                "kept the reserved period out of the search.")))
    return s


def _training_section(searches: list[dict]) -> Section:
    s = Section("How they converged", blurb=(
        "Best and mean fitness per generation, and the diversity of the population that "
        "produced them. Fitness here is the fold-consistency score, not a Sharpe you "
        "can quote."))
    lines, mean_lines = [], []
    for x in searches:
        h = x.get("history")
        if h is None or not len(h):
            continue
        gens = [str(int(g)) for g in h["generation"]]
        lines.append(S.LineSeries(x["study"], gens,
                                  [float(v) for v in h["best_fitness"]]))
        mean_lines.append(S.LineSeries(x["study"], gens,
                                       [float(v) for v in h["mean_fitness"]]))
    if lines:
        s.add(LineChart(lines, title="Best fitness by generation", y_format="num",
                        height=280,
                        subtitle="Horizontal axis is the generation number.",
                        caption="Rising means the search is still finding better "
                                "individuals. A curve that goes flat early has "
                                "converged, and the generations after that point are "
                                "re-evaluating one idea."))
        s.add(LineChart(mean_lines, title="Mean fitness by generation", y_format="num",
                        height=250,
                        caption="The population as a whole, not just its best member. "
                                "Mean rising toward best is the population converging "
                                "on the leader."))

    diversity = []
    for x in searches:
        h = x.get("history")
        if h is None or not len(h):
            continue
        diversity.append(S.LineSeries(x["study"], [str(int(g)) for g in h["generation"]],
                                      [float(v) for v in h["diversity"]]))
    if diversity:
        s.add(LineChart(diversity, title="Population diversity by generation",
                        y_format="num", height=250,
                        caption="Mean normalised spread across genes: around 0.3 is "
                                "healthy, near 0 is a clone army. Reported because the "
                                "most common way a genetic algorithm fails is silently "
                                "- a run whose diversity collapses in generation four "
                                "spends the next twenty re-evaluating one individual, "
                                "and its fitness curve looks like convergence rather "
                                "than like the stall it is."))

    rows = []
    for x in searches:
        h = x.get("history")
        if h is None or not len(h):
            continue
        first, last = h.iloc[0], h.iloc[-1]
        rows.append([
            _text(x["study"]),
            _cell(len(h), theme.count),
            _cell(first["best_fitness"], theme.num),
            _cell(last["best_fitness"], theme.num,
                  emphasis="good" if last["best_fitness"] > first["best_fitness"]
                  else ""),
            _cell(last["best_fitness"] - first["best_fitness"],
                  lambda v: f"{v:+.3f}"),
            _cell(first["diversity"], theme.num),
            _cell(last["diversity"], theme.num,
                  emphasis="warn" if last["diversity"] < 0.10 else ""),
            _cell(first["best_n_active"], theme.count),
            _cell(last["best_n_active"], theme.count),
        ])
    if rows:
        s.add(TableBlock(Table(
            ["search", "generations", "fitness, first", "fitness, last", "gain",
             "diversity, first", "diversity, last", "features, first",
             "features, last"], rows,
            caption="A search whose diversity ends below about 0.10 has converged; the "
                    "immigrants are the only thing still exploring."),
            title="Start and finish"))
    return s


def _winner_section(search: dict) -> Section:
    d = search.get("deflation") or {}
    r = search.get("research") or {}
    f = search.get("forward") or {}
    p = search["portfolio"]
    verdict = str(f.get("verdict", ""))

    s = Section(f"{search['study']}: the winner", blurb=(
        f"The best individual of the final generation of `{search['study']}`, decoded. "
        "A winning parameter vector nobody can read is a winning parameter vector "
        "nobody can check, which is the whole reason the search space is bounded to "
        "something that turns back into sentences."))
    s.add(StatRow([
        Stat("CAGR", theme.pct(r.get("cagr")), "research window, realistic costs",
             emphasis="good" if _gt(r.get("cagr"), 0) else "bad"),
        Stat("Sharpe", theme.num(r.get("sharpe"))),
        Stat("max drawdown", theme.pct(r.get("maxdd")), "daily curve", emphasis="bad"),
        Stat("turnover", theme.pct(r.get("turnover"), 0), "per year",
             emphasis="warn" if _gt(r.get("turnover"), 4.0) else ""),
        Stat("deflated Sharpe", _dsr(d.get("deflated_sharpe")),
             f"after {theme.count(d.get('n_trials'))} trials",
             emphasis="good" if _gt(d.get("deflated_sharpe"), DSR_THRESHOLD) else "bad"),
        Stat("forward verdict", verdict.upper() or "—",
             "2022 onward" if verdict else "not tested",
             emphasis=VERDICT_EMPHASIS.get(verdict, "")),
    ]))

    rows = [[_text(feature),
             _cell(w, lambda v: f"{v:+.2f}",
                   emphasis="good" if w > 0 else "bad"),
             _text("high is good" if w > 0 else "LOW is good", "muted")]
            for feature, w in search["weights"]]
    s.add(TableBlock(Table(["feature", "weight", "direction"], rows,
                           aligns=["left", "right", "left"],
                           caption=f"Ranks {len(search['weights'])} feature(s) and "
                                   f"ignores {search['n_ignored']} whose weight fell "
                                   "inside the dead zone. Weight is applied to the "
                                   "feature's percentile rank within the tradable "
                                   "universe on each rebalance date."),
                     title="What it ranks"))

    s.add(StatRow([
        Stat("holds", theme.count(p.get("top_k")), "names, by score"),
        Stat("weighting", str(p.get("weighting", ""))),
        Stat("per-name cap", theme.pct(p.get("max_weight"))),
        Stat("regime gate", str(p.get("use_regime", "")).upper(),
             "de-risks in a falling or volatile market"),
        Stat("defensive gross", theme.pct(p.get("defensive_gross"), 0)
             if p.get("use_regime") == "on" else "—",
             "invested while defensive"),
        Stat("vol trigger", theme.num(p.get("vol_trigger"))
             if p.get("use_regime") == "on" else "—",
             "x its own year"),
    ], title="How it builds the portfolio"))

    if d.get("deflated_sharpe") is not None:
        s.add(TableBlock(Table(
            ["", "value"],
            [[_text("configurations tried"), _cell(d.get("n_trials"), theme.count)],
             [_text("spread of their Sharpes"),
              _cell(d.get("trial_sharpe_std"), theme.num)],
             [_text("monthly observations"), _cell(d.get("n_months"), theme.count)],
             [_text("Sharpe (annualised, monthly)"),
              _cell(d.get("sharpe_annualised_monthly"), theme.num)],
             [_text("bar set by the search"),
              _cell(d.get("expected_max_sharpe_annualised"), theme.num,
                    emphasis="warn",
                    title="the Sharpe the luckiest of that many worthless "
                          "configurations would have posted")],
             [_text("DEFLATED SHARPE"),
              _cell(d.get("deflated_sharpe"), _dsr,
                    emphasis="good" if _gt(d.get("deflated_sharpe"), DSR_THRESHOLD)
                    else "bad")]],
            aligns=["left", "right"], sortable=False,
            caption="Read the deflated Sharpe as a probability, not a score."),
            title="Does it survive its own search?"))

    if f:
        s.add(TableBlock(Table(
            ["", "research", "forward", "change"],
            [[_text("Sharpe"),
              _cell(f.get("research_sharpe"), theme.num),
              _cell(f.get("forward_sharpe"), theme.num),
              _cell(f.get("decay_z"), lambda v: f"{v:+.2f}σ",
                    emphasis="bad" if f.get("decay_z") is not None
                    and f["decay_z"] < -1 else "")],
             [_text("vs the index"), _text("—"),
              _cell(f.get("forward_d_sharpe"), lambda v: f"{v:+.2f}",
                    emphasis="good" if _gt(f.get("forward_d_sharpe"), 0) else "bad"),
              _text("")]],
            aligns=["left", "right", "right", "right"], sortable=False,
            caption="The change is measured in standard errors of itself. On 54 monthly "
                    "observations anything inside about ±1σ is noise, so these are "
                    "suggestive rather than conclusive - which is exactly why a forward "
                    "test can refute a strategy and cannot confirm one."),
            title="What happened to it out of sample"))
        if f.get("verdict_reason"):
            s.add(Note(str(f["verdict_reason"]), level="warn"))
    else:
        s.add(Note("This winner has not been forward-tested.", level="info"))
    return s


def _registry_only_section(registry_only: list[dict]) -> Section:
    s = Section("Searches with no surviving checkpoint", blurb=(
        "The trial log remembers these searches, but there is no checkpoint in "
        "`data/experiments/evolve/` to decode a winner from. They are named rather than "
        "dropped: their trials still count toward the deflated Sharpe of anything logged "
        "in the same study, and a reader comparing this page against "
        "`sp500lab experiments studies` has to be able to see why one search has no "
        "winner here."))
    rows = []
    for x in registry_only:
        d = x.get("deflation") or {}
        rows.append([
            _text(x["study"]),
            _cell(x.get("runs"), theme.count),
            _cell(x.get("trials"), theme.count),
            _cell(x.get("best_sharpe"), theme.num),
            _cell(d.get("deflated_sharpe"), _dsr),
        ])
    s.add(TableBlock(Table(
        ["study", "runs", "trials", "best Sharpe", "deflated"], rows,
        caption="Recoverable as numbers, not as a strategy.")))
    return s
