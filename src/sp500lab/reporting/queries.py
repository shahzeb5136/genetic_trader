"""What the report pages need from the registry, assembled as plain data.

This is the seam between "go and find out" and "draw it". Everything here reads the
experiment registry, the forward store, the feature panel or the strategy classes and
returns dicts, DataFrames and `AlgorithmEntry` records. Nothing here builds a chart
spec, prints, touches argparse or writes a file.

That split is the point. These functions used to live inside `cli.py` as private
helpers of the command handlers, which meant the only way to reach the Algorithm Book's
data was to construct an `argparse.Namespace` - and the old `cmd_all` really did, calling
`cmd_algorithms(Namespace(out=..., open_after=False))` so that one page could appear in
the report set. A command calling another command through its own argument parser is
the shape a missing module makes. This is that module.

Some of these WRITE to the registry as a side effect: `monthly_entries` and
`timing_entries` run any strategy that has no research-window row yet, logged under the
study "reports". That is deliberate - the book is meant to be complete on a cold
checkout - but it is why they take the loaded frame and hand back a reloaded one rather
than caching it.

The roster lives here too. `roster()` and `ga_winners()` decide which algorithms the two
report sets are sets OF, and they are defined once so `report backtest` and `report
forward` can never disagree about it (ADR-045).
"""

from __future__ import annotations

from pathlib import Path

from ..paths import BACKTEST_REPORTS_DIR
from .util import page_href

BOOK_FAMILIES = {
    "baselines": "The null hypotheses",
    "alpha": "The twelve hypotheses",
    "frontier": "The second wave",
    "learned": "Learned models",
}


DEFLATABLE_PREFIXES = ("ga-", "frontier", "mlp", "timing", "learned")


RESEARCH_END = "2021-12-31"


def build_strategy(name: str, *, start, end, holdout: str, costs: str,
                   log_run: bool = True, strategy=None, claim: str | None = None,
                   study: str | None = None):
    """Run one strategy under all three cost settings and gather its context.

    Takes an OBJECT as well as a name so an evolved winner - which lives in a search
    checkpoint rather than in the strategy registry - goes through exactly the same path
    as a hand-written one. The report cannot tell them apart, which is the same claim the
    backtest engine makes.

    Takes plain keywords rather than an argparse Namespace: the report set, the
    forward suite and the tests all need this and none of them have one.
    """
    from ..backtest import run_backtest
    from ..backtest.benchmark import over_window
    from ..backtest.strategy import get_strategy

    settings = ("optimistic", "realistic", "pessimistic")
    common = dict(start=start, end=end, holdout=holdout, log_run=log_run, study=study)
    others = [run_backtest(strategy or get_strategy(name), costs=c,
                           record_trades=(c == costs), **common)
              for c in settings]
    result = others[settings.index(costs)] if costs in settings else others[1]

    if claim is None:
        claim = claim_of(strategy or get_strategy(name))
    bench = over_window(result)
    return result, others, bench, claim, feature_coverage(), deflation_for(result)


def evolved_winners() -> list[dict]:
    """The winner of every genetic-algorithm search on disk, with a readable claim.

    The discovery and genome decoding live in `evolve.winners()` so the report set and
    the forward-test suite cannot decode one genome two different ways. This adds only
    the prose.
    """
    from ..evolve.engine import winners
    from ..strategies.genome import alpha_genome, describe_genome

    out = []
    for w in winners():
        claim = (
            f"Evolved, not written. The best of {w['n_population']} individuals in the "
            f"final generation of the search “{w['study']}”, found by a genetic "
            "algorithm over a bounded space of weighted feature ranks. "
            + describe_genome(alpha_genome(w["preset"]), w["vector"]).replace("\n", " "))
        out.append({"name": w["name"], "strategy": w["strategy"], "claim": claim,
                    "study": w["study"]})
    return out


# --------------------------------------------------------------------------
# The roster: what the two report sets are sets OF
# --------------------------------------------------------------------------

#: How many genetic-algorithm winners the report sets carry: the best `GA_WINNERS_SHOWN`
#: searches on disk, ranked by the research Sharpe of each study's best logged run. Every
#: search still exists in `data/experiments/evolve/` and in the registry; the sets show
#: the top of the list so the folder stays readable (ADR-045).
GA_WINNERS_SHOWN = 3


def roster(group: str = "all") -> tuple[str, ...]:
    """The hand-written algorithms a report set is a set OF.

    `all` is every built-in strategy PLUS the `custom` group. That is different from the
    engine's `GROUPS["all"]`, which deliberately leaves custom strategies off the
    scoreboard everyone else is measured against: the report set is where a strategy of
    your own is meant to be read next to the rest. Any other group name is taken from
    `GROUPS`; anything else is a comma-separated list of strategy names.
    """
    from ..strategies import GROUPS
    if group == "all":
        base = tuple(GROUPS["all"])
        return base + tuple(n for n in GROUPS["custom"] if n not in base)
    if group in GROUPS:
        return tuple(GROUPS[group])
    return tuple(n.strip() for n in str(group).split(",") if n.strip())


def ga_winners(limit: int | None = GA_WINNERS_SHOWN) -> list[dict]:
    """The winners of the best `limit` genetic-algorithm searches on disk.

    Ranked by the research-window Sharpe of each study's best logged run, so "top" means
    what the search itself was maximising. `None` means every search; 0 means none.
    Same entries as `evolved_winners()`: name, strategy, claim, study.
    """
    if limit is not None and limit <= 0:
        return []
    ranked = sorted(evolved_winners(), key=lambda w: -_best_sharpe(w["study"]))
    return ranked if limit is None else ranked[:limit]


def _best_sharpe(study: str) -> float:
    """The Sharpe of a study's best logged run; -inf when the registry has none."""
    from ..backtest import registry
    try:
        best = registry.best(study)
    except Exception:                                             # noqa: BLE001
        return float("-inf")
    value = maybe_float(best.get("sharpe")) if best is not None else None
    return float("-inf") if value is None else value


def forward_roster(records, group: str = "all",
                   ga: int | None = GA_WINNERS_SHOWN) -> list[str]:
    """The roster, restricted to what the forward store actually holds.

    The forward set shows the SAME algorithms as the backtest set. Nothing that was
    forward-tested leaves the record - the index still counts every look - but a page is
    written only for a roster member, and `report forward` names the roster members that
    have no forward record yet.
    """
    names = list(roster(group)) + [w["name"] for w in ga_winners(ga)]
    present = set(records["strategy"].astype(str)) if len(records) else set()
    out: list[str] = []
    for n in names:
        if n in present and n not in out:
            out.append(n)
    return out


def claim_of(strategy) -> str:
    """The first paragraph of a strategy's own docstring, as prose.

    Taken from the source rather than restated in the report, so the two cannot drift.
    A strategy whose docstring stops explaining what it claims will produce a report
    that stops explaining it too, which is the correct failure.
    """
    doc = (type(strategy).__doc__ or "").strip()
    if not doc:
        return ""
    para = doc.split("\n\n")[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())


def claim_for(name: str) -> str:
    """What this candidate claims, in its own words.

    A registered strategy's claim is its docstring, taken from the source so the two
    cannot drift. An evolved winner has no docstring - it was never written by anyone -
    so its claim is the decoded genome, which is the same text `report backtest` gives it and
    is the reason the search space is bounded to something readable (ADR-031).
    """
    from ..backtest.strategy import get_strategy
    try:
        return claim_of(get_strategy(name))
    except Exception:                                             # noqa: BLE001
        pass
    return next((w["claim"] for w in evolved_winners() if w["name"] == name), "")


def feature_panel():
    try:
        from ..features import build_features
        return build_features()
    except Exception:                                             # noqa: BLE001
        return None


def feature_coverage():
    fp = feature_panel()
    return fp.coverage() if fp is not None else None


def deflation_for(result) -> dict | None:
    """The deflated Sharpe for this run's study, if it was logged."""
    from ..backtest import registry
    run_id = result.config.get("run_id")
    if not run_id:
        return None
    try:
        return registry.deflate(run_id)
    except (KeyError, ValueError, TypeError):
        return None


def exits_for(record):
    """Forced exits for a run, if its full result was saved to disk.

    The registry stores counts, not the rows. A run saved with `--save` has them; one
    that was only logged does not, and the report simply omits the table rather than
    inventing it.
    """
    return None


def forward_trades_csv(subset, cost_model: str):
    """Path to the saved forward trade ledger for one candidate, if it was saved."""
    rows = subset[subset["cost_model"] == cost_model]
    if rows.empty:
        return None
    saved = str(rows.sort_values("logged_at").iloc[-1].get("saved_to") or "")
    if not saved:
        return None
    path = Path(saved) / "trades.csv"
    return str(path) if path.exists() else None


def paragraphs(obj) -> list[str]:
    """A class docstring as whitespace-collapsed paragraphs."""
    doc = (obj.__doc__ or "").strip()
    if not doc:
        return []
    out = []
    for para in doc.split("\n\n"):
        text = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if text:
            out.append(text)
    return out


def research_row(df, strategy: str, cost_model: str):
    """Latest research-window registry row for one (strategy, cost) pair, or None."""
    if df.empty:
        return None
    hit = df[(df["strategy"] == strategy) & (df["cost_model"] == cost_model)
             & (df["holdout_mode"] == "exclude") & (df["end"] == RESEARCH_END)]
    if hit.empty:
        return None
    return hit.sort_values("logged_at").iloc[-1]


def row_stats(row) -> dict:
    return {"cagr": float(row["cagr"]), "sharpe": float(row["sharpe"]),
            "maxdd": float(row["max_drawdown"]),
            "turnover": None if row.get("ann_turnover") is None
            else float(row["ann_turnover"]),
            "cost_drag": None if row.get("cost_drag") is None
            else float(row["cost_drag"])}


def bench_stats(window_cache: dict, start: str, end: str) -> dict | None:
    """SPY's own daily statistics over exactly [start, end], cached per window."""
    key = (start, end)
    if key in window_cache:
        return window_cache[key]
    from ..backtest import metrics
    from ..backtest.benchmark import benchmark_total_return
    try:
        series = benchmark_total_return("SPY")
        sliced = series[(series.index >= start) & (series.index <= end)].dropna()
        perf = metrics.compute(sliced)
        out = {"cagr": perf.cagr, "sharpe": perf.sharpe}
    except Exception:                                             # noqa: BLE001
        out = None
    window_cache[key] = out
    return out


def deflation_from(df, row) -> dict | None:
    """registry.deflate, computed from an already-loaded frame (one parse, not N)."""
    import numpy as np

    from ..backtest import metrics

    study = row.get("study") or ""
    if not any(str(study).startswith(p) for p in DEFLATABLE_PREFIXES):
        return None
    trials = df[df["study"] == study].drop_duplicates("fingerprint")
    n_trials = len(trials)
    spread = float(trials["sharpe_monthly"].dropna().std(ddof=1)) \
        if len(trials) > 1 else 0.0
    sr_m, n_months = float(row["sharpe_monthly"]), int(row["n_months"])
    if not np.isfinite(sr_m) or n_months < 6 or n_trials < 2:
        return None
    sr_obs = sr_m / np.sqrt(12.0)
    full_kurt = float(row["kurtosis_monthly"]) + 3.0
    return {"study": study, "n_trials": n_trials,
            "deflated_sharpe": metrics.deflated_sharpe(
                sr_obs, n_months, float(row["skew_monthly"]), full_kurt,
                n_trials, spread / np.sqrt(12.0))}


def forward_lookup() -> dict:
    """{strategy: forward info dict} for the realistic cost setting, latest look."""
    try:
        from ..forward import store
        records = store.load()
    except Exception:                                             # noqa: BLE001
        return {}
    if records is None or len(records) == 0:
        return {}
    real = records[records["cost_model"] == "realistic"]
    out = {}
    for _, r in real.iterrows():
        out[str(r["strategy"])] = {
            "verdict": str(r.get("verdict", "")),
            "seal_mode": str(r.get("seal_mode", "")),
            "research_sharpe": maybe_float(r.get("research_sharpe")),
            "forward_sharpe": maybe_float(r.get("forward_sharpe")),
            "forward_d_sharpe": maybe_float(r.get("forward_d_sharpe")),
            "decay_z": maybe_float(r.get("decay_z")),
        }
    return out


def maybe_float(v):
    import numpy as np
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def monthly_entries(df, window_cache, forward, curves_wanted):
    """AlgorithmEntry list for every registered monthly strategy, running any that
    have no research-window rows yet (logged under study 'reports', like cmd_backtest)."""
    from ..backtest import run_backtest
    from ..backtest.strategy import get_strategy
    from ..strategies import GROUPS
    from .algorithms_view import AlgorithmEntry

    entries = []
    for group, family in BOOK_FAMILIES.items():
        for name in GROUPS[group]:
            if name == "spy_buy_hold":
                continue
            rows = {}
            for cost in ("optimistic", "realistic", "pessimistic"):
                row = research_row(df, name, cost)
                if row is None:
                    res = run_backtest(name, costs=cost, study="reports",
                                       record_trades=False,
                                       notes="algorithm book fill-in")
                    row = research_row(reload_registry(), name, cost)
                    if row is None:                       # registry disabled
                        row = row_from_result(res)
                rows[cost] = row
            real = rows["realistic"]
            paras = paragraphs(type(get_strategy(name)))
            entry = AlgorithmEntry(
                name=name, family=family, engine="monthly",
                origin="written",
                claim=paras[0] if paras else "",
                explain=paras[1:3],
                window=f"{str(real['start'])[:7]} → {str(real['end'])[:7]}",
                settings={c: row_stats(r) for c, r in rows.items()},
                bench=bench_stats(window_cache, str(real["start"]),
                                   str(real["end"])),
                deflation=deflation_from(df, real),
                forward=forward.get(name),
                href=f"../{BACKTEST_REPORTS_DIR.name}/{page_href(name)}",
            )
            curves_wanted[name] = real.get("run_id")
            entries.append(entry)
    return entries


def reload_registry():
    from ..backtest import registry
    return registry.load()


def row_from_result(res):
    """A registry-row-alike for when trial logging is disabled (tests)."""
    import pandas as pd
    p = res.performance
    return pd.Series({
        "start": res.config.get("start", ""), "end": res.config.get("end", ""),
        "cagr": p.cagr, "sharpe": p.sharpe, "max_drawdown": p.max_drawdown,
        "ann_turnover": p.ann_turnover, "cost_drag": p.cost_drag,
        "sharpe_monthly": float("nan"), "n_months": 0, "skew_monthly": float("nan"),
        "kurtosis_monthly": float("nan"), "study": "", "run_id": None,
    })


def evolved_entries(df, window_cache, forward, curves_wanted):
    from .algorithms_view import AlgorithmEntry

    entries = []
    for w in evolved_winners():
        name = w["name"]
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = research_row(df, name, cost)
            if row is None:
                from ..backtest import run_backtest
                strat = w["strategy"]
                strat.name = name
                run_backtest(strat, costs=cost, study=w["study"],
                             record_trades=False,
                             notes="algorithm book: winner re-run, same fingerprint "
                                   "as its trial")
                row = research_row(reload_registry(), name, cost)
            if row is not None:
                rows[cost] = row
        if "realistic" not in rows:
            continue
        real = rows["realistic"]
        entries.append(AlgorithmEntry(
            name=name, family="Evolved by the genetic algorithm", engine="monthly",
            origin="searched",
            claim=w["claim"],
            window=f"{str(real['start'])[:7]} → {str(real['end'])[:7]}",
            settings={c: row_stats(r) for c, r in rows.items()},
            bench=bench_stats(window_cache, str(real["start"]), str(real["end"])),
            deflation=deflation_from(df, real) or study_deflation(w["study"]),
            forward=forward.get(name),
        ))
        curves_wanted[name] = real.get("run_id")
    return entries


def study_deflation(study: str) -> dict | None:
    from ..backtest import registry
    try:
        d = registry.deflate_best(study)
    except (KeyError, ValueError):
        return None
    if not d or d.get("n_trials") in (None, 0):
        return None
    return {"study": study, "n_trials": d["n_trials"],
            "deflated_sharpe": d.get("deflated_sharpe")}


def timing_entries(df, forward, curves_wanted):
    """Calendar rules as book entries, benched against tm_buy_hold's own row."""
    from ..timing.data import load_timing_data
    from ..timing.engine import run_timing_backtest
    from ..timing.strategies import TIMING_GROUPS, get_timing_strategy
    from .algorithms_view import AlgorithmEntry

    data = load_timing_data()
    lo = data.date_index("2007-04-02", side="next")
    hi = data.date_index(RESEARCH_END, side="prev")

    def rows_for(name):
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = research_row(df, name, cost)
            if row is None:
                run_timing_backtest(name, costs=cost, study="reports",
                                    notes="algorithm book fill-in")
                row = research_row(reload_registry(), name, cost)
            rows[cost] = row
        return rows

    bench_rows = rows_for("tm_buy_hold")
    bench_real = bench_rows["realistic"]
    bench = ({"cagr": float(bench_real["cagr"]),
              "sharpe": float(bench_real["sharpe"])}
             if bench_real is not None else None)

    entries = []
    for name in TIMING_GROUPS["all"]:
        rows = bench_rows if name == "tm_buy_hold" else rows_for(name)
        real = rows["realistic"]
        if real is None:
            continue
        strat = get_timing_strategy(name)
        on, intra = strat.legs(data)
        exposure = float((on[lo:hi] | intra[lo:hi]).mean())
        paras = paragraphs(type(strat))
        entries.append(AlgorithmEntry(
            name=name, family="The calendar rules", engine="daily legs",
            origin="rule",
            claim=paras[0] if paras else "",
            explain=paras[1:3],
            window=f"{str(real['start'])[:7]} → {str(real['end'])[:7]}",
            settings={c: row_stats(r) for c, r in rows.items() if r is not None},
            bench=bench,
            deflation=deflation_from(df, real),
            forward=forward.get(name),
            href="timing.html",
            exposure=f"In the market {exposure:.0%} of sessions.",
        ))
        curves_wanted[name] = real.get("run_id")
    return entries


def ga_summary(df, forward) -> dict:
    """The one-panel GA story: every registered search, plus the newest winner."""
    from ..backtest import registry

    searches = []
    night_winner = ""
    try:
        studies = registry.studies()
    except Exception:                                             # noqa: BLE001
        return {}
    ga_studies = [s for s in studies["study"].tolist()
                  if str(s).startswith("ga-")]
    for study in sorted(ga_studies):
        try:
            d = registry.deflate_best(study)
            best = registry.best(study)
        except (KeyError, ValueError):
            continue
        if best is None or not d.get("n_trials"):
            continue
        winner_name = f"{study}-best"
        fwd = forward.get(winner_name, {})
        searches.append({
            "study": study,
            "preset": preset_of(study),
            "n_trials": d.get("n_trials"),
            "window": f"{str(best['start'])[:7]} → {str(best['end'])[:7]}",
            "cagr": float(best["cagr"]),
            "sharpe_monthly": d.get("sharpe_annualised_monthly"),
            "expected_max": d.get("expected_max_sharpe_annualised"),
            "deflated_sharpe": d.get("deflated_sharpe"),
            "forward": fwd.get("verdict", ""),
        })
    try:
        from ..evolve.engine import winners
        from ..strategies.genome import alpha_genome, describe_genome
        for w in winners():
            if w["study"] == "ga-night-1":
                text = describe_genome(alpha_genome(w["preset"]), w["vector"])
                night_winner = (
                    "The night-preset search (price features plus the "
                    "overnight/intraday decomposition and the dividend calendar) "
                    "converged on: " + " ".join(text.split()))
    except Exception:                                             # noqa: BLE001
        pass
    return {"searches": searches, "night_winner": night_winner}


def preset_of(study: str) -> str:
    try:
        from ..evolve.engine import winners
        for w in winners():
            if w["study"] == study:
                return w["preset"]
    except Exception:                                             # noqa: BLE001
        pass
    return ""


def book_curves(entries, curves_wanted) -> dict:
    """Best curve per family plus SPY - a legible chart, not a haystack."""
    from ..backtest import registry

    best_by_family: dict = {}
    for e in entries:
        d = e.d_sharpe
        if d is None:
            continue
        cur = best_by_family.get(e.family)
        if cur is None or d > cur[0]:
            best_by_family[e.family] = (d, e.name)
    names = [name for _, name in best_by_family.values()]
    run_ids = {curves_wanted.get(n): n for n in names if curves_wanted.get(n)}
    if not run_ids:
        return {}
    stored = registry.load_curves(list(run_ids))
    curves = {}
    spy = None
    for rid, frame in stored.items():
        curves[run_ids[rid]] = frame["nav"]
        if spy is None and "benchmark" in frame.columns:
            spy = frame["benchmark"]
    if spy is not None:
        curves["SPY"] = spy
    return curves


def forward_context() -> dict:
    out = {}
    try:
        from ..forward import store
        from ..forward.windows import describe_power
        out["power_note"] = describe_power(54)
        bar = store.selection_bar()
        if bar and bar.get("n_forward_tests"):
            out["selection_note"] = (
                f"{bar['n_forward_tests']} candidates have now been forward-tested "
                f"under {bar['cost_model']} costs. The luckiest of that many "
                f"worthless strategies would post a forward Sharpe of about "
                f"{bar['bar']:.2f} over this window - any forward result below that "
                "bar is indistinguishable from selection. The bar rises with every "
                "candidate added, which is the honest cost of testing more ideas.")
    except Exception:                                             # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------
# The two composed pages
# --------------------------------------------------------------------------

def algorithm_book() -> dict:
    """Everything the Algorithm Book view needs, in one call.

    Reloads the registry between stages because `monthly_entries` and `evolved_entries`
    append fill-in runs, and the deflation for a later entry has to see them.
    """
    from ..backtest import registry

    df = registry.load()
    window_cache: dict = {}
    curves_wanted: dict = {}
    forward = forward_lookup()

    entries = monthly_entries(df, window_cache, forward, curves_wanted)
    df = reload_registry()           # fill-in runs may have appended
    entries += evolved_entries(df, window_cache, forward, curves_wanted)
    df = reload_registry()
    entries += timing_entries(df, forward, curves_wanted)

    return {
        "entries": entries,
        "curves": book_curves(entries, curves_wanted),
        "ga": ga_summary(df, forward),
        "forward_context": forward_context(),
    }


def calendar_lab() -> dict:
    """Everything the Calendar Lab view needs: the acceptance checks, every rule
    costed three ways, the gross and net leg curves, and the member decomposition."""
    from ..backtest import registry
    from ..timing.data import load_timing_data
    from ..timing.decompose import decompose_members, summarise
    from ..timing.engine import run_timing_backtest, timing_accept
    from ..timing.strategies import TIMING_GROUPS, get_timing_strategy

    accept = timing_accept()
    df = registry.load()
    data = load_timing_data()
    lo = data.date_index("2007-04-02", side="next")
    hi = data.date_index(RESEARCH_END, side="prev")

    rules, run_ids = [], {}
    bench_sharpe = None
    for name in TIMING_GROUPS["all"]:
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = research_row(df, name, cost)
            if row is None:
                run_timing_backtest(name, costs=cost, study="reports",
                                    notes="calendar lab fill-in")
                df = reload_registry()
                row = research_row(df, name, cost)
            if row is not None:
                rows[cost] = row
        real = rows.get("realistic")
        if real is None:
            continue
        if name == "tm_buy_hold":
            bench_sharpe = float(real["sharpe"])
        strat = get_timing_strategy(name)
        on, intra = strat.legs(data)
        paras = paragraphs(type(strat))
        rules.append({
            "name": name,
            "claim": paras[0] if paras else "",
            "explain": paras[1:2],
            "exposure": f"{float((on[lo:hi] | intra[lo:hi]).mean()):.0%}",
            "settings": {c: row_stats(r) for c, r in rows.items()},
            "_sharpe": float(real["sharpe"]),
        })
        run_ids[name] = real.get("run_id")
    for r in rules:
        r["d_sharpe"] = (r.pop("_sharpe") - bench_sharpe
                         if bench_sharpe is not None else None)

    stored = registry.load_curves([rid for rid in run_ids.values() if rid])
    by_name = {name: stored.get(rid) for name, rid in run_ids.items() if rid}
    gross_curves, net_curves = {}, {}
    labels = {"tm_buy_hold": "buy & hold", "tm_overnight": "overnight (close→open)",
              "tm_intraday": "intraday (open→close)"}
    for name, label in labels.items():
        frame = by_name.get(name)
        if frame is None:
            continue
        gross_curves[label] = frame["nav_gross"] if "nav_gross" in frame.columns             else frame["nav"]
        net_curves[label] = frame["nav"]
    for name in ("tm_turn_of_month", "tm_sell_in_may", "tm_vix_overnight"):
        frame = by_name.get(name)
        if frame is not None:
            net_curves[name.replace("tm_", "")] = frame["nav"]

    members = decompose_members(start="2007-04-01", end=RESEARCH_END)
    return {"accept": accept, "rules": rules,
            "gross_curves": gross_curves, "net_curves": net_curves,
            "members": members, "member_summary": summarise(members)}


# --------------------------------------------------------------------------
# The genetic-algorithm lab (ADR-046)
#
# Everything the three `reports/genetic_algorithm/` pages need, read from the search
# checkpoints, the trial log and the forward store. Nothing here re-runs a search or a
# backtest: a search is thousands of evaluations and its record is already on disk.
# --------------------------------------------------------------------------

#: Filenames the lab writes. One place, because three pages cross-link to each other.
GENETIC_PAGES = {
    "methodology": "methodology.html",
    "features": "features.html",
    "searches": "evolved-algorithms.html",
}


def genetic_lab() -> dict:
    """Everything the genetic-algorithm pages show, in one call.

    Returns `searches` (one entry per search with a checkpoint on disk, best research
    Sharpe first), `registry_only` (searches the trial log remembers but whose
    checkpoint is gone, so no winner can be decoded), `presets` and `genome`.
    """
    from ..strategies.genome import PRESETS

    searches = [_search_record(study) for study in _checkpointed_studies()]
    searches = [s for s in searches if s is not None]
    searches.sort(key=lambda s: -(s["research"] or {}).get("sharpe", float("-inf")))
    return {
        "searches": searches,
        "registry_only": _registry_only_studies({s["study"] for s in searches}),
        "presets": {name: tuple(features) for name, features in PRESETS.items()},
        "genome": genome_anatomy(),
    }


def _checkpointed_studies() -> list[str]:
    from ..evolve.engine import EVOLVE_DIR
    if not EVOLVE_DIR.exists():
        return []
    return sorted(p.stem for p in EVOLVE_DIR.glob("*.jsonl"))


def _registry_only_studies(known: set[str]) -> list[dict]:
    """GA studies the trial log holds but no checkpoint explains.

    Named rather than dropped. Those trials still count toward every deflated Sharpe in
    their own study, and a reader comparing the page against `experiments studies` has
    to be able to see why one of them has no winner.
    """
    from ..backtest import registry
    try:
        studies = registry.studies()
    except Exception:                                             # noqa: BLE001
        return []
    out = []
    for _, r in studies.iterrows():
        name = str(r["study"])
        if not name.startswith("ga-") or name in known:
            continue
        out.append({"study": name, "runs": int(r.get("runs") or 0),
                    "trials": int(r.get("trials") or 0),
                    "best_sharpe": maybe_float(r.get("best_sharpe")),
                    "deflation": study_deflation(name)})
    return out


def _search_record(study: str) -> dict | None:
    """One search: its settings, its training history, its winner, and what became of it."""
    import json

    import numpy as np

    from ..evolve.engine import EVOLVE_DIR, load_history, load_population, study_preset
    from ..strategies.genome import (DEAD_ZONE, active_features, alpha_genome,
                                     describe_genome)

    path = EVOLVE_DIR / f"{study}.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        config = json.loads(lines[0]).get("config", {})
    except json.JSONDecodeError:
        config = {}

    preset = study_preset(study)
    genome = alpha_genome(preset)
    population = [p for p in load_population(study) if p.get("fitness") is not None]
    if not population:
        return None
    winner = max(population, key=lambda p: p["fitness"])
    vector = np.asarray(winner["vector"], dtype=float)
    decoded = genome.decode(vector)

    weights = sorted(((n[2:], float(v)) for n, v in decoded.items()
                      if n.startswith("w_") and abs(v) >= DEAD_ZONE),
                     key=lambda kv: -abs(kv[1]))
    name = f"{study}-best"
    df = reload_registry()
    row = research_row(df, name, "realistic")
    return {
        "study": study,
        "preset": preset,
        "config": config,
        "history": load_history(study),
        "winner_name": name,
        "winner_fitness": maybe_float(winner.get("fitness")),
        "weights": weights,
        "n_ignored": len(PRESETS_LEN(preset)) - len(weights),
        "active": active_features(genome, vector),
        "portfolio": {k: decoded[k] for k in
                      ("top_k", "weighting", "max_weight", "use_regime",
                       "defensive_gross", "vol_trigger")},
        "prose": describe_genome(genome, vector),
        "usage": _feature_usage(population, genome),
        "n_population": len(population),
        "deflation": study_deflation(study),
        "research": row_stats(row) if row is not None else None,
        "window": (f"{str(row['start'])[:7]} → {str(row['end'])[:7]}"
                   if row is not None else ""),
        "forward": forward_lookup().get(name),
    }


def PRESETS_LEN(preset: str) -> tuple[str, ...]:
    from ..strategies.genome import PRESETS
    return PRESETS[preset]


def _feature_usage(population: list[dict], genome) -> dict[str, int]:
    """How many of the final population put a live weight on each feature.

    The winner is one draw. What the whole surviving population agrees on is the more
    honest statement about what the search actually converged toward, and it is the one
    number on these pages that a single lucky individual cannot move.
    """
    import numpy as np

    from ..strategies.genome import active_features

    counts: dict[str, int] = {}
    for individual in population:
        for feature in active_features(genome,
                                       np.asarray(individual["vector"], dtype=float)):
            counts[feature] = counts.get(feature, 0) + 1
    return counts


def genome_anatomy() -> dict:
    """The search space itself: the portfolio genes, the constants, the defaults.

    Read off `alpha_genome` and `EvolutionConfig` rather than restated, so the
    methodology page cannot drift from the code it describes.
    """
    from ..evolve.engine import EvolutionConfig
    from ..evolve.operators import BLX_ALPHA
    from ..strategies.genome import DEAD_ZONE, PRESETS, WEIGHTINGS, alpha_genome

    genome = alpha_genome("price")
    shape = [g for g in genome.genes if not g.name.startswith("w_")]
    return {
        "dead_zone": DEAD_ZONE,
        "blx_alpha": BLX_ALPHA,
        "weightings": WEIGHTINGS,
        "defaults": EvolutionConfig().as_dict(),
        "shape_genes": [{"name": g.name, "low": g.low, "high": g.high,
                         "integer": g.integer, "choices": g.choices, "note": g.note}
                        for g in shape],
        "sizes": {name: len(alpha_genome(name)) for name in PRESETS},
    }
