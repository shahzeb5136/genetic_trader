"""`sp500lab report ...` — the static HTML report sets, built from the registry.

Kept out of the top-level cli.py so importing the CLI stays cheap; pandas and the
reporting stack load only when a report is actually asked for.

    sp500lab report backtest --open     the research window   -> reports/backtest/
    sp500lab report forward  --open     2022 onward            -> reports/forward/
    sp500lab report genetic  --open     the GA lab             -> reports/genetic_algorithm/

Each set is one folder holding exactly two kinds of file: `index.html`, a scoreboard with
every algorithm's headline statistics, and one self-contained page per algorithm. Both
sets are built from the same roster - every built-in strategy, the `custom` group, and
the winners of the best three genetic-algorithm searches - so `backtest/low-vol.html`
and `forward/low-vol.html` are the same strategy on either side of the boundary
(ADR-045).

`report genetic` writes three pages about the genetic algorithm itself: how the search
works, what it is allowed to read, and every search that has run with its winner decoded
(ADR-046). It runs no backtest - a search is thousands of evaluations and its record is
already on disk.

Everything else is on demand and lands in reports/extra/, never inside the sets:

    sp500lab report strategy low_vol    one strategy, in full, from a fresh run
    sp500lab report features            what every feature is, and does it leak
    sp500lab report algorithms          the Algorithm Book
    sp500lab report timing              the Calendar Lab
    sp500lab report registry            everything tried, with deflated Sharpes
    sp500lab report honesty             coverage, exits and holdout exposure
    sp500lab report study baselines
    sp500lab report run 20260827T182514-586cb2
    sp500lab report compare 20260827T1825-a 20260827T1825-b
    sp500lab report trades momentum_12_1 --open
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from ..paths import (BACKTEST_REPORTS_DIR, EXTRA_REPORTS_DIR, FORWARD_REPORTS_DIR,
                     GENETIC_REPORTS_DIR, REPORTS_DIR)
from . import queries as Q
from .tables import feature_usage
from .util import page_href, slugify

#: Where an on-demand page links back to. Those pages live in reports/extra/, one level
#: beside the two sets, so "all strategies" is the backtest index next door.
_BACKTEST_INDEX = f"../{BACKTEST_REPORTS_DIR.name}/index.html"
_FORWARD_INDEX = f"../{FORWARD_REPORTS_DIR.name}/index.html"
_GENETIC_DIR = f"../{GENETIC_REPORTS_DIR.name}"


def _rel(path: Path) -> str:
    """`reports/backtest`, for help strings."""
    return f"{REPORTS_DIR.name}/{path.name}"


def add_parser(sub) -> None:
    p = sub.add_parser("report", help="build the static HTML report sets from the registry")
    rs = p.add_subparsers(dest="report_command", required=True)

    bt = rs.add_parser(
        "backtest", aliases=["all"],
        help="the research-window set: an index and one page per algorithm, into "
             f"{_rel(BACKTEST_REPORTS_DIR)}/ - start here")
    bt.add_argument("group", nargs="?", default="all",
                    help="all (every built-in strategy plus the custom group) | "
                         "baselines | alpha | frontier | learned | evolved | custom, "
                         "or a comma-separated list of strategy names")
    _strategy_args(bt)
    _set_args(bt, BACKTEST_REPORTS_DIR)
    bt.set_defaults(func=cmd_backtest)

    fw = rs.add_parser(
        "forward",
        help="the out-of-sample set: an index and one page per forward-tested "
             f"algorithm, into {_rel(FORWARD_REPORTS_DIR)}/")
    fw.add_argument("group", nargs="?", default="all",
                    help="same meaning as for `backtest`; only roster members with a "
                         "forward record get a page")
    fw.add_argument("--costs", default="realistic",
                    choices=["optimistic", "realistic", "pessimistic"],
                    help="which cost setting the index leads with. All three are always "
                         "shown on each algorithm's own page.")
    _set_args(fw, FORWARD_REPORTS_DIR)
    fw.set_defaults(func=cmd_forward)

    gen = rs.add_parser(
        "genetic",
        help="the genetic-algorithm lab: how the search works, what it may read, and "
             f"every search with its winner, into {_rel(GENETIC_REPORTS_DIR)}/")
    gen.add_argument("-o", "--out", default=None,
                     help=f"output DIRECTORY (default: {_rel(GENETIC_REPORTS_DIR)}/)")
    gen.add_argument("--open", action="store_true", dest="open_after",
                     help="open the methodology page in a browser when the set is written")
    gen.set_defaults(func=cmd_genetic)

    st2 = rs.add_parser("strategy", help="one strategy, in full, from a live run")
    st2.add_argument("strategy")
    _strategy_args(st2)
    _shared(st2)
    st2.set_defaults(func=cmd_strategy)

    ft = rs.add_parser("features",
                       help="what every feature is, and whether it reads the future")
    ft.add_argument("--no-check", action="store_true",
                    help="skip the leakage check (it rebuilds the panel twice)")
    _shared(ft)
    ft.set_defaults(func=cmd_features)

    st = rs.add_parser("study", help="compare every run in one study")
    st.add_argument("study")
    _shared(st)
    st.set_defaults(func=cmd_study)

    rn = rs.add_parser("run", help="deep dive on a single run")
    rn.add_argument("run_id", nargs="?", default=None,
                    help="run id; omit to use the best run of --study")
    rn.add_argument("--study", default=None, help="pick the best run of this study")
    _shared(rn)
    rn.set_defaults(func=cmd_run)

    rg = rs.add_parser("registry", help="everything tried, with deflated Sharpes")
    _shared(rg)
    rg.set_defaults(func=cmd_registry)

    cp = rs.add_parser("compare", help="compare specific runs or strategies")
    cp.add_argument("targets", nargs="+",
                    help="run ids, or strategy names (latest run of each is used)")
    _shared(cp)
    cp.set_defaults(func=cmd_compare)

    tr = rs.add_parser(
        "trades",
        help="run a strategy and publish its orders as a downloadable page")
    tr.add_argument("strategy")
    tr.add_argument("--start", default="2007-04-01")
    tr.add_argument("--end", default=None)
    tr.add_argument("--costs", default="realistic",
                    choices=["optimistic", "realistic", "pessimistic", "free"])
    tr.add_argument("--capital", type=float, default=100_000.0)
    tr.add_argument("--top-k", type=int, default=None)
    tr.add_argument("--holdout", default="exclude",
                    choices=["exclude", "include", "only"])
    tr.add_argument("--no-log", action="store_true")
    _shared(tr)
    tr.set_defaults(func=cmd_trades)

    hs = rs.add_parser("honesty", help="coverage, exits and holdout exposure")
    _shared(hs)
    hs.set_defaults(func=cmd_honesty)

    ab = rs.add_parser(
        "algorithms",
        help="the Algorithm Book: every competitor explained in its own words, then "
             "scored against the index over its own window, on one page")
    _shared(ab)
    ab.set_defaults(func=cmd_algorithms)

    tl = rs.add_parser(
        "timing",
        help="the Calendar Lab: the overnight/intraday decomposition and every "
             "calendar rule, costed three ways")
    _shared(tl)
    tl.set_defaults(func=cmd_timing)


def _strategy_args(parser) -> None:
    """The flags that decide what a strategy report is a report OF."""
    parser.add_argument("--start", default="2007-04-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--costs", default="realistic",
                        choices=["optimistic", "realistic", "pessimistic"],
                        help="which setting the headline uses; all three are always "
                             "reported")
    parser.add_argument("--holdout", default="exclude",
                        choices=["exclude", "include", "only"],
                        help="'exclude' stops before the reserved 2022 holdout. The "
                             "other two are permanently recorded.")
    parser.add_argument("--study", default="reports",
                        help="study to log these runs under")
    parser.add_argument("--no-log", action="store_true",
                        help="do not record these runs as trials")


def _set_args(parser, default_dir: Path) -> None:
    """The flags both sets share: which GA winners, where to write, whether to open."""
    parser.add_argument("--ga-winners", type=int, default=Q.GA_WINNERS_SHOWN, metavar="N",
                        help="how many genetic-algorithm winners to include, best "
                             f"research Sharpe first (default {Q.GA_WINNERS_SHOWN})")
    parser.add_argument("--no-evolved", action="store_true",
                        help="include no genetic-algorithm winners")
    parser.add_argument("-o", "--out", default=None,
                        help=f"output DIRECTORY (default: {_rel(default_dir)}/)")
    parser.add_argument("--open", action="store_true", dest="open_after",
                        help="open the index in a browser when the set is written")


def _shared(parser) -> None:
    parser.add_argument("-o", "--out", default=None,
                        help=f"output path (default: {_rel(EXTRA_REPORTS_DIR)}/<name>.html)")
    parser.add_argument("--open", action="store_true", dest="open_after",
                        help="open the report in a browser when it is written")


# --------------------------------------------------------------------------
# The two sets
# --------------------------------------------------------------------------

def cmd_backtest(args) -> int:
    """Publish the research-window set: one page per algorithm, and an index.

    One command and one folder, because the point of the set is that it is a set: a
    folder of pages that open in a browser and link to each other is the closest this
    project gets to a product, and it is the answer to "I do not want to read the code".

    The folder holds exactly two kinds of file, the index and the algorithm pages, and
    nothing else (ADR-045). Everything that used to sit beside them - the feature layer,
    the registry, the honesty panel, the Algorithm Book, the Calendar Lab, the trade
    CSVs - is still one command away and lands in reports/extra/.
    """
    from . import index_report, strategy_report
    from .render.html import write as write_html
    from .views import _headline_claim

    out_dir = Path(args.out or BACKTEST_REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [{"name": n} for n in Q.roster(args.group)]
    found = Q.ga_winners(_ga_count(args))
    if found:
        print(f"  including {len(found)} genetic-algorithm winner(s): "
              + ", ".join(w["name"] for w in found))
    targets += found

    specs, curves, failed, written = [], {}, [], []
    for target in targets:
        name = target["name"]
        try:
            opts = _run_opts(args)
            opts["study"] = target.get("study") or opts["study"]
            result, others, bench, claim, cov, deflation = Q.build_strategy(
                name, strategy=target.get("strategy"),
                claim=target.get("claim"), **opts)
        except Exception as exc:                                  # noqa: BLE001
            failed.append(f"{name}: {exc}")
            continue
        href = page_href(name)
        page = write_html(strategy_report(result, results_by_cost=others,
                                          benchmark=bench, claim=claim,
                                          feature_coverage=cov, deflation=deflation),
                          out_dir / href)
        written.append(page)
        p = result.performance
        specs.append({
            "name": name, "href": href,
            "claim": _headline_claim(claim),
            "evolved": "strategy" in target,
            "n_trials": (deflation or {}).get("n_trials"),
            "deflated_sharpe": (deflation or {}).get("deflated_sharpe"),
            "window": f"{str(result.config.get('start'))[:7]}–"
                      f"{str(result.config.get('end'))[:7]}",
            "cagr": p.cagr, "ann_vol": p.ann_vol, "sharpe": p.sharpe,
            "max_drawdown": p.max_drawdown,
            "d_sharpe": (p.sharpe - bench.sharpe) if bench else None,
        })
        curves[name] = result.equity.dropna()
        print(f"  {name:20s} {p.cagr * 100:6.2f}%  Sharpe {p.sharpe:5.2f}  "
              f"-> {page.name}")

    if not specs:
        for line in failed:
            print(f"  FAILED {line}", file=sys.stderr)
        return 1
    specs.sort(key=lambda s: (s["d_sharpe"] is not None, s["d_sharpe"] or -99),
               reverse=True)

    cards = _sibling_cards(
        args,
        (_FORWARD_INDEX, "Forward tests: 2022 onward →",
         "What the same algorithms did after the research window ended - prediction "
         "against outcome."),
        _GENETIC_CARD)
    index = index_report(specs, curves=curves, extra_cards=cards,
                         subtitle=f"{len(specs)} algorithms, {args.costs} costs, each "
                                  "scored against the index over its own window.")
    path = write_html(index, out_dir / "index.html")
    written.append(path)
    if args.out is None:
        _prune(out_dir, written)

    print()
    print(f"  {len(specs)} algorithm page(s) + index -> {out_dir}")
    print(f"  open {path}")
    if failed:
        print()
        for line in failed:
            print(f"  FAILED {line}", file=sys.stderr)
    if args.open_after:
        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_forward(args) -> int:
    """Publish the forward-test set: one page per forward-tested algorithm, and an index.

    Same shape as the backtest set, same roster, same file names, so a reader can flip
    between `backtest/low-vol.html` and `forward/low-vol.html`. A roster member with no
    forward record yet gets no page, and the command says which those are.

    Nothing here runs a backtest or reads the panel. Every number comes out of
    `forward/store.py`, which is the guarantee ADR-034 was written to give - a forward
    report has to be rebuildable years later from the record alone.
    """
    from ..forward import store
    from . import forward_index_report, forward_strategy_report
    from .forward_views import PRIMARY_COSTS, primary_rows, strategy_href
    from .render.html import write as write_html

    records = store.load()
    if records.empty:
        print("No forward test has been run, so there is nothing to report.")
        print("  python -m sp500lab forward suite all --seal-first")
        return 1

    out_dir = Path(args.out or FORWARD_REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = args.costs or PRIMARY_COSTS

    names = Q.forward_roster(records, args.group, ga=_ga_count(args))
    tested = set(records["strategy"].astype(str))
    untested = [n for n in Q.roster(args.group) if n not in tested]
    rows = primary_rows(records, costs)
    rows = rows[rows["strategy"].astype(str).isin(names)]
    if rows.empty:
        print("  none of the roster has a forward record yet.", file=sys.stderr)
        return 1

    print(f"Building the forward set for {len(rows)} algorithm(s) [{costs} costs] "
          f"into {out_dir}")
    written = []
    for name in rows["strategy"].astype(str):
        subset = records[records["strategy"] == name]
        report = forward_strategy_report(
            subset, cost_model=costs, claim=Q.claim_for(name),
            trades_csv=Q.forward_trades_csv(subset, costs))
        page = write_html(report, out_dir / strategy_href(name))
        written.append(page)
        print(f"  {name:24s} {page.name}")

    cards = _sibling_cards(
        args,
        (_BACKTEST_INDEX, "← Backtests: the research window",
         "The same algorithms on the data they were built against, one page each."),
        _GENETIC_CARD)
    index = forward_index_report(records, cost_model=costs,
                                 roster=list(rows["strategy"].astype(str)),
                                 extra_cards=cards)
    path = write_html(index, out_dir / "index.html")
    written.append(path)
    if args.out is None:
        _prune(out_dir, written)

    print()
    print(f"  {len(written) - 1} algorithm page(s) + index -> {out_dir}")
    if untested:
        print(f"  no forward record yet, so no page: {', '.join(untested)}")
    print(f"  open {path}")
    if args.open_after:
        webbrowser.open(path.resolve().as_uri())
    return 0


def _ga_count(args) -> int:
    return 0 if getattr(args, "no_evolved", False) else int(args.ga_winners)


#: The card every set index carries pointing at the genetic-algorithm lab. The evolved
#: winners are rows on both scoreboards, and this is where they came from.
_GENETIC_CARD = (f"{_GENETIC_DIR}/{Q.GENETIC_PAGES['methodology']}",
                 "The genetic algorithm",
                 "How the search works, what it is allowed to read, and every search "
                 "with its winning genome decoded.")


def _sibling_cards(args, *specs: tuple[str, str, str]) -> list[dict]:
    """Cards linking one set's index to its siblings.

    Only when the set is written to its default folder: that is the only case in which a
    sibling's location is known. A set written somewhere else with -o stands alone.
    """
    if args.out is not None:
        return []
    return [{"href": href, "title": title, "blurb": blurb}
            for href, title, blurb in specs]


def _prune(out_dir: Path, keep: list) -> list:
    """Remove pages an earlier build left behind, so the folder describes THIS build.

    Only `*.html` at the top of the folder, and only called for the default folder the
    set owns - a folder the user pointed at with -o is theirs, and nothing in it is
    touched.
    """
    keep_names = {Path(p).name for p in keep}
    stale = sorted(p for p in Path(out_dir).glob("*.html") if p.name not in keep_names)
    for p in stale:
        p.unlink()
    if stale:
        print(f"  removed {len(stale)} page(s) from an earlier build: "
              + ", ".join(p.name for p in stale))
    return stale


# --------------------------------------------------------------------------
# The genetic-algorithm lab
# --------------------------------------------------------------------------

def cmd_genetic(args) -> int:
    """Write the three genetic-algorithm pages.

    Three pages and no index: a fourth page whose only content is three links is not
    navigation, it is a file. Each page carries a link grid to the other two and back to
    the backtest scoreboard, which is where an evolved winner sits beside everything it
    was competing against.

    Nothing here runs a search or a backtest. Every figure comes from the checkpoints in
    `data/experiments/evolve/`, the trial log, and the forward store - so the pages
    rebuild in a second and a search never has to be repeated to be reported.
    """
    from datetime import datetime, timezone

    from .genetic_views import features_report, methodology_report, searches_report
    from .render.html import write as write_html

    lab = Q.genetic_lab()
    if not lab["searches"] and not lab["registry_only"]:
        print("No genetic-algorithm search has been run, so there is nothing to report.")
        print("  python -m sp500lab evolve run --study ga-1")
        return 1

    out_dir = Path(args.out or GENETIC_REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    hrefs = dict(Q.GENETIC_PAGES)
    hrefs["backtest"] = (_BACKTEST_INDEX if args.out is None
                         else f"{BACKTEST_REPORTS_DIR.name}/index.html")

    from ..features.catalog import describe
    pages = [
        ("methodology", methodology_report(lab["genome"], lab["searches"],
                                           generated_at=stamp, hrefs=hrefs)),
        ("features", features_report(lab["presets"], lab["searches"], describe,
                                     generated_at=stamp, hrefs=hrefs)),
        ("searches", searches_report(lab["searches"], lab["registry_only"],
                                     generated_at=stamp, hrefs=hrefs)),
    ]
    print(f"Building the genetic-algorithm lab for {len(lab['searches'])} search(es) "
          f"into {out_dir}")
    written = []
    for key, report in pages:
        page = write_html(report, out_dir / Q.GENETIC_PAGES[key])
        written.append(page)
        print(f"  {report.title:36s} {page.name}")
    if args.out is None:
        _prune(out_dir, written)

    start = out_dir / Q.GENETIC_PAGES["methodology"]
    print()
    print(f"  {len(written)} page(s) -> {out_dir}")
    if lab["registry_only"]:
        print("  no checkpoint on disk, so no winner: "
              + ", ".join(r["study"] for r in lab["registry_only"]))
    print(f"  open {start}")
    if args.open_after:
        webbrowser.open(start.resolve().as_uri())
    return 0


# --------------------------------------------------------------------------
# On demand: single pages, into reports/extra/
# --------------------------------------------------------------------------

def cmd_strategy(args) -> int:
    """Run one strategy and publish everything about it."""
    from . import strategy_report
    result, others, bench, claim, coverage, deflation = Q.build_strategy(
        args.strategy, **_run_opts(args))
    report = strategy_report(result, results_by_cost=others, benchmark=bench, claim=claim,
                             feature_coverage=coverage, deflation=deflation,
                             index_href=_BACKTEST_INDEX)
    print(f"  {result.strategy}: CAGR {result.performance.cagr * 100:.2f}%  "
          f"Sharpe {result.performance.sharpe:.2f}  "
          f"{len(result.trades):,} orders")
    return _write(report, args, f"strategy-{slugify(args.strategy)}")


def cmd_features(args) -> int:
    """Publish the feature-layer report."""
    from . import feature_report
    fp = Q.feature_panel()
    if fp is None:
        print("  no feature panel. `python -m sp500lab features build`", file=sys.stderr)
        return 1
    leakage = None
    if not args.no_check:
        from ..features import check_leakage
        print("  running the leakage check (rebuilds the panel twice)…")
        leakage = check_leakage()
        print(f"  {'PASS' if leakage['ok'] else 'FAIL'}: "
              f"{len(leakage['features'])} features compared")
    usage = feature_usage(Q.roster("all"))
    return _write(feature_report(fp, leakage=leakage, usage=usage,
                                 index_href=_BACKTEST_INDEX), args, "features")


def cmd_study(args) -> int:
    from ..backtest import registry
    from . import comparison_report
    runs = registry.load(args.study)
    if runs.empty:
        return _no_runs(f"study {args.study!r}")
    report = comparison_report(
        runs, title=f"Study: {args.study}",
        subtitle=f"{len(runs)} run(s), {runs['fingerprint'].nunique()} distinct trial(s).")
    return _write(report, args, f"study-{slugify(args.study)}")


def cmd_run(args) -> int:
    from ..backtest import registry
    from . import run_report
    if args.run_id:
        record = registry.get(args.run_id)
        if record is None:
            print(f"  no run with id {args.run_id!r}", file=sys.stderr)
            return 1
    else:
        study = args.study or "baselines"
        record = registry.best(study)
        if record is None:
            return _no_runs(f"study {study!r}")
        print(f"  using the best run of {study!r}: "
              f"{record['strategy']} ({record['run_id']})")
    report = run_report(record, exits=Q.exits_for(record))
    return _write(report, args, f"run-{record['strategy']}-{str(record['run_id'])[-6:]}")


def cmd_registry(args) -> int:
    from ..backtest import registry
    from . import registry_report
    runs = registry.load()
    if runs.empty:
        return _no_runs("the registry")
    studies = registry.studies()
    deflations = {}
    for name in studies["study"]:
        try:
            deflations[name] = registry.deflate_best(name)
        except (KeyError, ValueError):
            continue
    return _write(registry_report(runs, studies, deflations), args, "registry")


def cmd_compare(args) -> int:
    from ..backtest import registry
    from . import comparison_report
    all_runs = registry.load()
    if all_runs.empty:
        return _no_runs("the registry")

    picked, missing = [], []
    for target in args.targets:
        hit = all_runs[all_runs["run_id"] == target]
        if hit.empty:                       # fall back to the latest run of a strategy
            by_name = all_runs[all_runs["strategy"] == target]
            hit = by_name.tail(1) if len(by_name) else hit
        if hit.empty:
            missing.append(target)
        else:
            picked.append(hit.iloc[-1])
    if missing:
        print(f"  not found: {', '.join(missing)}", file=sys.stderr)
    if not picked:
        return 1

    import pandas as pd
    runs = pd.DataFrame(picked).reset_index(drop=True)
    return _write(comparison_report(runs, title="Comparison"), args, "comparison")


def cmd_trades(args) -> int:
    """Run a strategy and write a page whose whole purpose is handing over the orders."""
    from ..backtest import run_backtest
    from ..backtest.strategy import get_strategy
    from . import trades_report

    strat = get_strategy(args.strategy)
    if args.top_k is not None and getattr(strat, "construction", None) is not None:
        from dataclasses import replace
        strat.construction = replace(strat.construction, top_k=args.top_k)

    result = run_backtest(strat, start=args.start, end=args.end, costs=args.costs,
                          initial_capital=args.capital, holdout=args.holdout,
                          log_run=not args.no_log, record_trades=True)
    print(f"  {len(result.trades):,} orders over "
          f"{result.config['start']}..{result.config['end']}")
    return _write(trades_report(result), args, f"trades-{slugify(args.strategy)}")


def cmd_honesty(args) -> int:
    from ..backtest import registry
    from . import honesty_report
    runs = registry.load()
    if runs.empty:
        return _no_runs("the registry")
    return _write(honesty_report(runs), args, "honesty")


# --------------------------------------------------------------------------
# The Algorithm Book and the Calendar Lab
#
# Both pages are `queries.<gather>()` -> `<view>()` -> a Report. They are exposed as
# page builders rather than only as commands so a caller can have the Report without
# the side effect of a command.
# --------------------------------------------------------------------------

def algorithms_page():
    """The Algorithm Book, as a Report."""
    from datetime import datetime, timezone

    from .algorithms_view import algorithms_report

    book = Q.algorithm_book()
    return algorithms_report(
        book["entries"], ga=book["ga"], curves=book["curves"],
        forward_context=book["forward_context"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
        index_href=_BACKTEST_INDEX,
        doc_links=[
            {"title": "The Calendar Lab", "href": "timing.html",
             "blurb": "The overnight decomposition and every calendar rule, "
                      "costed three ways."},
            {"title": "How the GA works", "href": "../../docs/HOW_THE_GA_WORKS.md",
             "blurb": "The five-minute write-up: one page for an executive, one "
                      "for an engineer."},
            {"title": "Forward tests", "href": _FORWARD_INDEX,
             "blurb": "The 2022-2026 record: predictions against outcomes, one page "
                      "per algorithm."},
        ])


def calendar_lab_page():
    """The Calendar Lab, as a Report."""
    from datetime import datetime, timezone

    from .timing_views import timing_report

    lab = Q.calendar_lab()
    return timing_report(
        accept=lab["accept"], rules=lab["rules"],
        gross_curves=lab["gross_curves"], net_curves=lab["net_curves"],
        members=lab["members"], member_summary=lab["member_summary"],
        members_csv=lab["members"].to_csv(index=False),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
        index_href=_BACKTEST_INDEX)


def cmd_algorithms(args) -> int:
    return _write(algorithms_page(), args, "algorithms")


def cmd_timing(args) -> int:
    return _write(calendar_lab_page(), args, "timing")


# ------------------------------------------------------------------ helpers

def _run_opts(args) -> dict:
    """The backtest settings the CLI collects, as plain keywords for build_strategy."""
    return dict(start=args.start, end=args.end, holdout=args.holdout,
                costs=args.costs, log_run=not args.no_log,
                study=getattr(args, "study", None))


def _write(report, args, default_name: str) -> int:
    from .render.html import write
    path = args.out or (EXTRA_REPORTS_DIR / f"{default_name}.html")
    out = write(report, path)
    size = out.stat().st_size / 1024
    print(f"  wrote {out}  ({size:,.0f} KB, self-contained)")
    print(f"  {len(report.sections)} section(s): "
          f"{', '.join(s.title for s in report.sections)}")
    if args.open_after:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _no_runs(where: str) -> int:
    print(f"  no runs logged in {where}.", file=sys.stderr)
    print("  Run a backtest first — every run is logged automatically.", file=sys.stderr)
    return 1
