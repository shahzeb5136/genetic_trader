"""`sp500lab report ...` — build a static HTML report from the registry.

Kept out of the top-level cli.py so importing the CLI stays cheap; pandas and the
reporting stack load only when a report is actually asked for.

    sp500lab report all --open           <- start here: every strategy + an index
    sp500lab report strategy low_vol     one strategy, in full
    sp500lab report features             what every feature is, and does it leak
    sp500lab report study baselines
    sp500lab report run  20260827T182514-586cb2
    sp500lab report registry
    sp500lab report compare 20260827T1825-a 20260827T1825-b
    sp500lab report honesty
    sp500lab report trades momentum_12_1 --open

`report all` is the one that matters. It runs every strategy in a group, writes one
self-contained page each, and writes an index that links them together - so the whole set
travels as a folder and nobody has to open Python to read a result.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from ..paths import REPORTS_DIR
from . import queries as Q
from .tables import feature_usage
from .util import slugify


def add_parser(sub) -> None:
    p = sub.add_parser("report", help="build a static HTML report from the registry")
    rs = p.add_subparsers(dest="report_command", required=True)

    al = rs.add_parser(
        "all",
        help="every strategy, the feature layer and an index that links them - "
             "start here")
    al.add_argument("group", nargs="?", default="all",
                    help="all | baselines | alpha | learned | evolved, or a "
                         "comma-separated list of strategy names")
    _strategy_args(al)
    al.add_argument("--no-features", action="store_true",
                    help="skip the feature-layer report")
    al.add_argument("--no-evolved", action="store_true",
                    help="skip the winners of any genetic-algorithm searches found in "
                         "data/experiments/evolve/")
    al.add_argument("--check-features", action="store_true",
                    help="also run the leakage check (slow: rebuilds the panel twice)")
    al.add_argument("-o", "--out", default=None,
                    help=f"output directory (default: {REPORTS_DIR.name}/)")
    al.add_argument("--open", action="store_true", dest="open_after")
    al.set_defaults(func=cmd_all)

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

    fw = rs.add_parser(
        "forward",
        help="the whole forward-test set: an executive summary, one technical report "
             "per candidate, the cross-sectional decay analysis and an honesty page")
    fw.add_argument("-o", "--out", default=None,
                    help=f"output DIRECTORY (default: {REPORTS_DIR.name}/forward_tests)")
    fw.add_argument("--costs", default="realistic",
                    choices=["optimistic", "realistic", "pessimistic"],
                    help="which cost setting the summaries lead with. All three are "
                         "always shown in each candidate's own report.")
    fw.add_argument("--no-markdown", action="store_true",
                    help="skip EXECUTIVE_SUMMARY.md; write only the HTML set")
    fw.add_argument("--open", action="store_true", dest="open_after")
    fw.set_defaults(func=cmd_forward)

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


def _shared(parser) -> None:
    parser.add_argument("-o", "--out", default=None,
                        help=f"output path (default: {REPORTS_DIR.name}/<name>.html)")
    parser.add_argument("--open", action="store_true", dest="open_after",
                        help="open the report in a browser when it is written")


# ----------------------------------------------------------------- commands

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


# ------------------------------------------------------------------ helpers

def cmd_forward(args) -> int:
    """Write the whole forward-test report set into one directory.

    A directory rather than a file, and one page per candidate rather than one long
    page, for the same reason `report all` works that way: a self-contained file
    survives being sent on its own, and an index with relative links is the cheap fix
    for the navigation that costs.

    Nothing here runs a backtest or reads the panel. Every number comes out of
    `forward/store.py`, which is the guarantee ADR-034 was written to give - a forward
    report has to be rebuildable years later from the record alone.
    """
    from ..forward import store
    from . import (
        forward_decay_report,
        forward_honesty_report,
        forward_index_report,
        forward_strategy_report,
    )
    from .forward_views import PRIMARY_COSTS, primary_rows, strategy_href
    from .render.html import write as write_html
    from .render.markdown import write as write_md

    records = store.load()
    if records.empty:
        print("No forward test has been run, so there is nothing to report.")
        print("  python -m sp500lab forward suite all --seal-first")
        return 1

    out_dir = Path(args.out or (REPORTS_DIR / "forward_tests"))
    out_dir.mkdir(parents=True, exist_ok=True)
    md_dir = out_dir / "markdown"
    data_dir = out_dir / "data"
    written: list = []

    rows = primary_rows(records, args.costs or PRIMARY_COSTS)
    print(f"Building the forward-test set for {len(rows)} candidate(s) "
          f"[{args.costs} costs] into {out_dir}")
    print()

    for name in rows["strategy"]:
        subset = records[records["strategy"] == name]
        report = forward_strategy_report(
            subset, cost_model=args.costs, claim=Q.claim_for(str(name)),
            trades_csv=Q.forward_trades_csv(subset, args.costs))
        written.append(write_html(report, out_dir / strategy_href(str(name))))
        if not args.no_markdown:
            written.append(write_md(report, md_dir / f"{slugify(str(name))}.md"))
        print(f"  {str(name):24s} {written[-1].name}")

    decay = forward_decay_report(records, args.costs)
    honesty = forward_honesty_report(records)
    index = forward_index_report(records, cost_model=args.costs, extra_cards=[
        {"title": "Did research predict anything?",
         "href": "decay-analysis.html",
         "blurb": "The cross-section: research ranking against forward ranking. The "
                  "question no single-strategy report can answer."},
        {"title": "What limits all of this",
         "href": "honesty.html",
         "blurb": "Sample size, multiple testing, the regime, and everything the "
                  "engine already warns about. Read before quoting a number."},
    ])
    written.append(write_html(decay, out_dir / "decay-analysis.html"))
    written.append(write_html(honesty, out_dir / "honesty.html"))
    written.append(write_html(index, out_dir / "index.html"))
    print(f"  {'decay analysis':24s} decay-analysis.html")
    print(f"  {'honesty':24s} honesty.html")
    print(f"  {'executive summary':24s} index.html")

    if not args.no_markdown:
        written.append(write_md(index, out_dir / "EXECUTIVE_SUMMARY.md"))
        written.append(write_md(decay, out_dir / "DECAY_ANALYSIS.md"))
        written.append(write_md(honesty, out_dir / "HONESTY.md"))
        print(f"  {'markdown':24s} EXECUTIVE_SUMMARY.md, DECAY_ANALYSIS.md, "
              f"HONESTY.md, markdown/*.md")

    written += _forward_data_exports(records, data_dir)
    written.append(_forward_readme(records, rows, args.costs, out_dir))
    print(f"  {'raw data':24s} data/*.csv")
    print(f"  {'folder guide':24s} README.md")

    total = sum(p.stat().st_size for p in written) / 1e6
    print()
    print(f"  {len(written)} file(s), {total:.1f} MB total. Every HTML page is "
          "self-contained: no server, no CDN, no build step.")
    print(f"  start at {out_dir / 'index.html'}")
    if args.open_after:
        webbrowser.open((out_dir / "index.html").resolve().as_uri())
    return 0


def _forward_data_exports(records, data_dir) -> list:
    """The numbers behind the pages, as CSV, so nobody has to open Python to re-cut them.

    A report is an argument. The data underneath it should be separable from the
    argument, or the only way to disagree is to rebuild the pipeline.
    """
    import pandas as pd

    from ..forward import store

    data_dir.mkdir(parents=True, exist_ok=True)
    written = []

    flat = data_dir / "forward_tests.csv"
    records.to_csv(flat, index=False)
    written.append(flat)

    long_rows = []
    for fid, windows in store.load_curves(list(records["forward_id"])).items():
        strategy = records.loc[records["forward_id"] == fid, "strategy"]
        label = str(strategy.iloc[0]) if len(strategy) else ""
        for window, df in windows.items():
            block = df.reset_index()
            block.insert(0, "window", window)
            block.insert(0, "strategy", label)
            block.insert(0, "forward_id", fid)
            long_rows.append(block)
    if long_rows:
        curves = data_dir / "forward_curves.csv"
        pd.concat(long_rows, ignore_index=True).to_csv(curves, index=False)
        written.append(curves)

    from ..forward import seal as seal_module
    seals = seal_module.load_seals()
    if len(seals):
        path = data_dir / "seals.csv"
        seals.to_csv(path, index=False)
        written.append(path)
    return written


def _forward_readme(records, rows, cost_model: str, out_dir):
    """A guide to the folder, generated from what is actually in it."""

    counts = rows["verdict"].value_counts().to_dict()
    lines = [
        "# Forward tests: 2022 onward",
        "",
        f"Generated by `python -m sp500lab report forward` from "
        f"`data/experiments/forward/`. {len(records)} forward runs across "
        f"{records['strategy'].nunique()} strategies and "
        f"{records['cost_model'].nunique()} cost settings; the summaries lead with "
        f"**{cost_model}** costs.",
        "",
        "## Start here",
        "",
        "| File | What it is |",
        "|---|---|",
        "| [`index.html`](index.html) | **The executive summary.** Verdicts, the "
        "paired scoreboard, every curve, and links to everything else. |",
        "| [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) | The same page as Markdown, "
        "for reading without a browser. |",
        "| [`decay-analysis.html`](decay-analysis.html) | Did the research window "
        "predict anything? The cross-section of research rank against forward rank. |",
        "| [`honesty.html`](honesty.html) | What limits all of it: sample size, "
        "multiple testing, the regime, and the inherited engine caveats. |",
        "",
        "## One technical report per candidate",
        "",
        "Each is a self-contained page: the prediction against the outcome, the "
        "significance of the gap, all nine checks, the curve, every year, the monthly "
        "grid, all three cost settings, full provenance, and the orders it placed.",
        "",
    ]
    for _, r in rows.iterrows():
        name = str(r["strategy"])
        lines.append(
            f"- [`{name}`](forward-{slugify(name)}.html) — "
            f"**{str(r['verdict']).upper()}**, forward CAGR "
            f"{_pct_or_dash(r.get('forward_cagr'))}, Sharpe vs index "
            f"{_signed_or_dash(r.get('forward_d_sharpe'))}")
    lines += [
        "",
        "Markdown copies of every one are in [`markdown/`](markdown/).",
        "",
        "## Raw data",
        "",
        "| File | Contents |",
        "|---|---|",
        "| [`data/forward_tests.csv`](data/forward_tests.csv) | One row per forward "
        "run: both legs, the comparison, the verdict, the diagnostics. |",
        "| [`data/forward_curves.csv`](data/forward_curves.csv) | Month-end curves in "
        "long form: `forward_id`, `strategy`, `window`, `date`, `nav`, `nav_gross`, "
        "`benchmark`. |",
        "| [`data/seals.csv`](data/seals.csv) | Every pre-registration, with the "
        "prediction it recorded and when. |",
        "| [`run.log`](run.log) | The console output of the run that produced all of "
        "this. |",
        "",
        "Full order ledgers are under `results/forward/<forward_id>/` "
        "(`trades.csv`, `trades.parquet`, `holdings`, `weights`, `exits`) and are "
        "embedded in each candidate's HTML page.",
        "",
        "## Read this before quoting anything",
        "",
        f"The forward window is 54 monthly observations. That puts a ±0.9 band around "
        "an annualised Sharpe of 1.0, so **a forward test on this data can refute a "
        "strategy and cannot confirm one.** `held` means *not refuted*. "
        f"{counts.get('held', 0)} held, {counts.get('decayed', 0)} decayed, "
        f"{counts.get('failed', 0)} failed.",
        "",
        "Every run here is in the holdout ledger (`sp500lab experiments holdout`) and "
        "none of it can be withdrawn. The reserved period is spent.",
        "",
        "See [`docs/FORWARD_TEST.md`](../../docs/FORWARD_TEST.md), ADR-033 and ADR-034.",
        "",
    ]
    path = Path(out_dir) / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _pct_or_dash(v) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _signed_or_dash(v) -> str:
    try:
        return f"{float(v):+.2f}"
    except (TypeError, ValueError):
        return "—"






def _run_opts(args) -> dict:
    """The backtest settings the CLI collects, as plain keywords for build_strategy."""
    return dict(start=args.start, end=args.end, holdout=args.holdout,
                costs=args.costs, log_run=not args.no_log,
                study=getattr(args, "study", None))


def _write(report, args, default_name: str) -> int:
    from .render.html import write
    path = args.out or (REPORTS_DIR / f"{default_name}.html")
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


# --------------------------------------------------------------------------
# The report set: one page per strategy, one for the features, one index
# --------------------------------------------------------------------------

def cmd_strategy(args) -> int:
    """Run one strategy and publish everything about it."""
    from . import strategy_report
    result, others, bench, claim, coverage, deflation = Q.build_strategy(
        args.strategy, **_run_opts(args))
    report = strategy_report(result, results_by_cost=others, benchmark=bench, claim=claim,
                             feature_coverage=coverage, deflation=deflation)
    print(f"  {result.strategy}: CAGR {result.performance.cagr * 100:.2f}%  "
          f"Sharpe {result.performance.sharpe:.2f}  "
          f"{len(result.trades):,} orders")
    return _write(report, args, f"strategy-{slugify(args.strategy)}")


def cmd_all(args) -> int:
    """Publish the whole set: every strategy, the feature layer, and an index.

    One command, because the point of the set is that it is a set. A folder of reports
    that each open in a browser and link to each other is the closest this project gets
    to a product, and it is the answer to "I do not want to read the code".
    """
    from ..backtest import registry
    from ..strategies import GROUPS
    from . import feature_report, index_report, strategy_report
    from .render.html import write as write_html

    out_dir = Path(args.out or REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = (GROUPS[args.group] if args.group in GROUPS
             else tuple(n.strip() for n in args.group.split(",") if n.strip()))

    targets = [{"name": n} for n in names]
    if not args.no_evolved:
        found = Q.evolved_winners()
        if found:
            print(f"  including {len(found)} evolved winner(s): "
                  + ", ".join(w["name"] for w in found))
        targets += found

    specs, curves, failed = [], {}, []
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
        href = f"strategy-{slugify(name)}.html"
        # The complete ledger always lands beside the page, whatever the page embeds.
        # A capped download is a convenience; losing orders is not acceptable.
        csv_href = f"trades/{slugify(name)}.csv"
        (out_dir / "trades").mkdir(exist_ok=True)
        result.trades.to_csv(out_dir / csv_href, index=False)
        page = write_html(strategy_report(result, results_by_cost=others,
                                          benchmark=bench, claim=claim,
                                          feature_coverage=cov, deflation=deflation,
                                          full_csv_href=csv_href),
                          out_dir / href)
        p = result.performance
        from .views import _headline_claim
        specs.append({
            "name": name, "href": href,
            "claim": _headline_claim(claim),
            "evolved": "strategy" in target,
            "n_trials": (deflation or {}).get("n_trials"),
            "deflated_sharpe": (deflation or {}).get("deflated_sharpe"),
            "window": f"{str(result.config.get('start'))[:7]}–"
                      f"{str(result.config.get('end'))[:7]}",
            "cagr": p.cagr, "sharpe": p.sharpe,
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

    cards = []
    if not args.no_features:
        leakage = None
        if args.check_features:
            from ..features import check_leakage
            print("  checking the feature layer for leakage (this rebuilds it twice)…")
            leakage = check_leakage()
        fp = Q.feature_panel()
        if fp is not None:
            page = write_html(feature_report(fp, leakage=leakage,
                                             usage=feature_usage(names)),
                              out_dir / "features.html")
            cards.append({"title": "The feature layer", "href": "features.html",
                          "blurb": f"What all {fp.n_features} features are, where each "
                                   "comes from, and the leakage test that decides "
                                   "whether any of it can be trusted.",
                          "stats": [("features", str(fp.n_features)),
                                    ("leaks", "0" if (leakage or {}).get("ok")
                                     else "not checked")]})
            print(f"  {'features':20s} {fp.n_features} documented -> {page.name}")

    for name, label, blurb in (
            ("registry", "Everything tried", "Every backtest ever run here, grouped by "
             "study, with the deflated Sharpe for each search."),
            ("honesty", "What to distrust", "Price coverage, forced exits, unresolved "
             "delisting assumptions and holdout exposure, across every run.")):
        try:
            _write_registry_page(name, out_dir)
            cards.append({"title": label, "href": f"{name}.html", "blurb": blurb})
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skipped {name}.html: {exc}", file=sys.stderr)

    # The two cross-cutting pages. Generated from the registry the runs above just
    # wrote to, so they and the strategy pages describe the same numbers.
    for build, fname, label, blurb in (
            (algorithms_page, "algorithms", "The Algorithm Book",
             "Every competitor - baselines, hypotheses, learned models, GA winners "
             "and calendar rules - explained in its own words and scored against "
             "the index, on one page."),
            (calendar_lab_page, "timing", "The Calendar Lab",
             "Overnight vs intraday, weekends, month turns and holidays: where "
             "SPY's return actually happens, and what survives costs.")):
        try:
            write_html(build(), out_dir / f"{fname}.html")
            cards.append({"title": label, "href": f"{fname}.html", "blurb": blurb})
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skipped {fname}.html: {exc}", file=sys.stderr)

    studies = registry.studies()
    index = index_report(specs, curves=curves, studies=studies, extra_cards=cards,
                         subtitle=f"{len(specs)} strategies, {args.costs} costs, "
                                  "each scored against the index over its own window.")
    path = write_html(index, out_dir / "index.html")
    print()
    print(f"  wrote {len(specs)} strategy reports + index -> {path}")
    print(f"  open {path}")
    if failed:
        print()
        for line in failed:
            print(f"  FAILED {line}", file=sys.stderr)
    if args.open_after:
        webbrowser.open(path.resolve().as_uri())
    return 0


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
    from ..strategies import GROUPS
    usage = feature_usage(GROUPS["all"])
    return _write(feature_report(fp, leakage=leakage, usage=usage), args, "features")


# ------------------------------------------------------- shared machinery















def _write_registry_page(kind: str, out_dir) -> None:
    from ..backtest import registry
    from . import honesty_report, registry_report
    from .render.html import write as write_html

    runs = registry.load()
    if runs.empty:
        raise ValueError("the registry is empty")
    if kind == "honesty":
        write_html(honesty_report(runs), out_dir / "honesty.html")
        return
    studies = registry.studies()
    deflations = {}
    for study in studies["study"]:
        try:
            deflations[study] = registry.deflate_best(study)
        except (KeyError, ValueError):
            continue
    write_html(registry_report(runs, studies, deflations), out_dir / "registry.html")


# --------------------------------------------------------------------------
# The Algorithm Book and the Calendar Lab
#
# Both pages are `queries.<gather>()` -> `<view>()` -> a Report. They are exposed as
# page builders rather than only as commands because `cmd_all` needs the Report, not
# the side effect of a command: it used to reach them by fabricating an
# argparse.Namespace, which is what a missing seam looks like.
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
        doc_links=[
            {"title": "The Calendar Lab", "href": "timing.html",
             "blurb": "The overnight decomposition and every calendar rule, "
                      "costed three ways."},
            {"title": "How the GA works", "href": "../docs/HOW_THE_GA_WORKS.md",
             "blurb": "The five-minute write-up: one page for an executive, one "
                      "for an engineer."},
            {"title": "Forward tests", "href": "forward_tests/index.html",
             "blurb": "The 2022-2026 record: predictions against outcomes, with "
                      "the decay analysis."},
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
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"))


def cmd_algorithms(args) -> int:
    return _write(algorithms_page(), args, "algorithms")






def cmd_timing(args) -> int:
    return _write(calendar_lab_page(), args, "timing")
