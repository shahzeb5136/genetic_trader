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
    report = run_report(record, exits=_exits_for(record))
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
            subset, cost_model=args.costs, claim=_claim_for(str(name)),
            trades_csv=_forward_trades_csv(subset, args.costs))
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


def _claim_for(name: str) -> str:
    """What this candidate claims, in its own words.

    A registered strategy's claim is its docstring, taken from the source so the two
    cannot drift. An evolved winner has no docstring - it was never written by anyone -
    so its claim is the decoded genome, which is the same text `report all` gives it and
    is the reason the search space is bounded to something readable (ADR-031).
    """
    from ..backtest.strategy import get_strategy
    try:
        return _claim_of(get_strategy(name))
    except Exception:                                             # noqa: BLE001
        pass
    return next((w["claim"] for w in evolved_winners() if w["name"] == name), "")


def _forward_trades_csv(subset, cost_model: str):
    """Path to the saved forward trade ledger for one candidate, if it was saved."""
    rows = subset[subset["cost_model"] == cost_model]
    if rows.empty:
        return None
    saved = str(rows.sort_values("logged_at").iloc[-1].get("saved_to") or "")
    if not saved:
        return None
    path = Path(saved) / "trades.csv"
    return str(path) if path.exists() else None


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


def _exits_for(record):
    """Forced exits for a run, if its full result was saved to disk.

    The registry stores counts, not the rows. A run saved with `--save` has them; one
    that was only logged does not, and the report simply omits the table rather than
    inventing it.
    """
    return None


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
    result, others, bench, claim, coverage, deflation = _build_strategy(args, args.strategy)
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
    coverage = _feature_coverage()

    targets = [{"name": n} for n in names]
    if not args.no_evolved:
        found = evolved_winners()
        if found:
            print(f"  including {len(found)} evolved winner(s): "
                  + ", ".join(w["name"] for w in found))
        targets += found

    specs, curves, failed = [], {}, []
    for target in targets:
        name = target["name"]
        try:
            result, others, bench, claim, cov, deflation = _build_strategy(
                args, name, strategy=target.get("strategy"),
                claim=target.get("claim"), study=target.get("study"))
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
        fp = _feature_panel()
        if fp is not None:
            page = write_html(feature_report(fp, leakage=leakage,
                                             usage=_feature_usage(names, coverage)),
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
    from argparse import Namespace
    for cmd, fname, label, blurb in (
            (cmd_algorithms, "algorithms", "The Algorithm Book",
             "Every competitor - baselines, hypotheses, learned models, GA winners "
             "and calendar rules - explained in its own words and scored against "
             "the index, on one page."),
            (cmd_timing, "timing", "The Calendar Lab",
             "Overnight vs intraday, weekends, month turns and holidays: where "
             "SPY's return actually happens, and what survives costs.")):
        try:
            cmd(Namespace(out=out_dir / f"{fname}.html", open_after=False))
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
    fp = _feature_panel()
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
    usage = _feature_usage(GROUPS["all"], fp.coverage())
    return _write(feature_report(fp, leakage=leakage, usage=usage), args, "features")


# ------------------------------------------------------- shared machinery

def _build_strategy(args, name: str, strategy=None, claim: str | None = None,
                    study: str | None = None):
    """Run one strategy under all three cost settings and gather its context.

    Takes an OBJECT as well as a name so an evolved winner - which lives in a search
    checkpoint rather than in the strategy registry - goes through exactly the same path
    as a hand-written one. The report cannot tell them apart, which is the same claim the
    backtest engine makes.
    """
    from ..backtest import run_backtest
    from ..backtest.benchmark import over_window
    from ..backtest.strategy import get_strategy

    settings = ("optimistic", "realistic", "pessimistic")
    common = dict(start=args.start, end=args.end, holdout=args.holdout,
                  log_run=not args.no_log,
                  study=study or getattr(args, "study", None))
    others = [run_backtest(strategy or get_strategy(name), costs=c,
                           record_trades=(c == args.costs), **common)
              for c in settings]
    result = others[settings.index(args.costs)] if args.costs in settings else others[1]

    if claim is None:
        claim = _claim_of(strategy or get_strategy(name))
    bench = over_window(result)
    return result, others, bench, claim, _feature_coverage(), _deflation_for(result)


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


def _claim_of(strategy) -> str:
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


def _feature_panel():
    try:
        from ..features import build_features
        return build_features()
    except Exception:                                             # noqa: BLE001
        return None


def _feature_coverage():
    fp = _feature_panel()
    return fp.coverage() if fp is not None else None


def _feature_usage(names, coverage):
    """A table of which strategies read which features."""
    from ..backtest.strategy import get_strategy
    from .tables import Cell, Table, _text

    rows = []
    for name in names:
        try:
            strat = get_strategy(name)
        except Exception:                                         # noqa: BLE001
            continue
        needed = tuple(getattr(strat, "requires_features", ()) or ())
        rows.append([
            _text(name),
            Cell(str(len(needed)), len(needed)),
            _text(", ".join(needed) if needed
                  else "none — computes its own from the price panel",
                  "" if needed else "muted"),
        ])
    if not rows:
        return None
    return Table(["strategy", "features", "which"], rows,
                 aligns=["left", "right", "left"],
                 caption="A strategy with no shared features is not worse; it is older "
                         "than the feature layer.")


def _deflation_for(result) -> dict | None:
    """The deflated Sharpe for this run's study, if it was logged."""
    from ..backtest import registry
    run_id = result.config.get("run_id")
    if not run_id:
        return None
    try:
        return registry.deflate(run_id)
    except (KeyError, ValueError, TypeError):
        return None


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
# --------------------------------------------------------------------------

#: Registry family -> the book's display label. Order comes from the view.
_BOOK_FAMILIES = {
    "baselines": "The null hypotheses",
    "alpha": "The twelve hypotheses",
    "frontier": "The second wave",
    "learned": "Learned models",
}

#: Studies whose trial counts mean something for deflation. Grab-bag studies like
#: "reports" and "adhoc" mix unrelated runs; deflating against them would print a
#: number that answers no question anyone asked.
_DEFLATABLE_PREFIXES = ("ga-", "frontier", "mlp", "timing", "learned")

_RESEARCH_END = "2021-12-31"


def _paragraphs(obj) -> list[str]:
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


def _research_row(df, strategy: str, cost_model: str):
    """Latest research-window registry row for one (strategy, cost) pair, or None."""
    if df.empty:
        return None
    hit = df[(df["strategy"] == strategy) & (df["cost_model"] == cost_model)
             & (df["holdout_mode"] == "exclude") & (df["end"] == _RESEARCH_END)]
    if hit.empty:
        return None
    return hit.sort_values("logged_at").iloc[-1]


def _row_stats(row) -> dict:
    return {"cagr": float(row["cagr"]), "sharpe": float(row["sharpe"]),
            "maxdd": float(row["max_drawdown"]),
            "turnover": None if row.get("ann_turnover") is None
            else float(row["ann_turnover"]),
            "cost_drag": None if row.get("cost_drag") is None
            else float(row["cost_drag"])}


def _bench_stats(window_cache: dict, start: str, end: str) -> dict | None:
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


def _deflation_from(df, row) -> dict | None:
    """registry.deflate, computed from an already-loaded frame (one parse, not N)."""
    import numpy as np

    from ..backtest import metrics

    study = row.get("study") or ""
    if not any(str(study).startswith(p) for p in _DEFLATABLE_PREFIXES):
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


def _forward_lookup() -> dict:
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
            "research_sharpe": _maybe_float(r.get("research_sharpe")),
            "forward_sharpe": _maybe_float(r.get("forward_sharpe")),
            "forward_d_sharpe": _maybe_float(r.get("forward_d_sharpe")),
            "decay_z": _maybe_float(r.get("decay_z")),
        }
    return out


def _maybe_float(v):
    import numpy as np
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _monthly_entries(df, window_cache, forward, curves_wanted):
    """AlgorithmEntry list for every registered monthly strategy, running any that
    have no research-window rows yet (logged under study 'reports', like cmd_all)."""
    from ..backtest import run_backtest
    from ..backtest.strategy import get_strategy
    from ..strategies import GROUPS
    from .algorithms_view import AlgorithmEntry

    entries = []
    for group, family in _BOOK_FAMILIES.items():
        for name in GROUPS[group]:
            if name == "spy_buy_hold":
                continue
            rows = {}
            for cost in ("optimistic", "realistic", "pessimistic"):
                row = _research_row(df, name, cost)
                if row is None:
                    res = run_backtest(name, costs=cost, study="reports",
                                       record_trades=False,
                                       notes="algorithm book fill-in")
                    row = _research_row(_reload_registry(), name, cost)
                    if row is None:                       # registry disabled
                        row = _row_from_result(res)
                rows[cost] = row
            real = rows["realistic"]
            paras = _paragraphs(type(get_strategy(name)))
            entry = AlgorithmEntry(
                name=name, family=family, engine="monthly",
                origin="written",
                claim=paras[0] if paras else "",
                explain=paras[1:3],
                window=f"{str(real['start'])[:7]} → {str(real['end'])[:7]}",
                settings={c: _row_stats(r) for c, r in rows.items()},
                bench=_bench_stats(window_cache, str(real["start"]),
                                   str(real["end"])),
                deflation=_deflation_from(df, real),
                forward=forward.get(name),
                href=f"strategy-{slugify(name)}.html",
            )
            curves_wanted[name] = real.get("run_id")
            entries.append(entry)
    return entries


def _reload_registry():
    from ..backtest import registry
    return registry.load()


def _row_from_result(res):
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


def _evolved_entries(df, window_cache, forward, curves_wanted):
    from .algorithms_view import AlgorithmEntry

    entries = []
    for w in evolved_winners():
        name = w["name"]
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = _research_row(df, name, cost)
            if row is None:
                from ..backtest import run_backtest
                strat = w["strategy"]
                strat.name = name
                run_backtest(strat, costs=cost, study=w["study"],
                             record_trades=False,
                             notes="algorithm book: winner re-run, same fingerprint "
                                   "as its trial")
                row = _research_row(_reload_registry(), name, cost)
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
            settings={c: _row_stats(r) for c, r in rows.items()},
            bench=_bench_stats(window_cache, str(real["start"]), str(real["end"])),
            deflation=_deflation_from(df, real) or _study_deflation(w["study"]),
            forward=forward.get(name),
        ))
        curves_wanted[name] = real.get("run_id")
    return entries


def _study_deflation(study: str) -> dict | None:
    from ..backtest import registry
    try:
        d = registry.deflate_best(study)
    except (KeyError, ValueError):
        return None
    if not d or d.get("n_trials") in (None, 0):
        return None
    return {"study": study, "n_trials": d["n_trials"],
            "deflated_sharpe": d.get("deflated_sharpe")}


def _timing_entries(df, forward, curves_wanted):
    """Calendar rules as book entries, benched against tm_buy_hold's own row."""
    from ..timing.data import load_timing_data
    from ..timing.engine import run_timing_backtest
    from ..timing.strategies import TIMING_GROUPS, get_timing_strategy
    from .algorithms_view import AlgorithmEntry

    data = load_timing_data()
    lo = data.date_index("2007-04-02", side="next")
    hi = data.date_index(_RESEARCH_END, side="prev")

    def rows_for(name):
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = _research_row(df, name, cost)
            if row is None:
                run_timing_backtest(name, costs=cost, study="reports",
                                    notes="algorithm book fill-in")
                row = _research_row(_reload_registry(), name, cost)
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
        paras = _paragraphs(type(strat))
        entries.append(AlgorithmEntry(
            name=name, family="The calendar rules", engine="daily legs",
            origin="rule",
            claim=paras[0] if paras else "",
            explain=paras[1:3],
            window=f"{str(real['start'])[:7]} → {str(real['end'])[:7]}",
            settings={c: _row_stats(r) for c, r in rows.items() if r is not None},
            bench=bench,
            deflation=_deflation_from(df, real),
            forward=forward.get(name),
            href="timing.html",
            exposure=f"In the market {exposure:.0%} of sessions.",
        ))
        curves_wanted[name] = real.get("run_id")
    return entries


def _ga_summary(df, forward) -> dict:
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
            "preset": _preset_of(study),
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


def _preset_of(study: str) -> str:
    try:
        from ..evolve.engine import winners
        for w in winners():
            if w["study"] == study:
                return w["preset"]
    except Exception:                                             # noqa: BLE001
        pass
    return ""


def cmd_algorithms(args) -> int:
    """The Algorithm Book: gather everything, hand it to the pure view."""
    from datetime import datetime, timezone

    from ..backtest import registry
    from .algorithms_view import algorithms_report

    df = registry.load()
    window_cache: dict = {}
    forward = _forward_lookup()
    curves_wanted: dict = {}

    entries = _monthly_entries(df, window_cache, forward, curves_wanted)
    df = _reload_registry()          # fill-in runs may have appended
    entries += _evolved_entries(df, window_cache, forward, curves_wanted)
    df = _reload_registry()
    entries += _timing_entries(df, forward, curves_wanted)

    curves = _book_curves(entries, curves_wanted)
    ga = _ga_summary(df, forward)
    forward_context = _forward_context()

    report = algorithms_report(
        entries, ga=ga, curves=curves, forward_context=forward_context,
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
    return _write(report, args, "algorithms")


def _book_curves(entries, curves_wanted) -> dict:
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


def _forward_context() -> dict:
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


def cmd_timing(args) -> int:
    """The Calendar Lab page."""
    from datetime import datetime, timezone

    from ..backtest import registry
    from ..timing.data import load_timing_data
    from ..timing.decompose import decompose_members, summarise
    from ..timing.engine import run_timing_backtest, timing_accept
    from ..timing.strategies import TIMING_GROUPS, get_timing_strategy
    from .timing_views import timing_report

    accept = timing_accept()
    df = registry.load()
    data = load_timing_data()
    lo = data.date_index("2007-04-02", side="next")
    hi = data.date_index(_RESEARCH_END, side="prev")

    rules, run_ids = [], {}
    bench_sharpe = None
    for name in TIMING_GROUPS["all"]:
        rows = {}
        for cost in ("optimistic", "realistic", "pessimistic"):
            row = _research_row(df, name, cost)
            if row is None:
                run_timing_backtest(name, costs=cost, study="reports",
                                    notes="calendar lab fill-in")
                df = _reload_registry()
                row = _research_row(df, name, cost)
            if row is not None:
                rows[cost] = row
        real = rows.get("realistic")
        if real is None:
            continue
        if name == "tm_buy_hold":
            bench_sharpe = float(real["sharpe"])
        strat = get_timing_strategy(name)
        on, intra = strat.legs(data)
        paras = _paragraphs(type(strat))
        rules.append({
            "name": name,
            "claim": paras[0] if paras else "",
            "explain": paras[1:2],
            "exposure": f"{float((on[lo:hi] | intra[lo:hi]).mean()):.0%}",
            "settings": {c: _row_stats(r) for c, r in rows.items()},
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
        gross_curves[label] = frame["nav_gross"] if "nav_gross" in frame.columns \
            else frame["nav"]
        net_curves[label] = frame["nav"]
    for name in ("tm_turn_of_month", "tm_sell_in_may", "tm_vix_overnight"):
        frame = by_name.get(name)
        if frame is not None:
            net_curves[name.replace("tm_", "")] = frame["nav"]

    members = decompose_members(start="2007-04-01", end=_RESEARCH_END)
    report = timing_report(
        accept=accept, rules=rules,
        gross_curves=gross_curves, net_curves=net_curves,
        members=members, member_summary=summarise(members),
        members_csv=members.to_csv(index=False),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"))
    return _write(report, args, "timing")
