"""`sp500lab report ...` — build a static HTML report from the registry.

Kept out of the top-level cli.py so importing the CLI stays cheap; pandas and the
reporting stack load only when a report is actually asked for.

    sp500lab report study baselines
    sp500lab report run  20260827T182514-586cb2
    sp500lab report registry
    sp500lab report compare 20260827T1825-a 20260827T1825-b
    sp500lab report honesty
"""

from __future__ import annotations

import sys
import webbrowser

from ..paths import REPORTS_DIR


def add_parser(sub) -> None:
    p = sub.add_parser("report", help="build a static HTML report from the registry")
    rs = p.add_subparsers(dest="report_command", required=True)

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

    hs = rs.add_parser("honesty", help="coverage, exits and holdout exposure")
    _shared(hs)
    hs.set_defaults(func=cmd_honesty)


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
    return _write(report, args, f"study-{_slug(args.study)}")


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


def cmd_honesty(args) -> int:
    from ..backtest import registry
    from . import honesty_report
    runs = registry.load()
    if runs.empty:
        return _no_runs("the registry")
    return _write(honesty_report(runs), args, "honesty")


# ------------------------------------------------------------------ helpers

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


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in str(text))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "report"
