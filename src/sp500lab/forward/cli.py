"""`sp500lab forward ...` - spend the holdout once, deliberately, and write it down.

    sp500lab forward window                       what is available, and what it proves
    sp500lab forward seal low_vol --rationale "..."     pre-register, no look spent
    sp500lab forward seals                        every pre-registration
    sp500lab forward run low_vol --rationale "..."      <- THIS SPENDS THE HOLDOUT
    sp500lab forward run low_vol --dry-run        what it would cost, without paying it
    sp500lab forward suite alpha --rationale "..."      a whole group in one decision
    sp500lab forward list                         every forward test run so far
    sp500lab forward show fwd-2026...             one of them, in full
    sp500lab forward scoreboard                   prediction against outcome, ranked

Kept out of the top-level cli.py so importing the CLI stays cheap: the panel, the
strategy registry and numpy load only when a forward test is actually asked for - the
same reason `backtest/cli.py` and `evolve/cli.py` are separate.

The ordering of the commands above is the intended workflow and it is not decorative.
`window` tells you whether there is enough data to bother. `seal` writes the prediction
down while you still cannot see the answer. `run` is the irreversible step. Everything
after it is reading.
"""

from __future__ import annotations

import json
import sys

from ..paths import PROJECT_ROOT


def add_parser(sub) -> None:
    """Register the `forward` subcommand tree on the main parser."""
    p = sub.add_parser(
        "forward",
        help="out-of-sample testing after the research window - spends the holdout")
    fs = p.add_subparsers(dest="forward_command", required=True)

    w = fs.add_parser("window",
                      help="what forward data exists, and what a window that long "
                           "can actually prove")
    w.set_defaults(func=cmd_window)

    sl = fs.add_parser(
        "seal",
        help="pre-register a candidate - runs the research window only, spends nothing")
    sl.add_argument("strategy")
    sl.add_argument("--rationale", required=True,
                    help="why this candidate deserves a look. Required, because a "
                         "reason written before the answer reads differently from one "
                         "written after.")
    _shared_run_args(sl)
    sl.set_defaults(func=cmd_seal)

    ls = fs.add_parser("seals", help="every pre-registration, oldest first")
    ls.add_argument("--strategy", default=None)
    ls.set_defaults(func=cmd_seals)

    r = fs.add_parser(
        "run",
        help="forward-test one strategy. THIS LOOKS AT THE HOLDOUT and is recorded "
             "permanently.")
    r.add_argument("strategy")
    r.add_argument("--rationale", default="",
                   help="why this candidate. Recorded on the seal if one is written.")
    r.add_argument("--dry-run", action="store_true",
                   help="report exactly what would be spent and stop before spending "
                        "it. Nothing is run, nothing is recorded.")
    r.add_argument("--mode", default="paired", choices=["paired", "continuous"],
                   help="'paired' runs the forward window from a fresh 100k, as a new "
                        "deployment would; 'continuous' runs one unbroken backtest "
                        "across the boundary and slices it. Both are one look.")
    r.add_argument("--costs", default=None,
                   choices=["optimistic", "realistic", "pessimistic"],
                   help="one setting instead of all three. Not recommended: fetching "
                        "the others later costs a second look.")
    r.add_argument("--no-save", action="store_true",
                   help="do not write the forward result's artifacts to results/forward")
    r.add_argument("--save", default=None, metavar="DIR",
                   help="write the artifacts here instead of results/forward")
    _shared_run_args(r)
    r.set_defaults(func=cmd_run)

    su = fs.add_parser(
        "suite",
        help="forward-test a group in one decision. One look each, and the "
             "best-of-N correction is printed with the result.")
    su.add_argument("group", nargs="?", default="all",
                    help="all | baselines | alpha | learned | evolved, or a "
                         "comma-separated list of strategy names")
    su.add_argument("--rationale", default="",
                   help="why this group. Recorded on every seal written.")
    su.add_argument("--dry-run", action="store_true")
    su.add_argument("--mode", default="paired", choices=["paired", "continuous"])
    su.add_argument("--costs", default=None,
                    choices=["optimistic", "realistic", "pessimistic"])
    su.add_argument("--save", default=None, metavar="DIR",
                    help="write each forward result's artifacts here (off by default "
                         "for a suite; a single `forward run` saves them)")
    su.add_argument("--no-evolved", action="store_true",
                    help="skip the winners of any genetic-algorithm searches found in "
                         "data/experiments/evolve/. They are included by default: an "
                         "evolved winner is a candidate like any other, and leaving the "
                         "searched ones out of the out-of-sample test would be the most "
                         "flattering omission available.")
    su.add_argument("--seal-first", action="store_true",
                    help="pre-register every candidate before looking at any of them. "
                         "Costs one extra research backtest per candidate and buys the "
                         "one guarantee a same-session run can still offer: the "
                         "candidate SET was fixed before any forward result was seen.")
    _shared_run_args(su)
    su.set_defaults(func=cmd_suite)

    li = fs.add_parser("list", help="every forward test, newest last")
    li.add_argument("--strategy", default=None)
    li.add_argument("--costs", default=None,
                    help="filter to one cost setting")
    li.set_defaults(func=cmd_list)

    sh = fs.add_parser("show", help="everything recorded about one forward test")
    sh.add_argument("forward_id")
    sh.add_argument("--json", action="store_true", dest="as_json",
                    help="the raw record instead of the readable summary")
    sh.set_defaults(func=cmd_show)

    sb = fs.add_parser(
        "scoreboard",
        help="prediction against outcome for every candidate, with the "
             "multiple-testing bar the forward window has accumulated")
    sb.add_argument("--costs", default="realistic",
                    help="which cost setting to rank on; 'all' for every row")
    sb.set_defaults(func=cmd_scoreboard)


def _shared_run_args(parser) -> None:
    parser.add_argument("--start", default="2007-04-01",
                        help="start of the RESEARCH window (the prediction), not of "
                             "the forward one - that is fixed at the holdout boundary")
    parser.add_argument("--forward-end", default=None,
                        help="stop the forward window early. Only sensible for "
                             "reproducing an older vintage.")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--weighting", default=None,
                        choices=["equal", "score", "score_rank", "inverse_vol"])
    parser.add_argument("--liquidity-floor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--study", default=None,
                        help="study name for the forward runs in the trial log")
    parser.add_argument("--origin-study", default=None,
                        help="the search this candidate came out of. Recovered from "
                             "the trial log by fingerprint when omitted; pass it when "
                             "the candidate was not run under its search's own name.")
    parser.add_argument("--notes", default="")


# ----------------------------------------------------------------- commands

def cmd_window(args) -> int:
    """What forward data exists, how much of it is new, and what it can prove."""
    from ..backtest.panel import build_panel
    from ..backtest.registry import HOLDOUT_START, holdout_touch_count, research_end
    from . import store
    from .windows import describe_power, forward_window, research_window

    panel = build_panel()
    data_end = str(panel.dates[-1])
    research = research_window("2007-04-01")
    try:
        forward = forward_window(data_end)
    except ValueError as exc:
        print(f"  {exc}")
        return 1

    print("=" * 78)
    print("FORWARD WINDOW")
    print("=" * 78)
    print(f"  research (searchable)   {research.describe()}")
    print(f"  holdout opens           {HOLDOUT_START}  (research ends {research_end()})")
    print(f"  forward (out of sample) {forward.describe()}")
    print(f"  data runs to            {data_end}")
    print()
    print(f"  {describe_power(forward.n_months)}")
    print()
    print(f"  The holdout has been looked at {holdout_touch_count()} time(s) so far "
          "(`sp500lab experiments holdout`).")

    seals = _seal_frame()
    if len(seals):
        print()
        print(f"  {len(seals)} candidate(s) pre-registered. Fresh data since each was "
              "last tested:")
        print()
        print(_freshness_table(forward).to_string(index=False))

    tests = store.load()
    if len(tests):
        print()
        print(f"  {len(tests)} forward test(s) recorded across "
              f"{tests['strategy'].nunique()} strategy(s). "
              "`sp500lab forward scoreboard`.")
    else:
        print()
        print("  No forward test has been run. Nothing here is spent yet.")
    return 0


def cmd_seal(args) -> int:
    """Pre-register a candidate. Runs the research window; spends no holdout."""
    from .engine import seal_candidate

    strat = _resolve_strategy(args)
    seals = seal_candidate(
        strat, rationale=args.rationale, research_start=args.start,
        costs=("optimistic", "realistic", "pessimistic"),
        initial_capital=args.capital, liquidity_floor=args.liquidity_floor,
        seed=args.seed, benchmark=args.benchmark, origin_study=args.origin_study)

    print("=" * 78)
    print(f"SEALED  {getattr(strat, 'name', args.strategy)}")
    print("=" * 78)
    for s in seals:
        print(s.summary())
        print()
    print("  No holdout data was read. The forward test is a separate, deliberate act:")
    print(f"    python -m sp500lab forward run {args.strategy}")
    print()
    print("  The gap between the timestamp above and that run is the only evidence")
    print("  that this candidate was chosen before anybody saw how it did.")
    return 0


def cmd_seals(args) -> int:
    from . import seal as seal_module
    df = seal_module.load_seals(args.strategy)
    if df.empty:
        print("No candidate has been pre-registered yet.")
        print("  python -m sp500lab forward seal <strategy> --rationale \"...\"")
        return 0
    cols = [c for c in ("seal_id", "sealed_at", "seal_mode", "strategy", "cost_model",
                        "research_cagr", "research_sharpe", "n_trials",
                        "deflated_sharpe", "data_end", "rationale") if c in df.columns]
    out = df[cols].copy()
    if "research_cagr" in out.columns:
        out["research_cagr"] = out["research_cagr"].map(_pct)
    for c in ("research_sharpe", "deflated_sharpe"):
        if c in out.columns:
            out[c] = out[c].map(_num)
    if "rationale" in out.columns:
        out["rationale"] = out["rationale"].str.slice(0, 40)
    print(out.to_string(index=False))
    print()
    print(f"  {len(df)} line(s), {df['seal_id'].nunique()} distinct "
          f"configuration(s) over {df['strategy'].nunique()} strategy(s) "
          "(one seal per cost setting). "
          "The EARLIEST line for an id is the one that binds.")
    return 0


def cmd_run(args) -> int:
    """Forward-test one strategy. This is the irreversible command in this file."""
    from .engine import forward_test

    strat = _resolve_strategy(args)
    if args.dry_run:
        return _dry_run([strat], args)

    _announce([getattr(strat, "name", args.strategy)], args)
    save: bool | str = True
    if args.no_save:
        save = False
    elif args.save:
        save = args.save

    test = forward_test(
        strat, rationale=args.rationale, research_start=args.start,
        forward_end=args.forward_end,
        costs=(args.costs,) if args.costs else _all_costs(),
        mode=args.mode, initial_capital=args.capital,
        liquidity_floor=args.liquidity_floor, seed=args.seed,
        benchmark=args.benchmark, origin_study=args.origin_study,
        notes=args.notes, save=save,
        **({"study": args.study} if args.study else {}))

    print()
    print(test.summary())
    _print_footer(test)
    return 0


def cmd_suite(args) -> int:
    """Forward-test a group of candidates as one decision."""
    from ..strategies import GROUPS
    from .engine import forward_suite

    names = (list(GROUPS[args.group]) if args.group in GROUPS
             else [n.strip() for n in args.group.split(",") if n.strip()])
    candidates: list = list(names)
    # An evolved winner's configuration was never run under its search's own `--study`
    # in a form the trial log can match by fingerprint, so its origin has to be handed
    # over explicitly. Without it the two most heavily searched candidates in the set
    # would report `n_trials=0` - the single most flattering error available here.
    origins: dict[str, str] = {}
    if not args.no_evolved:
        from ..evolve.engine import winners
        evolved = winners()
        if evolved:
            print(f"  including {len(evolved)} evolved winner(s): "
                  + ", ".join(f"{w['name']} (from {w['study']})" for w in evolved))
            candidates += [w["strategy"] for w in evolved]
            origins |= {w["name"]: w["study"] for w in evolved}
    if not candidates:
        print(f"  no strategies in group {args.group!r}", file=sys.stderr)
        return 1

    labels = [c if isinstance(c, str) else getattr(c, "name", "?") for c in candidates]
    if args.dry_run:
        from ..backtest.strategy import get_strategy
        return _dry_run([get_strategy(c) if isinstance(c, str) else c
                         for c in candidates], args)

    if args.seal_first:
        _seal_all(candidates, args, origins)
    _announce(labels, args)
    tests = forward_suite(
        candidates, origin_studies=origins,
        rationale=args.rationale, research_start=args.start,
        forward_end=args.forward_end,
        costs=(args.costs,) if args.costs else _all_costs(),
        mode=args.mode, initial_capital=args.capital,
        liquidity_floor=args.liquidity_floor, seed=args.seed,
        benchmark=args.benchmark, origin_study=args.origin_study,
        notes=args.notes, save=args.save or False,
        **({"study": args.study} if args.study else {}))
    if not tests:
        print("  every candidate failed to run", file=sys.stderr)
        return 1

    print()
    print("=" * 108)
    print(f"FORWARD SUITE  {args.group}   mode={args.mode}   "
          f"{len(tests)} of {len(candidates)} candidate(s) tested")
    print("=" * 108)
    _print_scoreboard(cost_model=args.costs or "realistic")
    print()
    for test in tests:
        head = test.headline
        if head is None:
            continue
        print(f"  {test.strategy:22s} {head.verdict.upper():13s} "
              f"{head.comparison.verdict_reason}")
    print()
    print("  Full detail: python -m sp500lab forward show <forward_id>")
    print("  Every look:  python -m sp500lab experiments holdout")
    return 0


def cmd_list(args) -> int:
    from . import store
    df = store.load(strategy=args.strategy)
    if df.empty:
        print("No forward test has been run yet.")
        return 0
    if args.costs:
        df = df[df["cost_model"] == args.costs]
    cols = [c for c in ("forward_id", "logged_at", "strategy", "cost_model", "mode",
                        "verdict", "look_number", "fresh_months", "research_sharpe",
                        "forward_sharpe", "decay_z", "seal_mode")
            if c in df.columns]
    out = df[cols].copy()
    for c in ("research_sharpe", "forward_sharpe", "decay_z"):
        if c in out.columns:
            out[c] = out[c].map(_num)
    print(out.to_string(index=False))
    print()
    print(f"  {len(df)} forward test(s) over {df['strategy'].nunique()} "
          f"strategy(s) and {df['cost_model'].nunique()} cost setting(s).")
    return 0


def cmd_show(args) -> int:
    from . import store
    rec = store.get(args.forward_id)
    if rec is None:
        print(f"  no forward test with id {args.forward_id!r}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(rec.as_dict(), indent=2, sort_keys=True, default=str))
        return 0

    from .compare import compare
    c = compare(rec.research_leg(), rec.forward_leg())
    print("=" * 78)
    print(f"FORWARD TEST  {rec.forward_id}")
    print("=" * 78)
    print(f"  strategy        {rec.strategy}  [{rec.cost_model} costs, "
          f"mode={rec.mode}]")
    print(f"  seal            {rec.seal_id}  ({rec.seal_mode})")
    print(f"  rationale       {rec.rationale or '(none given)'}")
    print(f"  run at          {rec.logged_at}   look #{rec.look_number}")
    print(f"  data vintage    {rec.data_end}"
          + (f"  (previous look saw {rec.previous_data_end})"
             if rec.previous_data_end else ""))
    print(f"  fresh months    {rec.fresh_months} of {rec.forward_leg().n_months}")
    if rec.study:
        print(f"  search behind   study {rec.study}: {rec.n_trials} trial(s), "
              f"deflated Sharpe {_num(rec.deflated_sharpe_research)}")
    else:
        print("  search behind   UNKNOWN - no earlier run of this configuration is in "
              "the trial log,")
        print("                  so the research Sharpe below is not corrected for the "
              "search")
        print("                  that produced the candidate")
    print(f"  runs            research {rec.research_run_id or 'n/a'}   "
          f"forward {rec.forward_run_id or 'n/a'}")
    if rec.saved_to:
        print(f"  artifacts       {rec.saved_to}")
    print()
    print(c.summary())
    print()
    print("  CHECKS")
    for chk in c.checks:
        print(f"    {chk.mark:4s}  {chk.name:20s} {chk.detail}")
    print()
    print(f"  {rec.verdict.upper()}: {rec.verdict_reason}")
    print()
    print("  HONESTY")
    print(f"    price coverage        {_pct(rec.coverage_median)} median, "
          f"{_pct(rec.coverage_min)} worst")
    print(f"    forced exits          {rec.forced_exits} "
          f"({rec.unresolved_exits} unresolved)")
    print(f"    unfilled orders       {rec.unfilled_orders}")
    print(f"    spread fallbacks      {rec.spread_fallback_orders}")
    print(f"    total holdout looks   {rec.holdout_looks_total} at the time of this run")
    print(f"    git                   {rec.git_commit[:12]}"
          f"{'  DIRTY TREE' if rec.git_dirty else ''}")
    return 0


def cmd_scoreboard(args) -> int:
    _print_scoreboard(cost_model=None if args.costs == "all" else args.costs)
    return 0


# ------------------------------------------------------------------ helpers

def _all_costs() -> tuple[str, ...]:
    from .engine import DEFAULT_COSTS
    return DEFAULT_COSTS


def _resolve_strategy(args):
    """The candidate `forward run` was given: a registered name, or a search's deliverable.

    An evolved deliverable - `<study>-ensemble` or `<study>-best` - lives in a search
    checkpoint rather than in the strategy registry, so it is looked up through
    `evolve.winners()`, the same discovery the suite and the report sets use. Its origin
    study is filled in from the checkpoint unless `--origin-study` was given, because a
    forward record without the search behind the candidate would carry `n_trials=0` for
    the most heavily searched candidate there is.
    """
    from ..backtest.cli import _apply_construction_overrides
    from ..backtest.strategy import get_strategy
    try:
        strat = get_strategy(args.strategy)
    except KeyError:
        strat, origin = _evolved_by_name(args.strategy)
        if getattr(args, "origin_study", None) is None:
            args.origin_study = origin
    _apply_construction_overrides(strat, args)
    return strat


def _evolved_by_name(name: str):
    """(strategy, study) for a genetic search's deliverable, by the name it is reported under."""
    from ..evolve.engine import winners
    found = {w["name"]: w for w in winners()}
    if name not in found:
        raise SystemExit(
            f"unknown strategy {name!r}: not a registered strategy, and no search in "
            f"data/experiments/evolve/ hands over that name (have: {sorted(found)})")
    return found[name]["strategy"], found[name]["study"]


def _seal_frame():
    from . import seal as seal_module
    return seal_module.load_seals()


def _freshness_table(forward):
    """Per sealed candidate: when it was last tested and how much data is new since."""
    import pandas as pd

    from . import seal as seal_module
    from . import store
    from .windows import freshness

    rows = []
    for seal_id in seal_module.load_seals()["seal_id"].drop_duplicates():
        s = seal_module.get(seal_id)
        if s is None:
            continue
        previous = store.previous_data_end(seal_id, s.cost_model)
        fresh, months = freshness(forward, previous)
        rows.append({
            "seal_id": seal_id, "strategy": s.strategy, "costs": s.cost_model,
            "last_tested_to": previous or "never",
            "fresh_from": fresh.start if fresh else "-",
            "fresh_months": months,
        })
    return pd.DataFrame(rows)


def _seal_all(candidates: list, args, origins: dict[str, str] | None = None) -> None:
    """Pre-register every candidate BEFORE any of them is looked at.

    This is the part of pre-registration a same-session run can still honestly deliver.
    It cannot prove the candidates were chosen without knowledge of the forward window -
    only a seal written days earlier does that - but it does prove the candidate SET was
    fixed before any forward result existed, which is what stops "test twenty, report
    three".
    """
    from .engine import seal_candidate

    settings = (args.costs,) if args.costs else _all_costs()
    print(f"Sealing {len(candidates)} candidate(s) under {len(settings)} cost "
          "setting(s) before looking at anything.")
    print("This reads the research window only. Nothing is spent here.")
    print()
    written = 0
    for candidate in candidates:
        label = (candidate if isinstance(candidate, str)
                 else getattr(candidate, "name", "?"))
        try:
            seals = seal_candidate(
                candidate, rationale=args.rationale or "sealed as part of a suite",
                research_start=args.start, costs=settings,
                initial_capital=args.capital, liquidity_floor=args.liquidity_floor,
                seed=args.seed, benchmark=args.benchmark,
                origin_study=(origins or {}).get(label, args.origin_study))
            written += len(seals)
            print(f"  sealed {label:24s} " + " ".join(s.seal_id for s in seals))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  FAILED to seal {label}: {exc}", file=sys.stderr)
    print()
    print(f"  {written} seal(s) written, 0 looks spent. "
          "`python -m sp500lab forward seals`")
    print()


def _announce(names: list[str], args) -> None:
    """Say what is about to be spent, before spending it. Never silenced."""
    from ..backtest.registry import HOLDOUT_START, holdout_touch_count

    settings = [args.costs] if args.costs else list(_all_costs())
    print("!" * 78)
    print("!! ABOUT TO READ HOLDOUT DATA")
    print("!" * 78)
    print(f"  candidates      {len(names)}: {', '.join(names[:8])}"
          f"{' ...' if len(names) > 8 else ''}")
    print(f"  window          {HOLDOUT_START} onward - the only out-of-sample data "
          "this project has")
    print(f"  cost settings   {', '.join(settings)}")
    print(f"  looks so far    {holdout_touch_count()} recorded")
    print()
    print("  Every run below is appended to the holdout ledger and cannot be undone.")
    print("  Stop now and use `--dry-run` if you have not decided what you are testing.")
    print()


def _dry_run(strategies, args) -> int:
    """Report exactly what a real run would spend, without spending any of it."""
    from ..backtest.panel import build_panel
    from ..backtest.registry import holdout_touch_count
    from . import seal as seal_module
    from . import store
    from .engine import BASELINE_STUDY
    from .seal import seal_id_for
    from .windows import describe_power, forward_window, freshness

    panel = build_panel()
    forward = forward_window(str(panel.dates[-1]), end=args.forward_end)
    settings = [args.costs] if args.costs else list(_all_costs())

    print("=" * 78)
    print("DRY RUN - nothing below has been run and nothing has been recorded")
    print("=" * 78)
    print(f"  forward window  {forward.describe()}")
    print(f"  {describe_power(forward.n_months)}")
    print(f"  looks so far    {holdout_touch_count()} recorded")
    print()
    print(f"  A real run would add {len(strategies) * len(settings)} entries to the "
          "holdout ledger:")
    print()
    for strat in strategies:
        detail = strat.describe() if hasattr(strat, "describe") else {}
        for cost_model in settings:
            sid = seal_id_for(
                strategy_class=detail.get("class", type(strat).__name__),
                params=detail.get("params", {}),
                construction=detail.get("construction"),
                cost_model=cost_model, initial_capital=args.capital,
                liquidity_floor=args.liquidity_floor, seed=args.seed)
            existing = seal_module.get(sid)
            previous = store.previous_data_end(sid, cost_model, args.mode)
            _, fresh_months = freshness(forward, previous)
            state = (f"sealed {existing.sealed_at[:10]} ({existing.seal_mode})"
                     if existing else "NOT SEALED - would be auto-sealed at run time")
            print(f"  {getattr(strat, 'name', '?'):22s} {cost_model:12s} {sid}")
            print(f"      {state}")
            print(f"      look #{store.look_number(sid, cost_model, args.mode)}, "
                  f"{fresh_months} of {forward.n_months} months would be fresh "
                  f"evidence")
    print()
    print("  To pre-register without looking:")
    print("    python -m sp500lab forward seal <strategy> --rationale \"...\"")
    print(f"  Research legs would be logged under study {BASELINE_STUDY!r}; no look.")
    return 0


def _print_scoreboard(cost_model: str | None) -> None:
    from . import store
    board = store.scoreboard(cost_model=cost_model)
    if board.empty:
        print("No forward test has been run yet. Nothing is spent.")
        return

    out = board.copy()
    for c in ("research_cagr", "forward_cagr"):
        if c in out.columns:
            out[c] = out[c].map(_pct)
    for c in ("research_sharpe", "forward_sharpe", "research_d_sharpe",
              "forward_d_sharpe", "decay_sharpe_monthly", "decay_z",
              "psr_vs_research"):
        if c in out.columns:
            out[c] = out[c].map(_num)
    print(out.to_string(index=False))

    bar = store.selection_bar(cost_model=cost_model or "realistic")
    print()
    print("  Sorted by `forward_d_sharpe` - the forward Sharpe minus the index's over")
    print("  the SAME dates. Ranking on the raw Sharpe would rank the market.")
    print()
    print(f"  MULTIPLE TESTING ON THE FORWARD WINDOW ITSELF "
          f"[{bar['cost_model']} costs]")
    print(f"    candidates looked at        {bar['n_forward_tests']}")
    print(f"    spread of forward Sharpes   {bar['spread']}")
    print(f"    luckiest-of-N bar           {bar['bar']}")
    print(f"    best forward Sharpe         {bar['best_sharpe_monthly']}")
    if bar["n_forward_tests"] > 1:
        print()
        print("    The holdout stops a strategy being FITTED to 2022 onward. It does")
        print("    not stop it being CHOSEN there. The bar above is what the luckiest")
        print("    of these candidates would have posted with no skill at all, and it")
        print("    is the same correction `experiments deflate` applies to a search.")
        print()
        print("    python -m sp500lab experiments deflate forward-test")


def _print_footer(test) -> None:
    print()
    for outcome in test.outcomes:
        rec = outcome.record
        print(f"  recorded as {rec.forward_id}  [{rec.cost_model}]  "
              f"verdict={rec.verdict}"
              + (f"  artifacts -> {rec.saved_to}" if rec.saved_to else ""))
    print()
    print("  python -m sp500lab forward scoreboard")
    print("  python -m sp500lab experiments holdout")


def _pct(v) -> str:
    return "n/a" if v is None or v != v else f"{v * 100:.2f}%"


def _num(v) -> str:
    return "n/a" if v is None or v != v else f"{v:.2f}"


def default_results_dir() -> str:
    return str(PROJECT_ROOT / "results" / "forward")
