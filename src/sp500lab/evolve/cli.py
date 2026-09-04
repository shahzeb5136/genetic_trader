"""`sp500lab evolve ...` - run a search, read one back, and check whether it means anything.

    sp500lab evolve run --study ga-1 --seeds 3
    sp500lab evolve run --study ga-1 --preset families-price --seeds 3
    sp500lab evolve history ga-1
    sp500lab evolve ensemble ga-1 --all-costs --trades results/trades/ga-1
    sp500lab evolve best ga-1

The command that matters after any of these is not in this file:

    sp500lab experiments deflate ga-1

A search reports the best of N draws. Without the correction for N, that number is not
optimistic or conservative - it is meaningless (ADR-026). And what a search hands on is
its ENSEMBLE, not its champion (ADR-050): `evolve ensemble` is the one to read.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..paths import PROJECT_ROOT
from ..strategies.genome import all_presets
from .config import EvolutionConfig

#: The engine's own defaults, so the flags below cannot drift from them.
_D = EvolutionConfig()


def add_parser(sub) -> None:
    p = sub.add_parser("evolve", help="the genetic algorithm")
    es = p.add_subparsers(dest="evolve_command", required=True)

    r = es.add_parser("run", help="run a search")
    r.add_argument("--study", default="ga",
                   help="name of this search. Decides n_trials for the deflated "
                        "Sharpe, so give each search its own.")
    r.add_argument("--preset", default=_D.preset, choices=list(all_presets()),
                   help="'families' (default): nine prior-signed stories, at most three "
                        "live, from 2010-07; 'families-price': the five visible from "
                        "2007 without a filing; 'price' / 'full' / 'night': one free "
                        "weight per feature, the pre-2026-09 spaces")
    r.add_argument("--population", type=int, default=_D.population)
    r.add_argument("--generations", type=int, default=_D.generations)
    r.add_argument("--elite", type=int, default=_D.elite)
    r.add_argument("--immigrants", type=int, default=_D.immigrants)
    r.add_argument("--tournament-size", type=int, default=_D.tournament_size)
    r.add_argument("--crossover-rate", type=float, default=_D.crossover_rate)
    r.add_argument("--mutation-rate", type=float, default=_D.mutation_rate)
    r.add_argument("--mutation-sigma", type=float, default=_D.mutation_sigma)
    r.add_argument("--seed", type=int, default=_D.seed)
    r.add_argument("--seeds", type=int, default=_D.n_seeds, dest="n_seeds",
                   help="how many independent searches to run (seeds --seed, --seed+1, "
                        "...). They share the study and the objective, and the ensemble "
                        "pools their best individuals.")
    r.add_argument("--no-seed-baselines", action="store_true",
                   help="start from a purely random population instead of seeding it "
                        "with one individual per family (or momentum, low-vol and the "
                        "rest on a feature preset)")

    r.add_argument("--start", default=_D.start)
    r.add_argument("--end", default=None)
    r.add_argument("--costs", default=_D.costs,
                   choices=["optimistic", "realistic", "pessimistic"],
                   help="the cost setting the search is CHARGED. Pessimistic by default: "
                        "a rule that only works under a kind spread estimate never "
                        "scores well (ADR-049).")
    r.add_argument("--liquidity-floor", type=float, default=_D.liquidity_floor)
    r.add_argument("--holdout", default=_D.holdout,
                   choices=["exclude", "include", "only"],
                   help="searching inside the holdout destroys it; 'exclude' is the "
                        "only sane value and the other two are permanently recorded")

    r.add_argument("--metric", default=_D.metric,
                   choices=["sharpe_monthly", "sharpe", "excess_sharpe", "calmar",
                            "information_ratio"])
    r.add_argument("--fold-scheme", default=_D.fold_scheme,
                   choices=["random", "contiguous"],
                   help="'random': N sub-periods of --fold-years drawn once from "
                        "--fold-seed; 'contiguous': equal spans with an embargo")
    r.add_argument("--folds", type=int, default=_D.n_folds,
                   help="how many sub-periods each individual is scored on")
    r.add_argument("--fold-years", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=(_D.fold_min_years, _D.fold_max_years),
                   help="length range of a random sub-period, in years")
    r.add_argument("--fold-seed", type=int, default=_D.fold_seed,
                   help="seed the sub-periods are drawn from. Fixed across --seeds on "
                        "purpose, so every seed is scored on the same ones.")
    r.add_argument("--embargo-days", type=int, default=_D.embargo_days,
                   help="gap between contiguous folds; unused by the random scheme")
    r.add_argument("--aggregate", default=_D.aggregate,
                   choices=["whole", "mean", "min", "mean_minus_std", "quantile"],
                   help="how per-sub-period scores become one number. 'quantile' at "
                        "--quantile is the worst quarter by default; 'min' is the "
                        "worst; 'whole' ignores sub-periods and is the least robust.")
    r.add_argument("--quantile", type=float, default=_D.quantile)
    r.add_argument("--turnover-penalty", type=float, default=_D.turnover_penalty,
                   help="fitness charged per 100%% of annual turnover, on top of the "
                        "cost model")
    r.add_argument("--complexity-penalty", type=float, default=_D.complexity_penalty,
                   help="fitness charged per feature the individual reads")
    r.add_argument("--family-penalty", type=float, default=_D.family_penalty,
                   help="fitness charged per family the individual backs")
    r.add_argument("--gate-penalty", type=float, default=_D.gate_penalty,
                   help="fitness charged for switching the regime gate on")
    r.add_argument("--ensemble-size", type=int, default=_D.ensemble_size,
                   help="average the top N distinct individuals across every seed into "
                        "the search's deliverable; 0 keeps only the champion")
    r.add_argument("--max-evaluations", type=int, default=None)
    r.add_argument("--no-log", action="store_true",
                   help="do not log individuals as trials. The deflated Sharpe cannot "
                        "be computed afterwards if you do this.")
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(func=cmd_run)

    h = es.add_parser("history", help="per-generation statistics of a past search")
    h.add_argument("study")
    h.set_defaults(func=cmd_history)

    b = es.add_parser("best", help="the champion of a search, re-run and explained. "
                                   "Read `ensemble` first - the champion is the "
                                   "luckiest draw, not the deliverable.")
    b.add_argument("study")
    b.add_argument("--generation", type=int, default=None)
    b.add_argument("--all-costs", action="store_true")
    b.add_argument("--trades", default=None, metavar="DIR",
                   help="also export the champion's buys and sells to this directory")
    b.add_argument("--holdout", default="exclude",
                   choices=["exclude", "include", "only"],
                   help="'only' is the final test. You get to do it once, and it is "
                        "recorded permanently.")
    b.set_defaults(func=cmd_best)

    e = es.add_parser("ensemble", help="the ensemble of a search - its deliverable - "
                                       "re-run beside its champion")
    e.add_argument("study")
    e.add_argument("--rebuild", action="store_true",
                   help="(re)build the ensemble from the checkpoint, across every seed, "
                        "before running it. Needed for a search that predates "
                        "ensembles or was interrupted.")
    e.add_argument("--size", type=int, default=None,
                   help="with --rebuild: how many individuals to average")
    e.add_argument("--all-costs", action="store_true")
    e.add_argument("--trades", default=None, metavar="DIR",
                   help="also export the ensemble's buys and sells to this directory")
    e.add_argument("--holdout", default="exclude",
                   choices=["exclude", "include", "only"],
                   help="'only' is the final test. You get to do it once, and it is "
                        "recorded permanently.")
    e.set_defaults(func=cmd_ensemble)


# ----------------------------------------------------------------- commands

def cmd_run(args) -> int:
    from .engine import evolve

    lo, hi = args.fold_years
    config = EvolutionConfig(
        study=args.study, preset=args.preset,
        population=args.population, generations=args.generations,
        elite=args.elite, immigrants=args.immigrants,
        tournament_size=args.tournament_size, crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma,
        seed=args.seed, n_seeds=args.n_seeds,
        seed_with_baselines=not args.no_seed_baselines,
        start=args.start, end=args.end, costs=args.costs,
        liquidity_floor=args.liquidity_floor, holdout=args.holdout,
        metric=args.metric, fold_scheme=args.fold_scheme, n_folds=args.folds,
        fold_min_years=float(lo), fold_max_years=float(hi), fold_seed=args.fold_seed,
        embargo_days=args.embargo_days, aggregate=args.aggregate,
        quantile=args.quantile,
        turnover_penalty=args.turnover_penalty,
        complexity_penalty=args.complexity_penalty,
        family_penalty=args.family_penalty, gate_penalty=args.gate_penalty,
        ensemble_size=args.ensemble_size,
        log_runs=not args.no_log, max_evaluations=args.max_evaluations)

    result = evolve(config)
    print()
    print(result.summary())

    board = result.leaderboard(args.top)
    if len(board):
        print()
        print(f"TOP {len(board)} INDIVIDUALS")
        out = board.copy()
        for col in ("cagr", "max_drawdown", "turnover"):
            out[col] = out[col].map(lambda v: f"{v * 100:.2f}%")
        for col in ("fitness", "sharpe"):
            out[col] = out[col].map(lambda v: f"{v:.3f}")
        print(out.drop(columns=["error"]).to_string(index=False))
        print("  `id` is a short hash of the genome; `n_active` is how many features "
              "it reads, `n_families` how many stories it backs.")

    if len(result.history):
        print()
        print("BY GENERATION")
        cols = ["seed", "generation", "best_fitness", "mean_fitness", "best_sharpe",
                "best_n_active", "best_n_families", "diversity", "scorable"]
        hist = result.history[[c for c in cols if c in result.history.columns]]
        print(hist.round(4).to_string(index=False))
        _warn_on_stall(result)

    print()
    if args.no_log:
        print("  !! individuals were NOT logged, so the deflated Sharpe cannot be")
        print("     computed for this search. The numbers above are uncorrected.")
    else:
        print("  Now correct it for the size of the search that produced it:")
        print(f"    python -m sp500lab experiments deflate {args.study}")
    if result.ensemble:
        print("  Then read the deliverable, under all three cost settings:")
        print(f"    python -m sp500lab evolve ensemble {args.study} --all-costs")
    return 0


def cmd_history(args) -> int:
    from .engine import load_history
    hist = load_history(args.study)
    if hist.empty:
        print(f"  no recorded search named {args.study!r}", file=sys.stderr)
        return 1
    print(hist.round(4).to_string(index=False))
    return 0


def cmd_best(args) -> int:
    """Re-run the champion of a past search with full diagnostics."""
    import numpy as np

    from ..backtest import run_backtest
    from ..backtest.results import compare, format_compare
    from ..strategies.evolvable import from_vector
    from ..strategies.genome import alpha_genome, describe_genome
    from .engine import champion, load_ensemble, load_population, study_config

    if args.generation is None:
        winner = champion(args.study)
    else:
        population = [p for p in load_population(args.study, args.generation)
                      if p.get("fitness") is not None]
        winner = max(population, key=lambda p: p["fitness"]) if population else None
    if winner is None:
        print(f"  no recorded search named {args.study!r}, or no individual with a "
              "finite fitness", file=sys.stderr)
        return 1

    vector = np.array(winner["vector"], dtype=np.float64)
    config = study_config(args.study)
    preset = config.get("preset", "price")

    print("=" * 76)
    print(f"CHAMPION OF {args.study}"
          + (f"  generation {args.generation}" if args.generation is not None else
             f"  (seed {winner.get('seed', config.get('seed', 0))}, generation "
             f"{winner.get('generation')})"))
    print("=" * 76)
    print(describe_genome(alpha_genome(preset), vector))
    print()
    if load_ensemble(args.study):
        print("  This search has an ensemble, which is its deliverable. The champion is")
        print("  the luckiest single draw; read `sp500lab evolve ensemble "
              f"{args.study}` first.")
        print()

    strategy = from_vector(vector, preset)
    strategy.name = f"{args.study}-best"
    results = _run_settings(strategy, args, config)
    for res in results:
        print(res.summary())
        print()
    if len(results) > 1:
        print(format_compare(compare(results)))
    _export_trades(results, args)
    print()
    print(f"  python -m sp500lab experiments deflate {args.study}")
    return 0


def cmd_ensemble(args) -> int:
    """The ensemble of a search, re-run with full diagnostics beside its champion."""
    import numpy as np

    from ..backtest.results import compare, format_compare
    from ..strategies.evolvable import from_vector
    from .engine import (build_ensemble, champion, ensemble_strategy, load_ensemble,
                         study_config)

    config = study_config(args.study)
    if not config:
        print(f"  no recorded search named {args.study!r}", file=sys.stderr)
        return 1
    record = load_ensemble(args.study)
    if args.rebuild or record is None:
        if record is None and not args.rebuild:
            print(f"  {args.study} has no stored ensemble; building one from its "
                  "checkpoint.")
        record = build_ensemble(args.study, size=args.size)

    strategy = ensemble_strategy(args.study, record)
    print("=" * 76)
    print(f"ENSEMBLE OF {args.study}  ({record['size']} individuals, "
          f"seeds {record.get('seeds')})")
    print("=" * 76)
    print(strategy.explain())
    print()
    members = record["members"]
    print(f"  members' fitness: best {members[0]['fitness']}, "
          f"worst {members[-1]['fitness']}")
    ev = record.get("evaluation") or {}
    if ev:
        print(f"  at build time ({ev.get('costs')} costs): robust score "
              f"{ev.get('fitness')}, Sharpe {ev.get('sharpe')}, CAGR {ev.get('cagr')}")
    print()

    results = _run_settings(strategy, args, config)
    for res in results:
        print(res.summary())
        print()

    best = champion(args.study)
    if best is not None:
        champ = from_vector(np.array(best["vector"], dtype=float), config["preset"])
        champ.name = f"{args.study}-best"
        # The champion is re-run under the same setting(s) so the two can be read side
        # by side - the point of the ensemble is what it does to the champion's number.
        results += _run_settings(champ, args, config)
    if len(results) > 1:
        print(format_compare(compare(results)))
    _export_trades(results[:1], args)
    print()
    print(f"  python -m sp500lab experiments deflate {args.study}")
    return 0


# ------------------------------------------------------------------ helpers

def _run_settings(strategy, args, config: dict) -> list:
    from ..backtest import run_backtest
    settings = (("optimistic", "realistic", "pessimistic") if args.all_costs
                else (config.get("costs", "realistic"),))
    return [run_backtest(strategy, costs=c, start=config.get("start", "2007-04-01"),
                         end=config.get("end"), holdout=args.holdout,
                         study=args.study, notes=f"{strategy.name} re-run",
                         record_trades=bool(args.trades))
            for c in settings]


def _export_trades(results: list, args) -> None:
    if not args.trades or not results:
        return
    from ..backtest.trades import (format_reconcile, holdings, reconcile, summarise,
                                   write_csv)
    from ..reporting import trades_report
    from ..reporting.render.html import write as write_html

    res = results[-1] if len(results) == 1 else results[1]
    out = Path(args.trades)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(res.trades, out / "trades.csv")
    holdings(res).to_csv(out / "holdings.csv", index=False)
    page = write_html(trades_report(res), out / "trades.html")
    print()
    print(summarise(res.trades).to_string(index=False))
    print()
    print(format_reconcile(reconcile(res.trades, res)))
    print()
    print(f"  wrote {out / 'trades.csv'} ({len(res.trades):,} orders)")
    print(f"  wrote {out / 'holdings.csv'}")
    print(f"  wrote {page}  (self-contained page, CSV embedded in it)")


def _warn_on_stall(result) -> None:
    """Say so when the search stopped searching. This is the usual silent failure."""
    hist = result.history
    if hist.empty:
        return
    seeds = hist["seed"].unique() if "seed" in hist.columns else [None]
    for seed in seeds:
        h = hist if seed is None else hist[hist["seed"] == seed]
        if len(h) < 5:
            continue
        tag = "" if seed is None else f" (seed {int(seed)})"
        final = float(h["diversity"].iloc[-1])
        improved = float(h["best_fitness"].iloc[-1]
                         - h["best_fitness"].iloc[len(h) // 2])
        if final < 0.05:
            print()
            print(f"  !! diversity collapsed to {final:.3f}{tag}: the population "
                  "converged and the later")
            print("     generations were re-evaluating one individual. Raise "
                  "--immigrants or --mutation-rate.")
        elif improved <= 0:
            print()
            print(f"  !! the best fitness did not improve in the second half of the "
                  f"run{tag}. More")
            print("     generations will not help; a different search space might.")


def default_results_dir() -> str:
    return str(PROJECT_ROOT / "results" / "evolve")
