"""`sp500lab evolve ...` - run a search, read one back, and check whether it means anything.

    sp500lab evolve run --study ga-1 --generations 25 --population 60
    sp500lab evolve run --study ga-full --preset full --metric excess_sharpe
    sp500lab evolve history ga-1
    sp500lab evolve best ga-1 --trades results/trades/ga-1

The command that matters after any of these is not in this file:

    sp500lab experiments deflate ga-1

A search reports the best of N draws. Without the correction for N, that number is not
optimistic or conservative - it is meaningless (ADR-026).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..paths import PROJECT_ROOT


def add_parser(sub) -> None:
    p = sub.add_parser("evolve", help="the genetic algorithm")
    es = p.add_subparsers(dest="evolve_command", required=True)

    r = es.add_parser("run", help="run a search")
    r.add_argument("--study", default="ga",
                   help="name of this search. Decides n_trials for the deflated "
                        "Sharpe, so give each search its own.")
    r.add_argument("--preset", default="price", choices=["price", "full", "night"],
                   help="'price' uses the whole 2007 window; 'full' adds fundamentals "
                        "and starts in 2010")
    r.add_argument("--population", type=int, default=60)
    r.add_argument("--generations", type=int, default=25)
    r.add_argument("--elite", type=int, default=4)
    r.add_argument("--immigrants", type=int, default=4)
    r.add_argument("--tournament-size", type=int, default=3)
    r.add_argument("--crossover-rate", type=float, default=0.7)
    r.add_argument("--mutation-rate", type=float, default=0.15)
    r.add_argument("--mutation-sigma", type=float, default=0.15)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-seed-baselines", action="store_true",
                   help="start from a purely random population instead of seeding it "
                        "with momentum, low-vol and the rest")

    r.add_argument("--start", default="2007-04-01")
    r.add_argument("--end", default=None)
    r.add_argument("--costs", default="realistic",
                   choices=["optimistic", "realistic", "pessimistic"])
    r.add_argument("--liquidity-floor", type=float, default=0.0)
    r.add_argument("--holdout", default="exclude",
                   choices=["exclude", "include", "only"],
                   help="searching inside the holdout destroys it; 'exclude' is the "
                        "only sane value and the other two are permanently recorded")

    r.add_argument("--metric", default="sharpe_monthly",
                   choices=["sharpe_monthly", "sharpe", "excess_sharpe", "calmar",
                            "information_ratio"])
    r.add_argument("--aggregate", default="mean_minus_std",
                   choices=["whole", "mean", "min", "mean_minus_std"],
                   help="how per-fold scores become one number. 'whole' ignores folds "
                        "and is the least robust option.")
    r.add_argument("--folds", type=int, default=4)
    r.add_argument("--embargo-days", type=int, default=31)
    r.add_argument("--turnover-penalty", type=float, default=0.0)
    r.add_argument("--complexity-penalty", type=float, default=0.0,
                   help="fitness charged per active feature; biases the search toward "
                        "explanations rather than fits")
    r.add_argument("--max-evaluations", type=int, default=None)
    r.add_argument("--no-log", action="store_true",
                   help="do not log individuals as trials. The deflated Sharpe cannot "
                        "be computed afterwards if you do this.")
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(func=cmd_run)

    h = es.add_parser("history", help="per-generation statistics of a past search")
    h.add_argument("study")
    h.set_defaults(func=cmd_history)

    b = es.add_parser("best", help="the winner of a search, re-run and explained")
    b.add_argument("study")
    b.add_argument("--generation", type=int, default=None)
    b.add_argument("--all-costs", action="store_true")
    b.add_argument("--trades", default=None, metavar="DIR",
                   help="also export the winner's buys and sells to this directory")
    b.add_argument("--holdout", default="exclude",
                   choices=["exclude", "include", "only"],
                   help="'only' is the final test. You get to do it once, and it is "
                        "recorded permanently.")
    b.set_defaults(func=cmd_best)


# ----------------------------------------------------------------- commands

def cmd_run(args) -> int:
    from . import EvolutionConfig, evolve

    config = EvolutionConfig(
        study=args.study, preset=args.preset,
        population=args.population, generations=args.generations,
        elite=args.elite, immigrants=args.immigrants,
        tournament_size=args.tournament_size, crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma,
        seed=args.seed, seed_with_baselines=not args.no_seed_baselines,
        start=args.start, end=args.end, costs=args.costs,
        liquidity_floor=args.liquidity_floor, holdout=args.holdout,
        metric=args.metric, aggregate=args.aggregate, n_folds=args.folds,
        embargo_days=args.embargo_days,
        turnover_penalty=args.turnover_penalty,
        complexity_penalty=args.complexity_penalty,
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
              "it actually uses.")

    if len(result.history):
        print()
        print("BY GENERATION")
        hist = result.history[["generation", "best_fitness", "mean_fitness",
                               "best_sharpe", "best_n_active", "diversity",
                               "scorable"]]
        print(hist.round(4).to_string(index=False))
        _warn_on_stall(result)

    print()
    if args.no_log:
        print("  !! individuals were NOT logged, so the deflated Sharpe cannot be")
        print("     computed for this search. The number above is uncorrected.")
    else:
        print("  Now correct it for the size of the search that produced it:")
        print(f"    python -m sp500lab experiments deflate {args.study}")
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
    """Re-run the winner of a past search with full diagnostics."""
    import numpy as np

    from ..backtest import run_backtest
    from ..backtest.results import compare, format_compare
    from ..strategies.evolvable import from_vector
    from ..strategies.genome import alpha_genome, describe_genome
    from .engine import load_population

    population = load_population(args.study, args.generation)
    if not population:
        print(f"  no recorded search named {args.study!r}", file=sys.stderr)
        return 1
    scored = [p for p in population if p.get("fitness") is not None]
    if not scored:
        print("  that generation has no individual with a finite fitness",
              file=sys.stderr)
        return 1

    winner = max(scored, key=lambda p: p["fitness"])
    vector = np.array(winner["vector"], dtype=np.float64)
    config = _config_of(args.study)
    preset = config.get("preset", "price")

    print("=" * 76)
    print(f"BEST OF {args.study}"
          + (f"  generation {args.generation}" if args.generation is not None else ""))
    print("=" * 76)
    print(describe_genome(alpha_genome(preset), vector))
    print()

    strategy = from_vector(vector, preset)
    strategy.name = f"{args.study}-best"
    settings = (("optimistic", "realistic", "pessimistic") if args.all_costs
                else (config.get("costs", "realistic"),))
    results = [run_backtest(strategy, costs=c, start=config.get("start", "2007-04-01"),
                            end=config.get("end"), holdout=args.holdout,
                            study=args.study, notes=f"best of {args.study}",
                            record_trades=bool(args.trades))
               for c in settings]
    for res in results:
        print(res.summary())
        print()
    if len(results) > 1:
        print(format_compare(compare(results)))

    if args.trades:
        from ..backtest.trades import (format_reconcile, holdings, reconcile,
                                       summarise, write_csv)
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

    print()
    print(f"  python -m sp500lab experiments deflate {args.study}")
    return 0


# ------------------------------------------------------------------ helpers

def _config_of(study: str) -> dict:
    from .engine import EVOLVE_DIR, _slug
    path = EVOLVE_DIR / f"{_slug(study)}.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                return json.loads(line).get("config", {})
            except json.JSONDecodeError:
                continue
    return {}


def _warn_on_stall(result) -> None:
    """Say so when the search stopped searching. This is the usual silent failure."""
    hist = result.history
    if len(hist) < 5:
        return
    final = float(hist["diversity"].iloc[-1])
    improved = float(hist["best_fitness"].iloc[-1] - hist["best_fitness"].iloc[len(hist) // 2])
    if final < 0.05:
        print()
        print(f"  !! diversity collapsed to {final:.3f}: the population converged and "
              "the later")
        print("     generations were re-evaluating one individual. Raise --immigrants "
              "or --mutation-rate.")
    elif improved <= 0:
        print()
        print("  !! the best fitness did not improve in the second half of the run. "
              "More")
        print("     generations will not help; a different search space might.")


def default_results_dir() -> str:
    return str(PROJECT_ROOT / "results" / "evolve")
