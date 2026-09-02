"""What a backtest returns, and how it is written down.

A result is not a number. Reporting a CAGR without the turnover that produced it, the
costs that were charged, the fraction of the index that was actually priced, and the
assumptions that resolved the delistings, is how a backtest lies without anyone
lying. So `BacktestResult` carries the diagnostics next to the headline, `summary()`
prints them together, and there is no accessor that returns the CAGR alone.

Reproducibility
---------------
`save()` writes the equity curve, the per-rebalance ledger and a manifest containing
the strategy's parameters, the cost model, the panel metadata and the git commit. That
manifest is what makes a result re-derivable a year later - and it is the input the
experiment registry will need when the genetic algorithms start generating thousands
of these.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostBreakdown
from .metrics import Performance

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything one backtest produced.

    equity        daily NAV, indexed by session date. The headline series.
    gross_equity  the same run with costs switched off, for the cost drag
    rebalances    one row per rebalance: NAV, turnover, positions, costs, coverage
    weights       target weights per rebalance date (rebalance x security_id)
    trades        one row per order: side, shares, as-traded price, attributed costs.
                  The evidence for the equity curve - see trades.py
    exits         positions resolved outside a rebalance - delistings and price gaps
    diagnostics   the honesty section: coverage, fallbacks, unresolved assumptions
    """

    strategy: str
    config: dict
    equity: pd.Series
    performance: Performance
    rebalances: pd.DataFrame
    weights: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    exits: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    gross_equity: pd.Series | None = field(repr=False, default=None)
    benchmark: pd.Series | None = field(repr=False, default=None)
    costs: CostBreakdown = field(default_factory=CostBreakdown)
    diagnostics: dict = field(default_factory=dict)

    # ------------------------------------------------------------- reporting

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"{self.strategy}   [{self.config.get('cost_model', '?')} costs]",
            "=" * 72,
            self.performance.summary(),
        ]
        c = self.costs.as_dict()
        lines += [
            "",
            "COSTS",
            f"  total charged    ${c['total']:,.0f}  "
            f"({c['bps_of_traded']:.1f} bp of ${c['traded_notional']:,.0f} traded)",
            f"  commission       ${c['commission']:,.0f}   spread ${c['spread']:,.0f}",
            f"  orders           {c['n_orders']:,}  "
            f"({c['n_min_commission']:,} priced by the $ minimum, "
            f"{c['n_cap_commission']:,} by the 1% cap)",
        ]
        if c["n_spread_fallback"]:
            lines.append(f"  !! {c['n_spread_fallback']:,} orders used the FALLBACK "
                         "spread - the estimator had no value")
        lines += ["", "DIAGNOSTICS"] + [f"  {k:24s} {v}" for k, v in self.diagnostics.items()]
        return "\n".join(lines)

    def annual_table(self) -> pd.DataFrame:
        """Year-by-year strategy vs benchmark, which is where a curve stops flattering."""
        from .metrics import annual_returns
        out = pd.DataFrame({"strategy": annual_returns(self.equity)})
        if self.benchmark is not None:
            out["benchmark"] = annual_returns(self.benchmark)
            out["excess"] = out["strategy"] - out["benchmark"]
        out.index = [str(i.year) for i in out.index]
        return out

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "config": self.config,
            "performance": self.performance.as_dict(),
            "costs": self.costs.as_dict(),
            "diagnostics": self.diagnostics,
        }

    # ----------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        eq = pd.DataFrame({"date": self.equity.index, "nav": self.equity.to_numpy()})
        if self.gross_equity is not None:
            eq["nav_gross"] = self.gross_equity.reindex(self.equity.index).to_numpy()
        if self.benchmark is not None:
            eq["benchmark"] = self.benchmark.reindex(self.equity.index).to_numpy()
        eq.to_parquet(d / "equity.parquet", index=False)

        self.rebalances.to_parquet(d / "rebalances.parquet", index=False)
        if len(self.weights):
            w = self.weights.copy()
            w.index = w.index.rename("date")
            w.reset_index().to_parquet(d / "weights.parquet", index=False)
        if len(self.exits):
            self.exits.to_parquet(d / "exits.parquet", index=False)
        if len(self.trades):
            self.trades.to_parquet(d / "trades.parquet", index=False)
            # CSV as well as parquet, deliberately. The parquet is for this codebase;
            # the CSV is for whoever is checking it, who may have neither Python nor a
            # reason to trust ours.
            self.trades.to_csv(d / "trades.csv", index=False)

        manifest = self.as_dict() | {"git_commit": _git_commit()}
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        log.info("result saved: %s", d)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "BacktestResult":
        d = Path(directory)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        eq = pd.read_parquet(d / "equity.parquet")
        equity = pd.Series(eq["nav"].to_numpy(), index=eq["date"].to_numpy(), name="nav")
        gross = (pd.Series(eq["nav_gross"].to_numpy(), index=eq["date"].to_numpy())
                 if "nav_gross" in eq.columns else None)
        bench = (pd.Series(eq["benchmark"].to_numpy(), index=eq["date"].to_numpy())
                 if "benchmark" in eq.columns else None)

        weights = pd.DataFrame()
        if (d / "weights.parquet").exists():
            weights = pd.read_parquet(d / "weights.parquet").set_index("date")
        exits = (pd.read_parquet(d / "exits.parquet")
                 if (d / "exits.parquet").exists() else pd.DataFrame())
        trades = (pd.read_parquet(d / "trades.parquet")
                  if (d / "trades.parquet").exists() else pd.DataFrame())

        return cls(
            strategy=manifest["strategy"], config=manifest["config"],
            equity=equity, gross_equity=gross, benchmark=bench,
            performance=Performance(**manifest["performance"]),
            rebalances=pd.read_parquet(d / "rebalances.parquet"),
            weights=weights, exits=exits, trades=trades,
            costs=CostBreakdown(**{k: v for k, v in manifest["costs"].items()
                                   if k not in ("total", "bps_of_traded")}),
            diagnostics=manifest["diagnostics"],
        )


def compare(results: list[BacktestResult], benchmark_name: str = "") -> pd.DataFrame:
    """Side-by-side scoreboard. This is what the competition ultimately prints."""
    rows = []
    for r in results:
        p = r.performance
        rows.append({
            "strategy": r.strategy,
            "costs": r.config.get("cost_model", ""),
            "CAGR": p.cagr,
            "vol": p.ann_vol,
            "Sharpe": p.sharpe,
            "maxDD": p.max_drawdown,
            "Calmar": p.calmar,
            "turnover": p.ann_turnover,
            "cost_drag": p.cost_drag,
            "names": p.avg_positions,
            "IR": p.information_ratio,
        })
    df = pd.DataFrame(rows)
    if benchmark_name and benchmark_name in set(df["strategy"]):
        base = float(df.loc[df["strategy"] == benchmark_name, "CAGR"].iloc[0])
        df["vs_bench"] = df["CAGR"] - base
    return df.sort_values("Sharpe", ascending=False).reset_index(drop=True)


def suite(results: list["BacktestResult"], benchmark: str = "SPY") -> pd.DataFrame:
    """The honest scoreboard: every strategy against the index over ITS OWN window.

    `compare()` ranks strategies against each other, which is the right thing only when
    they all ran over the same dates. In this project they do not - see
    `benchmark.over_window`. So this table carries the benchmark's CAGR and Sharpe for
    each row's own span, and the two columns anyone should actually read are `excess`
    and `d_sharpe`.

    Sorted by `d_sharpe` rather than by Sharpe. A strategy that returned 17%/yr in a
    market that returned 16%/yr did not do better than one that returned 10% in a market
    that returned 10%, and sorting on the raw number says it did.
    """
    from .benchmark import over_window

    rows = []
    for r in results:
        p = r.performance
        b = over_window(r, benchmark)
        rows.append({
            "strategy": r.strategy,
            "start": str(r.config.get("start", ""))[:7],
            "end": str(r.config.get("end", ""))[:7],
            "CAGR": p.cagr, "Sharpe": p.sharpe, "maxDD": p.max_drawdown,
            "turnover": p.ann_turnover, "names": p.avg_positions,
            "bench_CAGR": b.cagr if b else float("nan"),
            "bench_Sharpe": b.sharpe if b else float("nan"),
            "excess": (p.cagr - b.cagr) if b else float("nan"),
            "d_sharpe": (p.sharpe - b.sharpe) if b else float("nan"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("d_sharpe", ascending=False).reset_index(drop=True)


def format_suite(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ("CAGR", "maxDD", "turnover", "bench_CAGR", "excess"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:.2f}%")
    for col in ("Sharpe", "bench_Sharpe", "d_sharpe", "names"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}"
                                    if col == "d_sharpe" else f"{v:.2f}")
    return out.to_string(index=False)


def format_compare(df: pd.DataFrame) -> str:
    """Render a comparison table with percentages where percentages belong."""
    out = df.copy()
    for col in ("CAGR", "vol", "maxDD", "turnover", "cost_drag", "vs_bench"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:.2f}%")
    for col in ("Sharpe", "Calmar", "IR", "names"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    return out.to_string(index=False)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def equity_from_weights(dates: np.ndarray, nav: np.ndarray) -> pd.Series:
    """Build the equity Series, dropping the leading NaNs before the first rebalance."""
    s = pd.Series(nav, index=dates, name="nav")
    first = s.first_valid_index()
    return s.loc[first:] if first is not None else s
