"""The feature panel: every strategy's inputs, computed once, versioned, point-in-time.

Why this has to exist before the competition does
--------------------------------------------------
The stated goal of this project is a competition between genetic algorithms, neural nets
and classical rules. If each competitor computes its own momentum, its own volatility and
its own valuation ratio, the competition partly measures who wrote better feature code -
which is not the question. So features are computed once, here, and every competitor
reads the same numbers.

For the genetic algorithm it is also a hard performance requirement. A GA evaluating
10,000 individuals recomputes nothing: `score()` becomes a dot product of a genome
against a row of this panel. Recomputing a 252-day rolling regression inside a fitness
evaluation would turn a 30-minute search into a two-day one.

The grid: month ends, not every session
----------------------------------------
Features are stored only on the sessions a strategy can act on. Under the ADR-016
mandate that is the ~232 month-end rebalances, not the 4,900 sessions between them.
The difference is not cosmetic: a daily grid of 45 features over 677 securities is
590 MB and a month-end grid is 28 MB, and nothing in a monthly-rebalanced strategy can
use the rows in between. `at()` refuses a row that was not stored rather than
interpolating one, because a silently-interpolated feature is a lookahead bug wearing a
convenience API.

Point-in-time, by construction and by test
-------------------------------------------
Two different mechanisms, because the two families of feature fail differently:

  * price features are computed from `panel.adj_close[:t + 1]` - trailing windows only,
    the same discipline as ADR-017. A forward-looking window here is a coding error.
  * fundamental features are filtered on `filed_date <= as_of`, never `period_end`.
    60.8% of (security, tag, period) combinations in this dataset have been restated,
    so this is the common case rather than an edge case.

`check_leakage()` tests both at once, and it is the reason to trust any of it: rebuild
the whole matrix from a panel that physically ends at date T, with every filing after T
deleted, and assert the rows for dates <= T are bit-identical. A feature with a forward
window, or a fundamental joined on the wrong date column, cannot survive that.

Storage
-------
`(R, S, F)` float32 in one compressed npz under `data/gold/features/`, keyed by a hash
of the panel range, the feature version and the grid. float32 because a feature is a
statistical estimate to three or four significant figures at best, and float64 would
double the file to store noise. The leakage test compares float32 to float32, so the
choice does not weaken it.

Versioning
----------
`FEATURE_VERSION` is stamped into the cache key and into every result's metadata. Change
a feature definition, bump it: a backtest run against v3 features and one run against v4
are not comparable, and the registry has no way to know that unless the version travels
with the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.panel import Panel, build_panel
from ..paths import GOLD_DIR

log = logging.getLogger(__name__)

#: Bump whenever any feature's definition changes. Travels with every run.
#: v3 added the overnight/intraday decomposition (mom_on_12_1, mom_id_12_1,
#: on_minus_id_252d) and the dividend calendar (div_due_1m).
FEATURE_VERSION = 3

FEATURE_CACHE_DIR = GOLD_DIR / "features"

#: Feature families, in build order. Price first because it needs nothing but the panel;
#: fundamentals and macro touch DuckDB.
FAMILIES = ("price", "events", "fundamental", "macro")


@dataclass
class FeaturePanel:
    """(R, S, F) point-in-time features on the rebalance grid.

    `at(t)` is the only accessor the engine uses, and it takes a PANEL row index so the
    two panels never have to be aligned by date at runtime.
    """

    dates: np.ndarray            # (R,) '<U10', the sessions features exist for
    rows: np.ndarray             # (R,) int32, row indices into the price panel
    security_ids: np.ndarray     # (S,) '<U16'
    names: tuple[str, ...]       # (F,)
    values: np.ndarray           # (R, S, F) float32
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = tuple(self.names)
        self._row_pos = {int(r): i for i, r in enumerate(np.asarray(self.rows).tolist())}
        self._name_pos = {n: i for i, n in enumerate(self.names)}

    # ---------------------------------------------------------------- access

    @property
    def n_features(self) -> int:
        return len(self.names)

    def at(self, t: int) -> np.ndarray:
        """(S, F) features knowable at panel row `t`. The engine's entry point."""
        i = self._row_pos.get(int(t))
        if i is None:
            raise KeyError(
                f"no features stored for panel row {t} ({'?'}). The feature panel is "
                f"built on the {len(self.rows)} rebalance sessions only - see the module "
                "docstring. Rebuild with a wider grid rather than interpolating.")
        return self.values[i]

    def matrix(self, name: str) -> np.ndarray:
        """(R, S) one feature across the whole grid. For analysis, not for the engine."""
        return self.values[:, :, self._index(name)]

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.matrix(name), index=self.dates,
                            columns=self.security_ids)

    def _index(self, name: str) -> int:
        try:
            return self._name_pos[name]
        except KeyError:
            raise KeyError(f"unknown feature {name!r}; have {self.names}") from None

    def has(self, name: str) -> bool:
        return name in self._name_pos

    def coverage(self) -> pd.DataFrame:
        """Per feature: how often it is actually a number rather than NaN.

        The honest counterpart to a feature list. A fundamental feature that is 40%
        populated in 2010 and 95% populated in 2020 will make any strategy using it look
        like it improved over time when only the data did.
        """
        finite = np.isfinite(self.values)
        rows = []
        for f, name in enumerate(self.names):
            col = finite[:, :, f]
            per_date = col.mean(axis=1)
            rows.append({
                "feature": name,
                "overall": float(col.mean()),
                "first_date": str(self.dates[int(np.argmax(per_date > 0))])
                if per_date.any() else "",
                "worst_year": _worst_year(self.dates, per_date),
                "recent": float(per_date[-12:].mean()) if len(per_date) else float("nan"),
            })
        return pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, dates=self.dates, rows=self.rows, security_ids=self.security_ids,
            names=np.array(self.names), values=self.values,
            meta=np.array(json.dumps(self.meta)))
        return path

    @classmethod
    def load(cls, path: Path) -> "FeaturePanel":
        z = np.load(path, allow_pickle=False)
        return cls(dates=z["dates"], rows=z["rows"], security_ids=z["security_ids"],
                   names=tuple(str(n) for n in z["names"]), values=z["values"],
                   meta=json.loads(str(z["meta"])))


def _worst_year(dates: np.ndarray, per_date: np.ndarray) -> str:
    if not len(dates):
        return ""
    years = pd.Series(per_date, index=[d[:4] for d in dates.tolist()])
    by_year = years.groupby(level=0).mean()
    return f"{by_year.idxmin()} ({by_year.min():.0%})"


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

_MEMO: dict[str, FeaturePanel] = {}


def build_features(
    panel: Panel | None = None,
    *,
    families: tuple[str, ...] = FAMILIES,
    data_cutoff: str | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
) -> FeaturePanel:
    """Compute (or load) the feature panel for a price panel.

    `data_cutoff` deletes every fundamental filing and macro observation published after
    that date before anything is computed. Nothing in normal use needs it - the builders
    already filter on `filed_date <= as_of` - and that is exactly why it exists: it lets
    `check_leakage` prove the claim instead of asserting it. A run with a cutoff is never
    cached, so it cannot contaminate a real one.
    """
    panel = panel or build_panel()
    rows = np.asarray(panel.rebalance_index, dtype=np.int32)
    key = _cache_key(panel, families, rows)

    if data_cutoff is None and use_cache and not rebuild:
        if key in _MEMO:
            return _MEMO[key]
        path = FEATURE_CACHE_DIR / f"{key}.npz"
        if path.exists():
            log.info("features: loading cache %s", path.name)
            fp = FeaturePanel.load(path)
            _MEMO[key] = fp
            return fp

    fp = _build(panel, rows, families, data_cutoff)
    if data_cutoff is None and use_cache:
        path = fp.save(FEATURE_CACHE_DIR / f"{key}.npz")
        log.info("features: cached to %s (%.1f MB)", path.name,
                 path.stat().st_size / 1e6)
        _MEMO[key] = fp
    return fp


def clear_memo() -> None:
    _MEMO.clear()


def _build(panel: Panel, rows: np.ndarray, families: tuple[str, ...],
           data_cutoff: str | None) -> FeaturePanel:
    from . import events, fundamental, macro, price

    builders = {"price": price.compute, "events": events.compute,
                "fundamental": fundamental.compute, "macro": macro.compute}
    unknown = set(families) - set(builders)
    if unknown:
        raise ValueError(f"unknown feature families {sorted(unknown)}; "
                         f"have {sorted(builders)}")

    columns: dict[str, np.ndarray] = {}
    for fam in families:
        log.info("features: computing %s", fam)
        produced = builders[fam](panel, rows, data_cutoff=data_cutoff)
        for name, mat in produced.items():
            if name in columns:
                raise ValueError(f"feature {name!r} produced by two families")
            columns[name] = _as_grid(mat, len(rows), panel.n_securities, name)

    names = tuple(columns)
    values = np.empty((len(rows), panel.n_securities, len(names)), dtype=np.float32)
    for f, name in enumerate(names):
        values[:, :, f] = columns[name]

    meta = {
        "feature_version": FEATURE_VERSION,
        "families": list(families),
        "n_rows": int(len(rows)), "n_securities": int(panel.n_securities),
        "n_features": len(names),
        "start": str(panel.dates[rows[0]]), "end": str(panel.dates[rows[-1]]),
        "panel_start": panel.meta.get("start"), "panel_end": panel.meta.get("end"),
        "data_cutoff": data_cutoff,
        "names": list(names),
    }
    return FeaturePanel(dates=panel.dates[rows], rows=rows,
                        security_ids=panel.security_ids, names=names,
                        values=values, meta=meta)


def _as_grid(mat: np.ndarray, r: int, s: int, name: str) -> np.ndarray:
    """Accept an (R,S) matrix or an (R,) per-date series, return (R,S) float32.

    Macro features are one number per date. Broadcasting them across securities costs
    0.6 MB each and buys a single uniform accessor - `ctx.feature("vix")` works the same
    as `ctx.feature("mom_12_1")`, and no strategy has to know which kind it asked for.
    """
    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape == (r, s):
        return arr
    if arr.shape == (r,):
        return np.repeat(arr[:, None], s, axis=1)
    raise ValueError(f"feature {name!r} has shape {arr.shape}, expected ({r},{s}) "
                     f"or ({r},)")


def _cache_key(panel: Panel, families: tuple[str, ...], rows: np.ndarray) -> str:
    """Identity of a feature build: the feature version, the families, and the panel.

    The panel's extent is read from its ARRAYS, not from `panel.meta`. `truncate_panel`
    slices the arrays and leaves `meta["end"]` and `meta["n_dates"]` describing the
    original - so a key built from meta alone would let a truncated panel load the full
    panel's cached features, which is the leak the truncation exists to test for.
    """
    payload = json.dumps({
        "v": FEATURE_VERSION,
        "families": sorted(families),
        "panel": {"start": panel.meta.get("start"),
                  "first": str(panel.dates[0]), "last": str(panel.dates[-1]),
                  "n_dates": int(panel.n_dates), "n_securities": int(len(panel.security_ids)),
                  "format_version": panel.meta.get("format_version")},
        "n_rows": int(len(rows)),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# The leakage test - the reason to trust any of the above
# --------------------------------------------------------------------------

def check_leakage(panel: Panel | None = None, *, cut_at: str = "2016-12-30",
                  families: tuple[str, ...] = FAMILIES) -> dict:
    """Rebuild with the future deleted and assert the past is unchanged.

    Two things are deleted, because a feature can reach forward in two different ways:

      * the price panel is physically truncated at `cut_at`, so a rolling window that
        centred instead of trailing has no rows to centre on
      * every filing and macro observation published after `cut_at` is dropped, so a
        fundamental joined on `period_end` instead of `filed_date` loses the value it
        was illegally reading

    Then every stored row on or before `cut_at` must be bit-identical. Not close -
    identical. A feature that differs by 1e-9 is still reading something it should not.

    Returns a per-feature report; `ok` is the only field that matters.
    """
    panel = panel or build_panel()
    full = build_features(panel, families=families, use_cache=False)

    keep = full.dates <= cut_at
    if not keep.any():
        raise ValueError(f"no feature rows on or before {cut_at}")
    cut_row = int(full.rows[keep][-1])

    truncated = truncate_panel(panel, cut_row)
    partial = build_features(truncated, families=families, data_cutoff=cut_at,
                             use_cache=False)

    n = int(keep.sum())
    if len(partial.rows) < n:
        raise ValueError(
            f"truncated build produced {len(partial.rows)} rows, expected at least {n}")

    report = {"cut_at": cut_at, "rows_compared": n,
              "securities": int(full.values.shape[1]), "features": []}
    for name in full.names:
        x = full.matrix(name)[:n]
        if not partial.has(name):
            # A whole feature can vanish when its source has nothing before the cut -
            # the ICE credit spreads only exist from 2023. That is fine ONLY if the full
            # build also has nothing there. If it does have values, the feature is
            # reading data published after the cut, which is the leak this looks for.
            leaked = bool(np.isfinite(x).any())
            report["features"].append({
                "feature": name, "identical": not leaked,
                "note": ("PRESENT before the cut but absent once later data is removed"
                         if leaked else "no data before the cut in either build"),
            })
            continue
        y = partial.matrix(name)[:n]
        same = np.array_equal(np.nan_to_num(x, nan=-9e30),
                              np.nan_to_num(y, nan=-9e30))
        entry = {"feature": name, "identical": bool(same)}
        if not same:
            diff = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
            entry["max_abs_diff"] = float(diff.max())
            entry["n_differing"] = int((diff > 0).sum())
            entry["first_differing_date"] = str(
                full.dates[int(np.argmax(diff.max(axis=1) > 0))])
        report["features"].append(entry)

    extra = [n_ for n_ in partial.names if not full.has(n_)]
    if extra:
        report["unexpected_in_truncated_build"] = extra
    report["ok"] = all(e["identical"] for e in report["features"])
    report["failed"] = [e["feature"] for e in report["features"] if not e["identical"]]
    return report


def truncate_panel(panel: Panel, last_row: int) -> Panel:
    """A copy of the panel that physically ends at `last_row`.

    Every (D, S) matrix is sliced, so a forward-looking window has nothing to look at
    rather than quietly returning a different answer. The per-security vectors that
    encode the FUTURE - `last_bar_index`, the delisting outcome - are clamped too:
    leaving them alone would tell a feature builder how a security's life ended.
    """
    from dataclasses import replace

    d = last_row + 1
    return replace(
        panel,
        dates=panel.dates[:d],
        adj_close=panel.adj_close[:d], adj_open=panel.adj_open[:d],
        raw_close=panel.raw_close[:d], raw_open=panel.raw_open[:d],
        cum_split=panel.cum_split[:d], dollar_volume=panel.dollar_volume[:d],
        half_spread=panel.half_spread[:d], in_index=panel.in_index[:d],
        has_price=panel.has_price[:d], index_size=panel.index_size[:d],
        last_bar_index=np.minimum(panel.last_bar_index, last_row),
        delist_return=np.zeros_like(panel.delist_return),
        delist_reason=np.full_like(panel.delist_reason, "unknown"),
        rebalance_index=panel.rebalance_index[panel.rebalance_index <= last_row],
        meta=dict(panel.meta) | {"truncated_at": str(panel.dates[last_row])},
    )


def format_leakage(report: dict) -> str:
    lines = ["=" * 72,
             f"FEATURE LEAKAGE CHECK   cut at {report['cut_at']}",
             "=" * 72,
             f"  {report['rows_compared']} rebalance rows x "
             f"{report['securities']} securities, "
             f"{len(report['features'])} features",
             ""]
    for e in report["features"]:
        mark = "ok  " if e["identical"] else "LEAK"
        if "max_abs_diff" in e:
            detail = (f"   max diff {e['max_abs_diff']:.6g} on {e['n_differing']} "
                      f"cell(s), first {e['first_differing_date']}")
        elif e.get("note"):
            detail = f"   {e['note']}"
        else:
            detail = ""
        lines.append(f"  [{mark}] {e['feature']}{detail}")
    lines += ["",
              "  PASS - deleting the future changed nothing about the past."
              if report["ok"] else
              f"  FAIL - {len(report['failed'])} feature(s) read data they could not "
              "have had: " + ", ".join(report["failed"])]
    return "\n".join(lines)
