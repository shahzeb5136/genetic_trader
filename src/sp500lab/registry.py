"""Security master: stable internal IDs that outlive ticker changes.

Why this exists
---------------
Tickers are not identifiers. They get reassigned (FB->META keeps the company but
changes the symbol; a delisted ticker can be recycled to an unrelated company years
later), and one company can have several listed share classes (GOOG and GOOGL share
CIK 1652044). Keying price history on a bare ticker silently merges unrelated
companies and splits related ones - a subtle way to corrupt a backtest.

Design
------
The natural key is **(cik, ticker)**, which handles share classes correctly and
disambiguates recycled tickers by issuer. Each distinct pair gets a surrogate
`security_id` ("SID000123") assigned once, on first sight, and never changed or
reused. The registry is append-only: re-running ingestion adds new securities and
refreshes `last_seen`, but existing IDs are frozen.

Securities with no SEC registrant (foreign issuers, some ETFs, very old delistings)
get cik = 0, so they still receive a stable ID keyed on ticker alone.
"""

from __future__ import annotations

import logging

import pandas as pd

from .storage import read_silver, silver_exists, write_silver

log = logging.getLogger(__name__)

DATASET = "reference/security_master"

COLUMNS = ["security_id", "cik", "ticker", "name", "exchange", "first_seen", "last_seen"]

NO_CIK = 0  # sentinel for securities with no SEC registrant


class SecurityRegistry:
    """Append-only (cik, ticker) -> security_id map."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            df = pd.DataFrame(columns=COLUMNS)
        self.df = df.copy()
        self._index: dict[tuple[int, str], str] = {
            (int(r.cik), str(r.ticker)): str(r.security_id)
            for r in self.df.itertuples(index=False)
        }
        nums = [int(s[3:]) for s in self.df["security_id"].astype(str) if s.startswith("SID")]
        self._next = max(nums, default=0) + 1

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls) -> "SecurityRegistry":
        if silver_exists(DATASET):
            return cls(read_silver(DATASET))
        return cls()

    def save(self) -> None:
        out = self.df.sort_values("security_id").reset_index(drop=True)
        write_silver(out, DATASET)

    # ------------------------------------------------------------- assignment

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """Canonical ticker form: uppercase, class separator as '.'.

        Vendors disagree on share-class punctuation (BRK.B / BRK-B / BRK/B).
        We standardise on '.' and keep the vendor's own spelling in bronze.
        """
        t = str(ticker).strip().upper()
        return t.replace("-", ".").replace("/", ".").replace(" ", "")

    def resolve(self, cik: int | None, ticker: str) -> str | None:
        return self._index.get((int(cik or NO_CIK), self.normalize_ticker(ticker)))

    def resolve_or_assign(
        self,
        cik: int | None,
        ticker: str,
        *,
        name: str = "",
        exchange: str = "",
        seen_date: str | None = None,
    ) -> str:
        """Return the existing ID for (cik, ticker), or mint a new one."""
        cik_i = int(cik or NO_CIK)
        tick = self.normalize_ticker(ticker)
        key = (cik_i, tick)

        if key in self._index:
            sid = self._index[key]
            if seen_date is not None:
                mask = self.df["security_id"] == sid
                cur = self.df.loc[mask, "last_seen"]
                if cur.empty or pd.isna(cur.iloc[0]) or seen_date > str(cur.iloc[0]):
                    self.df.loc[mask, "last_seen"] = seen_date
            return sid

        sid = f"SID{self._next:06d}"
        self._next += 1
        self._index[key] = sid
        self.df = pd.concat([self.df, pd.DataFrame([{
            "security_id": sid, "cik": cik_i, "ticker": tick,
            "name": name, "exchange": exchange,
            "first_seen": seen_date, "last_seen": seen_date,
        }])], ignore_index=True)
        return sid

    def bulk_assign(self, rows: pd.DataFrame, *, seen_date: str | None = None) -> pd.Series:
        """Vectorised-ish assignment for a frame with cik/ticker/name/exchange columns.

        Returns a Series of security_ids aligned to `rows`.
        """
        out = []
        for r in rows.itertuples(index=False):
            out.append(self.resolve_or_assign(
                getattr(r, "cik", None),
                r.ticker,
                name=getattr(r, "name", "") or "",
                exchange=getattr(r, "exchange", "") or "",
                seen_date=seen_date,
            ))
        return pd.Series(out, index=rows.index, name="security_id")

    # ----------------------------------------------------------------- lookup

    def tickers_for_cik(self, cik: int) -> list[str]:
        return self.df.loc[self.df["cik"] == int(cik), "ticker"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"<SecurityRegistry {len(self)} securities, next={self._next}>"
