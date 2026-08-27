# Data sources

Every source currently in use, what it costs, what it actually delivers, and where it bites.

**Current total cost: \$0/month.** Budget is \$20/month; the first spend is planned for a paid
EOD feed (see *Planned* below).

---

## Active sources

### SEC EDGAR — identity + fundamentals

| | |
|---|---|
| Cost | Free, no key |
| Endpoints | `www.sec.gov/files/company_tickers_exchange.json`, `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` |
| Rate limit | 10 req/s published ceiling — we run at 8 |
| Auth | **A descriptive `User-Agent` with a real contact address is mandatory.** Requests without one get 403. |

Provides the CIK↔ticker↔exchange spine (10,388 securities, 8,002 registrants) and
point-in-time XBRL fundamentals.

This is the **primary record** — the same filings commercial fundamental vendors parse and
resell. Every fact carries the accession's filing date, which is what makes genuine
point-in-time analysis possible at all.

**Gotchas**
- `company_tickers_exchange.json` lists *current* registrants only. Companies that delisted
  years ago keep their CIK in EDGAR but drop out of this file, so it cannot supply a
  survivorship-free universe on its own.
- Large filers' `companyfacts` payloads run 3–4 MB each; the full S&P 500 pull is ~1.5 GB of
  bronze. Cached with a 7-day TTL.
- Tag usage is inconsistent across filers and eras. `Revenues` vs
  `RevenueFromContractWithCustomerExcludingAssessedTax` vs `SalesRevenueNet` all mean revenue
  depending on the year and the filer. `DEFAULT_TAGS` covers the common variants; reconciling
  them into one series is a Phase 3 job.

---

### Wikipedia — index membership

| | |
|---|---|
| Cost | Free, no key |
| Endpoints | `en.wikipedia.org/w/api.php` (MediaWiki API) |
| Rate limit | Self-imposed 4 req/s |

Two distinct uses:

1. **Current constituents** — `List of S&P 500 companies` (503 names with GICS sector, CIK,
   date added).
2. **Point-in-time membership** — the revision history of that same page. 3,143 revisions
   since 2005-09; we take the last of each month and parse it. See ADR-001.
3. **Change events** — `Historical components of the S&P 500` (407 add/remove events with
   effective dates).

**Gotchas**
- **Old-revision latency is ~22s per request**, apparently fixed cost. The API returns up to
  **50 revisions per call**, so batching is essential — it turns a 90-minute job into ~2
  minutes. All revision content is cached forever (old revisions are immutable).
- **The table format changed repeatedly.** Plain ticker (2008), `{{NyseSymbol|X}}` template
  (~2009), newline-separated cells (~2020), and **Company-first column order in 2007**. The
  parser detects the ticker column from the header and asserts a plausible row count, so a
  future change fails loudly instead of silently returning 4 tickers.
- **No constituent table at all before 2007-03** — bulleted company names, no tickers. Hard
  floor on history. See ADR-004.
- **Change events are under-recorded before 2010** — ~7/year vs the ~20/year that actually
  occurred. See ADR-010.
- It is a volunteer-maintained secondary source. Best free option; not ground truth.

---

### Yahoo Finance (`yfinance`) — prices, **placeholder**

| | |
|---|---|
| Cost | Free, no key |
| Access | `yfinance` package (unofficial API) |
| Status | **Temporary.** To be replaced by a paid feed. |

Daily OHLCV + dividends + splits, 2000 → present. 3.71M bars across 677 securities.

**The critical gotcha — read before using this data**

With `auto_adjust=False`, the OHLC columns are **already split-adjusted**; only dividends are
left unadjusted. NVDA's 2024-06-06 close comes back as ~\$121, not the ~\$1,210 that actually
traded. Applying split factors again double-counts by 10×, and the failure is *silent* — it
looks like alpha. Handled via `SOURCE_CONVENTION = split_adjusted`. See ADR-007.

**Other gotchas**
- **Coverage is 69.6% of ever-members** (677 / 973). Delisted names are largely absent — LEH,
  XTO, ZMH, YHOO all return nothing. This is the single largest known weakness and the exact
  reason to buy a paid feed.
- **Ticker recycling is not handled by the vendor.** CPWR returns an OTC shell's prices
  (0–50 shares/day) spliced onto Compuware's index-era history. 155 tickers affected.
- Unofficial API. It can break without notice; it is not licensed for redistribution.
- Bypasses `http_cache` (manages its own session); fetch-once is enforced at the bronze-artifact
  level instead.

---

### FRED — macro

| | |
|---|---|
| Cost | Free. **No key needed** via the CSV endpoint. |
| Endpoint | `fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES` |

18 series, 127,457 observations: Treasury yields and spreads, fed funds, VIX, credit spreads,
the dollar index, WTI, plus CPI / unemployment / payrolls / industrial production / GDP /
recession indicator.

**Gotcha — revision risk.** CPI, GDP, payrolls and unemployment are **restated for months**
after first publication. Today's value for March 2020 is not what was on the screen in April
2020, so using it in a backtest is a genuine look-ahead leak. Market-based series (yields, VIX,
spreads) are final on publication.

Each series is tagged `revised: true/false` — 7 of 18 are flagged. See ADR-011.

**Gotcha — truncated history on licensed series.** The two ICE BofA option-adjusted spread
series (`BAMLH0A0HYM2`, `BAMLC0A0CM`) return only ~3 years (from 2023-08) via the keyless CSV
endpoint; everything else returns full history (DGS10 back to 1962, VIXCLS to 1990). They are
licensed third-party data and FRED caps bulk download. Flagged by `check_macro_history_depth`.
Do not use them for pre-2023 regime tagging — a full history needs a FRED API key or another
source.

Setting `FRED_API_KEY` unlocks the JSON API and ALFRED vintages, but nothing here requires it.

---

## Verified-available, not yet wired in

| Source | Endpoint | Notes |
|---|---|---|
| FINRA short volume | `cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt` | Free, official, daily. Pipe-delimited. Confirmed working. |
| SEC Financial Statement Data Sets | `sec.gov/files/dera/data/financial-statement-data-sets/{YYYY}q{Q}.zip` | Bulk quarterly XBRL. Faster than per-CIK for wide pulls. Confirmed working. |
| SEC EDGAR full-index | `sec.gov/Archives/edgar/full-index/{YYYY}/QTR{Q}/company.idx` | All filings by quarter — gives exact 8-K/earnings dates. Confirmed working. |
| SEC 13F | EDGAR | Institutional holdings, 45-day lag. |

---

## Rejected

**Stooq** — planned as a second price source for cross-validation. Now returns a JavaScript
bot-detection interstitial instead of CSV. Dropped (ADR-012).

---

## EODHD — ACTIVE (free tier)

Registered and ingesting. **Free tier measured limits: 20 calls/day, ~1 year of history,
`/user` is free.** See ADR-014.

Currently held (2 calls, in `data/vault/eodhd/`):

* `us_active.json` — 51,206 active US symbols
* `us_delisted.json` — 59,826 delisted US symbols

with code, name, exchange, currency, type and ISIN for each.

**Headline finding: 281 of 295 (95.3%) of the S&P 500 ever-members Yahoo cannot supply
are present in EODHD's delisted list.** That is the justification for the annual plan.

⚠️ **Price convention unverified.** Docs say OHLC is raw but volume is split-adjusted —
a mixed convention. `verify_price_convention()` returns `inconclusive` on the free tier
because the NVDA 2024 split is outside the 1-year window. Verify before trusting volume.

---

## Planned upgrade

### EODHD "EOD All World"

| | |
|---|---|
| Cost | ~€199.90/year ≈ **\$16.66/mo** on annual billing (~\$20/mo monthly) |
| Provides | EOD OHLCV + splits + dividends, 30+ years, **including delisted tickers** |
| Limits | 100k requests/day, 1k/minute |

**Why this one:** delisted coverage. That is the entire reason to spend money — it closes the
30% survivorship hole that free sources cannot.

**Not included:** historical index constituents live in the separate Fundamentals tier
(\$59.99/mo). Constituents continue to come from Wikipedia revision history.

**Worth checking:** EODHD has historically offered a ~50% educational/scientific discount on
request. That would halve this line.

**On arrival, before trusting anything:**

1. **Verify the price convention against a known split** (NVDA 2024-06-10 10:1). Do not assume
   `as_traded` — check it. See ADR-007.
2. Download into `data/vault/`, not `bronze/` — it is not re-fetchable after cancellation.
3. Cross-validate against yfinance on the ~677 overlapping tickers.
4. Diff the constituent history against the Wikipedia reconstruction.
5. Re-run the identical backtests and **measure the survivorship delta**.

Verify all pricing directly — it moves.

---

## Licensing

Personal research use. Redistribution is **not** permitted for most of these, and cheap
vendor tiers are typically personal-use only. If this ever becomes a product, re-read every
licence first — particularly around derived data and model training, which some vendors
restrict explicitly.
