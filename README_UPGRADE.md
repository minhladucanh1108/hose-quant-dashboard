
# HOSE Quant Trading V2 — Research Upgrade

This package is a clean research upgrade of the legacy Streamlit scanner.

## What is fixed

- Removes embedded Telegram credentials.
- Removes the synthetic/random 3-year “backtest”.
- Uses actual OHLCV data for trade-level evaluation.
- Eliminates same-close execution bias: signal is known at close `t`, entry is next available open `t+1`.
- Uses actual crossover logic instead of naming `EMA20 > EMA50` a crossover.
- Fetches enough history for EMA200 warm-up.
- Uses trading-session holding periods in the backtest.
- Includes buy/sell fees.
- Implements stop/take-profit in the evaluator rather than only displaying them.
- Does not use `st.session_state` as a persistent paper-trading database.
- Surfaces data errors instead of silently swallowing every exception.
- Labels the 68-name list as a curated universe, not “all HOSE”.
- Adds quantified Wyckoff-like event proxies: Spring, SOS and LPS.
- Adds an auditable news-sentiment overlay using RSS title/summary feeds.
- Adds an hourly weekday GitHub Actions news updater.

## Strategy variants for ablation tests

1. `EMA_VOL`: real EMA20/50 crossover + volume confirmation, no MA200 hard gate.
2. `EMA_VOL_MA200`: the same trigger, plus Close > EMA200.
3. `WYCKOFF`: quantified Spring/SOS/LPS event proxies.
4. `COMPOSITE`: technical trigger OR Wyckoff event, with news sentiment kept as a separate ranking overlay.

The V2 score is a research ranking score, **not** a calibrated return probability.

## Backtest discipline

- Signal observed at daily close.
- Entry at next session open.
- Lot/portfolio capital constraints are not modeled in this trade-level evaluator.
- Minimum holding sessions is configurable; default 2.
- Default stop / take profit: -5% / +15%.
- Buy fee 0.15%, sell fee 0.25%.
- If stop and take-profit are both touched in the same daily candle, the evaluator assumes the stop was hit first (conservative).
- Overlapping trades are prevented within the same ticker.
- Metrics are trade-level diagnostics, not portfolio CAGR.

## Wyckoff warning

The code does **not** claim to identify canonical Wyckoff phases. It quantifies observable price/volume events:
- Spring proxy
- Sign of Strength (SOS) proxy
- Last Point of Support (LPS) proxy
- Trading-range context

These features need ablation and forward validation before they are allowed to drive live execution.

## News sentiment

`news_worker.py` reads public RSS feeds and creates `news_sentiment.csv`.
Current sentiment is a transparent Vietnamese finance keyword baseline. It should be treated as an experimental overlay.

The GitHub workflow `.github/workflows/news_sentiment.yml` updates the file hourly on weekdays and commits only when the output changes.

## Recommended next research sequence

1. Freeze this V2 engine.
2. Compare no-MA200 vs MA200 on the same history.
3. Add Wyckoff and repeat the same test.
4. Add sentiment last and test its incremental contribution.
5. Split periods into development / validation / forward paper.
6. Only after the signal survives those tests should it be connected to DNSE execution.

## Production note

Yahoo Finance is retained only as a research fallback. Before live/paper automation, replace price/trading-calendar data with an exchange-grade source such as the approved DNSE market-data connection and persist state outside Streamlit.


## V2.1 frozen cohort model

The app now separates two concepts that must not be mixed:

### Live T+0
- Dynamic current-session research scanner.
- Builds a provisional daily bar from Yahoo 1-minute data when available.
- Price and signal may change during the session.
- This screen is **not** historical truth and is not frozen.
- DNSE should replace Yahoo before any live execution use.

### Frozen cohort
- After market close, `daily_cohort_worker.py` computes the official daily signal from completed EOD data.
- The recommendation set for that date is written once and never mutated by later refreshes.
- `reference_price` is currently EOD Close.
- When DNSE is connected, this can be upgraded to first-trigger timestamp/price if desired.

### T+ tracking
- T+ counts trading sessions, not calendar days.
- T+1/T+2/... use subsequent session **Close** prices.
- Each T mark is stored once per cohort/ticker/t-index key.
- At T+10, every ticker receives a final WIN/LOSS/FLAT summary.
- A new recommendation date creates a separate T+0 cohort without changing prior cohorts.

This directly fixes the legacy `st.session_state` behavior where old recommendations could be lost or overwritten and where T+ values were not true frozen session marks.


## V2.1.1 clarification

- `Live T+0` is dynamic and can change during the market session.
- The official frozen cohort is created only from the common latest completed EOD session.
- Stale tickers are not allowed into a newer cohort.
- Historical cohorts are never reranked or overwritten.
- T+1...T+10 are marked from subsequent trading-session CLOSE prices.
- Official reference price is currently the signal-day EOD Close for reproducibility.
- Exact first-trigger intraday price requires a persistent intraday worker; this should be implemented with DNSE once the API is approved.
