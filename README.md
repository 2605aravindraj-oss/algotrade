# algotrade

## Market data sources

`data_sources/` has minimal clients for two market data providers:

- **Upstox** (`data_sources/upstox_client.py`): historical OHLC candles are
  public and need no auth. Live quotes need an OAuth access token — set
  `UPSTOX_ACCESS_TOKEN` in the environment (see Upstox's
  [authentication docs](https://upstox.com/developer/api-documentation/authentication)).
- **Yahoo Finance** (`data_sources/yahoo_client.py`): chart/quote data is
  public, no key required.

Run `python query_markets.py` for a demo that fetches Reliance Industries'
last week of candles from Upstox and its latest price from Yahoo Finance
side by side.

```
pip install -r requirements.txt
python query_markets.py
```

## EMA crossover backtest

`backtest/ema_crossover.py` implements a long-only EMA crossover strategy:
go long on a golden cross (fast EMA crosses above slow EMA), exit on a
death cross. Signals are filled at the next bar's open to avoid lookahead
bias.

```
python run_ema_backtest.py --instrument "NSE_EQ|INE002A01018" \
    --from 2023-01-01 --fast 12 --slow 26
```

Data comes from Upstox's public historical-candle endpoint (no access
token needed). The script prints total return vs. buy & hold, max
drawdown, win rate, and the most recent trades.

## Downtrend reversal / break of structure + retest backtest

`backtest/structure_reversal.py` encodes a discretionary price-action
pattern instead of an indicator crossover:

1. Downtrend: swing highs and swing lows both descending (lower highs,
   lower lows).
2. A swing low prints *higher* than the prior swing low (a "higher low")
   while price is still below the last swing high -> momentum is
   weakening.
3. Break of structure: price closes above that last lower high.
4. Retest: price pulls back toward the broken level, holds above it, then
   closes higher again -> the level flips from resistance to support.
5. Entry: long at the next bar's open. Stop below the retest low; target
   a fixed reward:risk multiple of the resulting risk.

Swing points are detected with a centered fractal and are only used once
enough future bars exist to have confirmed them in real time, so the
backtest has no lookahead bias.

```
python run_structure_backtest.py --instrument "NSE_EQ|INE002A01018" \
    --from 2023-01-01 --lookback 3 --reward-risk 2.0
```

The script prints performance vs. buy & hold and every detected setup
(downtrend high, higher low, breakout, retest, entry/exit), including ones
that were invalidated before triggering a trade.

## Scanning multiple stocks

`data_sources/instruments.py` resolves NSE trading symbols (e.g. "TCS") to
Upstox instrument keys using Upstox's public instrument master (cached in
the OS temp dir for 24h). `scan_structure_reversal.py` runs the structure
reversal backtest across a basket of symbols and prints a comparison
table:

```
python scan_structure_reversal.py TCS INFY HDFCBANK RELIANCE
python scan_structure_reversal.py   # defaults to a 20-stock large-cap basket
```

## Expired options data

`data_sources/upstox_client.py` also wraps Upstox's **expired-instruments**
API (`get_expired_expiries`, `get_expired_option_chain`, `get_expired_candles`),
which requires an OAuth access token but gives full historical OHLC/OI for
any past NSE options expiry and strike, at up to 1-minute resolution — not
just daily bars for currently-listed contracts.

## Daily intraday iron condor backtest (NIFTY 50)

`backtest/iron_condor.py` backtests entering a 4-leg iron condor on NIFTY
50 weekly options every trading day at ~09:15 and squaring off all legs at
market close:

```
short call = ATM + short_distance   (sell)
short put  = ATM - short_distance   (sell)
long call  = short call + wing_width  (buy, protection)
long put   = short put  - wing_width  (buy, protection)
```

Each day uses whichever weekly expiry is nearest (from same-day/0DTE up to
the following week), with strikes chosen off that day's 09:15 spot price.
Requires `UPSTOX_ACCESS_TOKEN` since every expiry involved is in the past.

```
export UPSTOX_ACCESS_TOKEN=...
python run_iron_condor_backtest.py --from 2025-07-01 --to 2025-08-29 \
    --short-distance 150 --wing-width 100
```

Prints gross P&L, transaction costs, net P&L, win rate, average win/loss,
best/worst day, max drawdown, and the full daily trade log.

`backtest/costs.py` estimates real Indian F&O options transaction costs
per fill (2025/26 rates): flat Rs 20/order brokerage, STT (0.1% of premium,
sell side only), exchange transaction charges (~0.035%, both sides), SEBI
turnover fee, stamp duty (0.003%, buy side only), and 18% GST on
brokerage+exchange+SEBI. All 8 fills/day (4 legs x entry+exit) are costed
and netted against gross P&L.