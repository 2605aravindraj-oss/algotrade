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