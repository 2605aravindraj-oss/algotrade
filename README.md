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