# algotrade

Historical data + backtesting toolkit for strategies traded through
[Upstox](https://upstox.com). Fetches OHLCV candles via the Upstox API,
runs pluggable strategies against them, and reports performance metrics
(returns, Sharpe, drawdown, win rate).

This is a **backtesting** tool — it does not place live orders.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your Upstox credentials from the
[developer console](https://upstox.com/developer/apps):

```
UPSTOX_API_KEY=...
UPSTOX_API_SECRET=...
UPSTOX_REDIRECT_URI=...
UPSTOX_ACCESS_TOKEN=...
```

`UPSTOX_ACCESS_TOKEN` is the short-lived (single trading day) token produced
by completing Upstox's OAuth login flow. The historical-candle endpoint used
here generally works without a token too, but supplying one avoids rate
limits on the anonymous tier.

## Usage

Download and cache historical candles:

```bash
python -m algotrade.cli fetch-data --symbol RELIANCE --exchange NSE_EQ \
    --interval day --start 2023-01-01 --end 2024-01-01
```

Run a backtest:

```bash
python -m algotrade.cli backtest --symbol RELIANCE --exchange NSE_EQ \
    --interval day --start 2023-01-01 --end 2024-01-01 \
    --strategy sma_crossover --fast 20 --slow 50 \
    --capital 100000 --commission-bps 5 --plot equity.png
```

Available strategies (`--strategy`):

- `sma_crossover` — long while the fast SMA is above the slow SMA (`--fast`, `--slow`)
- `rsi` — long while RSI is in a mean-reversion dip (`--rsi-period`, `--oversold`, `--exit-level`)

Downloaded candles are cached under `data/cache/` so repeated backtests over
the same symbol/date range don't re-hit the API; pass `--no-cache` to force
a refresh.

## Project layout

```
algotrade/
  config.py          # loads Upstox credentials from .env
  instruments.py      # resolves trading symbols to Upstox instrument_keys
  upstox_client.py    # historical-candle REST client
  data.py              # fetch + cache OHLCV as a pandas DataFrame
  strategies/          # pluggable Strategy classes
  backtest/
    engine.py           # vectorized backtest loop
    metrics.py           # Sharpe, drawdown, win rate, etc.
  cli.py                # `python -m algotrade.cli ...`
tests/                  # pytest suite (synthetic data, no network calls)
```

## Adding a new strategy

Subclass `algotrade.strategies.base.Strategy` and implement
`generate_signals(df) -> pd.Series`, returning a series valued in `{-1, 0, 1}`
(short/flat/long) aligned to `df.index`. Register it in
`algotrade/strategies/__init__.py`'s `STRATEGY_REGISTRY` to expose it via the
CLI's `--strategy` flag. The backtest engine shifts signals by one bar
automatically, so `generate_signals` can safely use the current bar's data.

## Tests

```bash
pytest
```

## Roadmap

- Options/futures instrument support (option-chain data, margin-aware sizing)
- Live/paper order placement via the Upstox order API
- Portfolio-level backtesting across multiple instruments
