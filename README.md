# algotrade

## NIFTY 50 Options Chain Analyzer

Fetches the live NIFTY 50 index option chain from the **Upstox API** on a
recurring interval (default every 5 minutes during market hours) and prints
a set of trading insights: PCR, Max Pain, support/resistance from OI
concentration, ATM straddle (implied move), IV skew, and per-strike
OI/price buildup classification versus the previous snapshot.

### Why Upstox

NSE's own website/API blocks requests from most cloud/datacenter IPs at the
edge (Akamai), which makes scraping unreliable from hosted environments.
Upstox is an official, authenticated broker API that returns the same
underlying exchange data (OI, LTP, volume, IV, greeks) without that
restriction.

### Setup

1. Create an app at https://developer.upstox.com/ to get an **API Key**,
   **API Secret**, and register a **Redirect URI** (any URL you control,
   even `http://localhost:3000`).
2. Copy `.env.example` to `.env` and fill in:
   ```
   UPSTOX_API_KEY=...
   UPSTOX_API_SECRET=...
   UPSTOX_REDIRECT_URI=...
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Authenticate (Upstox access tokens expire daily around 3:30am IST, so
   repeat this once per trading day):
   ```
   python -m nifty_options login          # prints a login URL
   # open it, log in, copy the `code` param from the redirect
   python -m nifty_options auth --code <code>
   ```
   The resulting access token is cached in `.upstox_token.json` (gitignored).

### Run

```
# Live, every 5 minutes, only during NSE market hours (09:15-15:30 IST, Mon-Fri)
python -m nifty_options run --interval 300

# Single fetch, useful for testing
python -m nifty_options run --once

# No credentials/network needed - runs the analytics against bundled sample data
python -m nifty_options run --demo
```

Each run appends a JSON record to `data/insights_log.jsonl` and stores the
latest raw snapshot in `data/latest_snapshot.json` so the next run can
compute OI/price buildup deltas.

### What each metric means

- **PCR (OI/Volume)** - Put/Call ratio. >1.2 is read as put-heavy
  (bullish-leaning positioning), <0.7 as call-heavy (bearish-leaning).
- **Max Pain** - the strike at which option writers collectively lose the
  least at expiry; price has a mild tendency to drift toward it as expiry
  nears.
- **Support/Resistance** - strikes with the largest Put OI (support) and
  Call OI (resistance) concentrations.
- **ATM Straddle price** - CE + PE premium at the at-the-money strike;
  a rough market-implied expected move.
- **IV skew** - ATM vs OTM implied volatility for calls/puts.
- **OI/Price buildup** (needs two snapshots) - classifies each near-ATM
  strike's CE/PE using the standard OI+price rule: rising premium + rising
  OI = "Long Buildup" (aggressive buying), falling premium + rising OI =
  "Short Buildup" (aggressive writing), and the mirror cases as "Short
  Covering" / "Long Unwinding".

### Project layout

```
nifty_options/
  upstox_client.py  # OAuth + option chain/quote fetch
  analytics.py       # PCR, max pain, support/resistance, buildup, sentiment
  reporting.py        # human-readable report formatting
  cli.py               # login/auth/run commands, market-hours-aware scheduler
sample_data/           # two mock snapshots (t0, t1) used by --demo and tests
tests/test_analytics.py
```

Run tests with `python -m pytest tests/`.
