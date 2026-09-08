from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .analytics import analyze
from .reporting import format_report, to_log_record
from .upstox_client import NIFTY_INSTRUMENT_KEY, UpstoxAuthError, UpstoxClient

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
LATEST_SNAPSHOT = DATA_DIR / "latest_snapshot.json"
INSIGHTS_LOG = DATA_DIR / "insights_log.jsonl"


def is_market_open(now: datetime) -> bool:
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_t, close_t = now.replace(hour=9, minute=15, second=0, microsecond=0), now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def _load_json(path: Path) -> list | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _run_once(client: UpstoxClient | None, expiry: str | None, demo: bool) -> None:
    now = datetime.now(IST)
    timestamp = now.isoformat(timespec="seconds")

    if demo:
        chain = json.loads((SAMPLE_DIR / "nifty_sample_snapshot_t1.json").read_text())
        prev_chain = json.loads((SAMPLE_DIR / "nifty_sample_snapshot_t0.json").read_text())
    else:
        assert client is not None
        if expiry is None:
            expiry = client.nearest_expiry()
        chain = client.get_option_chain(expiry_date=expiry)
        prev_chain = _load_json(LATEST_SNAPSHOT)

    prev_pcr_oi = None
    if prev_chain:
        from .analytics import compute_pcr, to_dataframe

        prev_pcr_oi = compute_pcr(to_dataframe(prev_chain))["pcr_oi"]

    insights = analyze(chain, timestamp=timestamp, prev_chain=prev_chain, prev_pcr_oi=prev_pcr_oi)
    print(format_report(insights))

    if not demo:
        DATA_DIR.mkdir(exist_ok=True)
        LATEST_SNAPSHOT.write_text(json.dumps(chain))
        with INSIGHTS_LOG.open("a") as f:
            f.write(json.dumps(to_log_record(insights), default=str) + "\n")


def cmd_login(args: argparse.Namespace) -> None:
    client = UpstoxClient()
    try:
        print("Open this URL, log in to Upstox, then copy the `code` query param from the redirect URL:")
        print(client.login_url())
    except UpstoxAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_auth(args: argparse.Namespace) -> None:
    client = UpstoxClient()
    try:
        token = client.exchange_code_for_token(args.code)
        print(f"Access token cached. Valid until ~3:30am IST tomorrow.\n{token[:12]}...")
    except UpstoxAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    client = None
    if not args.demo:
        client = UpstoxClient()

    while True:
        now = datetime.now(IST)
        if args.demo or args.ignore_market_hours or is_market_open(now):
            try:
                _run_once(client, args.expiry, args.demo)
            except UpstoxAuthError as e:
                print(f"Auth error: {e}\nRun `login` then `auth --code ...` to refresh the token.", file=sys.stderr)
                if args.once or args.demo:
                    sys.exit(1)
            except Exception as e:  # keep the loop alive across transient API errors
                print(f"Fetch/analysis error: {e}", file=sys.stderr)
        else:
            print(f"[{now.isoformat(timespec='seconds')}] Market closed (open 09:15-15:30 IST, Mon-Fri). Sleeping.")

        if args.once or args.demo:
            return
        time.sleep(args.interval)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="NIFTY 50 options chain analyzer (Upstox API)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="print the Upstox OAuth login URL")
    p_login.set_defaults(func=cmd_login)

    p_auth = sub.add_parser("auth", help="exchange an OAuth `code` for an access token")
    p_auth.add_argument("--code", required=True)
    p_auth.set_defaults(func=cmd_auth)

    p_run = sub.add_parser("run", help="fetch + analyze the option chain on a recurring interval")
    p_run.add_argument("--interval", type=int, default=300, help="seconds between fetches (default 300 = 5 min)")
    p_run.add_argument("--expiry", default=None, help="YYYY-MM-DD; defaults to the nearest expiry")
    p_run.add_argument("--once", action="store_true", help="fetch a single snapshot and exit")
    p_run.add_argument("--demo", action="store_true", help="use bundled sample data, no credentials/network needed")
    p_run.add_argument("--ignore-market-hours", action="store_true", help="fetch even outside 09:15-15:30 IST")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
