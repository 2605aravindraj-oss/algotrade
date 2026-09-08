from __future__ import annotations

from dataclasses import asdict

from tabulate import tabulate

from .analytics import Insights


def format_report(ins: Insights) -> str:
    lines = []
    lines.append(f"===== NIFTY 50 Options Insight @ {ins.timestamp} =====")
    lines.append(f"Spot: {ins.spot}   ATM Strike: {ins.atm}")
    lines.append("")
    lines.append(f"PCR (OI): {ins.pcr['pcr_oi']}   PCR (Volume): {ins.pcr['pcr_volume']}")
    lines.append(f"Total CE OI: {ins.pcr['total_ce_oi']:,.0f}   Total PE OI: {ins.pcr['total_pe_oi']:,.0f}")
    lines.append(f"Max Pain: {ins.max_pain}  ({ins.sentiment['max_pain_gap_pct']:+.2f}% from spot)")
    lines.append("")

    lines.append(f"ATM Straddle ({ins.straddle['atm_strike']}): CE {ins.straddle['ce_ltp']} + PE {ins.straddle['pe_ltp']} = {ins.straddle['straddle_price']}  (implied move)")
    lines.append(
        f"IV skew: ATM CE {ins.iv_skew['atm_ce_iv']} / ATM PE {ins.iv_skew['atm_pe_iv']}  |  "
        f"OTM CE avg {ins.iv_skew['otm_ce_iv_avg']} / OTM PE avg {ins.iv_skew['otm_pe_iv_avg']}"
    )
    lines.append("")

    res = ins.support_resistance["resistance"]
    sup = ins.support_resistance["support"]
    lines.append("Resistance (top Call OI):  " + ", ".join(f"{r['strike']} ({r['ce_oi']:,.0f})" for r in res))
    lines.append("Support    (top Put OI):   " + ", ".join(f"{s['strike']} ({s['pe_oi']:,.0f})" for s in sup))
    lines.append("")

    if ins.buildup is not None and not ins.buildup.empty:
        lines.append("OI/Price buildup near ATM (vs previous snapshot):")
        lines.append(
            tabulate(
                ins.buildup,
                headers="keys",
                tablefmt="simple",
                showindex=False,
            )
        )
        lines.append("")

    lines.append(f"Overall bias: {ins.sentiment['bias']}")
    for note in ins.sentiment["notes"]:
        lines.append(f"  - {note}")
    lines.append("=" * 55)
    return "\n".join(lines)


def to_log_record(ins: Insights) -> dict:
    record = asdict(ins)
    if ins.buildup is not None:
        record["buildup"] = ins.buildup.to_dict("records")
    return record
