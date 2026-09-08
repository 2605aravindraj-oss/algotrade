"""Turns a raw Upstox option-chain payload into trading insights.

Expected input: the `data` list returned by UpstoxClient.get_option_chain(),
i.e. a list of rows shaped like:
{
  "strike_price": 24800,
  "underlying_spot_price": 24815.3,
  "call_options": {
      "market_data": {"ltp": .., "oi": .., "volume": .., "close_price": ..},
      "option_greeks": {"iv": .., "delta": .., "gamma": .., "theta": .., "vega": ..}
  },
  "put_options": { ... same shape ... }
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def to_dataframe(chain: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in chain:
        ce = row.get("call_options") or {}
        pe = row.get("put_options") or {}
        ce_md, ce_gr = ce.get("market_data") or {}, ce.get("option_greeks") or {}
        pe_md, pe_gr = pe.get("market_data") or {}, pe.get("option_greeks") or {}
        rows.append(
            {
                "strike": row.get("strike_price"),
                "spot": row.get("underlying_spot_price"),
                "ce_ltp": ce_md.get("ltp", 0) or 0,
                "ce_oi": ce_md.get("oi", 0) or 0,
                "ce_volume": ce_md.get("volume", 0) or 0,
                "ce_iv": ce_gr.get("iv", 0) or 0,
                "pe_ltp": pe_md.get("ltp", 0) or 0,
                "pe_oi": pe_md.get("oi", 0) or 0,
                "pe_volume": pe_md.get("volume", 0) or 0,
                "pe_iv": pe_gr.get("iv", 0) or 0,
            }
        )
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df


def atm_strike(df: pd.DataFrame, spot: float) -> float:
    return df.loc[(df["strike"] - spot).abs().idxmin(), "strike"]


def compute_pcr(df: pd.DataFrame) -> dict[str, float]:
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    total_ce_vol = df["ce_volume"].sum()
    total_pe_vol = df["pe_volume"].sum()
    return {
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "pcr_oi": round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else float("nan"),
        "pcr_volume": round(total_pe_vol / total_ce_vol, 3) if total_ce_vol else float("nan"),
    }


def compute_max_pain(df: pd.DataFrame) -> float:
    strikes = df["strike"].to_numpy()
    ce_oi = df["ce_oi"].to_numpy()
    pe_oi = df["pe_oi"].to_numpy()
    best_strike, min_pain = None, None
    for s in strikes:
        call_writer_loss = ((s - strikes).clip(min=0) * ce_oi).sum()
        put_writer_loss = ((strikes - s).clip(min=0) * pe_oi).sum()
        pain = call_writer_loss + put_writer_loss
        if min_pain is None or pain < min_pain:
            min_pain, best_strike = pain, s
    return float(best_strike)


def support_resistance(df: pd.DataFrame, top_n: int = 3) -> dict[str, list[dict[str, float]]]:
    resistance = df.nlargest(top_n, "ce_oi")[["strike", "ce_oi"]].to_dict("records")
    support = df.nlargest(top_n, "pe_oi")[["strike", "pe_oi"]].to_dict("records")
    return {"resistance": resistance, "support": support}


def atm_straddle(df: pd.DataFrame, atm: float) -> dict[str, float]:
    row = df.loc[df["strike"] == atm].iloc[0]
    return {
        "atm_strike": atm,
        "ce_ltp": row["ce_ltp"],
        "pe_ltp": row["pe_ltp"],
        "straddle_price": round(row["ce_ltp"] + row["pe_ltp"], 2),
    }


def iv_skew(df: pd.DataFrame, atm: float) -> dict[str, float]:
    row = df.loc[df["strike"] == atm].iloc[0]
    otm_calls = df.loc[df["strike"] > atm].head(3)
    otm_puts = df.loc[df["strike"] < atm].tail(3)
    return {
        "atm_ce_iv": row["ce_iv"],
        "atm_pe_iv": row["pe_iv"],
        "otm_ce_iv_avg": round(otm_calls["ce_iv"].mean(), 2) if len(otm_calls) else float("nan"),
        "otm_pe_iv_avg": round(otm_puts["pe_iv"].mean(), 2) if len(otm_puts) else float("nan"),
    }


_BUILDUP_LABELS = {
    (True, True): "Long Buildup",
    (True, False): "Short Covering",
    (False, True): "Short Buildup",
    (False, False): "Long Unwinding",
}


def _classify(price_delta: float, oi_delta: float) -> str:
    if price_delta == 0 and oi_delta == 0:
        return "No Change"
    return _BUILDUP_LABELS[(price_delta > 0, oi_delta > 0)]


def oi_buildup(df: pd.DataFrame, prev_df: pd.DataFrame, atm: float, window: int = 5) -> pd.DataFrame:
    """Classify OI+price action per strike near the ATM strike, for both CE and PE.

    Uses the standard OI/price interpretation applied to the option's own
    premium: rising premium + rising OI = aggressive buying ("Long Buildup"),
    falling premium + rising OI = aggressive writing ("Short Buildup"), etc.
    """
    strikes = sorted(df["strike"].unique())
    idx = strikes.index(atm)
    near = set(strikes[max(0, idx - window) : idx + window + 1])
    merged = df.merge(prev_df, on="strike", suffixes=("", "_prev"))
    merged = merged.loc[merged["strike"].isin(near)]

    out = []
    for _, r in merged.iterrows():
        ce_label = _classify(r["ce_ltp"] - r["ce_ltp_prev"], r["ce_oi"] - r["ce_oi_prev"])
        pe_label = _classify(r["pe_ltp"] - r["pe_ltp_prev"], r["pe_oi"] - r["pe_oi_prev"])
        out.append(
            {
                "strike": r["strike"],
                "ce_oi_chg": r["ce_oi"] - r["ce_oi_prev"],
                "ce_ltp_chg": round(r["ce_ltp"] - r["ce_ltp_prev"], 2),
                "ce_signal": ce_label,
                "pe_oi_chg": r["pe_oi"] - r["pe_oi_prev"],
                "pe_ltp_chg": round(r["pe_ltp"] - r["pe_ltp_prev"], 2),
                "pe_signal": pe_label,
            }
        )
    return pd.DataFrame(out).sort_values("strike").reset_index(drop=True)


def sentiment(pcr: dict, max_pain: float, spot: float, prev_pcr_oi: float | None) -> dict[str, str]:
    notes = []

    if pcr["pcr_oi"] > 1.2:
        bias = "Bullish"
        notes.append(f"PCR(OI)={pcr['pcr_oi']} > 1.2 -> heavy put writing, support-heavy positioning")
    elif pcr["pcr_oi"] < 0.7:
        bias = "Bearish"
        notes.append(f"PCR(OI)={pcr['pcr_oi']} < 0.7 -> heavy call writing, resistance-heavy positioning")
    else:
        bias = "Neutral"
        notes.append(f"PCR(OI)={pcr['pcr_oi']} in the 0.7-1.2 neutral band")

    pain_gap_pct = round((max_pain - spot) / spot * 100, 2)
    if abs(pain_gap_pct) >= 0.3:
        notes.append(f"Max Pain {max_pain} is {pain_gap_pct:+.2f}% from spot {spot} -> mild pull toward that strike into expiry")
    else:
        notes.append(f"Max Pain {max_pain} is close to spot {spot} -> no strong pin pressure yet")

    if prev_pcr_oi is not None:
        drift = round(pcr["pcr_oi"] - prev_pcr_oi, 3)
        if abs(drift) >= 0.05:
            notes.append(f"PCR(OI) moved {drift:+.3f} vs previous snapshot -> {'building bullish' if drift > 0 else 'building bearish'} tone")

    return {"bias": bias, "max_pain_gap_pct": pain_gap_pct, "notes": notes}


@dataclass
class Insights:
    timestamp: str
    spot: float
    atm: float
    pcr: dict = field(default_factory=dict)
    max_pain: float = 0.0
    support_resistance: dict = field(default_factory=dict)
    straddle: dict = field(default_factory=dict)
    iv_skew: dict = field(default_factory=dict)
    buildup: pd.DataFrame | None = None
    sentiment: dict = field(default_factory=dict)


def analyze(
    chain: list[dict[str, Any]],
    timestamp: str,
    prev_chain: list[dict[str, Any]] | None = None,
    prev_pcr_oi: float | None = None,
) -> Insights:
    df = to_dataframe(chain)
    spot = float(df["spot"].dropna().iloc[0])
    atm = atm_strike(df, spot)
    pcr = compute_pcr(df)
    max_pain = compute_max_pain(df)

    buildup_df = None
    if prev_chain is not None:
        prev_df = to_dataframe(prev_chain)
        buildup_df = oi_buildup(df, prev_df, atm)

    return Insights(
        timestamp=timestamp,
        spot=spot,
        atm=atm,
        pcr=pcr,
        max_pain=max_pain,
        support_resistance=support_resistance(df),
        straddle=atm_straddle(df, atm),
        iv_skew=iv_skew(df, atm),
        buildup=buildup_df,
        sentiment=sentiment(pcr, max_pain, spot, prev_pcr_oi),
    )
