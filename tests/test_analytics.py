import json
from pathlib import Path

from nifty_options.analytics import analyze, compute_max_pain, compute_pcr, to_dataframe

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
T0 = json.loads((SAMPLE_DIR / "nifty_sample_snapshot_t0.json").read_text())
T1 = json.loads((SAMPLE_DIR / "nifty_sample_snapshot_t1.json").read_text())


def test_to_dataframe_shape():
    df = to_dataframe(T0)
    assert len(df) == len(T0)
    assert list(df["strike"]) == sorted(df["strike"])


def test_pcr_within_expected_range():
    df = to_dataframe(T0)
    pcr = compute_pcr(df)
    assert pcr["total_ce_oi"] > 0
    assert pcr["total_pe_oi"] > 0
    assert pcr["pcr_oi"] > 0


def test_max_pain_is_one_of_the_strikes():
    df = to_dataframe(T0)
    mp = compute_max_pain(df)
    assert mp in set(df["strike"])


def test_analyze_end_to_end_with_prev_snapshot():
    ins = analyze(T1, timestamp="2026-09-08T10:05:00+05:30", prev_chain=T0, prev_pcr_oi=compute_pcr(to_dataframe(T0))["pcr_oi"])
    assert ins.spot == 24851.7
    assert ins.atm in set(to_dataframe(T1)["strike"])
    assert ins.buildup is not None and not ins.buildup.empty
    # near-ATM calls were seeded with rising price + rising OI -> Long Buildup
    atm_row = ins.buildup.loc[ins.buildup["strike"] == ins.atm].iloc[0]
    assert atm_row["ce_signal"] == "Long Buildup"
    assert ins.sentiment["bias"] in {"Bullish", "Bearish", "Neutral"}


def test_analyze_without_prev_snapshot_still_works():
    ins = analyze(T0, timestamp="2026-09-08T10:00:00+05:30")
    assert ins.buildup is None
    assert ins.max_pain > 0
