from datetime import date

import pandas as pd
import pytest

from algotrade import yahoo_client


def test_to_yahoo_ticker_adds_nse_suffix():
    assert yahoo_client.to_yahoo_ticker("reliance", "NSE_EQ") == "RELIANCE.NS"


def test_to_yahoo_ticker_adds_bse_suffix():
    assert yahoo_client.to_yahoo_ticker("reliance", "BSE_EQ") == "RELIANCE.BO"


def test_to_yahoo_ticker_does_not_double_suffix():
    assert yahoo_client.to_yahoo_ticker("RELIANCE.NS", "NSE_EQ") == "RELIANCE.NS"


def test_to_yahoo_ticker_passes_through_unknown_exchange():
    assert yahoo_client.to_yahoo_ticker("AAPL", "US_EQ") == "AAPL"


def _raw_yahoo_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [104.0, 105.0, 106.0],
            "Adj Close": [104.0, 105.0, 106.0],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )


def test_normalize_yahoo_dataframe_shapes_columns_and_index():
    normalized = yahoo_client.normalize_yahoo_dataframe(_raw_yahoo_frame())

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume", "oi"]
    assert normalized.index.name == "timestamp"
    assert str(normalized.index.tz) == "UTC"
    assert (normalized["oi"] == 0).all()
    assert normalized["close"].tolist() == [104.0, 105.0, 106.0]


def test_normalize_yahoo_dataframe_flattens_multiindex_columns():
    raw = _raw_yahoo_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["RELIANCE.NS"]])

    normalized = yahoo_client.normalize_yahoo_dataframe(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume", "oi"]


def test_fetch_historical_candles_rejects_invalid_interval():
    with pytest.raises(ValueError):
        yahoo_client.fetch_historical_candles(
            "RELIANCE", "NSE_EQ", "banana", date(2024, 1, 1), date(2024, 1, 10)
        )


def test_fetch_historical_candles_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(yahoo_client.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError, match="no data"):
        yahoo_client.fetch_historical_candles(
            "RELIANCE", "NSE_EQ", "day", date(2024, 1, 1), date(2024, 1, 10)
        )


def test_fetch_historical_candles_normalizes_mocked_download(monkeypatch):
    captured = {}

    def fake_download(ticker, start, end, interval, progress, auto_adjust):
        captured["args"] = (ticker, start, end, interval)
        return _raw_yahoo_frame()

    monkeypatch.setattr(yahoo_client.yf, "download", fake_download)

    df = yahoo_client.fetch_historical_candles(
        "RELIANCE", "NSE_EQ", "day", date(2024, 1, 1), date(2024, 1, 3)
    )

    assert captured["args"] == ("RELIANCE.NS", "2024-01-01", "2024-01-04", "1d")
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "oi"]
    assert len(df) == 3
