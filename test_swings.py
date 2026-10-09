"""
test_swings.py — pytest suite for swings.py using hand-constructed candles.

Run:  pytest test_swings.py -v
"""

import pandas as pd

from src.swings import (
    candle_direction,
    classify_trend,
    detect_bos,
    detect_swings,
    is_obvious_trend,
    label_swings,
)


def make_df(candles, start="2026-01-05 09:30", freq="5min"):
    """candles: list of (open, high, low, close)."""
    idx = pd.date_range(start=start, periods=len(candles), freq=freq, tz="America/New_York")
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"], index=idx)


# ---------- direction ----------

def test_direction_classification():
    df = make_df([
        (100, 105, 99, 104),   # up
        (104, 106, 100, 101),  # down
        (101, 103, 100, 101),  # doji
    ])
    d = candle_direction(df)
    assert list(d) == [1, -1, 0]


# ---------- swing detection ----------

def test_swing_high_basic():
    # up candle then down candle; first candle carries the higher high
    df = make_df([
        (100, 110, 99, 108),   # up, high=110
        (108, 109, 101, 102),  # down, high=109
    ])
    sw = detect_swings(df)
    assert len(sw) == 1
    assert sw.loc[0, "type"] == "high"
    assert sw.loc[0, "price"] == 110
    assert sw.loc[0, "extreme_at"] == df.index[0]
    assert sw.loc[0, "confirmed_at"] == df.index[1]


def test_swing_high_second_candle_higher():
    df = make_df([
        (100, 110, 99, 108),   # up
        (108, 112, 101, 102),  # down but wicked higher first: high=112
    ])
    sw = detect_swings(df)
    assert sw.loc[0, "price"] == 112
    assert sw.loc[0, "extreme_at"] == df.index[1]


def test_swing_low_basic():
    df = make_df([
        (100, 101, 90, 92),    # down, low=90
        (92, 99, 91, 98),      # up, low=91
    ])
    sw = detect_swings(df)
    assert len(sw) == 1
    assert sw.loc[0, "type"] == "low"
    assert sw.loc[0, "price"] == 90
    assert sw.loc[0, "extreme_at"] == df.index[0]


def test_doji_is_skipped_in_pairing():
    # up, DOJI, down  -> pairing over non-doji candles => still a swing high
    df = make_df([
        (100, 110, 99, 108),   # up
        (108, 111, 107, 108),  # doji (open==close), high=111
        (108, 109, 101, 102),  # down
    ])
    sw = detect_swings(df)
    assert len(sw) == 1
    assert sw.loc[0, "type"] == "high"
    # doji's high is NOT considered (pair is candle0 & candle2)
    assert sw.loc[0, "price"] == 110


def test_consecutive_same_direction_no_swing():
    df = make_df([
        (100, 105, 99, 104),
        (104, 108, 103, 107),
        (107, 112, 106, 111),
    ])
    assert detect_swings(df).empty


def test_alternating_sequence_multiple_swings():
    df = make_df([
        (100, 105, 99, 104),   # up
        (104, 106, 98, 99),    # down  -> swing high @106
        (99, 103, 97, 102),    # up    -> swing low  @97
        (103, 107, 101, 102),  # down -> swing high @107
        (102, 104, 96, 104),   # up   -> swing low @96
    ])
    sw = detect_swings(df)
    assert list(sw["type"]) == ["high", "low", "high", "low"]
    assert list(sw["price"]) == [106, 97, 107, 96]


# ---------- labeling ----------

def test_labels_hh_hl():
    sw = pd.DataFrame({
        "type": ["high", "low", "high", "low"],
        "price": [106.0, 97.0, 110.0, 99.0],
        "extreme_at": pd.date_range("2026-01-05", periods=4, freq="5min"),
        "confirmed_at": pd.date_range("2026-01-05", periods=4, freq="5min"),
    })
    lab = label_swings(sw)
    assert list(lab["label"]) == [None, None, "HH", "HL"]


def test_labels_equal_extremes():
    sw = pd.DataFrame({
        "type": ["high", "high"],
        "price": [106.0, 106.0],
        "extreme_at": pd.date_range("2026-01-05", periods=2, freq="5min"),
        "confirmed_at": pd.date_range("2026-01-05", periods=2, freq="5min"),
    })
    lab = label_swings(sw)
    assert list(lab["label"]) == [None, "EQH"]


# ---------- BOS ----------

def test_bullish_bos_on_close_not_wick():
    df = make_df([
        (100, 110, 99, 108),   # up
        (108, 109, 101, 102),  # down -> swing high 110 confirmed here
        (102, 111, 101, 104),  # WICK above 110 but close 104 -> no BOS
        (104, 115, 103, 112),  # CLOSE 112 > 110 -> bullish BOS
    ])
    sw = detect_swings(df)
    bos = detect_bos(df, sw)
    bulls = bos[bos["direction"] == "bull"]
    assert len(bulls) == 1
    assert bulls.iloc[0]["timestamp"] == df.index[3]
    assert bulls.iloc[0]["level"] == 110


def test_no_bos_before_swing_confirmed():
    # close above a future swing level on the same bar the swing confirms
    # must NOT trigger (swing activates only after its confirming bar)
    df = make_df([
        (100, 110, 99, 108),    # up
        (108, 120, 101, 118),   # up candle again -> no swing yet, close 118
    ])
    sw = detect_swings(df)
    bos = detect_bos(df, sw)
    assert bos.empty


def test_bearish_bos():
    df = make_df([
        (100, 101, 90, 92),    # down, low 90
        (92, 99, 91, 98),      # up -> swing low 90 confirmed
        (98, 99, 89, 88),      # close 88 < 90 -> bearish BOS
    ])
    sw = detect_swings(df)
    bos = detect_bos(df, sw)
    bears = bos[bos["direction"] == "bear"]
    assert len(bears) == 1
    assert bears.iloc[0]["level"] == 90


# ---------- trend ----------

def _labeled(seq):
    """seq: list of (type, price). Build labeled swing frame."""
    sw = pd.DataFrame({
        "type": [t for t, _ in seq],
        "price": [float(p) for _, p in seq],
        "extreme_at": pd.date_range("2026-01-05", periods=len(seq), freq="5min"),
        "confirmed_at": pd.date_range("2026-01-05", periods=len(seq), freq="5min"),
    })
    return label_swings(sw)


def test_trend_up():
    lab = _labeled([("high", 105), ("low", 100), ("high", 110), ("low", 103)])
    assert classify_trend(lab) == "up"


def test_trend_down():
    lab = _labeled([("high", 105), ("low", 100), ("high", 102), ("low", 96)])
    assert classify_trend(lab) == "down"


def test_trend_sideways_mixed():
    lab = _labeled([("high", 105), ("low", 100), ("high", 110), ("low", 95)])
    # HH but LL -> expansion both ways -> sideways
    assert classify_trend(lab) == "sideways"


def test_obvious_trend_true():
    lab = _labeled([
        ("high", 105), ("low", 100),
        ("high", 110), ("low", 104), ("high", 115),
    ])
    bos = pd.DataFrame(columns=["timestamp", "direction", "level", "swing_confirmed_at"])
    ok, direction = is_obvious_trend(lab, bos, n=3)
    assert ok and direction == "up"


def test_obvious_trend_killed_by_opposing_bos():
    lab = _labeled([
        ("high", 105), ("low", 100),
        ("high", 110), ("low", 104), ("high", 115),
    ])
    bos = pd.DataFrame({
        "timestamp": [lab["confirmed_at"].iloc[-1]],
        "direction": ["bear"],
        "level": [104.0],
        "swing_confirmed_at": [lab["confirmed_at"].iloc[-2]],
    })
    ok, _ = is_obvious_trend(lab, bos, n=3)
    assert not ok
