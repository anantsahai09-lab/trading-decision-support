"""
swings.py — Swing detection, structure labeling, BOS detection, trend classification.

Implements Strategy v2.0:
  - Swing definition (reference material, adopted in v2 §2):
      swing HIGH = up candle immediately followed by down candle
                   -> price = the higher of the two candles' highs
      swing LOW  = down candle immediately followed by up candle
                   -> price = the lower of the two candles' lows
      Doji candles (open == close) are ignored: pairing happens across
      consecutive NON-doji candles.
  - Structure labels: HH / LH for highs, HL / LL for lows (vs previous
    swing of the same type).
  - BOS: a candle CLOSE beyond the most recent confirmed swing in that
    direction (close > last swing high = bullish BOS; close < last swing
    low = bearish BOS). Wicks do not count.
  - Trend classification (v2 §2): uptrend if latest high is HH and latest
    low is HL; downtrend if LH and LL; otherwise sideways/mixed.
    "Obvious trend" (provisional, v2 §10.2): >= N consecutive
    same-direction swing labels with no opposing BOS in between (N=3).

Input contract (all functions):
  df: pandas DataFrame indexed by ascending pd.DatetimeIndex with columns
      ['open', 'high', 'low', 'close']  (floats). Volume optional, unused.

All timestamps in outputs refer to bar-open times of the input index.
No lookahead: a swing is only known at the CLOSE of its confirming
(second) candle; `confirmed_at` records that bar. Consumers must not act
on a swing before its confirmed_at bar has closed.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Direction(Enum):
    UP = 1
    DOWN = -1
    DOJI = 0


REQUIRED_COLS = ("open", "high", "low", "close")


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending")
    if df.index.has_duplicates:
        raise ValueError("Index has duplicate timestamps")


def candle_direction(df: pd.DataFrame) -> pd.Series:
    """UP if close>open, DOWN if close<open, DOJI if equal."""
    d = np.sign(df["close"].to_numpy() - df["open"].to_numpy()).astype(int)
    return pd.Series(d, index=df.index, name="direction")


def detect_swings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect swing highs/lows over consecutive non-doji candle pairs.

    Returns a DataFrame with one row per swing, columns:
      type          : 'high' | 'low'
      price         : float, the swing extreme
      extreme_at    : timestamp of the candle carrying the extreme
      confirmed_at  : timestamp of the second (confirming) candle
    Sorted by confirmed_at.
    """
    _validate(df)
    dirs = candle_direction(df)
    nz = df.loc[dirs != 0]              # drop dojis for pairing
    nz_dirs = dirs.loc[dirs != 0]

    rows = []
    idx = nz.index
    for i in range(len(nz) - 1):
        d1, d2 = nz_dirs.iloc[i], nz_dirs.iloc[i + 1]
        c1, c2 = nz.iloc[i], nz.iloc[i + 1]
        t1, t2 = idx[i], idx[i + 1]
        if d1 == 1 and d2 == -1:  # up then down -> swing high
            if c1["high"] >= c2["high"]:
                price, ext_t = c1["high"], t1
            else:
                price, ext_t = c2["high"], t2
            rows.append(("high", price, ext_t, t2))
        elif d1 == -1 and d2 == 1:  # down then up -> swing low
            if c1["low"] <= c2["low"]:
                price, ext_t = c1["low"], t1
            else:
                price, ext_t = c2["low"], t2
            rows.append(("low", price, ext_t, t2))

    out = pd.DataFrame(rows, columns=["type", "price", "extreme_at", "confirmed_at"])
    return out.sort_values("confirmed_at").reset_index(drop=True)


def label_swings(swings: pd.DataFrame) -> pd.DataFrame:
    """
    Add structure labels vs the previous swing of the SAME type:
      highs -> 'HH' (higher high) or 'LH' (lower high)
      lows  -> 'HL' (higher low)  or 'LL' (lower low)
    First swing of each type gets label None.
    Equal prices resolve as continuation-negative ('LH' for equal highs,
    'HL' for equal lows is NOT granted): equal high -> 'LH', equal low -> 'LL'?
    Decision: equal extremes are labeled 'EQH'/'EQL' and treated as
    non-directional; they also matter later as relative-equal liquidity.
    """
    sw = swings.copy()
    labels: list[str | None] = []
    last = {"high": None, "low": None}
    for _, row in sw.iterrows():
        prev = last[row["type"]]
        if prev is None:
            labels.append(None)
        elif row["type"] == "high":
            if row["price"] > prev:
                labels.append("HH")
            elif row["price"] < prev:
                labels.append("LH")
            else:
                labels.append("EQH")
        else:
            if row["price"] < prev:
                labels.append("LL")
            elif row["price"] > prev:
                labels.append("HL")
            else:
                labels.append("EQL")
        last[row["type"]] = row["price"]
    sw["label"] = pd.Series(labels, index=sw.index, dtype="object")
    return sw


def detect_bos(df: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    """
    Break of structure: candle CLOSE beyond the most recent confirmed swing.

    A swing becomes 'active' only after its confirmed_at bar. The most
    recent active swing high/low is the break level. On a bullish BOS the
    reference swing high is consumed (a new one must confirm before the
    next bullish BOS), and symmetrically for bearish.

    Returns DataFrame columns:
      timestamp   : bar whose close broke structure
      direction   : 'bull' | 'bear'
      level       : the swing price broken
      swing_confirmed_at : confirmed_at of the broken swing
    """
    _validate(df)
    events = []
    highs = swings[swings["type"] == "high"].reset_index(drop=True)
    lows = swings[swings["type"] == "low"].reset_index(drop=True)
    hi_i = lo_i = 0
    active_high: tuple[float, pd.Timestamp] | None = None
    active_low: tuple[float, pd.Timestamp] | None = None

    for t, row in df.iterrows():
        # activate any swings confirmed strictly BEFORE this bar
        while hi_i < len(highs) and highs.loc[hi_i, "confirmed_at"] < t:
            active_high = (highs.loc[hi_i, "price"], highs.loc[hi_i, "confirmed_at"])
            hi_i += 1
        while lo_i < len(lows) and lows.loc[lo_i, "confirmed_at"] < t:
            active_low = (lows.loc[lo_i, "price"], lows.loc[lo_i, "confirmed_at"])
            lo_i += 1

        close = row["close"]
        if active_high is not None and close > active_high[0]:
            events.append((t, "bull", active_high[0], active_high[1]))
            active_high = None  # consumed until a new high confirms
        if active_low is not None and close < active_low[0]:
            events.append((t, "bear", active_low[0], active_low[1]))
            active_low = None

    return pd.DataFrame(
        events, columns=["timestamp", "direction", "level", "swing_confirmed_at"]
    )


def classify_trend(labeled: pd.DataFrame) -> str:
    """
    'up' if latest high label is HH and latest low label is HL,
    'down' if LH and LL, else 'sideways'. EQH/EQL count as sideways-leaning
    (they break the directional requirement).
    """
    highs = labeled[labeled["type"] == "high"]["label"].dropna()
    lows = labeled[labeled["type"] == "low"]["label"].dropna()
    if highs.empty or lows.empty:
        return "sideways"
    hi_label, lo_label = highs.iloc[-1], lows.iloc[-1]
    if hi_label == "HH" and lo_label == "HL":
        return "up"
    if hi_label == "LH" and lo_label == "LL":
        return "down"
    return "sideways"


def is_obvious_trend(
    labeled: pd.DataFrame, bos: pd.DataFrame, n: int = 3
) -> tuple[bool, str]:
    """
    Provisional §10.2 rule: obvious trend = the last n swing labels (of any
    type, in confirmation order) are all trend-consistent in one direction
    (HH/HL for up, LH/LL for down) AND no opposing BOS occurred after the
    first of those n swings.

    Returns (is_obvious, direction) where direction in {'up','down',''}.
    """
    lab = labeled.dropna(subset=["label"])
    if len(lab) < n:
        return (False, "")
    tail = lab.tail(n)
    up_set, down_set = {"HH", "HL"}, {"LH", "LL"}
    if set(tail["label"]).issubset(up_set):
        direction, opposing = "up", "bear"
    elif set(tail["label"]).issubset(down_set):
        direction, opposing = "down", "bull"
    else:
        return (False, "")
    window_start = tail["confirmed_at"].iloc[0]
    if not bos.empty:
        recent = bos[bos["timestamp"] >= window_start]
        if (recent["direction"] == opposing).any():
            return (False, "")
    return (True, direction)
