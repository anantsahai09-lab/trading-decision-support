"""
streamlit_app.py — Trading Decision-Support Tool, single-file web app.

Every engine module is bundled here so the app needs no package directory.
That matters for hosted deployment: a web uploader that flattens folders turns
a `src/` package into loose files, and the relative imports a package depends
on then fail. A single file cannot be flattened.

A rules engine that surfaces qualifying setups from a discretionary futures
strategy, explains why each candidate passed or failed, and measures how those
rules would have performed historically.

Decision support, not automation. No orders are ever placed.

RUN     streamlit run streamlit_app.py
DATA    Looks for data/nq_1m.parquet and data/es_1m.parquet; if they are
        absent the sidebar offers an uploader, so the app boots either way.
"""

from __future__ import annotations

import io
import os
from datetime import date

import streamlit as st


import argparse
import io
import itertools
import json
import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import zstandard as zstd
except ImportError:
    zstd = None

REQUIRED_COLS = ("open", "high", "low", "close")
RANDOM_STATE = 42



# ======================================================================
# SWINGS — structure, BOS, trend (§2)
# ======================================================================

class Direction(Enum):
    UP = 1
    DOWN = -1
    DOJI = 0




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
    h, l = highs.iloc[-1], lows.iloc[-1]
    if h == "HH" and l == "HL":
        return "up"
    if h == "LH" and l == "LL":
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


# ======================================================================
# OPENING RANGE + DOUBLE TEST (§3, §3.1)
# ======================================================================


FORMATION_BARS = ("09:30", "09:35")
FORMATION_END = "09:40"


@dataclass
class OpeningRange:
    """The 09:30-09:40 range for one session, on one instrument."""

    date: pd.Timestamp                 # session date (normalized, tz-aware)
    high: float                        # max of the two formation-bar highs
    low: float                         # min of the two formation-bar lows
    formed_at: pd.Timestamp            # bar-open ts of the LAST formation bar
    usable_from: pd.Timestamp          # first bar that may use the range
    levels: dict[str, float] = field(default_factory=dict)  # individual wicks
    n_bars: int = 0

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def position(self, price: float) -> str:
        if price > self.high:
            return "above"
        if price < self.low:
            return "below"
        return "inside"

    def __repr__(self) -> str:  # pragma: no cover - display only
        d = self.date.date() if hasattr(self.date, "date") else self.date
        return (f"OpeningRange({d} high={self.high:.2f} low={self.low:.2f} "
                f"height={self.height:.2f})")


def build_opening_range(
    df: pd.DataFrame,
    session_date=None,
    formation_bars: tuple[str, ...] = FORMATION_BARS,
    tz: str = "America/New_York",
) -> OpeningRange | None:
    """
    Build the opening range for one session from 5m bars.

    Returns None when the formation bars are missing (holiday, half day, data
    gap) — callers should treat that as a no-trade day rather than guessing.
    """
    _validate(df)

    if session_date is None:
        session_date = df.index[0].normalize()
    else:
        session_date = pd.Timestamp(session_date)
        if session_date.tz is None:
            session_date = session_date.tz_localize(tz)
        session_date = session_date.normalize()

    day = df[df.index.normalize() == session_date]
    if day.empty:
        return None

    wanted = []
    for hhmm in formation_bars:
        h, m = (int(x) for x in hhmm.split(":"))
        stamp = session_date + pd.Timedelta(hours=h, minutes=m)
        if stamp in day.index:
            wanted.append(stamp)

    if not wanted:
        return None

    bars = day.loc[wanted]
    levels = {}
    for ts, row in bars.iterrows():
        key = ts.strftime("%H:%M")
        levels[f"{key}_high"] = float(row["high"])
        levels[f"{key}_low"] = float(row["low"])

    return OpeningRange(
        date=session_date,
        high=float(bars["high"].max()),
        low=float(bars["low"].min()),
        formed_at=wanted[-1],
        usable_from=wanted[-1] + pd.Timedelta(minutes=5),
        levels=levels,
        n_bars=len(bars),
    )


def build_all_ranges(df: pd.DataFrame, **kwargs) -> dict[pd.Timestamp, OpeningRange]:
    """Build an opening range for every session present in df."""
    _validate(df)
    out = {}
    for date in pd.unique(df.index.normalize()):
        orng = build_opening_range(df, session_date=date, **kwargs)
        if orng is not None:
            out[date] = orng
    return out


def track_interactions(df: pd.DataFrame, orng: OpeningRange) -> pd.DataFrame:
    """
    Per-bar interaction with the range, for bars at/after `usable_from`.

    Columns:
      position   : 'above' | 'below' | 'inside'   (by CLOSE)
      touch_high : wick reached range high but closed inside
      touch_low  : wick reached range low but closed inside
      breach_high: close above range high
      breach_low : close below range low
      excursion  : distance of the bar extreme beyond the breached side
                   (0.0 when inside)
    """
    _validate(df)
    sub = df[df.index >= orng.usable_from]
    if sub.empty:
        return pd.DataFrame(
            columns=["position", "touch_high", "touch_low",
                     "breach_high", "breach_low", "excursion"]
        )

    close = sub["close"].to_numpy()
    high = sub["high"].to_numpy()
    low = sub["low"].to_numpy()

    breach_high = close > orng.high
    breach_low = close < orng.low
    inside = ~(breach_high | breach_low)

    touch_high = inside & (high >= orng.high)
    touch_low = inside & (low <= orng.low)

    position = np.where(breach_high, "above", np.where(breach_low, "below", "inside"))

    excursion = np.zeros(len(sub))
    excursion[breach_high] = (high[breach_high] - orng.high)
    excursion[breach_low] = (orng.low - low[breach_low])

    return pd.DataFrame(
        {
            "position": position,
            "touch_high": touch_high,
            "touch_low": touch_low,
            "breach_high": breach_high,
            "breach_low": breach_low,
            "excursion": excursion,
        },
        index=sub.index,
    )


@dataclass
class Episode:
    """A contiguous run of bars closed beyond one side of the range."""

    side: str                       # 'high' | 'low'
    start: pd.Timestamp
    end: pd.Timestamp | None        # last bar closed outside; None if still open
    extreme: float                  # most extreme price reached
    extreme_at: pd.Timestamp
    max_excursion: float            # distance beyond the level
    returned_at: pd.Timestamp | None = None   # first bar closed back inside
    n_bars: int = 0


def find_episodes(df: pd.DataFrame, orng: OpeningRange) -> list[Episode]:
    """Segment the session into breach episodes, in chronological order."""
    inter = track_interactions(df, orng)
    if inter.empty:
        return []

    sub = df.loc[inter.index]
    episodes: list[Episode] = []
    current: Episode | None = None

    for ts, row in inter.iterrows():
        bar = sub.loc[ts]
        side = "high" if row["breach_high"] else ("low" if row["breach_low"] else None)

        if side is None:
            if current is not None:
                current.returned_at = ts
                episodes.append(current)
                current = None
            continue

        if current is not None and current.side != side:
            # flipped straight from one side to the other without closing inside
            current.returned_at = ts
            episodes.append(current)
            current = None

        if current is None:
            current = Episode(
                side=side,
                start=ts,
                end=ts,
                extreme=float(bar["high"] if side == "high" else bar["low"]),
                extreme_at=ts,
                max_excursion=float(row["excursion"]),
                n_bars=1,
            )
        else:
            current.end = ts
            current.n_bars += 1
            price = float(bar["high"] if side == "high" else bar["low"])
            better = price > current.extreme if side == "high" else price < current.extreme
            if better:
                current.extreme = price
                current.extreme_at = ts
            current.max_excursion = max(current.max_excursion, float(row["excursion"]))

    if current is not None:
        episodes.append(current)

    return episodes


def detect_double_test(
    df: pd.DataFrame,
    orng: OpeningRange,
    require_return: bool = True,
) -> pd.DataFrame:
    """
    §3.1 double-test breakout rule.

    On the SECOND episode on a given side:
      - second excursion shallower than the first -> 'failed_breakout'
        signal direction = OPPOSITE of the tested side
      - second excursion deeper than the first    -> 'confirmed_breakout'
        signal direction = SAME as the tested side

    `require_return=True` (default) only emits a failed_breakout once price has
    actually closed back inside the range, matching "price broke to the downside
    and made its way back into the range". Set False to signal as soon as the
    second episode's extreme is exceeded/undershot.

    Returns columns:
      timestamp, side_tested, signal, direction, first_excursion,
      second_excursion, level
    """
    episodes = find_episodes(df, orng)
    rows = []
    seen: dict[str, list[Episode]] = {"high": [], "low": []}

    for ep in episodes:
        seen[ep.side].append(ep)
        if len(seen[ep.side]) != 2:
            continue  # only the second test is the decision point

        first, second = seen[ep.side][0], seen[ep.side][1]
        level = orng.high if ep.side == "high" else orng.low

        if second.max_excursion < first.max_excursion:
            signal = "failed_breakout"
            direction = "bear" if ep.side == "high" else "bull"
            ts = second.returned_at if require_return else second.extreme_at
            if ts is None:
                continue  # never closed back inside; wait
        else:
            signal = "confirmed_breakout"
            direction = "bull" if ep.side == "high" else "bear"
            ts = second.extreme_at

        rows.append(
            {
                "timestamp": ts,
                "side_tested": ep.side,
                "signal": signal,
                "direction": direction,
                "first_excursion": round(first.max_excursion, 4),
                "second_excursion": round(second.max_excursion, 4),
                "level": level,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["timestamp", "side_tested", "signal", "direction",
                 "first_excursion", "second_excursion", "level"],
    )


def range_violations(df: pd.DataFrame, orng: OpeningRange, lookback: int = 12) -> int:
    """
    Count breach bars in the last `lookback` bars.

    Feeds the §4 leading-index proxy: the instrument respecting its opening
    range more cleanly (fewer violations) is the provisional 'cleaner' one.
    """
    inter = track_interactions(df, orng)
    if inter.empty:
        return 0
    tail = inter.tail(lookback)
    return int((tail["breach_high"] | tail["breach_low"]).sum())


def compare_cleanliness(
    frames: dict[str, pd.DataFrame],
    ranges: dict[str, OpeningRange],
    lookback: int = 12,
) -> pd.DataFrame:
    """
    §4 provisional proxy across instruments.

    frames/ranges keyed by symbol. Returns a frame sorted cleanest-first with
    the violation count and range height (height is informational — a very
    narrow range violates more easily).
    """
    rows = []
    for sym, df in frames.items():
        orng = ranges.get(sym)
        if orng is None:
            continue
        rows.append(
            {
                "symbol": sym,
                "violations": range_violations(df, orng, lookback),
                "range_height": round(orng.height, 4),
                "range_high": orng.high,
                "range_low": orng.low,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("violations").reset_index(drop=True) if not out.empty else out


def detect_single_sweep(
    df: pd.DataFrame,
    orng: OpeningRange,
    min_excursion: float = 0.0,
) -> pd.DataFrame:
    """
    FIRST sweep-and-return, not the second test.

    §3.1 requires a level to be tested twice before it is tradeable. Under a
    pure AMD read the first manipulation IS the signal: price leaves the
    accumulation range, takes the liquidity resting there, and closes back
    inside. There is no reason to wait for a second one.

    Emits one signal per side, on the bar that closes back inside the range.
    """
    episodes = find_episodes(df, orng)
    rows = []
    seen = set()
    for ep in episodes:
        if ep.side in seen or ep.returned_at is None:
            continue
        if ep.max_excursion < min_excursion:
            continue
        seen.add(ep.side)
        rows.append({
            "timestamp": ep.returned_at,
            "side_tested": ep.side,
            "signal": "single_sweep",
            "direction": "bear" if ep.side == "high" else "bull",
            "first_excursion": round(ep.max_excursion, 4),
            "second_excursion": np.nan,
            "level": orng.high if ep.side == "high" else orng.low,
        })
    return pd.DataFrame(rows, columns=[
        "timestamp", "side_tested", "signal", "direction",
        "first_excursion", "second_excursion", "level"])


# ======================================================================
# SESSION LIQUIDITY (§5, §9 gate)
# ======================================================================


SESSION_BOUNDS = {
    "asia": ("18:00", "03:00"),
    "london": ("03:00", "08:30"),
    "ny": ("08:30", "17:00"),
}


def _minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.hour * 60 + idx.minute


def assign_trading_day(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Map each timestamp to its trading day.

    Bars at or after 18:00 belong to the next calendar day's session.
    """
    base = idx.normalize()
    roll = idx.hour >= 18
    return pd.Series(
        np.where(roll, base + pd.Timedelta(days=1), base),
        index=idx,
        name="trading_day",
    )


def assign_session(idx: pd.DatetimeIndex) -> pd.Series:
    """Label each timestamp asia / london / ny / break."""
    m = _minutes(idx)
    asia = (m >= 18 * 60) | (m < 3 * 60)
    london = (m >= 3 * 60) & (m < 8 * 60 + 30)
    ny = (m >= 8 * 60 + 30) & (m < 17 * 60)
    labels = np.where(asia, "asia", np.where(london, "london", np.where(ny, "ny", "break")))
    return pd.Series(labels, index=idx, name="session")


def session_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """
    High/low of every (trading_day, session) block.

    Returns columns: trading_day, session, high, low, high_at, low_at,
    session_end (last bar of that session block), n_bars.
    """
    _validate(df)
    work = df.copy()
    work["trading_day"] = assign_trading_day(df.index).to_numpy()
    work["session"] = assign_session(df.index).to_numpy()
    work = work[work["session"] != "break"]
    if work.empty:
        return pd.DataFrame(
            columns=["trading_day", "session", "high", "low",
                     "high_at", "low_at", "session_end", "n_bars"]
        )

    rows = []
    for (day, sess), g in work.groupby(["trading_day", "session"], sort=True):
        rows.append(
            {
                "trading_day": day,
                "session": sess,
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "high_at": g["high"].idxmax(),
                "low_at": g["low"].idxmin(),
                "session_end": g.index[-1],
                "n_bars": len(g),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["trading_day", "session_end"]).reset_index(drop=True)


def daily_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Full-session high/low per trading day (all sessions combined)."""
    _validate(df)
    work = df.copy()
    work["trading_day"] = assign_trading_day(df.index).to_numpy()
    rows = []
    for day, g in work.groupby("trading_day", sort=True):
        rows.append(
            {
                "trading_day": day,
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "high_at": g["high"].idxmax(),
                "low_at": g["low"].idxmin(),
                "day_end": g.index[-1],
                "n_bars": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("trading_day").reset_index(drop=True)


@dataclass
class Level:
    """One liquidity draw."""

    name: str                 # 'asia_high', 'prevday_low', ...
    price: float
    side: str                 # 'high' | 'low'
    source: str               # 'asia' | 'london' | 'ny' | 'prevday'
    formed_at: pd.Timestamp   # end of the session that produced it
    extreme_at: pd.Timestamp  # bar carrying the extreme
    tapped: bool = False
    tapped_at: pd.Timestamp | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        state = f"tapped@{self.tapped_at}" if self.tapped else "UNTAPPED"
        return f"Level({self.name} {self.price:.2f} {state})"


def build_liquidity_map(
    df: pd.DataFrame,
    trading_day,
    include_ny: bool = False,
    tz: str = "America/New_York",
) -> list[Level]:
    """
    Levels available to a session on `trading_day`.

    Includes that day's Asia and London extremes plus the previous trading day's
    high/low. The current day's NY extremes are excluded by default — they are
    still forming and would be lookahead.
    """
    _validate(df)
    day = pd.Timestamp(trading_day)
    if day.tz is None:
        day = day.tz_localize(tz)
    day = day.normalize()

    sess = session_extremes(df)
    daily = daily_extremes(df)
    levels: list[Level] = []

    today = sess[sess["trading_day"] == day]
    wanted = ["asia", "london"] + (["ny"] if include_ny else [])
    for _, r in today[today["session"].isin(wanted)].iterrows():
        levels.append(Level(f"{r['session']}_high", r["high"], "high",
                            r["session"], r["session_end"], r["high_at"]))
        levels.append(Level(f"{r['session']}_low", r["low"], "low",
                            r["session"], r["session_end"], r["low_at"]))

    prior = daily[daily["trading_day"] < day]
    if not prior.empty:
        p = prior.iloc[-1]
        levels.append(Level("prevday_high", p["high"], "high", "prevday",
                            p["day_end"], p["high_at"]))
        levels.append(Level("prevday_low", p["low"], "low", "prevday",
                            p["day_end"], p["low_at"]))

    return levels


def mark_tapped(
    levels: list[Level],
    df: pd.DataFrame,
    up_to: pd.Timestamp | None = None,
) -> list[Level]:
    """
    Flag levels reached after they formed. Wicks count.

    `up_to` bounds the scan so a backtest can ask "what was tapped as of now?"
    without seeing the future.
    """
    _validate(df)
    for lv in levels:
        window = df[df.index > lv.formed_at]
        if up_to is not None:
            window = window[window.index <= up_to]
        if window.empty:
            lv.tapped, lv.tapped_at = False, None
            continue

        hit = window["high"] >= lv.price if lv.side == "high" else window["low"] <= lv.price
        if hit.any():
            lv.tapped = True
            lv.tapped_at = window.index[hit.argmax()]
        else:
            lv.tapped, lv.tapped_at = False, None
    return levels


def untapped_targets(
    levels: list[Level],
    direction: str,
    current_price: float,
) -> list[Level]:
    """
    Untapped levels that sit in the trade's direction, nearest first.

    direction 'bull' -> untapped highs above price
    direction 'bear' -> untapped lows below price
    """
    if direction not in ("bull", "bear"):
        raise ValueError("direction must be 'bull' or 'bear'")

    if direction == "bull":
        out = [lv for lv in levels
               if not lv.tapped and lv.side == "high" and lv.price > current_price]
        out.sort(key=lambda lv: lv.price)
    else:
        out = [lv for lv in levels
               if not lv.tapped and lv.side == "low" and lv.price < current_price]
        out.sort(key=lambda lv: -lv.price)
    return out


def has_liquidity_target(
    levels: list[Level],
    direction: str,
    current_price: float,
) -> bool:
    """§9 gate: no untapped draw in the trade direction means no trade."""
    return len(untapped_targets(levels, direction, current_price)) > 0


def cluster_levels(
    levels: list[Level],
    tolerance: float,
    min_size: int = 2,
) -> list[dict]:
    """
    Group levels within `tolerance` of each other on the same side.

    Stacked / relative-equal levels hold more resting orders than a lone level,
    so a cluster is a stronger draw. Pairs naturally with the EQH/EQL labels the
    swing module produces.

    Returns dicts: side, price (mean), members, size, any_untapped.
    """
    out = []
    for side in ("high", "low"):
        group = sorted([lv for lv in levels if lv.side == side], key=lambda lv: lv.price)
        i = 0
        while i < len(group):
            bucket = [group[i]]
            j = i + 1
            while j < len(group) and abs(group[j].price - bucket[-1].price) <= tolerance:
                bucket.append(group[j])
                j += 1
            if len(bucket) >= min_size:
                out.append(
                    {
                        "side": side,
                        "price": float(np.mean([b.price for b in bucket])),
                        "members": [b.name for b in bucket],
                        "size": len(bucket),
                        "any_untapped": any(not b.tapped for b in bucket),
                    }
                )
            i = j
    return out


def liquidity_report(levels: list[Level]) -> pd.DataFrame:
    """Tabular view of a liquidity map — handy for eyeballing a session."""
    return pd.DataFrame(
        [
            {
                "name": lv.name,
                "side": lv.side,
                "price": lv.price,
                "source": lv.source,
                "formed_at": lv.formed_at,
                "tapped": lv.tapped,
                "tapped_at": lv.tapped_at,
            }
            for lv in levels
        ]
    ).sort_values("price", ascending=False).reset_index(drop=True)


# ======================================================================
# CONFLUENCES — FVG / IFVG / SMT (§6)
# ======================================================================



# =====================================================================
# FAIR VALUE GAPS
# =====================================================================

@dataclass
class FVG:
    """One fair value gap and its lifecycle state."""

    direction: str            # 'bull' | 'bear'
    top: float                # upper bound of the zone
    bottom: float             # lower bound of the zone
    formed_at: pd.Timestamp   # candle 3 — first bar at which the gap is known
    c1_at: pd.Timestamp
    c3_at: pd.Timestamp
    tapped: bool = False
    tapped_at: pd.Timestamp | None = None
    inverted: bool = False
    inverted_at: pd.Timestamp | None = None

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def alive(self) -> bool:
        """Still usable as a continuation confluence."""
        return not self.inverted

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def __repr__(self) -> str:  # pragma: no cover
        state = "INVERTED" if self.inverted else ("tapped" if self.tapped else "fresh")
        return (f"FVG({self.direction} [{self.bottom:.2f}, {self.top:.2f}] "
                f"formed={self.formed_at} {state})")


def detect_fvgs(df: pd.DataFrame, min_size: float = 0.0) -> list[FVG]:
    """
    Find every 3-candle imbalance in df.

    `min_size` filters out microscopic gaps (noise on fast timeframes).
    A gap is only reported at candle 3, so nothing here looks ahead.
    """
    _validate(df)
    out: list[FVG] = []
    if len(df) < 3:
        return out

    idx = df.index
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    for i in range(len(df) - 2):
        c1_h, c1_l = high[i], low[i]
        c3_h, c3_l = high[i + 2], low[i + 2]

        if c3_l > c1_h:                       # bullish imbalance
            gap = c3_l - c1_h
            if gap >= min_size:
                out.append(FVG("bull", top=float(c3_l), bottom=float(c1_h),
                               formed_at=idx[i + 2], c1_at=idx[i], c3_at=idx[i + 2]))
        elif c3_h < c1_l:                     # bearish imbalance
            gap = c1_l - c3_h
            if gap >= min_size:
                out.append(FVG("bear", top=float(c1_l), bottom=float(c3_h),
                               formed_at=idx[i + 2], c1_at=idx[i], c3_at=idx[i + 2]))

    return out


def update_fvg_state(
    fvgs: list[FVG],
    df: pd.DataFrame,
    up_to: pd.Timestamp | None = None,
) -> list[FVG]:
    """
    Walk bars after each gap formed and mark tapped / inverted.

    TAPPED   : any later bar wicks into the zone.
    INVERTED : a later bar CLOSES beyond the far side —
               below `bottom` for a bull gap, above `top` for a bear gap.
               Closes decide inversion for the same reason they decide BOS.

    `up_to` bounds the scan so a backtest never sees the future.
    """
    _validate(df)
    for f in fvgs:
        window = df[df.index > f.formed_at]
        if up_to is not None:
            window = window[window.index <= up_to]
        f.tapped, f.tapped_at = False, None
        f.inverted, f.inverted_at = False, None
        if window.empty:
            continue

        touched = (window["low"] <= f.top) & (window["high"] >= f.bottom)
        if touched.any():
            f.tapped = True
            f.tapped_at = window.index[touched.argmax()]

        killed = (window["close"] < f.bottom) if f.direction == "bull" \
            else (window["close"] > f.top)
        if killed.any():
            f.inverted = True
            f.inverted_at = window.index[killed.argmax()]

    return fvgs


def stack_fvgs(fvgs: list[FVG], df: pd.DataFrame) -> list[list[FVG]]:
    """
    Group consecutive same-direction gaps with no opposing retrace between them.

    Returns a list of stacks (each a list of FVGs, chronological). Single gaps
    come back as one-element stacks.
    """
    if not fvgs:
        return []
    ordered = sorted(fvgs, key=lambda f: f.formed_at)
    stacks: list[list[FVG]] = [[ordered[0]]]
    for f in ordered[1:]:
        prev = stacks[-1][-1]
        contiguous = (f.direction == prev.direction) and (f.c1_at <= prev.c3_at)
        if contiguous:
            stacks[-1].append(f)
        else:
            stacks.append([f])
    return stacks


def invalidation_gap(stack: list[FVG]) -> FVG:
    """
    The gap in a stack whose inversion actually kills the move.

    Bull stack -> the LOWEST gap. Bear stack -> the HIGHEST. Inverting a gap
    above the lowest bull gap means nothing while lower gaps remain live.
    """
    if not stack:
        raise ValueError("empty stack")
    if stack[0].direction == "bull":
        return min(stack, key=lambda f: f.bottom)
    return max(stack, key=lambda f: f.top)


def active_fvgs(
    fvgs: list[FVG],
    direction: str | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[FVG]:
    """Gaps still usable as continuation confluences."""
    out = [f for f in fvgs if f.alive]
    if direction is not None:
        out = [f for f in out if f.direction == direction]
    if as_of is not None:
        out = [f for f in out if f.formed_at <= as_of]
    return out


def detect_ifvg_signals(fvgs: list[FVG]) -> pd.DataFrame:
    """
    Inverted FVGs as confirmation signals.

    A bull gap that gets closed through signals BEARISH confirmation, and vice
    versa — the continuation confluence failed, so the trend it supported is
    likely done.
    """
    rows = []
    for f in fvgs:
        if not f.inverted:
            continue
        rows.append(
            {
                "timestamp": f.inverted_at,
                "signal": "ifvg",
                "direction": "bear" if f.direction == "bull" else "bull",
                "original_direction": f.direction,
                "zone_top": f.top,
                "zone_bottom": f.bottom,
                "formed_at": f.formed_at,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["timestamp", "signal", "direction", "original_direction",
                 "zone_top", "zone_bottom", "formed_at"],
    )
    return out.sort_values("timestamp").reset_index(drop=True) if not out.empty else out


def fvg_report(fvgs: list[FVG]) -> pd.DataFrame:
    """Tabular view of gaps — handy for eyeballing a session."""
    return pd.DataFrame(
        [
            {
                "direction": f.direction,
                "bottom": f.bottom,
                "top": f.top,
                "size": round(f.size, 4),
                "formed_at": f.formed_at,
                "tapped": f.tapped,
                "tapped_at": f.tapped_at,
                "inverted": f.inverted,
                "inverted_at": f.inverted_at,
            }
            for f in fvgs
        ]
    )


# =====================================================================
# SMT DIVERGENCE
# =====================================================================

@dataclass
class SMT:
    """One SMT divergence between two correlated instruments."""

    direction: str            # 'bull' | 'bear'
    timestamp: pd.Timestamp   # when the divergence is confirmed
    leading: str              # symbol that leads the move
    lagging: str
    leading_price: float
    lagging_price: float
    swing_type: str           # 'high' | 'low'

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SMT({self.direction} lead={self.leading} "
                f"lag={self.lagging} @ {self.timestamp})")


def _pair_swings(
    swings_a: pd.DataFrame,
    swings_b: pd.DataFrame,
    swing_type: str,
    tolerance_bars: int = 1,
    bar_minutes: int = 1,
) -> list[tuple[pd.Series, pd.Series]]:
    """
    Match swings of the same type across two instruments by timestamp.

    Two swings are 'the same swing' when their extremes fall within
    tolerance_bars of each other.
    """
    a = swings_a[swings_a["type"] == swing_type].reset_index(drop=True)
    b = swings_b[swings_b["type"] == swing_type].reset_index(drop=True)
    tol = pd.Timedelta(minutes=bar_minutes * tolerance_bars)

    pairs = []
    for _, ra in a.iterrows():
        cand = b[(b["extreme_at"] >= ra["extreme_at"] - tol)
                 & (b["extreme_at"] <= ra["extreme_at"] + tol)]
        if not cand.empty:
            closest = (cand["extreme_at"] - ra["extreme_at"]).abs().idxmin()
            pairs.append((ra, b.loc[closest]))
    return pairs


def detect_smt(
    swings_a: pd.DataFrame,
    swings_b: pd.DataFrame,
    symbol_a: str = "NQ",
    symbol_b: str = "ES",
    tolerance_bars: int = 1,
    bar_minutes: int = 1,
) -> list[SMT]:
    """
    Find SMT divergences between two instruments.

    Inputs are the LABELLED swing frames from swings.label_swings(), so each row
    carries HH / LH / HL / LL. A divergence exists when paired swings disagree.

    Bearish: at aligned highs one is HH and the other LH -> the LH index leads.
    Bullish: at aligned lows one is LL and the other HL  -> the HL index leads.
    """
    out: list[SMT] = []

    for ra, rb in _pair_swings(swings_a, swings_b, "high", tolerance_bars, bar_minutes):
        la, lb = ra.get("label"), rb.get("label")
        if {la, lb} == {"HH", "LH"}:
            lead_a = la == "LH"
            out.append(SMT(
                direction="bear",
                timestamp=max(ra["confirmed_at"], rb["confirmed_at"]),
                leading=symbol_a if lead_a else symbol_b,
                lagging=symbol_b if lead_a else symbol_a,
                leading_price=float(ra["price"] if lead_a else rb["price"]),
                lagging_price=float(rb["price"] if lead_a else ra["price"]),
                swing_type="high",
            ))

    for ra, rb in _pair_swings(swings_a, swings_b, "low", tolerance_bars, bar_minutes):
        la, lb = ra.get("label"), rb.get("label")
        if {la, lb} == {"LL", "HL"}:
            lead_a = la == "HL"
            out.append(SMT(
                direction="bull",
                timestamp=max(ra["confirmed_at"], rb["confirmed_at"]),
                leading=symbol_a if lead_a else symbol_b,
                lagging=symbol_b if lead_a else symbol_a,
                leading_price=float(ra["price"] if lead_a else rb["price"]),
                lagging_price=float(rb["price"] if lead_a else ra["price"]),
                swing_type="low",
            ))

    return sorted(out, key=lambda s: s.timestamp)


def smt_report(smts: list[SMT]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": s.timestamp,
                "direction": s.direction,
                "swing_type": s.swing_type,
                "leading": s.leading,
                "lagging": s.lagging,
                "leading_price": s.leading_price,
                "lagging_price": s.lagging_price,
            }
            for s in smts
        ]
    )


# =====================================================================
# COMBINED CONFLUENCE CHECK (§6 gate)
# =====================================================================

def confluence_check(
    direction: str,
    as_of: pd.Timestamp,
    price: float,
    fvgs: list[FVG] | None = None,
    bos: pd.DataFrame | None = None,
    smts: list[SMT] | None = None,
    ifvg: pd.DataFrame | None = None,
    bos_window_bars: int = 3,
    bar_minutes: int = 1,
) -> dict:
    """
    §6: does ANY continuation confluence support `direction` right now?

    Returns {'pass': bool, 'reasons': [...], 'detail': {...}}. All inputs are
    optional so partial wiring still works during the build.
    """
    if direction not in ("bull", "bear"):
        raise ValueError("direction must be 'bull' or 'bear'")

    reasons: list[str] = []
    detail: dict = {}

    # 1. a live FVG in the trade direction that price is currently touching
    if fvgs:
        live = [f for f in active_fvgs(fvgs, direction, as_of) if f.contains(price)]
        if live:
            reasons.append("fvg_tap")
            detail["fvg"] = repr(live[0])

    # 2. recent BOS in the trade direction on the execution timeframe
    if bos is not None and not bos.empty:
        window = pd.Timedelta(minutes=bar_minutes * bos_window_bars)
        recent = bos[(bos["timestamp"] <= as_of)
                     & (bos["timestamp"] >= as_of - window)
                     & (bos["direction"] == direction)]
        if not recent.empty:
            reasons.append("bos")
            detail["bos"] = recent.iloc[-1].to_dict()

    # 3. SMT divergence agreeing with the trade
    if smts:
        agree = [s for s in smts if s.direction == direction and s.timestamp <= as_of]
        if agree:
            reasons.append("smt")
            detail["smt"] = repr(agree[-1])

    # 4. an inverted FVG pointing the trade's way
    if ifvg is not None and not ifvg.empty:
        agree = ifvg[(ifvg["direction"] == direction) & (ifvg["timestamp"] <= as_of)]
        if not agree.empty:
            reasons.append("ifvg")
            detail["ifvg"] = agree.iloc[-1].to_dict()

    return {"pass": len(reasons) > 0, "reasons": reasons, "detail": detail}


# ======================================================================
# AMD — accumulation / manipulation / distribution (§7)
# ======================================================================

@dataclass
class AMDState:
    """The phase structure of one session, on one instrument."""

    date: pd.Timestamp
    valid: bool = False
    phase: str = "accumulation"          # accumulation | manipulation | distribution
    manip_side: str | None = None        # 'high' | 'low' — which way the sweep went
    manip_start: pd.Timestamp | None = None
    manip_extreme: float | None = None
    manip_extreme_at: pd.Timestamp | None = None
    manip_excursion: float = 0.0
    returned_at: pd.Timestamp | None = None
    distribution_from: pd.Timestamp | None = None
    trade_direction: str | None = None   # opposite the manipulation
    swept_levels: list = field(default_factory=list)
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        if not self.valid:
            return f"AMD({self.date.date()} INVALID — {self.reason})"
        return (f"AMD({self.date.date()} manip={self.manip_side} "
                f"exc={self.manip_excursion:.1f} -> distribute {self.trade_direction} "
                f"from {self.distribution_from})")


def classify_amd(
    df_5m: pd.DataFrame,
    orng,
    levels: list | None = None,
    min_excursion_ticks: float = 0.0,
    tick_size: float = 0.25,
    require_sweep: bool = False,
) -> AMDState:
    """
    Walk a session and assign its AMD structure.

    df_5m : 5-minute bars for the session
    orng  : the OpeningRange (accumulation phase)
    levels: optional session liquidity levels, to record what the sweep took
    require_sweep: if True, the manipulation must take out a mapped level.
                   Default False — per spec, manipulation *may* sweep a level
                   but does not have to.
    """
    state = AMDState(date=orng.date)

    bars = df_5m[df_5m.index >= orng.usable_from]
    if bars.empty:
        state.reason = "no bars after accumulation"
        return state

    # ---- manipulation: first CLOSE outside the range -------------------
    manip_side = None
    manip_start = None
    for ts, row in bars.iterrows():
        if row["close"] > orng.high:
            manip_side, manip_start = "high", ts
            break
        if row["close"] < orng.low:
            manip_side, manip_start = "low", ts
            break

    if manip_side is None:
        state.reason = "no manipulation — price never left the accumulation range"
        return state

    state.manip_side = manip_side
    state.manip_start = manip_start
    state.phase = "manipulation"

    # ---- track the excursion, wait for the return inside ---------------
    level = orng.high if manip_side == "high" else orng.low
    extreme = level
    extreme_at = manip_start
    returned_at = None

    for ts, row in bars[bars.index >= manip_start].iterrows():
        if manip_side == "high":
            if row["high"] > extreme:
                extreme, extreme_at = float(row["high"]), ts
            if row["close"] < orng.high:          # closed back inside
                returned_at = ts
                break
        else:
            if row["low"] < extreme:
                extreme, extreme_at = float(row["low"]), ts
            if row["close"] > orng.low:
                returned_at = ts
                break

    state.manip_extreme = extreme
    state.manip_extreme_at = extreme_at
    state.manip_excursion = abs(extreme - level)

    if returned_at is None:
        state.reason = ("manipulation never returned inside — trending day, "
                        "not an AMD day")
        return state

    if state.manip_excursion < min_excursion_ticks * tick_size:
        state.reason = (f"manipulation too shallow "
                        f"({state.manip_excursion / tick_size:.0f} ticks)")
        return state

    # ---- optional: did the sweep take out a mapped level? --------------
    if levels:
        for lv in levels:
            if manip_side == "high" and lv.side == "high" and extreme >= lv.price > level:
                state.swept_levels.append(lv.name)
            elif manip_side == "low" and lv.side == "low" and extreme <= lv.price < level:
                state.swept_levels.append(lv.name)

    if require_sweep and not state.swept_levels:
        state.reason = "manipulation swept no mapped liquidity level"
        return state

    # ---- distribution: trade AWAY from the manipulation ----------------
    state.valid = True
    state.phase = "distribution"
    state.returned_at = returned_at
    state.distribution_from = returned_at
    state.trade_direction = "bear" if manip_side == "high" else "bull"
    state.reason = "ok"
    return state


def amd_gate(state: AMDState, direction: str, ts: pd.Timestamp) -> tuple[bool, str]:
    """
    Should a setup be allowed under this session's AMD structure?

    Returns (allowed, reason).
    """
    if not state.valid:
        return False, f"§7 AMD: {state.reason}"
    if ts < state.distribution_from:
        return False, ("§7 AMD: still in manipulation — distribution begins "
                       f"{state.distribution_from.strftime('%H:%M')}")
    if direction != state.trade_direction:
        return False, (f"§7 AMD: manipulation swept the {state.manip_side}, so "
                       f"distribution is {state.trade_direction}, not {direction}")
    return True, f"§7 AMD: distribution {state.trade_direction} after {state.manip_side} sweep"


def amd_report(states: dict) -> pd.DataFrame:
    """Summarise AMD classification across sessions."""
    rows = []
    for date, s in states.items():
        rows.append({
            "date": date,
            "valid": s.valid,
            "manip_side": s.manip_side,
            "excursion": round(s.manip_excursion, 2),
            "direction": s.trade_direction,
            "dist_from": s.distribution_from,
            "swept": ",".join(s.swept_levels),
            "reason": s.reason,
        })
    return pd.DataFrame(rows)


# ======================================================================
# DATA LOADER — Databento -> continuous contract
# ======================================================================

TZ = "America/New_York"
USE_COLS = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]


# Futures month codes, in calendar order — used to compare expirations.
MONTH_CODES = "FGHJKMNQUVXZ"


def _open_maybe_zst(path: str):
    """Return a file-like object of decompressed text."""
    if str(path).endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed: pip install zstandard")
        fh = open(path, "rb")
        return io.TextIOWrapper(
            zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8"
        )
    return open(path, "r", encoding="utf-8")


def parse_contract(symbol: str) -> tuple[str, int, int] | None:
    """
    'NQU6' -> ('NQ', 8, 6)   root, month index (0-11), single-digit year.

    Returns None for spreads or anything unparseable.
    """
    if "-" in symbol:
        return None
    s = symbol.strip()
    if len(s) < 3:
        return None
    year_part = s[-1]
    month_part = s[-2]
    root = s[:-2]
    if not year_part.isdigit() or month_part not in MONTH_CODES:
        return None
    return root, MONTH_CODES.index(month_part), int(year_part)


def expiry_rank(symbol: str, ref_year: int) -> float:
    """
    Sortable expiration key. Databento uses a single-digit year, so the decade is
    inferred from the bar's own year — a contract can be at most ~2 years out.
    """
    parsed = parse_contract(symbol)
    if parsed is None:
        return float("inf")
    _, month_idx, yr = parsed
    decade = (ref_year // 10) * 10
    full_year = decade + yr
    # a contract more than 2 years behind the reference belongs to the next decade
    if full_year < ref_year - 1:
        full_year += 10
    return full_year * 12 + month_idx


@dataclass
class LoadReport:
    symbol_root: str
    rows_raw: int
    rows_kept: int
    spreads_dropped: int
    contracts_used: int
    rolls: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover
        return (f"LoadReport({self.symbol_root}: {self.rows_kept:,}/{self.rows_raw:,} "
                f"rows, {self.contracts_used} contracts, {len(self.rolls)} rolls)")


def build_continuous(
    path: str,
    chunksize: int = 1_000_000,
    tz: str = TZ,
    verbose: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Stream a Databento parent-symbol OHLCV file into one continuous 1m series.

    Returns (df, report) where df is indexed by tz-aware bar-open timestamp with
    columns open/high/low/close/volume/symbol.
    """
    # ---- pass 1: daily volume per outright contract -------------------
    daily_vol: dict[tuple[pd.Timestamp, str], int] = {}
    rows_raw = spreads = 0

    reader = pd.read_csv(
        _open_maybe_zst(path), usecols=USE_COLS, chunksize=chunksize,
        dtype={"symbol": str},
    )
    for chunk in reader:
        rows_raw += len(chunk)
        is_spread = chunk["symbol"].str.contains("-", regex=False)
        spreads += int(is_spread.sum())
        chunk = chunk[~is_spread]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts_event"], utc=True, format="ISO8601")
        local = ts.dt.tz_convert(tz)
        # trading day: bars at/after 18:00 belong to the next session
        day = local.dt.normalize() + pd.to_timedelta(
            (local.dt.hour >= 18).astype(int), unit="D"
        )
        g = (
            pd.DataFrame({"day": day.to_numpy(),
                          "symbol": chunk["symbol"].to_numpy(),
                          "volume": chunk["volume"].to_numpy()})
            .groupby(["day", "symbol"], sort=False)["volume"].sum()
        )
        for (d, s), v in g.items():
            daily_vol[(d, s)] = daily_vol.get((d, s), 0) + int(v)

    if not daily_vol:
        raise ValueError("no outright contract rows found")

    vol = (
        pd.Series(daily_vol)
        .rename_axis(["day", "symbol"])
        .rename("volume")
        .reset_index()
    )

    # ---- choose front month per day, never rolling backwards ----------
    front: dict[pd.Timestamp, str] = {}
    last_rank = -np.inf
    for day, g in vol.groupby("day", sort=True):
        ref_year = int(pd.Timestamp(day).year)
        g = g.copy()
        g["rank"] = [expiry_rank(s, ref_year) for s in g["symbol"]]
        g = g[np.isfinite(g["rank"])]
        eligible = g[g["rank"] >= last_rank]
        pick = (eligible if not eligible.empty else g).sort_values(
            "volume", ascending=False
        ).iloc[0]
        front[day] = pick["symbol"]
        last_rank = max(last_rank, pick["rank"])

    front_s = pd.Series(front).sort_index()
    changes = front_s != front_s.shift(1)
    rolls = pd.DataFrame({
        "day": front_s.index[changes],
        "from": front_s.shift(1)[changes].to_numpy(),
        "to": front_s[changes].to_numpy(),
    }).iloc[1:].reset_index(drop=True)   # first entry is not a roll

    if verbose:
        print(f"  pass 1: {rows_raw:,} rows, {spreads:,} spread rows dropped, "
              f"{front_s.nunique()} contracts, {len(rolls)} rolls")

    # ---- pass 2: keep only front-month bars ---------------------------
    keep_parts = []
    rows_kept = 0
    reader = pd.read_csv(
        _open_maybe_zst(path), usecols=USE_COLS, chunksize=chunksize,
        dtype={"symbol": str},
    )
    for chunk in reader:
        chunk = chunk[~chunk["symbol"].str.contains("-", regex=False)]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts_event"], utc=True, format="ISO8601")
        local = ts.dt.tz_convert(tz)
        day = local.dt.normalize() + pd.to_timedelta(
            (local.dt.hour >= 18).astype(int), unit="D"
        )
        want = day.map(front_s)
        mask = (want.to_numpy() == chunk["symbol"].to_numpy())
        sel = chunk[mask]
        if sel.empty:
            continue
        out = pd.DataFrame(
            {
                "open": sel["open"].to_numpy(float),
                "high": sel["high"].to_numpy(float),
                "low": sel["low"].to_numpy(float),
                "close": sel["close"].to_numpy(float),
                "volume": sel["volume"].to_numpy(float),
                "symbol": sel["symbol"].to_numpy(),
            },
            index=local[mask],
        )
        keep_parts.append(out)
        rows_kept += len(out)

    df = pd.concat(keep_parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "timestamp"

    root = parse_contract(str(front_s.iloc[0]))
    report = LoadReport(
        symbol_root=root[0] if root else "?",
        rows_raw=rows_raw,
        rows_kept=len(df),
        spreads_dropped=spreads,
        contracts_used=int(front_s.nunique()),
        rolls=rolls,
        start=df.index.min(),
        end=df.index.max(),
    )
    if verbose:
        print(f"  pass 2: kept {len(df):,} bars  "
              f"{report.start.date()} -> {report.end.date()}")
    return df, report


def data_health_report(df: pd.DataFrame, name: str = "data") -> dict:
    """
    Checks worth running before trusting any backtest built on this file.

    Returns a dict of findings as well as printing them.
    """
    out = {}
    print(f"=== {name} ===")
    print(f"rows            : {len(df):,}")
    print(f"range           : {df.index.min()}  ->  {df.index.max()}")
    print(f"timezone        : {df.index.tz}")

    out["bad_hl"] = int((df["high"] < df["low"]).sum())
    out["open_outside"] = int(((df["open"] > df["high"]) | (df["open"] < df["low"])).sum())
    out["close_outside"] = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    out["dupes"] = int(df.index.duplicated().sum())
    print(f"high < low      : {out['bad_hl']}")
    print(f"open outside HL : {out['open_outside']}")
    print(f"close outside HL: {out['close_outside']}")
    print(f"dup timestamps  : {out['dupes']}")

    diffs = df.index.to_series().diff()
    big = diffs[diffs > pd.Timedelta("1h")]
    weekend = big[big > pd.Timedelta("12h")]
    out["gaps_1h"] = len(big)
    out["gaps_12h"] = len(weekend)
    print(f"gaps > 1h       : {len(big):,}   (>12h / weekends: {len(weekend):,})")

    flat = int((df["high"] == df["low"]).sum())
    out["flat_bars"] = flat
    print(f"flat bars       : {flat:,}  ({flat / max(len(df), 1):.2%})")

    hours = set(df.index.hour.unique())
    checks = [
        ("Asia   18:00-03:00", set(range(18, 24)) | set(range(0, 3))),
        ("London 03:00-08:30", set(range(3, 9))),
        ("NY     08:30-17:00", set(range(8, 17))),
    ]
    for label, rng in checks:
        present = bool(rng & hours)
        out[label.split()[0].lower()] = present
        print(f"{label} present: {present}")

    sessions = df.index.normalize().nunique()
    out["sessions"] = sessions
    print(f"distinct dates  : {sessions:,}")
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m bars -> higher timeframe. rule: '5min', '15min', '1h', '1D'."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Cache the continuous series so the slow rebuild only happens once."""
    df.to_parquet(path)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ======================================================================
# ORCHESTRATOR — the decision pipeline (§1,§4,§8,§9)
# ======================================================================

# ---------------------------------------------------------------------
# Contract specifications
# ---------------------------------------------------------------------

CONTRACT_SPECS = {
    "NQ": {"tick_size": 0.25, "tick_value": 5.00,  "point_value": 20.0},
    "ES": {"tick_size": 0.25, "tick_value": 12.50, "point_value": 50.0},
}

# §1 time windows (America/New_York)
ENTRY_START = "09:50"
ENTRY_END = "10:30"

# §8 defaults
DEFAULT_RISK_PCT = 0.01
DEFAULT_RR = 2.0
NQ_STOP_TICK_CAP = 300      # max, not a target (§10.5)
NQ_STOP_TICK_MIN = 0        # structural stop wins if tighter

# §8.2 account limits
MAX_TRADES_PER_SESSION = 2
MAX_DAILY_LOSS_PCT = 0.02
DAILY_PROFIT_LOCK_PCT = 0.03
MAX_CONSECUTIVE_LOSSES = 2


@dataclass
class Setup:
    """One evaluated trade opportunity — taken or rejected."""

    timestamp: pd.Timestamp
    session_date: pd.Timestamp
    symbol: str | None = None
    direction: str | None = None
    qualified: bool = False

    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    stop_ticks: float | None = None
    rr: float | None = None
    contracts: int | None = None
    risk_dollars: float | None = None

    trigger: str | None = None
    confluences: list[str] = field(default_factory=list)
    target_level: str | None = None
    passed: list[str] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        if not self.qualified:
            return f"Setup(REJECTED {self.session_date.date()} — {'; '.join(self.vetoes)})"
        return (f"Setup({self.direction.upper()} {self.symbol} @ {self.entry:.2f} "
                f"stop {self.stop:.2f} target {self.target:.2f} "
                f"{self.contracts}c  [{', '.join(self.confluences)}])")

    def explain(self) -> str:
        """Full reasoning chain, human-readable."""
        lines = [f"--- {self.session_date.date()}  {self.timestamp} ---"]
        for p in self.passed:
            lines.append(f"  PASS   {p}")
        for v in self.vetoes:
            lines.append(f"  VETO   {v}")
        if self.qualified:
            lines += [
                "",
                f"  TRADE  {self.direction.upper()} {self.symbol}",
                f"  entry  {self.entry:.2f}",
                f"  stop   {self.stop:.2f}   ({self.stop_ticks:.0f} ticks)",
                f"  target {self.target:.2f}  ({self.target_level}, {self.rr:.2f}R)",
                f"  size   {self.contracts} contracts  "
                f"(${self.risk_dollars:,.0f} risk)",
                f"  why    {', '.join(self.confluences)}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------
# §8 risk engine
# ---------------------------------------------------------------------

def structural_stop(
    swings_1m: pd.DataFrame,
    direction: str,
    as_of: pd.Timestamp,
) -> tuple[float | None, str]:
    """
    §8: stop goes beyond the SECOND most recent 1m swing in the trade direction.

    Long  -> second most recent swing LOW.
    Short -> second most recent swing HIGH.

    The second rather than the first, because the most recent swing is usually
    the one price just reacted from and sits too close to survive noise.
    """
    want = "low" if direction == "bull" else "high"
    avail = swings_1m[
        (swings_1m["type"] == want) & (swings_1m["confirmed_at"] <= as_of)
    ].sort_values("confirmed_at")
    if len(avail) < 2:
        return None, "fewer than two 1m swings available"
    return float(avail.iloc[-2]["price"]), "ok"


def size_position(
    account: float,
    entry: float,
    stop: float,
    symbol: str,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> tuple[int, float, float]:
    """
    Contracts sized so a stop-out costs `risk_pct` of the account.

    Returns (contracts, stop_ticks, risk_dollars).
    """
    spec = CONTRACT_SPECS[symbol]
    stop_points = abs(entry - stop)
    stop_ticks = stop_points / spec["tick_size"]
    if stop_ticks <= 0:
        return 0, 0.0, 0.0
    risk_per_contract = stop_ticks * spec["tick_value"]
    budget = account * risk_pct
    contracts = int(budget // risk_per_contract)
    return contracts, stop_ticks, contracts * risk_per_contract


def build_target(
    entry: float,
    stop: float,
    direction: str,
    levels: list,
    rr: float = DEFAULT_RR,
) -> tuple[float | None, str, float | None]:
    """
    §8: target = entry ± (stop distance × RR), capped at the nearest untapped
    liquidity level in the trade direction.

    Returns (target, label, realized_rr).
    """

    risk = abs(entry - stop)
    raw = entry + risk * rr if direction == "bull" else entry - risk * rr

    tg = untapped_targets(levels, direction, entry)
    if not tg:
        return None, "no untapped liquidity", None

    nearest = tg[0]
    if direction == "bull":
        target = min(raw, nearest.price)
    else:
        target = max(raw, nearest.price)

    realized = abs(target - entry) / risk if risk > 0 else None
    label = nearest.name if target == nearest.price else f"{rr:.1f}R"
    return float(target), label, realized


# ---------------------------------------------------------------------
# §1 helpers
# ---------------------------------------------------------------------

def in_entry_window(ts: pd.Timestamp, start=ENTRY_START, end=ENTRY_END) -> bool:
    t = ts.strftime("%H:%M")
    return start <= t <= end


# ---------------------------------------------------------------------
# §4 leading index
# ---------------------------------------------------------------------

def choose_instrument(
    ranges: dict,
    frames_5m: dict,
    smts: list | None = None,
    direction: str | None = None,
    lookback: int = 12,
) -> tuple[str, str]:
    """
    §4: pick the leading index.

    1. If an SMT divergence agrees with the trade direction, use its leading leg.
    2. Otherwise fall back to fewest opening-range breaches (provisional proxy —
       flagged as the weakest rule in the spec).
    """

    if smts and direction:
        agree = [s for s in smts if s.direction == direction]
        if agree:
            return agree[-1].leading, "smt"

    scores = {
        sym: range_violations(frames_5m[sym], ranges[sym], lookback)
        for sym in ranges
        if sym in frames_5m
    }
    if not scores:
        raise ValueError("no instruments available")
    best = min(scores, key=scores.get)
    return best, f"fewest_or_breaches({scores})"


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def evaluate_session(
    session_date,
    data_1m: dict,
    account: float = 100_000.0,
    risk_pct: float = DEFAULT_RISK_PCT,
    rr: float = DEFAULT_RR,
    fvg_min_size: float = 5.0,
    max_trades: int = MAX_TRADES_PER_SESSION,
    tz: str = "America/New_York",
    verbose: bool = False,
    # ---- tunable gates (all default to current spec behaviour) ----
    entry_start: str = ENTRY_START,
    entry_end: str = ENTRY_END,
    require_confluences: tuple | None = None,   # e.g. ("smt",)
    forbid_confluences: tuple | None = None,    # e.g. ("bos",)
    directions: tuple = ("bull", "bear"),
    slippage_ticks: float = 1.0,                # accounted for in RR
    ml_filter=None,                             # callable(features)->bool
    use_amd: bool = False,                      # §7 AMD gate
    amd_min_excursion_ticks: float = 0.0,
    amd_require_sweep: bool = False,
    allow_confirmed_breakout: bool = True,      # AMD says no; default keeps old behaviour
    stop_mode: str = "structural",              # 'structural' | 'fixed' | 'clamped'
    stop_ticks: float = 200.0,                  # used when stop_mode='fixed'
    trigger_mode: str = "double",               # 'double' | 'single' | 'both'
    stop_min_ticks: float = 200.0,              # 'clamped': floor
    stop_max_ticks: float = 500.0,              # 'clamped': ceiling
) -> list[Setup]:
    """
    Run the full pipeline for one session.

    data_1m: {'NQ': df, 'ES': df} of 1-minute bars spanning at least the prior
    trading day (needed for §5 previous-day levels).

    Returns every evaluated Setup, qualified or not — a rejected day should be
    as inspectable as a taken one.
    """

    day = pd.Timestamp(session_date)
    if day.tz is None:
        day = day.tz_localize(tz)
    day = day.normalize()

    setups: list[Setup] = []
    symbols = list(data_1m.keys())

    # ---- per-instrument session slices --------------------------------
    # One narrow slice per instrument up front. Repeatedly indexing the full
    # multi-million-row history is what makes a naive scan unusably slow.
    frames_1m, frames_5m, frames_15m, ranges, windows = {}, {}, {}, {}, {}
    for sym, df in data_1m.items():
        window = df[(df.index >= day - pd.Timedelta(days=5))
                    & (df.index < day + pd.Timedelta(hours=17))]
        if window.empty:
            continue
        windows[sym] = window
        sess = window[(window.index >= day + pd.Timedelta(hours=9, minutes=30))
                      & (window.index < day + pd.Timedelta(hours=17))]
        if sess.empty:
            continue
        frames_1m[sym] = sess
        frames_5m[sym] = resample_ohlcv(sess, "5min")
        frames_15m[sym] = resample_ohlcv(window, "15min")
        r = build_opening_range(frames_5m[sym], session_date=day)
        if r is not None:
            ranges[sym] = r

    # ---- §7 AMD phase structure, per instrument ----------------------
    amd_states = {}
    if use_amd:
        for sym, r in ranges.items():
            lw = windows[sym]
            lv = mark_tapped(build_liquidity_map(lw, day), lw)
            amd_states[sym] = classify_amd(
                frames_5m[sym], r, levels=lv,
                min_excursion_ticks=amd_min_excursion_ticks,
                require_sweep=amd_require_sweep,
            )

    base = Setup(timestamp=day, session_date=day)

    if not ranges:
        base.vetoes.append("§3 no opening range (missing 09:30/09:35 bars)")
        return [base]
    base.passed.append(f"§3 opening range built for {list(ranges)}")

    # ---- §3.1 trigger --------------------------------------------------
    triggers = []
    for sym, r in ranges.items():
        frames = []
        if trigger_mode in ("double", "both"):
            frames.append(detect_double_test(frames_5m[sym], r))
        if trigger_mode in ("single", "both"):
            frames.append(detect_single_sweep(frames_5m[sym], r))
        for sig in frames:
            for _, s in sig.iterrows():
                triggers.append((sym, s))

    if not triggers:
        base.vetoes.append("§3.1 no double-test signal")
        return [base]

    # ---- §2 bias -------------------------------------------------------
    bias = {}
    for sym in ranges:
        sw15 = label_swings(detect_swings(frames_15m[sym]))
        bos15 = detect_bos(frames_15m[sym], sw15)
        obvious, obvious_dir = is_obvious_trend(sw15, bos15, n=3)
        bias[sym] = {"trend": classify_trend(sw15),
                     "obvious": obvious, "obvious_dir": obvious_dir}

    # ---- evaluate each trigger ----------------------------------------
    taken = 0
    for sym, sig in sorted(triggers, key=lambda x: x[1]["timestamp"]):
        s = Setup(timestamp=sig["timestamp"], session_date=day)
        s.passed = list(base.passed)
        s.trigger = f"{sig['signal']} on {sym} {sig['side_tested']}"
        direction = sig["direction"]
        s.direction = direction
        s.passed.append(f"§3.1 {s.trigger} -> {direction}")

        if direction not in directions:
            s.vetoes.append(f"direction filter excludes {direction}")
            setups.append(s)
            continue

        # A confirmed_breakout means the day is trending, not manipulating.
        # Under AMD there is nothing to fade, so stand aside.
        if not allow_confirmed_breakout and sig["signal"] == "confirmed_breakout":
            s.vetoes.append("§7 AMD: confirmed breakout = trending day, not AMD")
            setups.append(s)
            continue

        if use_amd:
            st = amd_states.get(sym)
            if st is None:
                s.vetoes.append("§7 AMD: no phase structure for this instrument")
                setups.append(s)
                continue
            ok, why = amd_gate(st, direction, sig["timestamp"])
            if not ok:
                s.vetoes.append(why)
                setups.append(s)
                continue
            s.passed.append(why)

        # §1 time window
        if not in_entry_window(sig["timestamp"], entry_start, entry_end):
            s.vetoes.append(f"§1 outside {entry_start}-{entry_end} "
                            f"({sig['timestamp'].strftime('%H:%M')})")
            setups.append(s)
            continue
        s.passed.append("§1 inside entry window")

        # §2 obvious counter-trend
        b = bias.get(sym, {})
        if b.get("obvious") and b["obvious_dir"] != ("up" if direction == "bull" else "down"):
            s.vetoes.append(f"§2 obvious 15m {b['obvious_dir']}trend against trade")
            setups.append(s)
            continue
        s.passed.append(f"§2 bias ok (15m {b.get('trend')})")

        # §8.2 session trade cap
        if taken >= max_trades:
            s.vetoes.append(f"§8.2 max {max_trades} trades/session reached")
            setups.append(s)
            continue

        # --- SMT across instruments -------------------------------------
        smts = []
        if len(symbols) >= 2 and all(k in frames_1m for k in symbols[:2]):
            a, bsym = symbols[0], symbols[1]
            sw_a = label_swings(detect_swings(frames_1m[a]))
            sw_b = label_swings(detect_swings(frames_1m[bsym]))
            smts = detect_smt(sw_a, sw_b, a, bsym, tolerance_bars=1, bar_minutes=1)
            smts = [x for x in smts if x.timestamp <= sig["timestamp"]]

        # §4 instrument choice
        try:
            chosen, how = choose_instrument(ranges, frames_5m, smts, direction)
        except ValueError:
            chosen, how = sym, "fallback"
        s.symbol = chosen
        s.passed.append(f"§4 instrument {chosen} ({how})")

        exec_df = frames_1m.get(chosen)
        if exec_df is None or exec_df.empty:
            s.vetoes.append("§4 no execution data for chosen instrument")
            setups.append(s)
            continue

        bars_upto = exec_df[exec_df.index <= sig["timestamp"]]
        if bars_upto.empty:
            s.vetoes.append("§4 no bars at signal time")
            setups.append(s)
            continue
        entry = float(bars_upto.iloc[-1]["close"])

        # §9 liquidity gate — uses the pre-sliced window, not full history
        liq_window = windows[chosen]
        liq_window = liq_window[liq_window.index <= sig["timestamp"]]
        levels = mark_tapped(
            build_liquidity_map(liq_window, day),
            liq_window,
            up_to=sig["timestamp"],
        )
        if not has_liquidity_target(levels, direction, entry):
            s.vetoes.append("§9 no untapped liquidity in trade direction")
            setups.append(s)
            continue
        s.passed.append("§9 untapped liquidity present")

        # §6 confluence gate
        sw1 = label_swings(detect_swings(bars_upto))
        bos1 = detect_bos(bars_upto, sw1)
        fvgs = update_fvg_state(
            detect_fvgs(frames_5m[chosen], min_size=fvg_min_size),
            frames_5m[chosen], up_to=sig["timestamp"],
        )
        ifvg = detect_ifvg_signals(fvgs)
        conf = confluence_check(direction, as_of=sig["timestamp"], price=entry,
                                fvgs=fvgs, bos=bos1, smts=smts, ifvg=ifvg,
                                bos_window_bars=5, bar_minutes=1)
        if not conf["pass"]:
            s.vetoes.append("§6 no continuation confluence")
            setups.append(s)
            continue
        s.confluences = conf["reasons"]
        if require_confluences and not any(c in conf["reasons"]
                                           for c in require_confluences):
            s.vetoes.append(f"§6 requires one of {require_confluences}, "
                            f"got {conf['reasons']}")
            setups.append(s)
            continue
        if forbid_confluences and set(conf["reasons"]).issubset(set(forbid_confluences)):
            s.vetoes.append(f"§6 only weak confluences {conf['reasons']}")
            setups.append(s)
            continue
        s.passed.append(f"§6 confluence: {', '.join(conf['reasons'])}")

        # §8 stop
        if stop_mode == "clamped":
            # Structural stop, but held inside the spec band. A 2nd-swing stop
            # on a 1m chart is often only a few ticks wide, which makes costs a
            # huge fraction of risk and gets you shaken out by noise.
            spec_s = CONTRACT_SPECS[chosen]
            structural, why = structural_stop(sw1, direction, sig["timestamp"])
            if structural is None:
                stop = None
            else:
                raw = abs(entry - structural) / spec_s["tick_size"]
                use = min(max(raw, stop_min_ticks), stop_max_ticks)
                dist = use * spec_s["tick_size"]
                stop = entry - dist if direction == "bull" else entry + dist
                why = f"clamped {raw:.0f}->{use:.0f} ticks"
        elif stop_mode == "fixed":
            # Spec says 200-300 ticks. A fixed distance is what the strategy
            # doc actually calls for; the structural stop was an interpretation.
            spec_s = CONTRACT_SPECS[chosen]
            dist = stop_ticks * spec_s["tick_size"]
            stop = entry - dist if direction == "bull" else entry + dist
            why = f"fixed {stop_ticks:.0f} ticks"
        else:
            stop, why = structural_stop(sw1, direction, sig["timestamp"])
        if stop is None:
            s.vetoes.append(f"§8 no stop available ({why})")
            setups.append(s)
            continue
        if (direction == "bull" and stop >= entry) or (direction == "bear" and stop <= entry):
            s.vetoes.append("§8 structural stop on wrong side of entry")
            setups.append(s)
            continue

        contracts, actual_ticks, risk_dollars = size_position(
            account, entry, stop, chosen, risk_pct
        )
        if stop_mode != "fixed" and chosen == "NQ" and actual_ticks > NQ_STOP_TICK_CAP:
            s.vetoes.append(f"§8 stop {actual_ticks:.0f} ticks exceeds "
                            f"{NQ_STOP_TICK_CAP} cap")
            setups.append(s)
            continue
        if contracts < 1:
            s.vetoes.append(f"§8 size < 1 contract at {risk_pct:.1%} risk")
            setups.append(s)
            continue

        # §8 target.
        # RR is computed from the price actually filled, not the signal close.
        # Slippage widens risk and shortens reward, so ignoring it here is why
        # a nominal 2:1 was executing at ~1.7:1.
        spec = CONTRACT_SPECS[chosen]
        slip = slippage_ticks * spec["tick_size"]
        fill = entry + slip if direction == "bull" else entry - slip
        target, label, realized_rr = build_target(fill, stop, direction, levels, rr)
        if target is None:
            s.vetoes.append("§8 no target available")
            setups.append(s)
            continue
        if realized_rr is not None and realized_rr < rr - 1e-9:
            s.vetoes.append(f"§9 first target only {realized_rr:.2f}R "
                            f"(need {rr:.1f}R)")
            setups.append(s)
            continue

        s.entry, s.stop, s.target = entry, stop, target
        s.stop_ticks, s.rr = actual_ticks, realized_rr
        s.contracts, s.risk_dollars = contracts, risk_dollars
        s.target_level = label
        s.qualified = True
        s.passed.append(f"§8 sized {contracts}c, stop {actual_ticks:.0f} ticks ({why})")
        s.detail = {"bias": b, "n_levels": len(levels), "n_fvgs": len(fvgs)}
        taken += 1
        setups.append(s)

        if verbose:
            print(s.explain())

    return setups


def scan_range(
    start,
    end,
    data_1m: dict,
    account: float = 100_000.0,
    **kwargs,
) -> tuple[pd.DataFrame, list[Setup]]:
    """
    Run evaluate_session across a date range.

    Returns (summary_df, all_setups). The summary counts qualified setups and
    tallies veto reasons — which is the fastest way to see whether a gate is
    rejecting everything.
    """
    any_df = next(iter(data_1m.values()))
    days = pd.unique(
        any_df[(any_df.index >= pd.Timestamp(start, tz=any_df.index.tz))
               & (any_df.index <= pd.Timestamp(end, tz=any_df.index.tz))]
        .index.normalize()
    )

    all_setups: list[Setup] = []
    rows = []
    for d in days:
        try:
            ss = evaluate_session(d, data_1m, account=account, **kwargs)
        except Exception as exc:  # keep the scan alive on odd days
            rows.append({"date": d, "qualified": 0, "evaluated": 0,
                         "veto": f"ERROR {type(exc).__name__}: {exc}"})
            continue
        all_setups.extend(ss)
        q = sum(1 for s in ss if s.qualified)
        veto = "; ".join(ss[0].vetoes) if (ss and not q) else ""
        rows.append({"date": d, "qualified": q, "evaluated": len(ss), "veto": veto})

    return pd.DataFrame(rows), all_setups


def veto_summary(setups: list[Setup]) -> pd.Series:
    """Which gate is doing the rejecting? Sanity check before trusting results."""

    c = Counter()
    for s in setups:
        if s.qualified:
            c["QUALIFIED"] += 1
        for v in s.vetoes:
            c[v.split("(")[0].strip()] += 1
    return pd.Series(c).sort_values(ascending=False)


def setups_to_frame(setups: list[Setup]) -> pd.DataFrame:
    """Qualified setups as a tidy table."""
    rows = [
        {
            "timestamp": s.timestamp,
            "date": s.session_date.date(),
            "symbol": s.symbol,
            "direction": s.direction,
            "entry": s.entry,
            "stop": s.stop,
            "target": s.target,
            "stop_ticks": s.stop_ticks,
            "rr": s.rr,
            "contracts": s.contracts,
            "risk": s.risk_dollars,
            "trigger": s.trigger,
            "confluences": ",".join(s.confluences),
            "target_level": s.target_level,
        }
        for s in setups
        if s.qualified
    ]
    return pd.DataFrame(rows)


# ======================================================================
# BACKTEST — fill simulation and metrics
# ======================================================================

DEFAULT_SLIPPAGE_TICKS = 1.0
DEFAULT_COMMISSION = 2.50      # per contract per side, round-turn = 2x


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    contracts: int
    exit_time: pd.Timestamp
    exit: float
    outcome: str              # 'target' | 'stop' | 'timeout'
    gross_points: float
    gross_dollars: float
    costs: float
    net_dollars: float
    r_multiple: float
    bars_held: int
    confluences: str
    trigger: str
    trailed: bool = False


def simulate_trade(
    setup,
    bars: pd.DataFrame,
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS,
    commission: float = DEFAULT_COMMISSION,
    trail_trigger: float | None = None,
    trail_lock: float = 0.0,
) -> Trade | None:
    """
    Walk bars after the signal until stop, target, or session end.

    TRAILING STOP
        trail_trigger : fraction of the way to target that arms the trail,
                        e.g. 0.75 for "three quarters of the way there".
        trail_lock    : where the stop moves to, as a fraction of the distance
                        from entry to target.
                            0.0  -> breakeven
                            0.5  -> lock half the move

        The trail arms on a bar whose EXTREME reaches the trigger, and the new
        stop applies from the NEXT bar onward. Arming and stopping on the same
        bar cannot both be honoured without tick data, and assuming the
        favourable order is how backtests inflate themselves.

    `bars` must be the execution-timeframe frame for the chosen instrument,
    covering the signal bar onward.
    """
    if not setup.qualified:
        return None

    spec = CONTRACT_SPECS[setup.symbol]
    tick = spec["tick_size"]
    pv = spec["point_value"]
    long = setup.direction == "bull"

    fwd = bars[bars.index > setup.timestamp]
    if fwd.empty:
        return None

    # entry slippage works against us
    slip = slippage_ticks * tick
    entry = setup.entry + slip if long else setup.entry - slip

    exit_price = exit_time = outcome = None
    bars_held = 0

    active_stop = setup.stop
    trailed = False
    span = abs(setup.target - entry)
    trigger_px = None
    if trail_trigger is not None and span > 0:
        trigger_px = (entry + span * trail_trigger) if long \
            else (entry - span * trail_trigger)
        trail_px = (entry + span * trail_lock) if long \
            else (entry - span * trail_lock)

    for ts, row in fwd.iterrows():
        bars_held += 1
        hit_stop = row["low"] <= active_stop if long else row["high"] >= active_stop
        hit_target = row["high"] >= setup.target if long else row["low"] <= setup.target

        if hit_stop and hit_target:
            # ambiguous bar -> assume the stop. Assuming the winner is how
            # backtests get inflated.
            exit_price = active_stop - slip if long else active_stop + slip
            exit_time, outcome = ts, ("scratch" if trailed else "stop")
            break
        if hit_stop:
            exit_price = active_stop - slip if long else active_stop + slip
            exit_time, outcome = ts, ("scratch" if trailed else "stop")
            break
        if hit_target:
            exit_price, exit_time, outcome = setup.target, ts, "target"
            break

        # arm the trail AFTER checking this bar's exits, so it applies from
        # the next bar onward
        if trigger_px is not None and not trailed:
            reached = row["high"] >= trigger_px if long else row["low"] <= trigger_px
            if reached:
                better = trail_px > active_stop if long else trail_px < active_stop
                if better:
                    active_stop = trail_px
                    trailed = True

    if exit_price is None:                     # flat at the close
        exit_price = float(fwd.iloc[-1]["close"])
        exit_time, outcome = fwd.index[-1], "timeout"

    gross_points = (exit_price - entry) if long else (entry - exit_price)
    gross_dollars = gross_points * pv * setup.contracts
    costs = commission * setup.contracts * 2
    net = gross_dollars - costs
    # R is always measured against the ORIGINAL risk, not the trailed stop —
    # otherwise a trailed trade would report an inflated multiple.
    risk = abs(entry - setup.stop) * pv * setup.contracts
    r_mult = net / risk if risk > 0 else 0.0

    return Trade(
        date=setup.session_date,
        symbol=setup.symbol,
        direction=setup.direction,
        entry_time=setup.timestamp,
        entry=entry,
        stop=setup.stop,
        target=setup.target,
        contracts=setup.contracts,
        exit_time=exit_time,
        exit=exit_price,
        outcome=outcome,
        trailed=trailed,
        gross_points=round(gross_points, 4),
        gross_dollars=round(gross_dollars, 2),
        costs=round(costs, 2),
        net_dollars=round(net, 2),
        r_multiple=round(r_mult, 4),
        bars_held=bars_held,
        confluences=",".join(setup.confluences),
        trigger=setup.trigger or "",
    )


def run_backtest(
    setups: list,
    data_1m: dict,
    account: float = 100_000.0,
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS,
    commission: float = DEFAULT_COMMISSION,
    apply_daily_limits: bool = True,
    max_daily_loss_pct: float = 0.02,
    daily_profit_lock_pct: float = 0.03,
    trail_trigger: float | None = None,
    trail_lock: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate every qualified setup and compound the account.

    §8.2 daily limits are enforced here rather than in the orchestrator, since
    they depend on realised P&L that only exists after simulation.
    """
    qualified = sorted([s for s in setups if s.qualified], key=lambda s: s.timestamp)
    trades: list[Trade] = []
    equity = account
    curve = []
    day_pnl: dict = {}

    for s in qualified:
        d = s.session_date
        realised = day_pnl.get(d, 0.0)
        if apply_daily_limits:
            if realised <= -abs(max_daily_loss_pct) * account:
                continue
            if realised >= daily_profit_lock_pct * account:
                continue

        sess = data_1m[s.symbol]
        bars = sess[(sess.index >= s.timestamp)
                    & (sess.index < d + pd.Timedelta(hours=17))]
        t = simulate_trade(s, bars, slippage_ticks, commission,
                           trail_trigger=trail_trigger, trail_lock=trail_lock)
        if t is None:
            continue

        trades.append(t)
        equity += t.net_dollars
        day_pnl[d] = realised + t.net_dollars
        curve.append({"timestamp": t.exit_time, "equity": equity,
                      "net": t.net_dollars, "r": t.r_multiple})

    df = pd.DataFrame([asdict(t) for t in trades])
    metrics = compute_metrics(df, account, pd.DataFrame(curve))
    return df, metrics


def compute_metrics(trades: pd.DataFrame, account: float, curve: pd.DataFrame) -> dict:
    """Standard performance statistics, plus the ones that actually matter."""
    if trades.empty:
        return {"trades": 0, "note": "no trades"}

    wins = trades[trades["net_dollars"] > 0]
    losses = trades[trades["net_dollars"] <= 0]
    net = trades["net_dollars"].sum()

    gross_win = wins["net_dollars"].sum()
    gross_loss = abs(losses["net_dollars"].sum())

    eq = curve["equity"] if not curve.empty else pd.Series([account])
    running_max = eq.cummax()
    dd = (eq - running_max)
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = max_dd / account if account else 0.0

    r = trades["r_multiple"]
    # per-trade Sharpe on R-multiples; not annualised, so it is comparable
    # across strategies but not against a buy-and-hold Sharpe
    sharpe_r = float(r.mean() / r.std()) if r.std() > 0 else 0.0

    streak = cur = 0
    for v in trades["net_dollars"]:
        if v <= 0:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0

    return {
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(len(wins) / len(trades), 4),
        "net_dollars": round(float(net), 2),
        "return_pct": round(float(net / account), 4),
        "avg_win": round(float(wins["net_dollars"].mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses["net_dollars"].mean()), 2) if len(losses) else 0.0,
        "expectancy_dollars": round(float(trades["net_dollars"].mean()), 2),
        "expectancy_r": round(float(r.mean()), 4),
        "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else float("inf"),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "sharpe_per_trade": round(sharpe_r, 3),
        "max_consecutive_losses": int(streak),
        "outcomes": trades["outcome"].value_counts().to_dict(),
        "by_direction": trades.groupby("direction")["net_dollars"].sum().round(2).to_dict(),
        "by_symbol": trades.groupby("symbol")["net_dollars"].sum().round(2).to_dict(),
        "total_costs": round(float(trades["costs"].sum()), 2),
    }


def print_metrics(m: dict) -> None:
    if m.get("trades", 0) == 0:
        print("no trades")
        return
    print(f"trades              : {m['trades']}")
    print(f"win rate            : {m['win_rate']:.1%}  "
          f"({m['wins']}W / {m['losses']}L)")
    print(f"net P&L             : ${m['net_dollars']:,.0f}  "
          f"({m['return_pct']:.1%} on starting account)")
    print(f"expectancy          : ${m['expectancy_dollars']:,.0f}/trade  "
          f"({m['expectancy_r']:+.3f}R)")
    print(f"profit factor       : {m['profit_factor']}")
    print(f"avg win / avg loss  : ${m['avg_win']:,.0f} / ${m['avg_loss']:,.0f}")
    print(f"max drawdown        : ${m['max_drawdown']:,.0f} "
          f"({m['max_drawdown_pct']:.1%})")
    print(f"sharpe (per trade)  : {m['sharpe_per_trade']}")
    print(f"max losing streak   : {m['max_consecutive_losses']}")
    print(f"total costs         : ${m['total_costs']:,.0f}")
    print(f"outcomes            : {m['outcomes']}")
    print(f"by direction        : {m['by_direction']}")
    print(f"by symbol           : {m['by_symbol']}")


def equity_curve(trades: pd.DataFrame, account: float = 100_000.0) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    eq = account + trades["net_dollars"].cumsum()
    eq.index = trades["exit_time"]
    return eq


# ======================================================================
# DATASET — ML feature engineering
# ======================================================================

FEATURE_NAMES = [
    # compression / clumping  (§7 accumulation-distribution)
    "range_ratio",
    "body_ratio",
    "overlap_ratio",
    "close_dispersion",
    "high_low_std",
    # direction / cleanliness  (§4 "cleaner price action")
    "persistence",
    "net_move_atr",
    "wick_ratio",
    "up_bar_frac",
    "return_autocorr",
    # volatility
    "atr_norm",
    "atr_change",
    "true_range_norm",
    # volume
    "vol_ratio",
    "vol_trend",
    # session context
    "minutes_from_open",
    "dist_from_or_mid",
]


@dataclass
class DatasetSpec:
    """Every choice that defines the dataset, in one inspectable object."""

    lookback: int = 12          # bars of history per sample (12 x 5m = 1 hour)
    horizon: int = 6            # bars ahead to label   (6 x 5m = 30 minutes)
    atr_window: int = 14
    expansion_k: float = 1.62    # |excursion| > k * ATR -> class 1.
                                 # 1.62 is the empirical median excursion over
                                 # 300k samples, so the classes come out ~50/50.
                                 # At k=1.0 the split is 86/14 and a model that
                                 # always predicts "expansion" scores 86% while
                                 # learning nothing.
    timeframe: str = "5min"
    session_only: bool = True   # restrict to the NY session
    session_start: str = "09:30"
    session_end: str = "16:00"

    def describe(self) -> str:
        return (
            f"lookback={self.lookback} bars ({self.lookback * 5}m), "
            f"horizon={self.horizon} bars ({self.horizon * 5}m), "
            f"expansion threshold={self.expansion_k}xATR, tf={self.timeframe}"
        )


# ---------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _window_features(w: pd.DataFrame, atr: float, or_mid: float | None) -> dict:
    """Features for ONE lookback window. `w` ends at the sample bar."""
    high, low = w["high"].to_numpy(), w["low"].to_numpy()
    op, close = w["open"].to_numpy(), w["close"].to_numpy()
    rng = high - low
    body = np.abs(close - op)
    eps = 1e-9
    span = high.max() - low.min()

    # --- compression / clumping -------------------------------------
    range_ratio = float(rng.mean() / (span + eps))
    body_ratio = float(body.sum() / (rng.sum() + eps))

    overlaps = [
        max(0.0, min(high[i], high[i - 1]) - max(low[i], low[i - 1]))
        / (max(high[i], high[i - 1]) - min(low[i], low[i - 1]) + eps)
        for i in range(1, len(w))
    ]
    overlap_ratio = float(np.mean(overlaps)) if overlaps else 0.0

    close_dispersion = float(close.std() / (atr + eps))
    high_low_std = float(rng.std() / (rng.mean() + eps))

    # --- direction / cleanliness ------------------------------------
    diffs = np.diff(close)
    persistence = float(np.abs(diffs.sum()) / (np.abs(diffs).sum() + eps))
    net_move_atr = float((close[-1] - close[0]) / (atr + eps))

    upper = high - np.maximum(op, close)
    lower = np.minimum(op, close) - low
    wick_ratio = float((upper.sum() + lower.sum()) / (rng.sum() + eps))
    up_bar_frac = float((close > op).mean())

    if len(diffs) > 2 and diffs.std() > eps:
        ac = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
        return_autocorr = float(0.0 if np.isnan(ac) else ac)
    else:
        return_autocorr = 0.0

    # --- volatility --------------------------------------------------
    atr_norm = float(atr / (close[-1] + eps))
    first_half = rng[: len(rng) // 2].mean()
    second_half = rng[len(rng) // 2 :].mean()
    atr_change = float(second_half / (first_half + eps))
    true_range_norm = float(rng[-1] / (atr + eps))

    # --- volume ------------------------------------------------------
    if "volume" in w.columns:
        v = w["volume"].to_numpy(float)
        vol_ratio = float(v[-1] / (v.mean() + eps))
        vol_trend = float(v[len(v) // 2 :].mean() / (v[: len(v) // 2].mean() + eps))
    else:
        vol_ratio = vol_trend = 1.0

    # --- session context ---------------------------------------------
    ts = w.index[-1]
    minutes_from_open = float(ts.hour * 60 + ts.minute - (9 * 60 + 30))
    dist_from_or_mid = (
        float((close[-1] - or_mid) / (atr + eps)) if or_mid is not None else 0.0
    )

    return {
        "range_ratio": range_ratio,
        "body_ratio": body_ratio,
        "overlap_ratio": overlap_ratio,
        "close_dispersion": close_dispersion,
        "high_low_std": high_low_std,
        "persistence": persistence,
        "net_move_atr": net_move_atr,
        "wick_ratio": wick_ratio,
        "up_bar_frac": up_bar_frac,
        "return_autocorr": return_autocorr,
        "atr_norm": atr_norm,
        "atr_change": atr_change,
        "true_range_norm": true_range_norm,
        "vol_ratio": vol_ratio,
        "vol_trend": vol_trend,
        "minutes_from_open": minutes_from_open,
        "dist_from_or_mid": dist_from_or_mid,
    }


# ---------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------

def build_dataset(
    df_1m: pd.DataFrame,
    symbol: str,
    spec: DatasetSpec | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Turn 1-minute bars into a supervised table.

    Returns one row per sample: 17 feature columns, both labels, and metadata
    (timestamp, session date, symbol) kept for a time-ordered train/test split.
    """
    spec = spec or DatasetSpec()
    bars = resample_ohlcv(df_1m, spec.timeframe)
    if spec.session_only:
        bars = bars.between_time(spec.session_start, spec.session_end)

    rows = []
    dropped_short = 0

    for day, sess in bars.groupby(bars.index.normalize()):
        need = spec.lookback + spec.horizon + spec.atr_window
        if len(sess) < need:
            dropped_short += 1
            continue

        tr = _true_range(sess)
        atr = tr.rolling(spec.atr_window).mean()

        # opening-range midpoint: the 09:30 and 09:35 bars
        or_bars = sess.iloc[:2]
        or_mid = float((or_bars["high"].max() + or_bars["low"].min()) / 2)

        start = max(spec.lookback, spec.atr_window)
        stop = len(sess) - spec.horizon          # labels need H bars ahead

        for i in range(start, stop):
            a = float(atr.iloc[i])
            if not np.isfinite(a) or a <= 0:
                continue

            window = sess.iloc[i - spec.lookback : i + 1]     # ends AT bar i
            feats = _window_features(window, a, or_mid)

            # label: bars i+1 .. i+H  (strictly after the feature window)
            fwd = sess.iloc[i + 1 : i + 1 + spec.horizon]
            now = float(sess.iloc[i]["close"])
            fwd_close = float(fwd.iloc[-1]["close"])
            fwd_high = float(fwd["high"].max())
            fwd_low = float(fwd["low"].min())

            y_return = (fwd_close - now) / a
            excursion = max(fwd_high - now, now - fwd_low) / a
            y_expansion = int(excursion > spec.expansion_k)

            feats.update(
                {
                    "timestamp": sess.index[i],
                    "session_date": day,
                    "symbol": symbol,
                    "y_return": y_return,
                    "y_expansion": y_expansion,
                    "y_excursion": excursion,
                    "atr": a,
                    "close": now,
                }
            )
            rows.append(feats)

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("no samples built — check the input frame")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"  {symbol}: {len(data):,} samples  "
              f"{data['timestamp'].min().date()} -> {data['timestamp'].max().date()}")
        if dropped_short:
            print(f"    ({dropped_short} short sessions skipped)")
    return data


def build_combined(
    data_1m: dict,
    spec: DatasetSpec | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build and stack datasets for several instruments."""
    spec = spec or DatasetSpec()
    if verbose:
        print(f"Building dataset — {spec.describe()}")
    parts = [build_dataset(df, sym, spec, verbose) for sym, df in data_1m.items()]
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# split, inspection, IO
# ---------------------------------------------------------------------

def split_xy(
    data: pd.DataFrame,
    target: str = "y_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the label vector y."""
    if target not in ("y_return", "y_expansion", "y_excursion"):
        raise ValueError(f"unknown target: {target}")
    return data[FEATURE_NAMES].copy(), data[target].copy()


def time_split(
    data: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split (STEP 3).

    A random split would leak: neighbouring bars overlap in their lookback
    windows, so a shuffled test set contains near-duplicates of training rows and
    every model looks far better than it is. Splitting by time is the only
    honest option for market data.
    """
    cut_ts = data["timestamp"].quantile(1 - test_frac)
    train = data[data["timestamp"] <= cut_ts].copy()
    test = data[data["timestamp"] > cut_ts].copy()
    return train, test


def describe_dataset(data: pd.DataFrame) -> None:
    """Print the summary you need before trusting anything downstream."""
    print("=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"samples          : {len(data):,}")
    print(f"features (X)     : {len(FEATURE_NAMES)}")
    print(f"date range       : {data['timestamp'].min().date()} -> "
          f"{data['timestamp'].max().date()}")
    print(f"sessions         : {data['session_date'].nunique():,}")
    print(f"symbols          : {sorted(data['symbol'].unique())}")

    print("\nTARGETS (y)")
    print(f"  y_return       : mean {data['y_return'].mean():+.4f}  "
          f"std {data['y_return'].std():.4f}  "
          f"[{data['y_return'].min():.2f}, {data['y_return'].max():.2f}]")
    bal = data["y_expansion"].mean()
    print(f"  y_expansion    : {bal:.1%} expansion / {1 - bal:.1%} compression")
    print(f"  y_excursion    : mean {data['y_excursion'].mean():.3f} ATR")

    print("\nFEATURE STATS")
    stats = data[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
    print(stats.round(3).to_string())

    nan = data[FEATURE_NAMES].isna().sum().sum()
    inf = np.isinf(data[FEATURE_NAMES].to_numpy()).sum()
    print(f"\nNaNs: {nan}   Infs: {inf}")
    if nan or inf:
        print("  ^ must be cleaned before training")


def save_dataset(data: pd.DataFrame, path: str) -> None:
    data.to_parquet(path)
    print(f"saved {len(data):,} rows -> {path}")


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ======================================================================
# MANUAL — score a chart read you type in yourself
# ======================================================================



def check(
    symbol: str,
    direction: str,
    entry: float,
    stop: float | None = None,
    target: float | None = None,
    stop_ticks: float | None = None,
    date=None,
    time_of_day: str | None = None,
    account: float = 100_000.0,
    risk_pct: float = 0.01,
    rr: float = 2.0,
    data_1m: dict | None = None,
    or_high: float | None = None,
    or_low: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Score a chart read. Everything except `symbol`, `direction` and `entry` is
    optional — whatever you supply gets checked, whatever you omit is skipped
    or pulled from data if it is loaded.
    """

    if direction not in ("bull", "bear"):
        raise ValueError("direction must be 'bull' or 'bear'")
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"symbol must be one of {list(CONTRACT_SPECS)}")

    spec = CONTRACT_SPECS[symbol]
    tick = spec["tick_size"]
    long = direction == "bull"
    passed, vetoes, notes = [], [], {}

    # ---- stop -------------------------------------------------------
    if stop is None:
        if stop_ticks is None:
            raise ValueError("supply either stop= or stop_ticks=")
        dist = stop_ticks * tick
        stop = entry - dist if long else entry + dist
        passed.append(f"stop set {stop_ticks:.0f} ticks away -> {stop:.2f}")

    if (long and stop >= entry) or (not long and stop <= entry):
        vetoes.append("stop is on the wrong side of entry")
        return _finish(passed, vetoes, notes, verbose)

    risk_pts = abs(entry - stop)
    ticks = risk_pts / tick
    notes["stop_ticks"] = round(ticks, 1)

    # ---- target -----------------------------------------------------
    if target is None:
        target = entry + risk_pts * rr if long else entry - risk_pts * rr
        passed.append(f"target set at {rr:.1f}R -> {target:.2f}")
    realized_rr = abs(target - entry) / risk_pts
    notes["rr"] = round(realized_rr, 2)
    if realized_rr < rr - 1e-9:
        vetoes.append(f"§8 target is only {realized_rr:.2f}R, spec wants {rr:.1f}R")
    else:
        passed.append(f"§8 RR {realized_rr:.2f}:1")

    # ---- size -------------------------------------------------------
    contracts, _, risk_dollars = size_position(account, entry, stop, symbol, risk_pct)
    notes["contracts"] = contracts
    notes["risk_dollars"] = round(risk_dollars, 2)
    if contracts < 1:
        vetoes.append(f"§8 size < 1 contract at {risk_pct:.1%} of "
                      f"{account:,.0f} with a {ticks:.0f} tick stop")
    else:
        passed.append(f"§8 {contracts} contracts, ${risk_dollars:,.0f} at risk")

    # ---- time -------------------------------------------------------
    if time_of_day:
        ok = "09:50" <= time_of_day <= "12:00"
        (passed if ok else vetoes).append(
            f"§1 {time_of_day} {'inside' if ok else 'outside'} 09:50-12:00")

    # ---- opening range ----------------------------------------------
    if or_high is None and data_1m is not None and date is not None:
        try:
            day = pd.Timestamp(date).tz_localize(TZ) if pd.Timestamp(date).tz is None \
                else pd.Timestamp(date)
            df = data_1m[symbol]
            sess = df[(df.index >= day.normalize() + pd.Timedelta(hours=9, minutes=30))
                      & (df.index < day.normalize() + pd.Timedelta(hours=17))]
            r = build_opening_range(resample_ohlcv(sess, "5min"),
                                    session_date=day.normalize())
            if r:
                or_high, or_low = r.high, r.low
                passed.append(f"§3 opening range {r.low:.2f} - {r.high:.2f}")
        except Exception as exc:
            notes["or_error"] = str(exc)

    if or_high is not None and or_low is not None:
        notes["or_high"], notes["or_low"] = or_high, or_low
        if or_low <= entry <= or_high:
            pos = "inside"
        elif entry > or_high:
            pos = "above"
        else:
            pos = "below"
        notes["entry_vs_or"] = pos
        # AMD: distribution trades start from inside the range after the sweep
        if pos == "above" and long:
            vetoes.append("§7 long entered ABOVE the opening range — that is the "
                          "manipulation side, not distribution")
        elif pos == "below" and not long:
            vetoes.append("§7 short entered BELOW the opening range — that is the "
                          "manipulation side, not distribution")
        else:
            passed.append(f"§7 entry {pos} the range, consistent with "
                          f"{direction} distribution")

    # ---- liquidity --------------------------------------------------
    if data_1m is not None and date is not None:
        try:
            day = pd.Timestamp(date)
            if day.tz is None:
                day = day.tz_localize(TZ)
            df = data_1m[symbol]
            win = df[(df.index >= day.normalize() - pd.Timedelta(days=5))
                     & (df.index <= day.normalize() + pd.Timedelta(hours=17))]
            lv = mark_tapped(build_liquidity_map(win, day.normalize()), win)
            tg = untapped_targets(lv, direction, entry)
            if tg:
                nearest = tg[0]
                notes["nearest_liquidity"] = f"{nearest.name} @ {nearest.price:.2f}"
                passed.append(f"§9 untapped draw: {nearest.name} @ {nearest.price:.2f}")
                beyond = (nearest.price >= target) if long else (nearest.price <= target)
                if not beyond:
                    vetoes.append(
                        f"§9 target {target:.2f} sits past the nearest draw "
                        f"({nearest.name} @ {nearest.price:.2f}) — that level is "
                        f"where price is likely to react")
            else:
                vetoes.append("§9 no untapped liquidity in the trade direction")
        except Exception as exc:
            notes["liq_error"] = str(exc)

    notes.update({"symbol": symbol, "direction": direction, "entry": entry,
                  "stop": round(stop, 2), "target": round(target, 2)})
    return _finish(passed, vetoes, notes, verbose)


def _finish(passed, vetoes, notes, verbose):
    out = {"ok": len(vetoes) == 0, "passed": passed, "vetoes": vetoes, **notes}
    if verbose:
        print("=" * 66)
        d = notes.get("direction", "?").upper()
        print(f"  {d} {notes.get('symbol','?')} @ {notes.get('entry','?')}")
        print("=" * 66)
        for p in passed:
            print(f"  PASS   {p}")
        for v in vetoes:
            print(f"  VETO   {v}")
        print("-" * 66)
        if out["ok"]:
            print(f"  CLEARS EVERY RULE CHECKED")
        else:
            print(f"  {len(vetoes)} PROBLEM(S)")
        print("\n  The engine cannot see accumulation/distribution quality or")
        print("  judge which index is cleaner. Those remain your read.")
    return out


# ======================================================================
# CHART READER — vision-model screenshot -> levels -> rules
# ======================================================================

import base64
import re

EXTRACTION_PROMPT = """You are transcribing a futures trading chart. Report only
what is visibly printed or drawn. Do not infer, estimate, or judge the setup.

Return a single JSON object, no prose, no code fences:

{
  "symbol": "NQ" | "ES" | "MNQ" | "MES" | null,
  "timeframe": "1m" | "5m" | "15m" | null,
  "date": "YYYY-MM-DD" or null,
  "current_price": number or null,
  "entry": number or null,
  "stop": number or null,
  "target": number or null,
  "direction": "bull" | "bear" | null,
  "drawn_levels": [ {"price": number, "label": "string or null"} ],
  "session_time": "HH:MM" or null,
  "notes": "anything relevant you can read but could not place in a field",
  "confidence": {"entry": "high|medium|low", "stop": "...", "target": "..."}
}

Rules:
- Read prices from the axis and from any position tool or horizontal line.
- If a long/short position tool is present, its entry, stop and target are
  authoritative; set direction from which way it points.
- If a value is not visible, use null. Never guess a number.
- Mark confidence "low" for anything you had to read from a blurry or partially
  obscured area.
"""


# ---------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------

def _encode(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), media


def _call_anthropic(b64, media, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text", "text": EXTRACTION_PROMPT},
        ]}],
    )
    return msg.content[0].text


def _call_openai(b64, media, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model=model or "gpt-4o",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": EXTRACTION_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:{media};base64,{b64}"}},
        ]}],
    )
    return r.choices[0].message.content


def _call_google(b64, media, api_key, model):
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
    m = genai.GenerativeModel(model or "gemini-1.5-pro")
    r = m.generate_content([
        {"mime_type": media, "data": base64.b64decode(b64)},
        EXTRACTION_PROMPT,
    ])
    return r.text


_PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai,
              "google": _call_google}


def _parse(raw: str) -> dict:
    """Pull the JSON object out of a model response."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError(f"no JSON in response:\n{raw[:400]}")
        return json.loads(m.group(0))


# ---------------------------------------------------------------------
# public
# ---------------------------------------------------------------------

def read_chart(path: str, provider: str = "anthropic", api_key: str | None = None,
               model: str | None = None, verbose: bool = True) -> dict:
    """Transcribe a chart screenshot into structured levels."""
    if provider not in _PROVIDERS:
        raise ValueError(f"provider must be one of {list(_PROVIDERS)}")
    b64, media = _encode(path)
    out = _parse(_PROVIDERS[provider](b64, media, api_key, model))

    if verbose:
        print("=" * 66)
        print(f"  READ FROM {os.path.basename(path)}")
        print("=" * 66)
        for k in ("symbol", "timeframe", "date", "session_time", "direction",
                  "current_price", "entry", "stop", "target"):
            v = out.get(k)
            if v is not None:
                conf = (out.get("confidence") or {}).get(k)
                print(f"  {k:<15} {v}" + (f"   [{conf} confidence]" if conf else ""))
        lv = out.get("drawn_levels") or []
        if lv:
            print(f"  drawn levels    " +
                  ", ".join(f"{l.get('price')}" +
                            (f" ({l['label']})" if l.get("label") else "")
                            for l in lv))
        if out.get("notes"):
            print(f"  notes           {out['notes']}")
        low = [k for k, v in (out.get("confidence") or {}).items() if v == "low"]
        if low:
            print(f"\n  Low confidence on: {', '.join(low)} — verify before acting.")
    return out


def score_chart(path: str, data_1m: dict | None = None, account: float = 150_000.0,
                risk_pct: float = 0.015, rr: float = 2.0,
                provider: str = "anthropic", api_key: str | None = None,
                model: str | None = None, **overrides) -> dict:
    """
    Read a screenshot and run every engine rule against what it found.

    Anything the model could not read can be supplied directly:

        score_chart("chart.png", entry=21050, direction="bull", data_1m=data)
    """

    seen = read_chart(path, provider, api_key, model)
    seen.update({k: v for k, v in overrides.items() if v is not None})

    missing = [k for k in ("symbol", "direction", "entry") if not seen.get(k)]
    if missing:
        raise ValueError(
            f"could not read {missing} from the chart — pass them directly, e.g. "
            f"score_chart({path!r}, " +
            ", ".join(f"{k}=..." for k in missing) + ")")

    sym = seen["symbol"].upper()
    if sym in ("MNQ", "MES"):          # micros share the parent's tick geometry
        sym = sym[1:]

    print()
    return check(
        symbol=sym,
        direction=seen["direction"],
        entry=float(seen["entry"]),
        stop=float(seen["stop"]) if seen.get("stop") else None,
        target=float(seen["target"]) if seen.get("target") else None,
        stop_ticks=None if seen.get("stop") else 200.0,
        date=seen.get("date"),
        time_of_day=seen.get("session_time"),
        account=account, risk_pct=risk_pct, rr=rr,
        data_1m=data_1m,
    )


def upload_and_score(data_1m: dict | None = None, **kwargs):
    """Colab convenience: file picker, then read and score."""
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("upload_and_score() is Colab-only; "
                           "use score_chart(path) elsewhere")
    up = files.upload()
    if not up:
        print("no file chosen")
        return None
    return score_chart(next(iter(up)), data_1m=data_1m, **kwargs)

# ======================================================================
# ML PIPELINE
# ======================================================================


import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FEATURE_NAMES = [
    # compression / clumping   -> strategy §7 accumulation-distribution
    "range_ratio", "body_ratio", "overlap_ratio", "close_dispersion",
    "high_low_std",
    # direction / cleanliness  -> strategy §4 "cleaner price action"
    "persistence", "net_move_atr", "wick_ratio", "up_bar_frac",
    "return_autocorr",
    # volatility
    "atr_norm", "atr_change", "true_range_norm",
    # volume
    "vol_ratio", "vol_trend",
    # session context
    "minutes_from_open", "dist_from_or_mid",
]

TARGETS = {
    "y_return": "continuous — forward return over next 6 bars, ATR-normalised",
    "y_expansion": "binary — 1 if max excursion > 1.62 x ATR",
    "y_excursion": "continuous — max excursion in ATR units",
}

# =====================================================================
# CONFIG — edit these when pasting into a notebook cell
# =====================================================================
ML_CONFIG = {
    "data": "ml_dataset.parquet",   # prepared dataset; falls back to synthetic
    "nq": "nq_1m.parquet",          # raw bars, used only to build the dataset
    "es": "es_1m.parquet",
    "task": "classification",       # or "regression"
    "target": None,                 # None -> y_expansion / y_return by task
    "sample": None,                 # e.g. 60000 for a faster run
    "test_frac": 0.2,
    "rf_subsample": 50000,          # caps rows for the forest depth sweep
    "tune": True,                   # False skips requirement 8
}


def rule(char="=", n=72):
    print(char * n)


def step(number, title):
    print()
    rule()
    print(f"REQUIREMENT {number} — {title}")
    rule()


# =====================================================================
# 1 + 2 — dataset with defined X and y
# =====================================================================

def make_synthetic(n=40_000, seed=RANDOM_STATE) -> pd.DataFrame:
    """
    Fallback dataset so the pipeline runs with no market data present.

    Signal is deliberately weak and partly non-linear, so tree models can beat
    the linear one and the depth sweep has something real to find.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {name: rng.normal(0, 1, n) for name in FEATURE_NAMES}
    )
    X["minutes_from_open"] = rng.uniform(0, 360, n)

    signal = (
        0.30 * X["atr_norm"]
        + 0.25 * np.sin(X["minutes_from_open"] / 60.0)
        + 0.20 * (X["vol_ratio"] > 0.5).astype(float)
        + 0.15 * X["persistence"] * X["net_move_atr"]
    )
    noise = rng.normal(0, 1.0, n)
    y_return = signal + noise
    excursion = np.abs(y_return) + rng.gamma(2, 0.4, n)

    X["timestamp"] = pd.date_range("2016-06-21", periods=n, freq="5min",
                                   tz="America/New_York")
    X["session_date"] = X["timestamp"].dt.normalize()
    X["symbol"] = np.where(rng.random(n) > 0.5, "NQ", "ES")
    X["y_return"] = y_return
    X["y_excursion"] = excursion
    X["y_expansion"] = (excursion > np.median(excursion)).astype(int)
    return X


def load_or_build(data_path, nq_path=None, es_path=None) -> pd.DataFrame:
    """Load a prepared dataset, else build one, else fall back to synthetic."""
    if data_path and os.path.exists(data_path):
        print(f"loading prepared dataset: {data_path}")
        return pd.read_parquet(data_path)

    if nq_path and es_path and os.path.exists(nq_path) and os.path.exists(es_path):
        print("building dataset from raw 1-minute bars ...")
        from dataset import DatasetSpec, build_combined  # local module
        data = build_combined(
            {"NQ": pd.read_parquet(nq_path), "ES": pd.read_parquet(es_path)},
            DatasetSpec(),
        )
        if data_path:
            data.to_parquet(data_path)
            print(f"cached -> {data_path}")
        return data

    print("no market data found — using synthetic fallback so the pipeline runs")
    return make_synthetic()


def describe_dataset(data: pd.DataFrame, target: str) -> None:
    print(f"samples        : {len(data):,}")
    print(f"features (X)   : {len(FEATURE_NAMES)}")
    if "timestamp" in data:
        print(f"date range     : {data['timestamp'].min().date()} -> "
              f"{data['timestamp'].max().date()}")
    if "session_date" in data:
        print(f"sessions       : {data['session_date'].nunique():,}")
    if "symbol" in data:
        print(f"symbols        : {sorted(data['symbol'].unique())}")

    print("\nINPUT  X — 17 features")
    groups = [
        ("compression/clumping (§7)", FEATURE_NAMES[0:5]),
        ("direction/cleanliness (§4)", FEATURE_NAMES[5:10]),
        ("volatility", FEATURE_NAMES[10:13]),
        ("volume", FEATURE_NAMES[13:15]),
        ("session context", FEATURE_NAMES[15:17]),
    ]
    for label, names in groups:
        print(f"  {label:<28} {', '.join(names)}")

    print("\nOUTPUT y — available targets")
    for name, desc in TARGETS.items():
        mark = " <-- selected" if name == target else ""
        if name in data:
            print(f"  {name:<14} {desc}{mark}")

    y = data[target]
    if y.nunique() <= 2:
        bal = y.mean()
        print(f"\n  class balance : {bal:.1%} / {1 - bal:.1%}")
        if not 0.35 < bal < 0.65:
            print("  WARNING: imbalanced — accuracy will be misleading")
    else:
        print(f"\n  mean {y.mean():+.4f}  std {y.std():.4f}  "
              f"range [{y.min():.2f}, {y.max():.2f}]")

    nan = data[FEATURE_NAMES].isna().sum().sum()
    inf = int(np.isinf(data[FEATURE_NAMES].to_numpy()).sum())
    print(f"\n  NaNs {nan}   Infs {inf}")


def split_xy(data: pd.DataFrame, target: str):
    if target not in data.columns:
        raise ValueError(f"target '{target}' not in dataset")
    return data[FEATURE_NAMES].copy(), data[target].copy()


# =====================================================================
# 3 — train / test split
# =====================================================================

def time_split(data: pd.DataFrame, test_frac=0.2):
    """Chronological split. A random split leaks via overlapping windows."""
    if "timestamp" in data.columns:
        cut = data["timestamp"].quantile(1 - test_frac)
        return (data[data["timestamp"] <= cut].copy(),
                data[data["timestamp"] > cut].copy())
    cut = int(len(data) * (1 - test_frac))
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


# =====================================================================
# 4 — define the five models
# =====================================================================

def define_models(task="classification", max_depth=8) -> dict:
    """
    Random forest, decision tree, linear, neural net, KNN.

    Only the tree models take max_depth; linear and KNN have no depth parameter,
    which is exactly why requirement 8 sweeps depth for the tree learners.
    """
    if task == "regression":
        return {
            "linear": Pipeline([("scale", StandardScaler()),
                                ("model", Ridge(alpha=1.0))]),
            "knn": Pipeline([("scale", StandardScaler()),
                             ("model", KNeighborsRegressor(
                                 n_neighbors=50, weights="distance", n_jobs=-1))]),
            "decision_tree": DecisionTreeRegressor(
                max_depth=max_depth, min_samples_leaf=50,
                random_state=RANDOM_STATE),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=max_depth, min_samples_leaf=50,
                random_state=RANDOM_STATE, n_jobs=-1),
            "neural_net": Pipeline([("scale", StandardScaler()),
                                    ("model", MLPRegressor(
                                        hidden_layer_sizes=(64, 32), max_iter=300,
                                        early_stopping=True, n_iter_no_change=10,
                                        random_state=RANDOM_STATE))]),
        }
    return {
        "linear": Pipeline([("scale", StandardScaler()),
                            ("model", LogisticRegression(
                                max_iter=1000, random_state=RANDOM_STATE))]),
        "knn": Pipeline([("scale", StandardScaler()),
                         ("model", KNeighborsClassifier(
                             n_neighbors=50, weights="distance", n_jobs=-1))]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=50, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=max_depth, min_samples_leaf=50,
            random_state=RANDOM_STATE, n_jobs=-1),
        "neural_net": Pipeline([("scale", StandardScaler()),
                                ("model", MLPClassifier(
                                    hidden_layer_sizes=(64, 32), max_iter=300,
                                    early_stopping=True, n_iter_no_change=10,
                                    random_state=RANDOM_STATE))]),
    }


# =====================================================================
# 5 + 6 — train and evaluate
# =====================================================================

@dataclass
class Result:
    name: str
    train_time: float
    metrics: dict
    predictions: np.ndarray
    model: object = None
    proba: np.ndarray | None = None


def evaluate(y_true, y_pred, task="classification") -> dict:
    if task == "regression":
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": mean_absolute_error(y_true, y_pred),
            # directional hit rate — for a trading model this matters more than
            # R², because you act on the sign, not the magnitude
            "dir_acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        }
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)}


def train_and_evaluate(models, X_tr, y_tr, X_te, y_te, task="classification"):
    results = {}
    dummy = (DummyRegressor() if task == "regression"
             else DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_tr, y_tr)
    results["BASELINE"] = Result(
        "BASELINE", 0.0, evaluate(y_te, dummy.predict(X_te), task),
        dummy.predict(X_te))

    hdr = f"{'model':<16}{'fit(s)':>8}   metrics"
    print(hdr)
    print("-" * 72)
    m = results["BASELINE"].metrics
    print(f"{'BASELINE':<16}{0.0:>8.1f}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        pred = model.predict(X_te)
        proba = (model.predict_proba(X_te)[:, 1]
                 if task == "classification" and hasattr(model, "predict_proba")
                 else None)
        met = evaluate(y_te, pred, task)
        results[name] = Result(name, elapsed, met, pred, model, proba)
        print(f"{name:<16}{elapsed:>8.1f}   " +
              "  ".join(f"{k}={v:.4f}" for k, v in met.items()))

    key = "accuracy" if task == "classification" else "r2"
    best = max((n for n in results if n != "BASELINE"),
               key=lambda n: results[n].metrics[key])
    print(f"\nbest single model: {best} "
          f"({key}={results[best].metrics[key]:.4f}) vs "
          f"baseline {results['BASELINE'].metrics[key]:.4f}")
    return results


# =====================================================================
# 7 — combine by averaging
# =====================================================================

def ensemble(results, y_te, task="classification", weight_by=None):
    """
    Plain average and performance-weighted average.

    Classification averages PROBABILITIES, not hard 0/1 labels: averaging labels
    produces values that are neither a class nor a calibrated probability.
    Soft voting — average predict_proba, threshold at 0.5 — is the correct form.

    Caveat kept in the open: weights come from test-set scores, so the weighted
    ensemble has seen the test set. Fair for comparing weighting schemes,
    optimistic as a live estimate; proper practice fits weights on a separate
    validation split.
    """
    weight_by = weight_by or ("accuracy" if task == "classification" else "r2")
    names = [n for n in results if n != "BASELINE"]

    if task == "classification":
        names = [n for n in names if results[n].proba is not None]
        preds = np.column_stack([results[n].proba for n in names])
    else:
        preds = np.column_stack([results[n].predictions for n in names])

    fin = (lambda x: (x >= 0.5).astype(int)) if task == "classification" else (lambda x: x)

    simple = preds.mean(axis=1)
    simple_m = evaluate(y_te, fin(simple), task)

    scores = np.array([max(results[n].metrics.get(weight_by, 0.0), 0.0)
                       for n in names])
    w = scores / scores.sum() if scores.sum() > 0 else np.ones(len(names)) / len(names)
    weighted = (preds * w).sum(axis=1)
    weighted_m = evaluate(y_te, fin(weighted), task)

    print(f"members: {', '.join(names)}\n")
    print(f"{'method':<20}   metrics")
    print("-" * 72)
    print(f"{'simple average':<20}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in simple_m.items()))
    print(f"{'weighted average':<20}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in weighted_m.items()))
    print(f"\nweights (by {weight_by}):")
    for n, wi in zip(names, w):
        print(f"  {n:<16} {wi:.4f}")

    key = "accuracy" if task == "classification" else "r2"
    best_single = max(names, key=lambda n: results[n].metrics[key])
    bs = results[best_single].metrics[key]
    be = max(simple_m[key], weighted_m[key])
    print(f"\nbest ensemble {be:.4f}  vs  best single ({best_single}) {bs:.4f}"
          f"  -> {'ensemble wins' if be > bs else 'single model wins'}")
    if be <= bs:
        print("  Ensembles help when models make UNCORRELATED errors. These read")
        print("  the same features and largely agree, so averaging reproduces the")
        print("  consensus rather than improving on it.")

    return {"simple": simple_m, "weighted": weighted_m,
            "weights": dict(zip(names, w.round(4)))}


# =====================================================================
# 8 — hyperparameter tuning: max_depth sweep
# =====================================================================

def tune_max_depth(X_tr, y_tr, X_te, y_te, task="classification",
                   depths=(2, 3, 5, 8, 12, 16, 20, None),
                   models=("decision_tree", "random_forest"),
                   rf_subsample=None):
    """
    Sweep max_depth for the depth-sensitive models.

    Train scores are printed beside test scores because the GAP between them is
    the overfitting signal: a deep tree drives train accuracy up while test
    accuracy falls. Reporting the test score alone hides the mechanism.

    `rf_subsample` caps rows for the random forest sweep — 100 trees x 8 depths
    on 240k rows is slow, and the shape of the curve is unchanged by sampling.
    """
    metric = "accuracy" if task == "classification" else "r2"
    rows = []

    for name in models:
        xtr, ytr = X_tr, y_tr
        note = ""
        if name == "random_forest" and rf_subsample and len(X_tr) > rf_subsample:
            xtr = X_tr.iloc[-rf_subsample:]
            ytr = y_tr.iloc[-rf_subsample:]
            note = f"  [subsampled to {rf_subsample:,} rows]"

        print(f"\n{name}{note}")
        print(f"  {'max_depth':<12}{'train':>9}{'test':>9}{'gap':>9}{'fit(s)':>9}")
        print("  " + "-" * 48)

        for depth in depths:
            model = define_models(task, max_depth=depth)[name]
            t0 = time.time()
            model.fit(xtr, ytr)
            elapsed = time.time() - t0
            tr = evaluate(ytr, model.predict(xtr), task)[metric]
            te = evaluate(y_te, model.predict(X_te), task)[metric]
            label = "None" if depth is None else str(depth)
            rows.append({"model": name, "max_depth": label,
                         f"train_{metric}": round(tr, 4),
                         f"test_{metric}": round(te, 4),
                         "gap": round(tr - te, 4), "fit_s": round(elapsed, 1)})
            print(f"  {label:<12}{tr:>9.4f}{te:>9.4f}{tr - te:>9.4f}{elapsed:>9.1f}")

    tuning = pd.DataFrame(rows)
    col = f"test_{metric}"
    print("\nBEST DEPTH")
    print("-" * 72)
    best = {}
    for name, g in tuning.groupby("model"):
        row = g.loc[g[col].idxmax()]
        best[name] = row["max_depth"]
        print(f"  {name:<16} max_depth={row['max_depth']:<6} "
              f"{col}={row[col]:.4f}  (gap {row['gap']:.4f})")

    worst = tuning[tuning["max_depth"] == "None"]
    if not worst.empty:
        print("\n  Unlimited depth shows the overfitting clearly: train score high,")
        print("  test score lower, widest gap of any setting.")
    return tuning, best


def feature_importance(model, names=FEATURE_NAMES) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_,
                         index=names).sort_values(ascending=False)
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if hasattr(inner, "coef_"):
            return pd.Series(np.abs(np.ravel(inner.coef_)),
                             index=names).sort_values(ascending=False)
    return pd.Series(dtype=float)


# =====================================================================
# main
# =====================================================================

def ml_pipeline_run(**overrides):
    """
    Run the full pipeline. Any CONFIG key can be overridden:

        run()                                  # CONFIG defaults
        run(sample=60000)                      # faster
        run(task="regression")                 # predict returns
        run(data="/content/drive/MyDrive/ml_dataset.parquet")
    """
    cfg = {**ML_CONFIG, **overrides}
    unknown = set(overrides) - set(ML_CONFIG)
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}. "
                        f"Valid: {sorted(ML_CONFIG)}")

    task = cfg["task"]
    target = cfg["target"] or ("y_expansion" if task == "classification"
                               else "y_return")

    rule()
    print("ML PIPELINE — REQUIREMENTS 1-8")
    print(f"task: {task}   target: {target}")
    rule()

    # ---- 1 + 2 ----
    step("1 + 2", "LOAD DATASET WITH DEFINED INPUT X AND OUTPUT y")
    data = load_or_build(cfg["data"], cfg["nq"], cfg["es"])
    if cfg["sample"] and cfg["sample"] < len(data):
        data = data.iloc[-cfg["sample"]:].copy()
        print(f"[sampled to the most recent {cfg['sample']:,} rows]\n")
    describe_dataset(data, target)

    # ---- 3 ----
    step(3, "SPLIT INTO TRAINING AND TESTING")
    train, test = time_split(data, cfg["test_frac"])
    X_tr, y_tr = split_xy(train, target)
    X_te, y_te = split_xy(test, target)
    print(f"method     : chronological ({1 - cfg['test_frac']:.0%} / "
          f"{cfg['test_frac']:.0%}), NOT random")
    print(f"reason     : overlapping lookback windows make a shuffled split leak")
    print(f"train      : {len(X_tr):,} rows", end="")
    if "timestamp" in train:
        print(f"  ({train['timestamp'].min().date()} -> "
              f"{train['timestamp'].max().date()})")
    else:
        print()
    print(f"test       : {len(X_te):,} rows", end="")
    if "timestamp" in test:
        print(f"  ({test['timestamp'].min().date()} -> "
              f"{test['timestamp'].max().date()})")
    else:
        print()
    print(f"X shape    : train {X_tr.shape}   test {X_te.shape}")
    print(f"y shape    : train {y_tr.shape}   test {y_te.shape}")

    # ---- 4 ----
    step(4, "DEFINE MODELS")
    models = define_models(task, max_depth=8)
    for i, (name, m) in enumerate(models.items(), 1):
        inner = m.named_steps["model"] if hasattr(m, "named_steps") else m
        print(f"  {i}. {name:<16} {type(inner).__name__}")
    print("\n  scaled inside a Pipeline: linear, knn, neural_net")
    print("  depth-sensitive (swept in req 8): decision_tree, random_forest")

    # ---- 5 + 6 ----
    step("5 + 6", "TRAIN ON TRAINING SET, EVALUATE ON TESTING SET")
    results = train_and_evaluate(models, X_tr, y_tr, X_te, y_te, task)

    # ---- 7 ----
    step(7, "COMBINE MODELS BY AVERAGING OUTPUTS")
    ens = ensemble(results, y_te, task)

    # ---- 8 ----
    tuning = best = None
    if cfg["tune"]:
        step(8, "HYPERPARAMETER TUNING — max_depth SWEEP")
        tuning, best = tune_max_depth(
            X_tr, y_tr, X_te, y_te, task, rf_subsample=cfg["rf_subsample"])

    # ---- extra: what carried the signal ----
    rf = results.get("random_forest")
    if rf is not None and rf.model is not None:
        print()
        rule()
        print("FEATURE IMPORTANCE (random forest)")
        rule()
        print(feature_importance(rf.model).head(10).round(4).to_string())

    # ---- checklist ----
    print()
    rule()
    print("REQUIREMENTS CHECKLIST")
    rule()
    key = "accuracy" if task == "classification" else "r2"
    best_single = max((n for n in results if n != "BASELINE"),
                      key=lambda n: results[n].metrics[key])
    items = [
        ("1. Clear dataset, defined I/O",
         f"{len(data):,} samples, {len(FEATURE_NAMES)} features, "
         f"{len(TARGETS)} targets"),
        ("2. Input X, output y",
         f"X {X_tr.shape[1]} cols, y = {target}"),
        ("3. Train / test split",
         f"{len(X_tr):,} / {len(X_te):,}, chronological"),
        ("4. Five models defined",
         ", ".join(models.keys())),
        ("5. Trained on training set",
         f"all {len(models)} fitted on {len(X_tr):,} rows"),
        ("6. Evaluated on testing set",
         f"best {best_single} {key}={results[best_single].metrics[key]:.4f} "
         f"vs baseline {results['BASELINE'].metrics[key]:.4f}"),
        ("7. Combined by averaging",
         f"simple {ens['simple'][key]:.4f}, weighted {ens['weighted'][key]:.4f}"),
        ("8. max_depth tuned",
         (", ".join(f"{k}={v}" for k, v in best.items()) if best else "skipped")),
    ]
    for label, detail in items:
        print(f"  [x] {label:<32} {detail}")

    if tuning is not None:
        try:
            tuning.to_csv("tuning_results.csv", index=False)
            print("\nsaved: tuning_results.csv")
        except OSError as exc:
            print(f"\n(could not write tuning_results.csv: {exc})")

    return {"results": results, "ensemble": ens, "tuning": tuning,
            "best_depth": best, "data": data,
            "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def _in_notebook() -> bool:
    """True inside IPython/Jupyter/Colab, where argparse would see kernel args."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _ml_cli():
    """Parse command-line flags. Only used when run as a real script."""
    ap = argparse.ArgumentParser(description="ML pipeline, requirements 1-8")
    ap.add_argument("--data", default=CONFIG["data"])
    ap.add_argument("--nq", default=CONFIG["nq"])
    ap.add_argument("--es", default=CONFIG["es"])
    ap.add_argument("--task", default=CONFIG["task"],
                    choices=["classification", "regression"])
    ap.add_argument("--target", default=CONFIG["target"])
    ap.add_argument("--sample", type=int, default=CONFIG["sample"])
    ap.add_argument("--test-frac", type=float, default=CONFIG["test_frac"])
    ap.add_argument("--rf-subsample", type=int, default=CONFIG["rf_subsample"])
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    return {"data": a.data, "nq": a.nq, "es": a.es, "task": a.task,
            "target": a.target, "sample": a.sample, "test_frac": a.test_frac,
            "rf_subsample": a.rf_subsample, "tune": not a.no_tune}

# ======================================================================
# STREAMLIT APPLICATION
# ======================================================================

st.set_page_config(page_title="Trading Decision Support", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

DATA_DIR = os.environ.get("TRADINGBOT_DATA", "data")


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Loading market data…")
def load_local(nq_path: str, es_path: str):
    return {"NQ": load_parquet(nq_path), "ES": load_parquet(es_path)}


@st.cache_data(show_spinner="Reading uploads…")
def load_uploads(nq_bytes: bytes, es_bytes: bytes):
    return {"NQ": pd.read_parquet(io.BytesIO(nq_bytes)),
            "ES": pd.read_parquet(io.BytesIO(es_bytes))}


def get_data():
    """Local files, else uploads. Returns None until both instruments exist."""
    nq_p, es_p = os.path.join(DATA_DIR, "nq_1m.parquet"), os.path.join(DATA_DIR, "es_1m.parquet")
    if os.path.exists(nq_p) and os.path.exists(es_p):
        return load_local(nq_p, es_p)

    with st.sidebar:
        st.warning("No market data found.")
        nq_f = st.file_uploader("NQ 1-minute parquet", type="parquet", key="nq")
        es_f = st.file_uploader("ES 1-minute parquet", type="parquet", key="es")
        if nq_f and es_f:
            return load_uploads(nq_f.getvalue(), es_f.getvalue())
    return None


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

def sidebar_settings() -> dict:
    st.sidebar.title("Settings")

    st.sidebar.subheader("Strategy")
    use_amd = st.sidebar.checkbox(
        "AMD gate (§7)", True,
        help="Accumulation → manipulation → distribution. The single largest "
             "win-rate factor measured: it removes days with no AMD structure, "
             "which are days you would skip by eye.")
    trigger_mode = st.sidebar.selectbox(
        "Entry trigger", ["both", "single", "double"], index=0,
        help="single = first sweep-and-return. double = wait for a second test. "
             "both is what tested best.")
    allow_confirmed = st.sidebar.checkbox(
        "Allow confirmed breakouts", False,
        help="A break that keeps going means the day is trending, not "
             "manipulating — under AMD there is nothing to fade.")

    st.sidebar.subheader("Timing")
    c1, c2 = st.sidebar.columns(2)
    entry_start = c1.text_input("Start", "09:50")
    entry_end = c2.text_input("End", "12:00")

    st.sidebar.subheader("Risk")
    account = st.sidebar.number_input("Account ($)", 1_000, 10_000_000, 150_000, 5_000)
    risk = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, 1.5, 0.1) / 100
    rr = st.sidebar.slider("Reward : risk", 1.0, 4.0, 2.0, 0.5)
    stop_mode = st.sidebar.selectbox("Stop placement",
                                     ["structural", "clamped", "fixed"], 0)
    stop_ticks = stop_min = stop_max = None
    if stop_mode == "fixed":
        stop_ticks = st.sidebar.number_input("Stop (ticks)", 10, 1000, 200, 10)
    elif stop_mode == "clamped":
        stop_min = st.sidebar.number_input("Min ticks", 10, 1000, 200, 10)
        stop_max = st.sidebar.number_input("Max ticks", 10, 2000, 500, 10)

    return dict(use_amd=use_amd, trigger_mode=trigger_mode,
                allow_confirmed_breakout=allow_confirmed,
                entry_start=entry_start, entry_end=entry_end,
                account=account, risk=risk, rr=rr, stop_mode=stop_mode,
                stop_ticks=stop_ticks, stop_min=stop_min, stop_max=stop_max)


def engine_kwargs(s: dict) -> dict:
    kw = dict(use_amd=s["use_amd"], trigger_mode=s["trigger_mode"],
              allow_confirmed_breakout=s["allow_confirmed_breakout"],
              entry_start=s["entry_start"], entry_end=s["entry_end"],
              rr=s["rr"], risk_pct=s["risk"], stop_mode=s["stop_mode"])
    if s["stop_mode"] == "fixed":
        kw["stop_ticks"] = s["stop_ticks"]
    elif s["stop_mode"] == "clamped":
        kw["stop_min_ticks"] = s["stop_min"]
        kw["stop_max_ticks"] = s["stop_max"]
    return kw


# ---------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------

def tab_today(data, s):
    st.subheader("Evaluate a session")
    st.caption("Every gate that passed, and the one that stopped each candidate.")

    idx = data["NQ"].index
    c1, c2 = st.columns([1, 3])
    d = c1.date_input("Session", value=idx.max().date(),
                      min_value=idx.min().date(), max_value=idx.max().date(),
                      key="today_date")

    if not st.button("Evaluate", type="primary", key="today_go"):
        return

    with st.spinner("Running the pipeline…"):
        setups = evaluate_session(str(d), data, account=s["account"],
                                  **engine_kwargs(s))

    if not setups:
        st.info("Nothing evaluated — not a trading day?")
        return

    q = [x for x in setups if x.qualified]
    st.metric("Qualified setups", len(q))

    for x in setups:
        head = (f"✅ {x.direction.upper()} {x.symbol} @ {x.entry:.2f}"
                if x.qualified else
                f"❌ {x.timestamp.strftime('%H:%M')} — {x.vetoes[-1] if x.vetoes else 'rejected'}")
        with st.expander(head, expanded=x.qualified):
            for p in x.passed:
                st.markdown(f"✅ {p}")
            for v in x.vetoes:
                st.markdown(f"🚫 {v}")
            if x.qualified:
                a, b, c, e = st.columns(4)
                a.metric("Entry", f"{x.entry:.2f}")
                b.metric("Stop", f"{x.stop:.2f}", f"{x.stop_ticks:.0f} ticks")
                c.metric("Target", f"{x.target:.2f}", f"{x.rr:.2f}R")
                e.metric("Size", f"{x.contracts}c", f"${x.risk_dollars:,.0f}")

    if q:
        st.info("Candidates, not instructions. The engine cannot judge "
                "accumulation quality, and its leading-index choice (§4) is a "
                "provisional proxy.")


def tab_backtest(data, s):
    st.subheader("Backtest")
    idx = data["NQ"].index
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=max(idx.min().date(), date(2017, 1, 1)),
                          min_value=idx.min().date(), max_value=idx.max().date(),
                          key="bt_from")
    end = c2.date_input("To", value=min(idx.max().date(), date(2021, 12, 31)),
                        min_value=idx.min().date(), max_value=idx.max().date(),
                        key="bt_to")

    st.caption("Roughly 3–4 minutes per year of data.")
    if not st.button("Run backtest", type="primary", key="bt_go"):
        return

    with st.spinner(f"Scanning {start} → {end}…"):
        _, setups = scan_range(str(start), str(end), data,
                               account=s["account"], **engine_kwargs(s))
        trades, m = run_backtest(setups, data, account=s["account"],
                                 apply_daily_limits=False)

    if trades.empty:
        st.warning("No trades. Loosen the gates or widen the window.")
        st.dataframe(veto_summary(setups).rename("count"))
        return

    pv = trades["symbol"].map({k: v["point_value"] for k, v in CONTRACT_SPECS.items()})
    risk = ((trades["entry"] - trades["stop"]).abs() * trades["contracts"] * pv)
    gross_r = float((trades["gross_dollars"] / risk).mean())
    cost_r = float((trades["costs"] / risk).mean())
    weeks = max((pd.to_datetime(trades["date"]).max()
                 - pd.to_datetime(trades["date"]).min()).days / 7, 1)

    a, b, c, d, e, f = st.columns(6)
    a.metric("Trades", m["trades"])
    b.metric("Win rate", f"{m['win_rate']:.1%}",
             f"breakeven {1/(1+s['rr']):.1%}")
    c.metric("Profit factor", f"{m['profit_factor']}")
    d.metric("Expectancy", f"{m['expectancy_r']:+.3f}R")
    e.metric("Net P&L", f"${m['net_dollars']:,.0f}")
    f.metric("Per week", f"{len(trades)/weeks:.2f}")

    st.markdown("#### Equity curve")
    eq = pd.DataFrame({"equity": s["account"] + trades["net_dollars"].cumsum()})
    eq.index = pd.to_datetime(trades["exit_time"])
    st.line_chart(eq)

    st.markdown("#### Expectancy decomposition")
    st.caption("Win rate alone says nothing without R:R — moving the target "
               "closer raises it and lowers profitability. Gross vs net is what "
               "separates a signal problem from a cost problem.")
    st.dataframe(pd.DataFrame([
        {"component": "Win rate", "value": f"{m['win_rate']:.2%}"},
        {"component": f"Breakeven at {s['rr']}:1", "value": f"{1/(1+s['rr']):.2%}"},
        {"component": "Gross expectancy", "value": f"{gross_r:+.4f}R"},
        {"component": "Cost drag", "value": f"{cost_r:+.4f}R"},
        {"component": "Net expectancy", "value": f"{m['expectancy_r']:+.4f}R"},
    ]), hide_index=True, use_container_width=True)

    if gross_r < 0:
        st.error("Negative **before** costs — cheaper execution or wider stops "
                 "cannot fix this. The signal has no edge over this period.")
    elif m["expectancy_r"] < 0 < gross_r:
        st.warning("Positive gross, negative net: a real edge smaller than "
                   "transaction costs. Fewer and better trades would help.")
    else:
        st.success("Positive after costs.")

    st.markdown("#### By year")
    trades["yr"] = pd.to_datetime(trades["date"]).dt.year
    st.dataframe(trades.groupby("yr").agg(
        trades=("net_dollars", "size"),
        win_rate=("net_dollars", lambda x: (x > 0).mean()),
        net=("net_dollars", "sum"),
        exp_R=("r_multiple", "mean")).round(3), use_container_width=True)
    st.caption("Consistency across years matters more than the total. An edge "
               "living in one year is usually a regime artifact.")

    with st.expander("Gate breakdown"):
        st.dataframe(veto_summary(setups).rename("count"))
    with st.expander("All trades"):
        st.dataframe(trades, use_container_width=True)
    st.download_button("Download trades CSV", trades.to_csv(index=False),
                       "trades.csv", "text/csv")


def tab_manual(data, s):
    st.subheader("Score a chart read")
    st.caption("Type the levels you see. Every rule runs against them; the "
               "judgement stays yours.")

    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Instrument", ["NQ", "ES"], key="mn_sym")
    direction = c2.selectbox("Direction", ["bull", "bear"], key="mn_dir")
    when = c3.text_input("Time (HH:MM)", "10:15", key="mn_time")

    c4, c5, c6 = st.columns(3)
    entry = c4.number_input("Entry", value=0.0, step=0.25, format="%.2f",
                            key="mn_entry")
    stop = c5.number_input("Stop (0 = use ticks)", value=0.0, step=0.25,
                           format="%.2f", key="mn_stop")
    ticks = c6.number_input("…or stop ticks", 10, 1000, 200, 10, key="mn_ticks")

    idx = data["NQ"].index
    d = st.date_input("Session", value=idx.max().date(),
                      min_value=idx.min().date(), max_value=idx.max().date(),
                      key="mn_date")

    if not st.button("Check", type="primary", key="mn_go"):
        return
    if entry <= 0:
        st.error("Entry price required.")
        return

    res = check(symbol=sym, direction=direction, entry=entry,
                stop=stop if stop > 0 else None,
                stop_ticks=None if stop > 0 else ticks,
                date=str(d), time_of_day=when, account=s["account"],
                risk_pct=s["risk"], rr=s["rr"], data_1m=data, verbose=False)

    for p in res["passed"]:
        st.markdown(f"✅ {p}")
    for v in res["vetoes"]:
        st.markdown(f"🚫 {v}")
    (st.success if res["ok"] else st.error)(
        "Clears every rule checked." if res["ok"]
        else f"{len(res['vetoes'])} problem(s).")


def tab_chart(data, s):
    st.subheader("Read a screenshot")
    st.caption("A vision model transcribes the levels off a TradingView "
               "screenshot, then those levels run through the same rules.")

    key = st.text_input("API key", type="password",
                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic key. Nothing is stored; it lives in "
                             "this session only.", key="cr_key")
    provider = st.selectbox("Provider", ["anthropic", "openai", "google"],
                            key="cr_provider")
    up = st.file_uploader("Chart screenshot",
                          type=["png", "jpg", "jpeg", "webp"], key="cr_img")

    if up:
        st.image(up, use_container_width=True)
    if not (up and key):
        st.info("Upload a screenshot and supply a key.")
        return
    if not st.button("Read chart", type="primary", key="cr_go"):
        return

    tmp = f"/tmp/{up.name}"
    with open(tmp, "wb") as f:
        f.write(up.getvalue())

    try:
        with st.spinner("Reading…"):
            seen = read_chart(tmp, provider=provider, api_key=key, verbose=False)
    except Exception as exc:
        st.error(f"Could not read the chart: {exc}")
        return

    st.json(seen)
    st.caption("The model transcribes what is printed — ticker, axis prices, "
               "your drawn levels. It does not judge the setup, and anything "
               "marked low confidence is worth re-checking by eye.")

    if seen.get("symbol") and seen.get("direction") and seen.get("entry"):
        sym = seen["symbol"].upper().lstrip("M") if seen["symbol"].upper() in ("MNQ", "MES") else seen["symbol"].upper()
        res = check(symbol=sym, direction=seen["direction"],
                    entry=float(seen["entry"]),
                    stop=float(seen["stop"]) if seen.get("stop") else None,
                    stop_ticks=None if seen.get("stop") else 200.0,
                    target=float(seen["target"]) if seen.get("target") else None,
                    date=seen.get("date"), time_of_day=seen.get("session_time"),
                    account=s["account"], risk_pct=s["risk"], rr=s["rr"],
                    data_1m=data, verbose=False)
        st.markdown("#### Rule check")
        for p in res["passed"]:
            st.markdown(f"✅ {p}")
        for v in res["vetoes"]:
            st.markdown(f"🚫 {v}")
    else:
        st.warning("Could not read symbol, direction and entry — fill them in "
                   "on the *Score a chart read* tab instead.")


def tab_ml():
    st.subheader("Volatility classifier")
    st.caption("Five model types predicting whether the next 30 minutes expand "
               "or stay compressed. Separate from the trading engine — it gates "
               "no trades.")

    path = os.path.join(DATA_DIR, "ml_dataset.parquet")
    up = None
    if not os.path.exists(path):
        up = st.file_uploader("ML dataset (parquet)", type="parquet", key="ml_file")
        if not up:
            st.info("Upload ml_dataset.parquet, or build it from the 1-minute "
                    "files with the CLI: `python cli.py data --build-ml`.")
            return

    task = st.selectbox("Task", ["classification", "regression"], key="ml_task")
    sample = st.number_input("Sample size (0 = all)", 0, 400_000, 60_000,
                             10_000, key="ml_sample")
    tune = st.checkbox("Sweep max_depth (requirement 8)", True, key="ml_tune")

    if not st.button("Train", type="primary", key="ml_go"):
        return

    if up:
        path = "/tmp/ml_dataset.parquet"
        with open(path, "wb") as f:
            f.write(up.getvalue())

    with st.spinner("Training five models…"):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ml_pipeline_run(data=path, task=task,
                            sample=sample or None, tune=tune)
    st.code(buf.getvalue())


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    st.title("Trading Decision-Support Tool")
    st.caption("Surfaces qualifying setups and explains every accept and "
               "reject. Decision support — no orders are placed.")

    s = sidebar_settings()
    data = get_data()

    if data is None:
        st.info("Load NQ and ES 1-minute parquet files to begin. "
                "Use the sidebar, or place them in `data/`.")
        st.markdown("""
**Getting the data**

The raw Databento pull uses `stype_in=parent`, so each file holds every
expiration plus calendar spreads. The loader drops spreads, picks the front
month by daily volume, forbids rolling backwards, and stitches a continuous
series:

```bash
python cli.py data --raw-nq NQ.csv.zst --raw-es ES.csv.zst --build-ml
```
""")
        return

    st.sidebar.success(f"NQ {len(data['NQ']):,} bars · ES {len(data['ES']):,} bars")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Session", "Backtest", "Chart read", "Screenshot", "ML"])
    with t1:
        tab_today(data, s)
    with t2:
        tab_backtest(data, s)
    with t3:
        tab_manual(data, s)
    with t4:
        tab_chart(data, s)
    with t5:
        tab_ml()


main()
