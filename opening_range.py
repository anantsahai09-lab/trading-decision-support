"""
opening_range.py — Opening range construction, interaction tracking, and the
double-test breakout rule.

Implements Strategy v2.0 §3 and §3.1.

§3  On the 5m chart, mark the high and low wicks of the 09:30 and 09:35 candles
    (the 09:30–09:40 formation window) on BOTH NQ and ES. These levels act as
    support/resistance liquidity; price is drawn to them.

§3.1 Real breakouts usually test a level TWICE before breaking out.
     - First breach: not trusted.
     - Second breach: the decision point.
         * shallower than the first, and price closes back inside
           -> failed breakout -> expect resolution the OTHER way
         * deeper than the first -> genuine breakout, continuation
     Look for entries on the second test.

Definitions used here (consistent with swings.py conventions):
  TOUCH   : bar wick reaches/pierces a level but the bar CLOSES inside the range
  BREACH  : bar CLOSES beyond a level
  EPISODE : a contiguous run of bars closed beyond the same side, ending when a
            bar closes back inside the range

Closes are used for breaches for the same reason BOS uses closes: wicks through
a level are exactly the liquidity-grab behaviour the strategy expects to see,
not a structural break.

No lookahead: the range is not usable until `formed_at` (the close of the 09:35
bar). All interaction tracking and signals are evaluated only on bars strictly
after that.

Input contract:
  df: DataFrame indexed by ascending tz-aware DatetimeIndex, columns
      ['open','high','low','close']. Should be 5-minute bars for range
      construction; interaction tracking works on any timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")

FORMATION_BARS = ("09:30", "09:35")
FORMATION_END = "09:40"


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("Index must be timezone-aware (use America/New_York)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending")


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
