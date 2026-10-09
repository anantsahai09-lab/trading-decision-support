"""
sessions.py — Session mapping and liquidity draws.

Implements Strategy v2.0 §5 (liquidity targets) and the §9 gate:
"no untapped liquidity in trade direction -> no trade".

SESSION WINDOWS (all America/New_York)
    Asia    18:00 -> 03:00   (spans midnight)
    London  03:00 -> 08:30
    NY      08:30 -> 17:00
    break   17:00 -> 18:00   (spread hour, no market)

TRADING DAY
    CME futures run Sun 18:00 -> Fri 17:00. A bar at or after 18:00 belongs to
    the NEXT calendar day's trading session. So the Asia session that starts
    Monday 18:00 is part of Tuesday's trading day — which is what makes
    "previous day high/low" mean the right thing.

LEVELS produced per trading day
    asia_high / asia_low
    london_high / london_low
    ny_high / ny_low            (only meaningful intraday or for prior days)
    prevday_high / prevday_low  (previous trading day's full-session extremes)

TAPPED vs UNTAPPED
    A level is TAPPED once price trades through it after the level was formed:
    for a high, any later bar with high >= level; for a low, any later bar with
    low <= level. Wicks count here — unlike breaches in opening_range.py — because
    the question is whether resting orders at that level were reached, not whether
    structure broke.

No lookahead: a level's `formed_at` is the end of the session that produced it,
and tap-checking only scans bars strictly after that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")

SESSION_BOUNDS = {
    "asia": ("18:00", "03:00"),
    "london": ("03:00", "08:30"),
    "ny": ("08:30", "17:00"),
}


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

        hit = (window["high"] >= lv.price if lv.side == "high"
               else window["low"] <= lv.price)
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
