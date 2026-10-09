"""
amd.py — Accumulation / Manipulation / Distribution phase model (§7).

THE MODEL
    Accumulation  the 09:30-09:40 opening range. Orders build inside a tight
                  range; bars overlap heavily.
    Manipulation  price breaks OUT of that range and then closes back INSIDE.
                  The judas swing. It may sweep a session level but need not.
    Distribution  the real move, in the OPPOSITE direction to the manipulation,
                  toward the actual draw on liquidity. This is the trade.

WHY IT IS A PRECONDITION, NOT A CONFLUENCE
    Under AMD the whole session has a shape. A day that never leaves the
    accumulation range, or leaves and never returns, is not an AMD day and there
    is nothing to trade. Treating this as a gate rather than a bonus is the
    difference between "did this setup appear" and "is this day the right
    shape at all".

CONSEQUENCE FOR SIGNALS
    A `confirmed_breakout` (second push deeper than the first) is NOT a trade
    under AMD. A genuine break in the manipulation direction means the day is
    trending, not manipulating — stand aside. Only `failed_breakout` survives,
    and it is traded AWAY from the manipulation.

NO LOOKAHEAD
    A phase is only assigned once the bars that define it have closed.
    `distribution_from` is the first bar at which an entry may be considered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


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
