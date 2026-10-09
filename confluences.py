"""
confluences.py — Continuation confluence detectors (Strategy v2.0 §6).

Any ONE of these confirms an entry. SMT is optional, not a gate.

  1. FVG      — 5m fair value gap (imbalance), tap and respect
  2. IFVG     — inverted FVG: a gap that gets disrespected flips polarity and
                becomes a CONFIRMATION signal in the opposite direction
  3. SMT      — NQ/ES divergence at aligned swings
  4. 1m BOS   — lives in swings.py (detect_bos)

FVG GEOMETRY
    Three consecutive candles. The middle one expands; the gap is the space the
    outer two never traded through.

    Bullish FVG:  candle3.low  >  candle1.high   -> zone [c1.high, c3.low]
    Bearish FVG:  candle3.high <  candle1.low    -> zone [c3.high, c1.low]

    The colour of candles 1 and 3 is irrelevant — only the wick gap matters.
    Any overlap of the outer wicks means there is no imbalance.

FVG LIFECYCLE
    formed      -> known only at candle 3's CLOSE (no lookahead)
    tapped      -> price wicks back into the zone
    respected   -> price taps the zone and continues in the FVG's direction
    inverted    -> a candle CLOSES beyond the far side of the zone; the gap is
                   dead as a continuation signal and flips into a confirmation
                   signal the other way (IFVG)

STACKED FVGs
    Consecutive same-direction gaps with no retrace between them form one
    imbalance block. Per the reference material, the *last* gap in a bullish
    stack (the lowest) is the one whose inversion kills the move — inverting an
    upper gap in the stack means nothing while lower gaps remain live.

SMT DIVERGENCE
    Bearish: at aligned swing highs, one index makes a higher high while the
             other makes a lower high. The index making the LOWER high leads.
    Bullish: at aligned swing lows, one makes a lower low while the other makes
             a higher low. The index making the HIGHER low leads.

    Alignment tolerance is +/-1 bar by default (§10.7, provisional). SMT is only
    meaningful at a significant liquidity sweep — callers should gate on §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending")


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
