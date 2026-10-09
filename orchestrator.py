"""
orchestrator.py — Signal pipeline and risk engine (Strategy v2.1 §1, §4, §8, §9).

Runs the full decision sequence for one session and emits qualified setups.

PIPELINE (each stage can veto)
    1. §2  bias          — 15m order flow; 'obvious trend' overrides
    2. §3  opening range — built from the 09:30/09:35 5m bars on BOTH instruments
    3. §3.1 trigger      — double-test signal supplies direction
    4. §1  time window   — 09:50-10:30 ET only
    5. §9  liquidity     — an untapped draw must exist in the trade direction
    6. §6  confluence    — FVG / IFVG / SMT / 1m BOS, any ONE
    7. §4  instrument    — leading index (SMT geometry, else fewest OR breaches)
    8. §8  risk          — stop at 2nd most recent 1m swing, RR target capped at
                           liquidity, position size at 1% risk
    9. §8.2 limits       — per-session trade cap and daily loss/profit stops

Everything is evaluated as of a specific bar, using only data at or before it.
The detectors already enforce their own no-lookahead rules; this layer never
reaches past `as_of`.

OUTPUT
    A Setup carries the full reasoning chain — every gate that passed, every
    veto — so a rejected day is as inspectable as a taken trade. That is the
    point of a decision-support tool: you want to know why it said no.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

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
    from .sessions import untapped_targets  # local import keeps modules decoupled

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
    from .opening_range import range_violations

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
    from .confluences import (
        confluence_check,
        detect_fvgs,
        detect_ifvg_signals,
        detect_smt,
        update_fvg_state,
    )
    from .data_loader import resample_ohlcv
    from .opening_range import build_opening_range, detect_double_test
    from .sessions import build_liquidity_map, has_liquidity_target, mark_tapped
    from .swings import (
        classify_trend,
        detect_bos,
        detect_swings,
        is_obvious_trend,
        label_swings,
    )

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
        from .amd import amd_gate, classify_amd
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
            from .opening_range import detect_single_sweep
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
    from collections import Counter

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
