"""
manual.py — evaluate a setup you read off a chart yourself.

WHY THIS EXISTS
    The engine cannot read a TradingView screenshot: that needs a vision model
    at runtime, which is an API dependency this tool does not have. What it can
    do is take the levels YOU read off the chart and run every rule against
    them, so the judgement stays with you and the arithmetic and rule-checking
    do not.

WHAT IT CHECKS
    §1 entry window        is it inside the tradeable hours
    §3 opening range       does the price sit inside / outside the 09:30-09:40 range
    §7 AMD                 has manipulation completed, and is your direction the
                           distribution side
    §5/§9 liquidity        is there an untapped draw in your direction
    §8 risk                stop distance, contracts at your risk %, RR to target,
                           and whether the nearest liquidity beats your target

USE
    from .manual import check
    check(symbol="NQ", direction="bull", entry=21050, stop=21000,
          date="2026-07-15", account=100_000)

    # or supply your own levels when you do not want the engine's
    check(symbol="ES", direction="bear", entry=5605.50, stop=5615.75,
          target=5585, or_high=5612, or_low=5598)
"""

from __future__ import annotations

import pandas as pd

TZ = "America/New_York"


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
    from .orchestrator import CONTRACT_SPECS, size_position

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
            from .data_loader import resample_ohlcv
            from .opening_range import build_opening_range
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
            from .sessions import (
                build_liquidity_map,
                mark_tapped,
                untapped_targets,
            )
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
            print("  CLEARS EVERY RULE CHECKED")
        else:
            print(f"  {len(vetoes)} PROBLEM(S)")
        print("\n  The engine cannot see accumulation/distribution quality or")
        print("  judge which index is cleaner. Those remain your read.")
    return out
