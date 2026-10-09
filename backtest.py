"""
backtest.py — Trade simulation and performance metrics.

Takes qualified Setups from orchestrator.evaluate_session and walks forward bar
by bar to find the outcome: stop, target, or time-based exit at the close.

FILL ASSUMPTIONS (deliberately pessimistic)
    entry   : the close of the signal bar. The signal is only known at that
              close, so this is the earliest honest fill.
    stop    : filled at the stop price when a bar's LOW (long) or HIGH (short)
              reaches it. Slippage is applied.
    target  : filled at the target price when a bar's HIGH (long) or LOW
              (short) reaches it. No slippage — a limit order fills at its
              price or better.
    both in one bar : assume the STOP filled. Without tick data there is no way
              to know which came first, and assuming the winner would inflate
              every result. This is the single most common way backtests lie.
    timeout : flat at the session close if neither level is hit.

Costs default to 1 tick of slippage on entry and stop, plus commission per
contract per side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .orchestrator import CONTRACT_SPECS

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
