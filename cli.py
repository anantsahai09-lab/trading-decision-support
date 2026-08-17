"""
cli.py — command-line entry point.

Everything the Streamlit app does, from a terminal. Useful for the long jobs:
a ten-year backtest takes far longer than a browser session will stay open.

    python cli.py check
    python cli.py data --raw-nq NQ.csv.zst --raw-es ES.csv.zst --build-ml
    python cli.py today --date 2026-07-15
    python cli.py backtest --start 2017-01-01 --end 2021-12-31 --out trades.csv
    python cli.py ml --sample 60000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from src import (CONTRACT_SPECS, evaluate_session, load_parquet, print_metrics,
                 run_backtest, scan_range, setups_to_frame, veto_summary)

APP = "Trading Decision-Support Tool"
VERSION = "1.0.0"
DATA_DIR = os.environ.get("TRADINGBOT_DATA", "data")
BANNER = f"\n{'=' * 72}\n  {APP}  v{VERSION}\n{'=' * 72}"


def _p(*parts):
    return os.path.join(DATA_DIR, *parts)


def _need(path, what):
    if not os.path.exists(path):
        sys.exit(f"\nMissing {what}: {path}\nRun `python cli.py check`.")
    return path


def _load(nq=None, es=None):
    return {"NQ": load_parquet(_need(nq or _p("nq_1m.parquet"), "NQ bars")),
            "ES": load_parquet(_need(es or _p("es_1m.parquet"), "ES bars"))}


def _engine_kwargs(a):
    kw = dict(use_amd=not a.no_amd, trigger_mode=a.trigger,
              allow_confirmed_breakout=a.allow_confirmed,
              entry_start=a.entry_start, entry_end=a.entry_end,
              rr=a.rr, risk_pct=a.risk, stop_mode=a.stop_mode)
    if a.stop_mode == "fixed":
        kw["stop_ticks"] = a.stop_ticks
    elif a.stop_mode == "clamped":
        kw["stop_min_ticks"] = a.stop_min
        kw["stop_max_ticks"] = a.stop_max
    return kw


def _decompose(trades):
    """Gross edge vs cost drag — a cost problem and a signal problem need
    different fixes, and only this split tells them apart."""
    pv = trades["symbol"].map({k: v["point_value"] for k, v in CONTRACT_SPECS.items()})
    risk = ((trades["entry"] - trades["stop"]).abs()
            * trades["contracts"] * pv).replace(0, np.nan)
    return {"gross_r": float((trades["gross_dollars"] / risk).mean()),
            "cost_r": float((trades["costs"] / risk).mean()),
            "net_r": float(trades["r_multiple"].mean())}


# ---------------------------------------------------------------------

def cmd_check(a):
    print(BANNER)
    print("\nDEPENDENCIES")
    for mod in ("pandas", "numpy", "sklearn", "pyarrow", "zstandard",
                "streamlit", "matplotlib", "anthropic"):
        try:
            m = __import__(mod)
            print(f"  [ok]   {mod:<14} {getattr(m, '__version__', '?')}")
        except ImportError:
            req = mod in ("pandas", "numpy", "pyarrow", "sklearn")
            print(f"  [--]   {mod:<14} missing ({'required' if req else 'optional'})")

    print(f"\nDATA  ({os.path.abspath(DATA_DIR)})")
    ok = False
    for name, desc in [("nq_1m.parquet", "NQ 1-minute continuous"),
                       ("es_1m.parquet", "ES 1-minute continuous"),
                       ("ml_dataset.parquet", "ML feature dataset")]:
        p = _p(name)
        if os.path.exists(p):
            print(f"  [ok]   {name:<22} {os.path.getsize(p)/1e6:>7.1f} MB   {desc}")
            ok = True
        else:
            print(f"  [--]   {name:<22} {'':>7}     {desc} — missing")
    if not ok:
        print("\n  Build them:")
        print("    python cli.py data --raw-nq NQ.csv.zst --raw-es ES.csv.zst --build-ml")
    print()


def cmd_data(a):
    from src import (build_continuous, data_health_report, save_parquet)
    from src.dataset import DatasetSpec, build_combined

    print(BANNER)
    os.makedirs(DATA_DIR, exist_ok=True)
    for sym, raw in (("NQ", a.raw_nq), ("ES", a.raw_es)):
        if not raw:
            continue
        print(f"\nBuilding {sym} from {raw}")
        df, rep = build_continuous(_need(raw, f"{sym} raw file"))
        print(f"  {rep}")
        out = _p(f"{sym.lower()}_1m.parquet")
        save_parquet(df, out)
        print(f"  saved: {out}\n")
        data_health_report(df, f"{sym} 1m")
        if len(rep.rolls):
            rp = _p(f"{sym.lower()}_rolls.csv")
            rep.rolls.to_csv(rp, index=False)
            print(f"\n  {len(rep.rolls)} rolls -> {rp}")
            print("  Roll days carry a price discontinuity: the previous day's")
            print("  high/low came from a different contract.")

    if a.build_ml:
        print("\nBuilding ML dataset")
        ds = build_combined(_load(), DatasetSpec())
        out = _p("ml_dataset.parquet")
        ds.to_parquet(out)
        print(f"  saved: {out}  ({len(ds):,} samples)")


def cmd_today(a):
    print(BANNER)
    data = _load(a.nq, a.es)
    d = a.date or str(pd.Timestamp.now(tz="America/New_York").date())
    print(f"\nSESSION {d}\n" + "=" * 72)
    setups = evaluate_session(d, data, account=a.account, **_engine_kwargs(a))
    for s in setups:
        print()
        print(s.explain())
    q = [s for s in setups if s.qualified]
    print("\n" + "=" * 72)
    print(f"{len(q)} QUALIFIED SETUP(S)" if q else "NO QUALIFYING SETUP")


def cmd_scan(a):
    print(BANNER)
    data = _load(a.nq, a.es)
    print(f"\nSCAN {a.start} -> {a.end}\n" + "=" * 72)
    t0 = time.time()
    summary, setups = scan_range(a.start, a.end, data, account=a.account,
                                 **_engine_kwargs(a))
    q = int(summary["qualified"].sum())
    print(f"\nsessions   : {len(summary):,}")
    print(f"qualified  : {q}")
    print(f"frequency  : {q / max(len(summary)/5, 1):.2f} per week")
    print(f"elapsed    : {time.time()-t0:.0f}s")
    print("\nGATE BREAKDOWN\n" + "-" * 72)
    print(veto_summary(setups).to_string())
    tf = setups_to_frame(setups)
    if not tf.empty and a.out:
        tf.to_csv(a.out, index=False)
        print(f"\nsaved {len(tf)} setups -> {a.out}")


def cmd_backtest(a):
    print(BANNER)
    data = _load(a.nq, a.es)
    print(f"\nBACKTEST {a.start} -> {a.end}")
    print(f"account ${a.account:,.0f} · risk {a.risk:.1%} · RR {a.rr}:1")
    print("=" * 72)
    t0 = time.time()
    _, setups = scan_range(a.start, a.end, data, account=a.account,
                           **_engine_kwargs(a))
    trades, m = run_backtest(setups, data, account=a.account,
                             slippage_ticks=a.slippage,
                             apply_daily_limits=not a.no_limits)
    print(f"\nPERFORMANCE  ({time.time()-t0:.0f}s)\n" + "-" * 72)
    print_metrics(m)

    if trades.empty:
        return
    d = _decompose(trades)
    print("\nEXPECTANCY DECOMPOSITION\n" + "-" * 72)
    print(f"  win rate           {m['win_rate']:.1%}")
    print(f"  breakeven at {a.rr}:1  {1/(1+a.rr):.1%}")
    print(f"  gross expectancy   {d['gross_r']:+.4f}R   (before costs)")
    print(f"  cost drag          {d['cost_r']:+.4f}R")
    print(f"  net expectancy     {d['net_r']:+.4f}R")
    if d["gross_r"] < 0:
        print("\n  Negative BEFORE costs — the signal has no edge here.")
    elif d["net_r"] < 0 < d["gross_r"]:
        print("\n  Positive gross, negative net: a real but sub-cost edge.")

    print("\nBY YEAR\n" + "-" * 72)
    trades["yr"] = pd.to_datetime(trades["date"]).dt.year
    print(trades.groupby("yr").agg(
        n=("net_dollars", "size"),
        win=("net_dollars", lambda x: (x > 0).mean()),
        net=("net_dollars", "sum"),
        exp_r=("r_multiple", "mean")).round(3).to_string())

    if a.out:
        trades.to_csv(a.out, index=False)
        print(f"\nsaved {len(trades)} trades -> {a.out}")


def cmd_ml(a):
    from src import ml_pipeline
    print(BANNER)
    ml_pipeline.run(data=a.data or _p("ml_dataset.parquet"), task=a.task,
                    sample=a.sample, tune=not a.no_tune,
                    rf_subsample=a.rf_subsample)


def build_parser():
    ap = argparse.ArgumentParser(prog="cli", description=f"{APP} v{VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, dates=True):
        p.add_argument("--nq")
        p.add_argument("--es")
        p.add_argument("--account", type=float, default=150_000)
        p.add_argument("--risk", type=float, default=0.015)
        p.add_argument("--rr", type=float, default=2.0)
        p.add_argument("--entry-start", default="09:50")
        p.add_argument("--entry-end", default="12:00")
        p.add_argument("--trigger", default="both",
                       choices=["both", "single", "double"])
        p.add_argument("--no-amd", action="store_true")
        p.add_argument("--allow-confirmed", action="store_true")
        p.add_argument("--stop-mode", default="structural",
                       choices=["structural", "fixed", "clamped"])
        p.add_argument("--stop-ticks", type=float, default=200)
        p.add_argument("--stop-min", type=float, default=200)
        p.add_argument("--stop-max", type=float, default=500)
        if dates:
            p.add_argument("--start", default="2017-01-01")
            p.add_argument("--end", default="2021-12-31")
        p.add_argument("--out")

    p = sub.add_parser("check"); p.set_defaults(func=cmd_check)

    p = sub.add_parser("data")
    p.add_argument("--raw-nq"); p.add_argument("--raw-es")
    p.add_argument("--build-ml", action="store_true")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("today"); common(p, dates=False)
    p.add_argument("--date"); p.set_defaults(func=cmd_today)

    p = sub.add_parser("scan"); common(p); p.set_defaults(func=cmd_scan)

    p = sub.add_parser("backtest"); common(p)
    p.add_argument("--slippage", type=float, default=1.0)
    p.add_argument("--no-limits", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("ml")
    p.add_argument("--data"); p.add_argument("--sample", type=int)
    p.add_argument("--task", default="classification",
                   choices=["classification", "regression"])
    p.add_argument("--rf-subsample", type=int, default=50_000)
    p.add_argument("--no-tune", action="store_true")
    p.set_defaults(func=cmd_ml)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
