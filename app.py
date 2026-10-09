"""
app.py — Streamlit interface for the Trading Decision-Support Tool.

Deployed on Railway or Streamlit Cloud. Everything the CLI does, in a browser.

DATA
    The parquet files are ~110 MB combined, which is too large to commit. The
    app looks for them in this order:
        1. ./data/            (local development)
        2. $DATA_URL          (a signed URL or public bucket, fetched on boot)
        3. file uploader      (the user supplies them in the browser)
    Nothing is required at build time, so the image stays small and the deploy
    stays fast.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit is launched from another directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (  # noqa: E402
    CONTRACT_SPECS,
    check,
    evaluate_session,
    load_parquet,
    run_backtest,
    scan_range,
    veto_summary,
)

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

    return {
        "use_amd": use_amd, "trigger_mode": trigger_mode,
        "allow_confirmed_breakout": allow_confirmed,
        "entry_start": entry_start, "entry_end": entry_end,
        "account": account, "risk": risk, "rr": rr,
        "stop_mode": stop_mode, "stop_ticks": stop_ticks,
        "stop_min": stop_min, "stop_max": stop_max,
    }


def engine_kwargs(s: dict) -> dict:
    kw = {
        "use_amd": s["use_amd"], "trigger_mode": s["trigger_mode"],
        "allow_confirmed_breakout": s["allow_confirmed_breakout"],
        "entry_start": s["entry_start"], "entry_end": s["entry_end"],
        "rr": s["rr"], "risk_pct": s["risk"], "stop_mode": s["stop_mode"],
    }
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
    c1, _ = st.columns([1, 3])
    d = c1.date_input("Session", value=idx.max().date(),
                      min_value=idx.min().date(), max_value=idx.max().date())

    if not st.button("Evaluate", type="primary"):
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
                          min_value=idx.min().date(), max_value=idx.max().date())
    end = c2.date_input("To", value=min(idx.max().date(), date(2021, 12, 31)),
                        min_value=idx.min().date(), max_value=idx.max().date())

    st.caption("Roughly 3–4 minutes per year of data.")
    if not st.button("Run backtest", type="primary"):
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
    sym = c1.selectbox("Instrument", ["NQ", "ES"])
    direction = c2.selectbox("Direction", ["bull", "bear"])
    when = c3.text_input("Time (HH:MM)", "10:15")

    c4, c5, c6 = st.columns(3)
    entry = c4.number_input("Entry", value=0.0, step=0.25, format="%.2f")
    stop = c5.number_input("Stop (0 = use ticks)", value=0.0, step=0.25, format="%.2f")
    ticks = c6.number_input("…or stop ticks", 10, 1000, 200, 10)

    idx = data["NQ"].index
    d = st.date_input("Session", value=idx.max().date(),
                      min_value=idx.min().date(), max_value=idx.max().date())

    if not st.button("Check", type="primary"):
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
                             "this session only.")
    provider = st.selectbox("Provider", ["anthropic", "openai", "google"])
    up = st.file_uploader("Chart screenshot", type=["png", "jpg", "jpeg", "webp"])

    if up:
        st.image(up, use_container_width=True)
    if not (up and key):
        st.info("Upload a screenshot and supply a key.")
        return
    if not st.button("Read chart", type="primary"):
        return

    from src.chart_reader import read_chart
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
        up = st.file_uploader("ML dataset (parquet)", type="parquet")
        if not up:
            st.info("Upload ml_dataset.parquet, or build it from the 1-minute "
                    "files with the CLI: `python cli.py data --build-ml`.")
            return

    task = st.selectbox("Task", ["classification", "regression"])
    sample = st.number_input("Sample size (0 = all)", 0, 400_000, 60_000, 10_000)
    tune = st.checkbox("Sweep max_depth (requirement 8)", True)

    if not st.button("Train", type="primary"):
        return

    from src import ml_pipeline
    if up:
        path = "/tmp/ml_dataset.parquet"
        with open(path, "wb") as f:
            f.write(up.getvalue())

    with st.spinner("Training five models…"):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ml_pipeline.run(data=path, task=task,
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


if __name__ == "__main__":
    main()
