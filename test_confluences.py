"""
test_confluences.py — tests for confluences.py.

Run:  pytest test_confluences.py -v
"""

import pandas as pd

from src.confluences import (
    FVG,
    SMT,
    active_fvgs,
    confluence_check,
    detect_fvgs,
    detect_ifvg_signals,
    detect_smt,
    invalidation_gap,
    smt_report,
    stack_fvgs,
    update_fvg_state,
)
from src.swings import detect_swings, label_swings

TZ = "America/New_York"


def make_df(candles, start="2026-01-05 09:30", freq="5min"):
    idx = pd.date_range(start=start, periods=len(candles), freq=freq, tz=TZ)
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"], index=idx)


# ---------- FVG detection ----------

def test_bullish_fvg_detected():
    # c1 high 105, c3 low 110 -> gap [105, 110]
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),   # expansion
        (116, 120, 110, 118),
    ])
    f = detect_fvgs(df)
    assert len(f) == 1
    assert f[0].direction == "bull"
    assert f[0].bottom == 105 and f[0].top == 110
    assert f[0].size == 5
    assert f[0].formed_at == df.index[2]   # known only at candle 3


def test_bearish_fvg_detected():
    # c1 low 110, c3 high 105 -> gap [105, 110]
    df = make_df([
        (118, 120, 110, 112),
        (112, 113,  98, 100),   # expansion down
        (100, 105,  95,  98),
    ])
    f = detect_fvgs(df)
    assert len(f) == 1
    assert f[0].direction == "bear"
    assert f[0].bottom == 105 and f[0].top == 110


def test_wick_overlap_is_not_a_gap():
    df = make_df([
        (100, 110,  99, 108),
        (108, 118, 107, 116),
        (116, 120, 109, 118),   # c3 low 109 < c1 high 110 -> overlap
    ])
    assert detect_fvgs(df) == []


def test_candle_colour_irrelevant():
    """First and third candle direction must not matter."""
    down_c1_c3 = make_df([
        (105, 105,  99, 100),   # down candle
        (100, 118,  99, 116),
        (120, 120, 110, 112),   # down candle
    ])
    f = detect_fvgs(down_c1_c3)
    assert len(f) == 1 and f[0].direction == "bull"


def test_min_size_filter():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 106, 118),   # gap of only 1
    ])
    assert len(detect_fvgs(df, min_size=0.0)) == 1
    assert detect_fvgs(df, min_size=5.0) == []


def test_too_few_candles():
    df = make_df([(100, 105, 99, 104), (104, 118, 103, 116)])
    assert detect_fvgs(df) == []


# ---------- FVG lifecycle ----------

def test_fvg_tapped_by_wick():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),   # bull gap [105, 110]
        (118, 119, 108, 117),   # wick to 108 -> inside the zone
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    assert f[0].tapped
    assert f[0].tapped_at == df.index[3]
    assert not f[0].inverted


def test_fvg_inverted_by_close_not_wick():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),   # bull gap [105, 110]
        (118, 119, 100, 107),   # wick BELOW 105 but closes 107 -> not inverted
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    assert f[0].tapped and not f[0].inverted

    df2 = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),
        (118, 119, 100, 102),   # CLOSES 102 < 105 -> inverted
    ])
    f2 = update_fvg_state(detect_fvgs(df2), df2)
    assert f2[0].inverted
    assert f2[0].inverted_at == df2.index[3]
    assert not f2[0].alive


def test_bear_fvg_inverted_by_close_above():
    df = make_df([
        (118, 120, 110, 112),
        (112, 113,  98, 100),
        (100, 105,  95,  98),   # bear gap [105, 110]
        ( 98, 115,  97, 112),   # closes 112 > 110 -> inverted
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    assert f[0].inverted


def test_up_to_bounds_state_scan():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),
        (118, 119, 115, 117),   # nothing yet
        (117, 118, 100, 102),   # inverts here
    ])
    early = update_fvg_state(detect_fvgs(df), df, up_to=df.index[3])
    assert not early[0].inverted
    late = update_fvg_state(detect_fvgs(df), df, up_to=df.index[4])
    assert late[0].inverted


def test_active_fvgs_filters_inverted_and_direction():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),   # bull gap
        (118, 119, 100, 102),   # inverts it
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    assert active_fvgs(f) == []
    assert active_fvgs(f, direction="bear") == []


# ---------- stacking ----------

def test_stack_groups_contiguous_gaps():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 122, 110, 120),   # gap1 [105,110]
        (120, 135, 119, 133),
        (133, 140, 125, 138),   # gap2 [122,125] — contiguous
    ])
    f = detect_fvgs(df)
    assert len(f) >= 2
    stacks = stack_fvgs(f, df)
    assert any(len(s) >= 2 for s in stacks)


def test_invalidation_gap_is_lowest_for_bull_stack():
    t = pd.Timestamp("2026-01-05 09:30", tz=TZ)
    stack = [
        FVG("bull", top=110, bottom=105, formed_at=t, c1_at=t, c3_at=t),
        FVG("bull", top=125, bottom=122, formed_at=t, c1_at=t, c3_at=t),
    ]
    assert invalidation_gap(stack).bottom == 105


def test_invalidation_gap_is_highest_for_bear_stack():
    t = pd.Timestamp("2026-01-05 09:30", tz=TZ)
    stack = [
        FVG("bear", top=110, bottom=105, formed_at=t, c1_at=t, c3_at=t),
        FVG("bear", top=95, bottom=90, formed_at=t, c1_at=t, c3_at=t),
    ]
    assert invalidation_gap(stack).top == 110


# ---------- IFVG signals ----------

def test_ifvg_signal_flips_direction():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),   # BULL gap
        (118, 119, 100, 102),   # inverted
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    sig = detect_ifvg_signals(f)
    assert len(sig) == 1
    assert sig.iloc[0]["direction"] == "bear"          # flipped
    assert sig.iloc[0]["original_direction"] == "bull"
    assert sig.iloc[0]["timestamp"] == df.index[3]


def test_no_ifvg_when_nothing_inverted():
    df = make_df([
        (100, 105,  99, 104),
        (104, 118, 103, 116),
        (116, 120, 110, 118),
    ])
    f = update_fvg_state(detect_fvgs(df), df)
    assert detect_ifvg_signals(f).empty


# ---------- SMT ----------

def _swing_frame(rows, start="2026-01-05 09:30"):
    """rows: list of (type, price, offset_minutes)."""
    base = pd.Timestamp(start, tz=TZ)
    df = pd.DataFrame({
        "type": [r[0] for r in rows],
        "price": [float(r[1]) for r in rows],
        "extreme_at": [base + pd.Timedelta(minutes=r[2]) for r in rows],
        "confirmed_at": [base + pd.Timedelta(minutes=r[2] + 1) for r in rows],
    })
    return label_swings(df)


def test_bearish_smt_lower_high_leads():
    # NQ makes HH, ES makes LH -> ES leads down
    nq = _swing_frame([("high", 100, 0), ("high", 110, 10)])   # HH
    es = _swing_frame([("high", 100, 0), ("high",  95, 10)])   # LH
    smts = detect_smt(nq, es, "NQ", "ES")
    assert len(smts) == 1
    assert smts[0].direction == "bear"
    assert smts[0].leading == "ES"
    assert smts[0].lagging == "NQ"


def test_bullish_smt_higher_low_leads():
    # NQ makes LL, ES makes HL -> ES leads up
    nq = _swing_frame([("low", 100, 0), ("low",  90, 10)])   # LL
    es = _swing_frame([("low", 100, 0), ("low", 105, 10)])   # HL
    smts = detect_smt(nq, es, "NQ", "ES")
    assert len(smts) == 1
    assert smts[0].direction == "bull"
    assert smts[0].leading == "ES"


def test_no_smt_when_both_agree():
    nq = _swing_frame([("high", 100, 0), ("high", 110, 10)])   # HH
    es = _swing_frame([("high", 100, 0), ("high", 112, 10)])   # HH
    assert detect_smt(nq, es) == []


def test_smt_requires_timestamp_alignment():
    nq = _swing_frame([("high", 100, 0), ("high", 110, 10)])
    es = _swing_frame([("high", 100, 0), ("high",  95, 60)])   # an hour later
    assert detect_smt(nq, es, tolerance_bars=1, bar_minutes=5) == []


def test_smt_tolerance_allows_one_bar_drift():
    nq = _swing_frame([("high", 100, 0), ("high", 110, 10)])
    es = _swing_frame([("high", 100, 0), ("high",  95, 11)])   # 1 min later
    assert len(detect_smt(nq, es, tolerance_bars=1, bar_minutes=1)) == 1


def test_smt_report_shape():
    nq = _swing_frame([("high", 100, 0), ("high", 110, 10)])
    es = _swing_frame([("high", 100, 0), ("high",  95, 10)])
    rep = smt_report(detect_smt(nq, es))
    assert set(rep.columns) >= {"timestamp", "direction", "leading", "lagging"}


# ---------- combined gate ----------

def test_confluence_passes_on_fvg_tap():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    f = [FVG("bull", top=110, bottom=105, formed_at=t, c1_at=t, c3_at=t)]
    res = confluence_check("bull", as_of=t, price=107, fvgs=f)
    assert res["pass"] and "fvg_tap" in res["reasons"]


def test_confluence_fails_with_nothing():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    res = confluence_check("bull", as_of=t, price=107)
    assert not res["pass"] and res["reasons"] == []


def test_confluence_passes_on_recent_bos():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    bos = pd.DataFrame({
        "timestamp": [t - pd.Timedelta(minutes=1)],
        "direction": ["bull"],
        "level": [100.0],
        "swing_confirmed_at": [t - pd.Timedelta(minutes=5)],
    })
    res = confluence_check("bull", as_of=t, price=107, bos=bos, bos_window_bars=3)
    assert res["pass"] and "bos" in res["reasons"]


def test_confluence_ignores_stale_bos():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    bos = pd.DataFrame({
        "timestamp": [t - pd.Timedelta(minutes=30)],
        "direction": ["bull"],
        "level": [100.0],
        "swing_confirmed_at": [t - pd.Timedelta(minutes=35)],
    })
    res = confluence_check("bull", as_of=t, price=107, bos=bos, bos_window_bars=3)
    assert not res["pass"]


def test_confluence_ignores_opposite_direction():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    f = [FVG("bear", top=110, bottom=105, formed_at=t, c1_at=t, c3_at=t)]
    res = confluence_check("bull", as_of=t, price=107, fvgs=f)
    assert not res["pass"]


def test_confluence_collects_multiple_reasons():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    f = [FVG("bull", top=110, bottom=105, formed_at=t, c1_at=t, c3_at=t)]
    bos = pd.DataFrame({
        "timestamp": [t],
        "direction": ["bull"],
        "level": [100.0],
        "swing_confirmed_at": [t],
    })
    smts = [SMT("bull", t, "ES", "NQ", 100.0, 95.0, "low")]
    res = confluence_check("bull", as_of=t, price=107, fvgs=f, bos=bos, smts=smts)
    assert res["pass"]
    assert set(res["reasons"]) == {"fvg_tap", "bos", "smt"}


def test_confluence_no_lookahead():
    """A confluence that forms later must not count now."""
    now = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    later = now + pd.Timedelta(minutes=30)
    smts = [SMT("bull", later, "ES", "NQ", 100.0, 95.0, "low")]
    assert not confluence_check("bull", as_of=now, price=107, smts=smts)["pass"]


def test_bad_direction_raises():
    t = pd.Timestamp("2026-01-05 10:00", tz=TZ)
    try:
        confluence_check("sideways", as_of=t, price=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------- integration with real swing detection ----------

def test_smt_from_detected_swings():
    """End-to-end: raw candles -> swings -> labels -> SMT."""
    nq_df = make_df([
        (100, 105,  99, 104),
        (104, 106,  98,  99),   # swing high 106
        ( 99, 103,  97, 102),
        (102, 112, 101, 111),
        (111, 115, 108, 109),   # swing high 115 -> HH
    ], freq="1min")
    es_df = make_df([
        (100, 105,  99, 104),
        (104, 106,  98,  99),   # swing high 106
        ( 99, 103,  97, 102),
        (102, 104, 101, 103),
        (103, 104, 100, 101),   # swing high 104 -> LH
    ], freq="1min")

    nq_sw = label_swings(detect_swings(nq_df))
    es_sw = label_swings(detect_swings(es_df))
    smts = detect_smt(nq_sw, es_sw, "NQ", "ES", tolerance_bars=1, bar_minutes=1)

    assert len(smts) == 1
    assert smts[0].direction == "bear"
    assert smts[0].leading == "ES"     # ES made the lower high
