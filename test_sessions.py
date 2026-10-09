"""
test_sessions.py — tests for sessions.py.

Run:  pytest test_sessions.py -v
"""

import pandas as pd

from src.sessions import (
    Level,
    assign_session,
    assign_trading_day,
    build_liquidity_map,
    cluster_levels,
    daily_extremes,
    has_liquidity_target,
    liquidity_report,
    mark_tapped,
    session_extremes,
    untapped_targets,
)

TZ = "America/New_York"


def bars(specs):
    """specs: list of (timestamp_str, open, high, low, close)."""
    idx = pd.DatetimeIndex([pd.Timestamp(s[0], tz=TZ) for s in specs])
    return pd.DataFrame(
        [s[1:] for s in specs],
        columns=["open", "high", "low", "close"],
        index=idx,
    ).sort_index()


# ---------- trading day / session assignment ----------

def test_evening_bars_roll_to_next_trading_day():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-05 17:30", tz=TZ),   # break, still Jan 5
        pd.Timestamp("2026-01-05 18:00", tz=TZ),   # Asia -> Jan 6
        pd.Timestamp("2026-01-05 23:00", tz=TZ),   # Asia -> Jan 6
        pd.Timestamp("2026-01-06 02:00", tz=TZ),   # Asia -> Jan 6
        pd.Timestamp("2026-01-06 10:00", tz=TZ),   # NY   -> Jan 6
    ])
    td = assign_trading_day(idx)
    assert str(td.iloc[0].date()) == "2026-01-05"
    assert str(td.iloc[1].date()) == "2026-01-06"
    assert str(td.iloc[2].date()) == "2026-01-06"
    assert str(td.iloc[3].date()) == "2026-01-06"
    assert str(td.iloc[4].date()) == "2026-01-06"


def test_session_labels():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-05 19:00", tz=TZ),   # asia
        pd.Timestamp("2026-01-06 01:00", tz=TZ),   # asia (past midnight)
        pd.Timestamp("2026-01-06 04:00", tz=TZ),   # london
        pd.Timestamp("2026-01-06 08:00", tz=TZ),   # london
        pd.Timestamp("2026-01-06 09:30", tz=TZ),   # ny
        pd.Timestamp("2026-01-06 17:30", tz=TZ),   # break
    ])
    s = assign_session(idx)
    assert list(s) == ["asia", "asia", "london", "london", "ny", "break"]


def test_session_boundaries_are_inclusive_lower():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-06 03:00", tz=TZ),   # london starts
        pd.Timestamp("2026-01-06 08:30", tz=TZ),   # ny starts
        pd.Timestamp("2026-01-06 18:00", tz=TZ),   # asia starts
    ])
    assert list(assign_session(idx)) == ["london", "ny", "asia"]


# ---------- session extremes ----------

def _two_session_day():
    return bars([
        ("2026-01-05 19:00", 100, 105,  99, 103),   # asia -> Jan 6
        ("2026-01-05 22:00", 103, 108,  97, 101),   # asia  (high 108, low 97)
        ("2026-01-06 04:00", 101, 106, 100, 104),   # london
        ("2026-01-06 06:00", 104, 110, 102, 108),   # london (high 110, low 100)
        ("2026-01-06 09:30", 108, 112, 106, 110),   # ny
        ("2026-01-06 10:00", 110, 113, 105, 107),   # ny (high 113, low 105)
    ])


def test_session_extremes_split_correctly():
    se = session_extremes(_two_session_day())
    asia = se[se["session"] == "asia"].iloc[0]
    london = se[se["session"] == "london"].iloc[0]
    ny = se[se["session"] == "ny"].iloc[0]

    assert asia["high"] == 108 and asia["low"] == 97
    assert london["high"] == 110 and london["low"] == 100
    assert ny["high"] == 113 and ny["low"] == 105
    # all three belong to the Jan 6 trading day
    assert se["trading_day"].nunique() == 1


def test_break_bars_excluded():
    df = bars([
        ("2026-01-06 10:00", 100, 105, 99, 103),
        ("2026-01-06 17:30", 103, 200, 50, 104),   # spread hour, absurd values
    ])
    se = session_extremes(df)
    assert set(se["session"]) == {"ny"}
    assert se.iloc[0]["high"] == 105     # the 17:30 bar is ignored


def test_daily_extremes_span_all_sessions():
    d = daily_extremes(_two_session_day())
    assert len(d) == 1
    assert d.iloc[0]["high"] == 113
    assert d.iloc[0]["low"] == 97


# ---------- liquidity map ----------

def _two_day_frame():
    day1 = bars([
        ("2026-01-05 19:00", 100, 105,  99, 103),
        ("2026-01-06 04:00", 103, 106, 100, 104),
        ("2026-01-06 10:00", 104, 120,  90, 110),   # prev-day extremes 120 / 90
    ])
    day2 = bars([
        ("2026-01-06 19:00", 110, 115, 108, 112),   # asia  -> Jan 7
        ("2026-01-07 04:00", 112, 118, 111, 114),   # london -> Jan 7
        ("2026-01-07 09:30", 114, 116, 113, 115),   # ny
    ])
    return pd.concat([day1, day2])


def test_liquidity_map_contents():
    df = _two_day_frame()
    levels = build_liquidity_map(df, "2026-01-07")
    names = {lv.name for lv in levels}
    assert names == {
        "asia_high", "asia_low",
        "london_high", "london_low",
        "prevday_high", "prevday_low",
    }
    by = {lv.name: lv.price for lv in levels}
    assert by["asia_high"] == 115 and by["asia_low"] == 108
    assert by["london_high"] == 118 and by["london_low"] == 111
    assert by["prevday_high"] == 120 and by["prevday_low"] == 90


def test_ny_excluded_by_default_no_lookahead():
    df = _two_day_frame()
    assert all(lv.source != "ny" for lv in build_liquidity_map(df, "2026-01-07"))
    with_ny = build_liquidity_map(df, "2026-01-07", include_ny=True)
    assert any(lv.source == "ny" for lv in with_ny)


def test_first_day_has_no_prevday_levels():
    df = _two_day_frame()
    levels = build_liquidity_map(df, "2026-01-06")
    assert all("prevday" not in lv.name for lv in levels)


# ---------- tapping ----------

def test_mark_tapped_wick_counts():
    df = bars([
        ("2026-01-06 19:00", 100, 110,  95, 105),   # asia -> Jan 7, high 110
        ("2026-01-07 04:00", 105, 108, 100, 106),   # london
        ("2026-01-07 09:30", 106, 111, 104, 108),   # NY wick 111 taps asia_high
    ])
    levels = mark_tapped(build_liquidity_map(df, "2026-01-07"), df)
    asia_high = next(lv for lv in levels if lv.name == "asia_high")
    assert asia_high.tapped
    assert asia_high.tapped_at == df.index[2]


def test_untapped_level_stays_false():
    df = bars([
        ("2026-01-06 19:00", 100, 110,  95, 105),   # asia high 110
        ("2026-01-07 04:00", 105, 108, 100, 106),
        ("2026-01-07 09:30", 106, 109, 104, 108),   # never reaches 110
    ])
    levels = mark_tapped(build_liquidity_map(df, "2026-01-07"), df)
    asia_high = next(lv for lv in levels if lv.name == "asia_high")
    assert not asia_high.tapped and asia_high.tapped_at is None


def test_up_to_bounds_the_scan():
    df = bars([
        ("2026-01-06 19:00", 100, 110,  95, 105),
        ("2026-01-07 04:00", 105, 108, 100, 106),
        ("2026-01-07 09:30", 106, 107, 104, 105),
        ("2026-01-07 10:00", 105, 115, 104, 112),   # taps 110 here
    ])
    levels = build_liquidity_map(df, "2026-01-07")

    early = mark_tapped(list(levels), df, up_to=df.index[2])
    assert not next(lv for lv in early if lv.name == "asia_high").tapped

    later = mark_tapped(build_liquidity_map(df, "2026-01-07"), df, up_to=df.index[3])
    assert next(lv for lv in later if lv.name == "asia_high").tapped


def test_level_not_tapped_by_its_own_session():
    """The bar that MADE the high must not count as tapping it."""
    df = bars([
        ("2026-01-06 19:00", 100, 110,  95, 105),   # makes asia high 110
        ("2026-01-07 04:00", 105, 106, 100, 102),   # london stays below
    ])
    levels = mark_tapped(build_liquidity_map(df, "2026-01-07"), df)
    assert not next(lv for lv in levels if lv.name == "asia_high").tapped


# ---------- targets and the §9 gate ----------

def _levels_fixture():
    t = pd.Timestamp("2026-01-07 08:00", tz=TZ)
    return [
        Level("asia_high", 115, "high", "asia", t, t),
        Level("london_high", 118, "high", "london", t, t),
        Level("prevday_high", 130, "high", "prevday", t, t, tapped=True),
        Level("asia_low", 108, "low", "asia", t, t),
        Level("london_low", 100, "low", "london", t, t),
    ]


def test_untapped_targets_bull_sorted_nearest_first():
    tg = untapped_targets(_levels_fixture(), "bull", current_price=112)
    assert [lv.name for lv in tg] == ["asia_high", "london_high"]
    assert tg[0].price < tg[1].price          # nearest first
    assert all(lv.name != "prevday_high" for lv in tg)   # tapped excluded


def test_untapped_targets_bear_sorted_nearest_first():
    tg = untapped_targets(_levels_fixture(), "bear", current_price=112)
    assert [lv.name for lv in tg] == ["asia_low", "london_low"]
    assert tg[0].price > tg[1].price          # nearest below first


def test_targets_exclude_levels_behind_price():
    tg = untapped_targets(_levels_fixture(), "bull", current_price=119)
    assert tg == []                           # both untapped highs are below


def test_gate_blocks_when_no_target():
    levels = _levels_fixture()
    assert has_liquidity_target(levels, "bull", 112)
    assert not has_liquidity_target(levels, "bull", 119)
    assert has_liquidity_target(levels, "bear", 112)
    assert not has_liquidity_target(levels, "bear", 99)


def test_bad_direction_raises():
    try:
        untapped_targets(_levels_fixture(), "sideways", 100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------- clustering ----------

def test_cluster_groups_nearby_levels():
    t = pd.Timestamp("2026-01-07 08:00", tz=TZ)
    levels = [
        Level("asia_high", 115.0, "high", "asia", t, t),
        Level("london_high", 115.5, "high", "london", t, t),
        Level("prevday_high", 140.0, "high", "prevday", t, t),
    ]
    cl = cluster_levels(levels, tolerance=1.0)
    assert len(cl) == 1
    assert cl[0]["size"] == 2
    assert set(cl[0]["members"]) == {"asia_high", "london_high"}
    assert abs(cl[0]["price"] - 115.25) < 1e-9


def test_cluster_ignores_isolated_levels():
    t = pd.Timestamp("2026-01-07 08:00", tz=TZ)
    levels = [
        Level("a", 100.0, "high", "asia", t, t),
        Level("b", 120.0, "high", "london", t, t),
    ]
    assert cluster_levels(levels, tolerance=1.0) == []


def test_cluster_tracks_untapped_state():
    t = pd.Timestamp("2026-01-07 08:00", tz=TZ)
    levels = [
        Level("a", 100.0, "high", "asia", t, t, tapped=True),
        Level("b", 100.2, "high", "london", t, t, tapped=True),
    ]
    assert cluster_levels(levels, tolerance=1.0)[0]["any_untapped"] is False


# ---------- report ----------

def test_liquidity_report_sorted_desc():
    rep = liquidity_report(_levels_fixture())
    assert list(rep["price"]) == sorted(rep["price"], reverse=True)
    assert set(rep.columns) >= {"name", "side", "price", "tapped"}
