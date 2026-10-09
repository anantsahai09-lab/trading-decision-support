"""
test_opening_range.py — tests for opening_range.py using hand-built 5m candles.

Run:  pytest test_opening_range.py -v
"""

import pandas as pd

from src.opening_range import (
    build_all_ranges,
    build_opening_range,
    compare_cleanliness,
    detect_double_test,
    find_episodes,
    range_violations,
    track_interactions,
)

TZ = "America/New_York"


def session(candles, date="2026-01-05", start="09:30", freq="5min"):
    """candles: list of (open, high, low, close) starting at `start`."""
    idx = pd.date_range(start=f"{date} {start}", periods=len(candles),
                        freq=freq, tz=TZ)
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"], index=idx)


# ---------- range construction ----------

def test_range_from_two_bars():
    df = session([
        (100, 105, 98, 103),   # 09:30  high 105 low 98
        (103, 107, 101, 104),  # 09:35  high 107 low 101
        (104, 106, 102, 105),  # 09:40
    ])
    r = build_opening_range(df)
    assert r.high == 107
    assert r.low == 98
    assert r.height == 9
    assert r.n_bars == 2
    assert r.formed_at == df.index[1]
    assert r.usable_from == df.index[2]   # 09:40 is the first usable bar


def test_individual_wick_levels_recorded():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),
        (104, 106, 102, 105),
    ])
    r = build_opening_range(df)
    assert r.levels["09:30_high"] == 105
    assert r.levels["09:30_low"] == 98
    assert r.levels["09:35_high"] == 107
    assert r.levels["09:35_low"] == 101


def test_missing_formation_bars_returns_none():
    # session starts late (10:00) -> no 09:30/09:35 bars
    df = session([(100, 105, 98, 103), (103, 107, 101, 104)], start="10:00")
    assert build_opening_range(df) is None


def test_mid_and_position():
    df = session([
        (100, 110, 100, 105),
        (105, 110, 90, 100),
        (100, 101, 99, 100),
    ])
    r = build_opening_range(df)
    assert r.high == 110 and r.low == 90
    assert r.mid == 100
    assert r.position(115) == "above"
    assert r.position(85) == "below"
    assert r.position(100) == "inside"


def test_build_all_ranges_multiple_days():
    d1 = session([(100, 105, 98, 103), (103, 107, 101, 104)], date="2026-01-05")
    d2 = session([(200, 205, 198, 203), (203, 207, 201, 204)], date="2026-01-06")
    df = pd.concat([d1, d2])
    ranges = build_all_ranges(df)
    assert len(ranges) == 2
    highs = sorted(r.high for r in ranges.values())
    assert highs == [107, 207]


# ---------- interactions ----------

def test_no_lookahead_formation_bars_excluded():
    df = session([
        (100, 105, 98, 103),   # 09:30 formation
        (103, 107, 101, 104),  # 09:35 formation
        (104, 106, 102, 105),  # 09:40 first usable
    ])
    r = build_opening_range(df)
    inter = track_interactions(df, r)
    assert len(inter) == 1
    assert inter.index[0] == df.index[2]


def test_touch_vs_breach():
    df = session([
        (100, 105, 98, 103),   # formation
        (103, 107, 101, 104),  # formation -> range 98..107
        (104, 107, 102, 105),  # wick to 107, close inside -> TOUCH high
        (105, 110, 104, 109),  # close 109 > 107 -> BREACH high
        (109, 110, 96, 97),    # close 97 < 98 -> BREACH low
    ])
    r = build_opening_range(df)
    inter = track_interactions(df, r)
    assert bool(inter.iloc[0]["touch_high"]) and not bool(inter.iloc[0]["breach_high"])
    assert bool(inter.iloc[1]["breach_high"])
    assert inter.iloc[1]["position"] == "above"
    assert bool(inter.iloc[2]["breach_low"])
    assert inter.iloc[2]["position"] == "below"


def test_excursion_measures_wick_beyond_level():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),   # range 98..107
        (104, 112, 103, 109),   # breach high, wick 112 -> excursion 5
    ])
    r = build_opening_range(df)
    inter = track_interactions(df, r)
    assert inter.iloc[0]["excursion"] == 5


# ---------- episodes ----------

def test_single_episode_then_return():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 112, 103, 110),    # breach high (extreme 112)
        (110, 113, 108, 111),    # still above (extreme 113)
        (111, 112, 100, 102),    # back inside -> episode closes
    ])
    r = build_opening_range(df)
    eps = find_episodes(df, r)
    assert len(eps) == 1
    assert eps[0].side == "high"
    assert eps[0].extreme == 113
    assert eps[0].max_excursion == 6      # 113 - 107
    assert eps[0].n_bars == 2
    assert eps[0].returned_at == df.index[4]


def test_two_episodes_same_side():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 112, 103, 110),    # ep1 breach high, excursion 5
        (110, 111, 100, 102),    # back inside
        (102, 110, 101, 109),    # ep2 breach high, excursion 3
        (109, 110, 100, 101),    # back inside
    ])
    r = build_opening_range(df)
    eps = find_episodes(df, r)
    assert len(eps) == 2
    assert [e.side for e in eps] == ["high", "high"]
    assert eps[0].max_excursion == 5
    assert eps[1].max_excursion == 3


def test_side_flip_without_returning_inside():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 112, 103, 110),    # above
        (110, 111, 90, 92),      # straight to below, no inside close
    ])
    r = build_opening_range(df)
    eps = find_episodes(df, r)
    assert len(eps) == 2
    assert [e.side for e in eps] == ["high", "low"]


# ---------- double test (§3.1) ----------

def test_failed_breakout_shallower_second_test():
    """Second push is shallower -> fake breakout -> expect the opposite way."""
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 105, 92, 93),      # ep1 below, low 92 -> excursion 6
        ( 93, 104, 92, 102),     # back inside
        (102, 103, 95, 96),      # ep2 below, low 95 -> excursion 3 (shallower)
        ( 96, 104, 95, 103),     # back inside -> signal here
    ])
    r = build_opening_range(df)
    sig = detect_double_test(df, r)
    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["side_tested"] == "low"
    assert row["signal"] == "failed_breakout"
    assert row["direction"] == "bull"          # broke down twice, failed -> up
    assert row["first_excursion"] == 6
    assert row["second_excursion"] == 3
    assert row["timestamp"] == df.index[5]     # the bar that closed back inside


def test_confirmed_breakout_deeper_second_test():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 110, 103, 109),    # ep1 above, excursion 3
        (109, 110, 100, 102),    # back inside
        (102, 115, 101, 114),    # ep2 above, excursion 8 (deeper)
    ])
    r = build_opening_range(df)
    sig = detect_double_test(df, r)
    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["signal"] == "confirmed_breakout"
    assert row["direction"] == "bull"
    assert row["second_excursion"] > row["first_excursion"]


def test_first_breach_alone_emits_nothing():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),
        (104, 112, 103, 110),    # single breach only
        (110, 111, 100, 102),
    ])
    r = build_opening_range(df)
    assert detect_double_test(df, r).empty


def test_third_test_does_not_re_signal():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 112, 103, 110),    # ep1 exc 5
        (110, 111, 100, 102),
        (102, 110, 101, 109),    # ep2 exc 3 -> failed_breakout
        (109, 110, 100, 101),
        (101, 109, 100, 108),    # ep3 — must not emit again
        (108, 109, 100, 102),
    ])
    r = build_opening_range(df)
    sig = detect_double_test(df, r)
    assert len(sig) == 1


def test_require_return_false_signals_earlier():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),
        (104, 105, 92, 93),      # ep1 exc 6
        ( 93, 104, 92, 102),
        (102, 103, 95, 96),      # ep2 exc 3, extreme at this bar
        ( 96, 104, 95, 103),     # returns inside
    ])
    r = build_opening_range(df)
    early = detect_double_test(df, r, require_return=False)
    late = detect_double_test(df, r, require_return=True)
    assert early.iloc[0]["timestamp"] == df.index[4]
    assert late.iloc[0]["timestamp"] == df.index[5]
    assert early.iloc[0]["timestamp"] < late.iloc[0]["timestamp"]


def test_both_sides_tracked_independently():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),    # range 98..107
        (104, 112, 103, 110),    # high ep1 exc 5
        (110, 111, 100, 102),
        (102, 110, 101, 109),    # high ep2 exc 3 -> failed (bear)
        (109, 110, 100, 101),
        (101, 102, 92, 93),      # low ep1 exc 6
        ( 93, 104, 92, 102),
        (102, 103, 95, 96),      # low ep2 exc 3 -> failed (bull)
        ( 96, 104, 95, 103),
    ])
    r = build_opening_range(df)
    sig = detect_double_test(df, r)
    assert len(sig) == 2
    assert set(sig["side_tested"]) == {"high", "low"}
    assert set(sig["direction"]) == {"bear", "bull"}


# ---------- §4 cleanliness proxy ----------

def test_range_violations_counts_breaches():
    df = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),   # range 98..107
        (104, 112, 103, 110),   # breach
        (110, 111, 100, 102),   # inside
        (102, 110, 101, 109),   # breach
    ])
    r = build_opening_range(df)
    assert range_violations(df, r, lookback=12) == 2


def test_compare_cleanliness_ranks_quieter_instrument_first():
    noisy = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),
        (104, 112, 103, 110),   # breach
        (110, 111, 100, 109),   # breach
    ])
    calm = session([
        (100, 105, 98, 103),
        (103, 107, 101, 104),
        (104, 106, 102, 105),   # inside
        (105, 106, 103, 104),   # inside
    ])
    ranges = {"NOISY": build_opening_range(noisy), "CALM": build_opening_range(calm)}
    out = compare_cleanliness({"NOISY": noisy, "CALM": calm}, ranges)
    assert out.iloc[0]["symbol"] == "CALM"
    assert out.iloc[0]["violations"] == 0
    assert out.iloc[1]["violations"] == 2
