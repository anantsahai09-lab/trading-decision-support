"""
dataset.py — STEP 1: build a clear supervised dataset (X, y) from bar data.

WHY BARS AND NOT TRADES
    The backtest produces ~400 setups across 10 years. Training five model types
    with hyperparameter tuning on 400 rows fits noise, not signal. Bar data gives
    ~700k 5-minute bars per instrument — enough that the models are learning
    structure rather than memorising.

WHAT THE MODEL LEARNS
    "Given how price has been behaving over the last N bars, what happens next?"

    This directly operationalises two open questions in the strategy spec that
    could not be written as rules:
      §7  accumulation / distribution — what does "price clumps up" actually mean
      §4  "cleaner price action"      — what does clean vs choppy actually mean

INPUT  X  (17 features, all computed from bars at or BEFORE time t)
    Compression / clumping        range_ratio, body_ratio, overlap_ratio,
                                  close_dispersion, high_low_std
    Direction / cleanliness       persistence, net_move_atr, wick_ratio,
                                  up_bar_frac, return_autocorr
    Volatility                    atr_norm, atr_change, true_range_norm
    Volume                        vol_ratio, vol_trend
    Session context               minutes_from_open, dist_from_or_mid

OUTPUT y
    regression      forward return over the next H bars, normalised by ATR
                    (an ATR-normalised target is comparable across 2016 prices
                    and 2026 prices — raw points are not)
    classification  1 if |forward move| exceeds `expansion_k` * ATR  (EXPANSION)
                    0 otherwise                                     (COMPRESSION)

NO LOOKAHEAD
    Features use a trailing window ending at bar t. The label uses bars t+1 to
    t+H. The two never overlap. The final H rows of each session are dropped,
    since their labels would need bars from the next session.

SESSION-AWARE
    Every window is built inside a single trading day. No feature or label ever
    spans the overnight break, which would otherwise create fake gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    # compression / clumping  (§7 accumulation-distribution)
    "range_ratio",
    "body_ratio",
    "overlap_ratio",
    "close_dispersion",
    "high_low_std",
    # direction / cleanliness  (§4 "cleaner price action")
    "persistence",
    "net_move_atr",
    "wick_ratio",
    "up_bar_frac",
    "return_autocorr",
    # volatility
    "atr_norm",
    "atr_change",
    "true_range_norm",
    # volume
    "vol_ratio",
    "vol_trend",
    # session context
    "minutes_from_open",
    "dist_from_or_mid",
]


@dataclass
class DatasetSpec:
    """Every choice that defines the dataset, in one inspectable object."""

    lookback: int = 12          # bars of history per sample (12 x 5m = 1 hour)
    horizon: int = 6            # bars ahead to label   (6 x 5m = 30 minutes)
    atr_window: int = 14
    expansion_k: float = 1.62    # |excursion| > k * ATR -> class 1.
                                 # 1.62 is the empirical median excursion over
                                 # 300k samples, so the classes come out ~50/50.
                                 # At k=1.0 the split is 86/14 and a model that
                                 # always predicts "expansion" scores 86% while
                                 # learning nothing.
    timeframe: str = "5min"
    session_only: bool = True   # restrict to the NY session
    session_start: str = "09:30"
    session_end: str = "16:00"

    def describe(self) -> str:
        return (
            f"lookback={self.lookback} bars ({self.lookback * 5}m), "
            f"horizon={self.horizon} bars ({self.horizon * 5}m), "
            f"expansion threshold={self.expansion_k}xATR, tf={self.timeframe}"
        )


# ---------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m bars -> higher timeframe. Defined here so this module stands alone."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _window_features(w: pd.DataFrame, atr: float, or_mid: float | None) -> dict:
    """Features for ONE lookback window. `w` ends at the sample bar."""
    high, low = w["high"].to_numpy(), w["low"].to_numpy()
    op, close = w["open"].to_numpy(), w["close"].to_numpy()
    rng = high - low
    body = np.abs(close - op)
    eps = 1e-9
    span = high.max() - low.min()

    # --- compression / clumping -------------------------------------
    range_ratio = float(rng.mean() / (span + eps))
    body_ratio = float(body.sum() / (rng.sum() + eps))

    overlaps = [
        max(0.0, min(high[i], high[i - 1]) - max(low[i], low[i - 1]))
        / (max(high[i], high[i - 1]) - min(low[i], low[i - 1]) + eps)
        for i in range(1, len(w))
    ]
    overlap_ratio = float(np.mean(overlaps)) if overlaps else 0.0

    close_dispersion = float(close.std() / (atr + eps))
    high_low_std = float(rng.std() / (rng.mean() + eps))

    # --- direction / cleanliness ------------------------------------
    diffs = np.diff(close)
    persistence = float(np.abs(diffs.sum()) / (np.abs(diffs).sum() + eps))
    net_move_atr = float((close[-1] - close[0]) / (atr + eps))

    upper = high - np.maximum(op, close)
    lower = np.minimum(op, close) - low
    wick_ratio = float((upper.sum() + lower.sum()) / (rng.sum() + eps))
    up_bar_frac = float((close > op).mean())

    if len(diffs) > 2 and diffs.std() > eps:
        ac = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
        return_autocorr = float(0.0 if np.isnan(ac) else ac)
    else:
        return_autocorr = 0.0

    # --- volatility --------------------------------------------------
    atr_norm = float(atr / (close[-1] + eps))
    first_half = rng[: len(rng) // 2].mean()
    second_half = rng[len(rng) // 2 :].mean()
    atr_change = float(second_half / (first_half + eps))
    true_range_norm = float(rng[-1] / (atr + eps))

    # --- volume ------------------------------------------------------
    if "volume" in w.columns:
        v = w["volume"].to_numpy(float)
        vol_ratio = float(v[-1] / (v.mean() + eps))
        vol_trend = float(v[len(v) // 2 :].mean() / (v[: len(v) // 2].mean() + eps))
    else:
        vol_ratio = vol_trend = 1.0

    # --- session context ---------------------------------------------
    ts = w.index[-1]
    minutes_from_open = float(ts.hour * 60 + ts.minute - (9 * 60 + 30))
    dist_from_or_mid = (
        float((close[-1] - or_mid) / (atr + eps)) if or_mid is not None else 0.0
    )

    return {
        "range_ratio": range_ratio,
        "body_ratio": body_ratio,
        "overlap_ratio": overlap_ratio,
        "close_dispersion": close_dispersion,
        "high_low_std": high_low_std,
        "persistence": persistence,
        "net_move_atr": net_move_atr,
        "wick_ratio": wick_ratio,
        "up_bar_frac": up_bar_frac,
        "return_autocorr": return_autocorr,
        "atr_norm": atr_norm,
        "atr_change": atr_change,
        "true_range_norm": true_range_norm,
        "vol_ratio": vol_ratio,
        "vol_trend": vol_trend,
        "minutes_from_open": minutes_from_open,
        "dist_from_or_mid": dist_from_or_mid,
    }


# ---------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------

def build_dataset(
    df_1m: pd.DataFrame,
    symbol: str,
    spec: DatasetSpec | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Turn 1-minute bars into a supervised table.

    Returns one row per sample: 17 feature columns, both labels, and metadata
    (timestamp, session date, symbol) kept for a time-ordered train/test split.
    """
    spec = spec or DatasetSpec()
    bars = resample_ohlcv(df_1m, spec.timeframe)
    if spec.session_only:
        bars = bars.between_time(spec.session_start, spec.session_end)

    rows = []
    dropped_short = 0

    for day, sess in bars.groupby(bars.index.normalize()):
        need = spec.lookback + spec.horizon + spec.atr_window
        if len(sess) < need:
            dropped_short += 1
            continue

        tr = _true_range(sess)
        atr = tr.rolling(spec.atr_window).mean()

        # opening-range midpoint: the 09:30 and 09:35 bars
        or_bars = sess.iloc[:2]
        or_mid = float((or_bars["high"].max() + or_bars["low"].min()) / 2)

        start = max(spec.lookback, spec.atr_window)
        stop = len(sess) - spec.horizon          # labels need H bars ahead

        for i in range(start, stop):
            a = float(atr.iloc[i])
            if not np.isfinite(a) or a <= 0:
                continue

            window = sess.iloc[i - spec.lookback : i + 1]     # ends AT bar i
            feats = _window_features(window, a, or_mid)

            # label: bars i+1 .. i+H  (strictly after the feature window)
            fwd = sess.iloc[i + 1 : i + 1 + spec.horizon]
            now = float(sess.iloc[i]["close"])
            fwd_close = float(fwd.iloc[-1]["close"])
            fwd_high = float(fwd["high"].max())
            fwd_low = float(fwd["low"].min())

            y_return = (fwd_close - now) / a
            excursion = max(fwd_high - now, now - fwd_low) / a
            y_expansion = int(excursion > spec.expansion_k)

            feats.update(
                {
                    "timestamp": sess.index[i],
                    "session_date": day,
                    "symbol": symbol,
                    "y_return": y_return,
                    "y_expansion": y_expansion,
                    "y_excursion": excursion,
                    "atr": a,
                    "close": now,
                }
            )
            rows.append(feats)

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("no samples built — check the input frame")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"  {symbol}: {len(data):,} samples  "
              f"{data['timestamp'].min().date()} -> {data['timestamp'].max().date()}")
        if dropped_short:
            print(f"    ({dropped_short} short sessions skipped)")
    return data


def build_combined(
    data_1m: dict,
    spec: DatasetSpec | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build and stack datasets for several instruments."""
    spec = spec or DatasetSpec()
    if verbose:
        print(f"Building dataset — {spec.describe()}")
    parts = [build_dataset(df, sym, spec, verbose) for sym, df in data_1m.items()]
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# split, inspection, IO
# ---------------------------------------------------------------------

def split_xy(
    data: pd.DataFrame,
    target: str = "y_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the label vector y."""
    if target not in ("y_return", "y_expansion", "y_excursion"):
        raise ValueError(f"unknown target: {target}")
    return data[FEATURE_NAMES].copy(), data[target].copy()


def time_split(
    data: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split (STEP 3).

    A random split would leak: neighbouring bars overlap in their lookback
    windows, so a shuffled test set contains near-duplicates of training rows and
    every model looks far better than it is. Splitting by time is the only
    honest option for market data.
    """
    cut_ts = data["timestamp"].quantile(1 - test_frac)
    train = data[data["timestamp"] <= cut_ts].copy()
    test = data[data["timestamp"] > cut_ts].copy()
    return train, test


def describe_dataset(data: pd.DataFrame) -> None:
    """Print the summary you need before trusting anything downstream."""
    print("=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"samples          : {len(data):,}")
    print(f"features (X)     : {len(FEATURE_NAMES)}")
    print(f"date range       : {data['timestamp'].min().date()} -> "
          f"{data['timestamp'].max().date()}")
    print(f"sessions         : {data['session_date'].nunique():,}")
    print(f"symbols          : {sorted(data['symbol'].unique())}")

    print("\nTARGETS (y)")
    print(f"  y_return       : mean {data['y_return'].mean():+.4f}  "
          f"std {data['y_return'].std():.4f}  "
          f"[{data['y_return'].min():.2f}, {data['y_return'].max():.2f}]")
    bal = data["y_expansion"].mean()
    print(f"  y_expansion    : {bal:.1%} expansion / {1 - bal:.1%} compression")
    print(f"  y_excursion    : mean {data['y_excursion'].mean():.3f} ATR")

    print("\nFEATURE STATS")
    stats = data[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
    print(stats.round(3).to_string())

    nan = data[FEATURE_NAMES].isna().sum().sum()
    inf = np.isinf(data[FEATURE_NAMES].to_numpy()).sum()
    print(f"\nNaNs: {nan}   Infs: {inf}")
    if nan or inf:
        print("  ^ must be cleaned before training")


def save_dataset(data: pd.DataFrame, path: str) -> None:
    data.to_parquet(path)
    print(f"saved {len(data):,} rows -> {path}")


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
